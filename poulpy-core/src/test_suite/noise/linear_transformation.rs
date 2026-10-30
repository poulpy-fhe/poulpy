use crate::test_suite::keys::TransformKeys;
use crate::test_suite::poisoned_scratch;
use crate::test_suite::refusal::{assert_refusal, glwe_state};
use crate::{GLWEBytesOf, ScratchArenaTakeCore};
use poulpy_hal::layouts::CnvPVecLToBackendRef;
use poulpy_hal::layouts::CnvPVecRToBackendMut;
use poulpy_hal::layouts::CnvPVecRToBackendRef;
use poulpy_hal::layouts::VecZnxBigToBackendMut;
use poulpy_hal::layouts::VecZnxBigToBackendRef;
use poulpy_hal::layouts::VecZnxDftToBackendMut;
use poulpy_hal::layouts::VecZnxDftToBackendRef;
use poulpy_hal::layouts::{CnvPVecLToBackendMut, DataView};
use std::collections::HashMap;

use poulpy_hal::{
    api::{
        CnvOps, CnvPVecAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxFillUniformSource, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, GaloisElement, HostDataMut, HostDataRef, Module, PrepareHint, ScratchOwned, VecZnx},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::api::GLWESecretSampling;
use crate::{
    GLWEAdd, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWELinearTransformations,
    GLWENoise, LinearTransformationBabySteps, LinearTransformationPrepared,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory},
    encryption::DEFAULT_SIGMA_XE,
    error::{CoreError, Result},
    layouts::{
        Dsize, GGLWEInfos, GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWELayout, GLWEPlaintext, GLWESecret,
        GLWETensorKey, GLWETensorKeyLayout, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LinearTransformation,
        LinearTransformationDiagonal, LinearTransformationGiantStep, LinearTransformationLayout, LinearTransformationPlan,
        LinearTransformationStrategy, ModuleCoreAlloc, SetK, TorusPrecision, k_digit_of,
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
            GLWESecretPrepared, GLWETensorKeyPrepared, PreparedDiagonal,
        },
    },
    noise::GGLWENoiseModel,
    test_suite::noise::{assert_refuses_uncovered_input, glwe_noise_checked},
};
use crate::{GLWEPreparedTensoring, GLWETensorKeyEncryptSk, api::GLWETensorKeyPreparedFactory};

/// The stored keys, with the rotations listed in `coarse_ps` answered through a
/// coarser `dsize`: one baby-step key set carrying two different layouts.
struct MixedDsize<'a, BE: Backend> {
    atks: &'a HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    coarse: TorusPrecision,
    coarse_ps: &'a [i64],
}

impl<BE: Backend> GetAutomorphismKey<BE> for MixedDsize<'_, BE> {
    fn lookup_automorphism_key(&self, p: i64, _k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let key = self.atks.get(&p).ok_or(CoreError::GGLWEKeyUse {
            op: "get_automorphism_key",
            detail: format!("no automorphism key for p={p}"),
        })?;
        if self.coarse_ps.contains(&p) {
            key.with_k_digit(self.coarse)
        } else {
            Ok(key.to_backend_ref())
        }
    }
}

/// The rotation at position `i` of `baby_steps` (the unrotated one aside)
/// answered by the keys of layout `i - 1`: one key set over several layouts.
struct MixedLayouts<'a, BE: Backend, G: Fn(i64) -> i64> {
    maps: &'a [HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>],
    baby_steps: &'a [i64],
    galois: G,
}

impl<BE: Backend, G: Fn(i64) -> i64> GetAutomorphismKey<BE> for MixedLayouts<'_, BE, G> {
    fn lookup_automorphism_key(&self, p: i64, _k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let position = self
            .baby_steps
            .iter()
            .filter(|&&rot| rot != 0)
            .position(|&rot| (self.galois)(rot) == p)
            .ok_or(CoreError::GGLWEKeyUse {
                op: "get_automorphism_key",
                detail: format!("no automorphism key for p={p}"),
            })?;
        let map = &self.maps[position % self.maps.len()];
        Ok(map[&p].to_backend_ref())
    }
}

pub fn test_glwe_hoisted_baby_rotations_match_automorphism<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + CnvPVecAlloc<BE>
        + CnvOps<BE>
        + GLWECopy<BE>
        + GLWEEncryptSk<BE>
        + GLWELinearTransformations<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let rank = 3usize;
    // Pinned rather than taken from `params`: at this radix the stored and the
    // coarsened key resolve to different product widths, which is what makes
    // the mixed key set below a real test of the scratch the query promises.
    let _ = params;
    let in_base2k: usize = 12;
    let key_base2k = in_base2k;
    let k_in = 4 * in_base2k + 1;
    let dsize = 2;

    let ct_infos = GLWELayout {
        n: n.into(),
        base2k: in_base2k.into(),
        k: k_in.into(),
        rank: rank.into(),
    };
    // Keys at whole-limb digits, at digits off a limb boundary, and at a third
    // width off it, for a key set that mixes all three.
    let layout_at = |k_digit: usize| GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: key_base2k.into(),
        // Rows enough that the key still covers the input when read at twice its
        // digit width, which the mixed key set below does: that read keeps one
        // stored digit as guard.
        dnum: (2 * k_in.div_ceil(2 * k_digit) + 1).into(),
        k_aux: (k_digit + module.log_n()).into(),
        rank: rank.into(),
        k_digit: (k_digit as u32).into(),
    };
    let widths = [dsize * key_base2k, dsize * key_base2k + 1, 3 * key_base2k + 1];
    let layouts: Vec<GLWEAutomorphismKeyLayout> = widths.iter().map(|&k_digit| layout_at(k_digit)).collect();
    // Each stored layout is also read at twice its digit width.
    let coarse: Vec<crate::layouts::GGLWELayout> = layouts[..2]
        .iter()
        .map(|layout| {
            layout
                .gglwe_layout()
                .at_k_digit(TorusPrecision(2 * layout.k_digit.0))
                .unwrap()
        })
        .collect();

    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    let mut source_xs = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);
    let mut source_xa = Source::new([5u8; 32]);
    let baby_steps = vec![0, 1, 3, 5];
    let product_size = ct_infos.size() + pt.size();

    let mut tmp_bytes = module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
        | module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size())
        | module.cnv_apply_dft_tmp_bytes(0, product_size, ct_infos.size(), pt.size())
        | module.vec_znx_big_normalize_tmp_bytes()
        | module.vec_znx_idft_apply_tmp_bytes();
    // The public query sizes one layout at a time, so a caller holding several
    // takes the max over them; each key set below runs on exactly its own max.
    let plan = LinearTransformationPlan {
        baby_steps: baby_steps.clone(),
        ..Default::default()
    };
    let mut stored_bytes = Vec::new();
    for layout in &layouts {
        tmp_bytes |= module.glwe_automorphism_key_encrypt_sk_tmp_bytes(layout)
            | module.glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, layout)
            | module.glwe_automorphism_key_prepare_tmp_bytes(layout);
        stored_bytes.push(module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &plan, layout));
    }
    let mut coarse_bytes = Vec::new();
    for layout in &coarse {
        tmp_bytes |= module.glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, layout);
        coarse_bytes.push(module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &plan, layout));
    }
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);

    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    module.vec_znx_fill_uniform_source(
        in_base2k,
        pt.k().as_usize(),
        &mut vec_znx_backend_mut::<BE>(&mut pt.data),
        0,
        &mut source_xa,
    );
    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        &sk_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    // Automorphism keys are indexed by Galois element throughout the engine.
    let mut maps: Vec<HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>> = Vec::new();
    for layout in &layouts {
        let mut atks = HashMap::new();
        for &rot in baby_steps.iter().filter(|&&rot| rot != 0) {
            let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_automorphism_key_alloc_from_infos(layout);
            module.glwe_automorphism_key_encrypt_sk(
                &mut atk,
                module.galois_element(rot),
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            let mut prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(layout);
            module.glwe_automorphism_key_prepare(&mut prepared, &atk, &mut scratch.borrow());
            atks.insert(module.galois_element(rot), prepared);
        }
        maps.push(atks);
    }

    // Over the stored keys of each of the first two layouts: all rotations as
    // stored, then half of them read through the coarser view.
    let half_coarse: Vec<i64> = baby_steps
        .iter()
        .filter(|&&rot| rot != 0)
        .skip(1)
        .map(|&rot| module.galois_element(rot))
        .collect();
    for (i, (atks, layout)) in maps.iter().zip(&layouts).take(2).enumerate() {
        for coarse_ps in [Vec::new(), half_coarse.clone()] {
            let keys = MixedDsize::<BE> {
                atks,
                coarse: TorusPrecision(2 * layout.k_digit.0),
                coarse_ps: &coarse_ps,
            };
            // The mixed set reads two layouts, so its budget is the max of their queries.
            let prep_bytes = if coarse_ps.is_empty() {
                stored_bytes[i]
            } else {
                stored_bytes[i].max(coarse_bytes[i])
            };
            assert_hoisted_babies_match(
                module,
                &keys,
                &ct,
                &pt,
                &baby_steps,
                rank,
                product_size,
                prep_bytes,
                &mut scratch,
            );
        }
    }

    // One rotation from each layout: whole-limb digits and two widths off a limb
    // boundary in one key set.
    let keys = MixedLayouts::<BE, _> {
        maps: &maps,
        baby_steps: &baby_steps,
        galois: |rot| module.galois_element(rot),
    };
    assert_hoisted_babies_match(
        module,
        &keys,
        &ct,
        &pt,
        &baby_steps,
        rank,
        product_size,
        stored_bytes.iter().copied().fold(0, usize::max),
        &mut scratch,
    );
}

/// Prepares the baby steps of `ct` through `keys` in `prep_bytes` of scratch and
/// checks every one against the automorphism of `ct` by its own key.
#[expect(
    clippy::too_many_arguments,
    reason = "the comparison takes the modules, the operands, the keys and their layouts"
)]
fn assert_hoisted_babies_match<BE: crate::test_suite::noise::TestBackend, K: GetAutomorphismKey<BE>>(
    module: &Module<BE>,
    keys: &K,
    ct: &GLWE<BE::OwnedBuf, BE::ZnxWord>,
    pt: &GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord>,
    baby_steps: &[i64],
    rank: usize,
    product_size: usize,
    prep_bytes: usize,
    scratch: &mut ScratchOwned<BE>,
) where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + CnvPVecAlloc<BE>
        + CnvOps<BE>
        + GLWECopy<BE>
        + GLWEEncryptSk<BE>
        + GLWELinearTransformations<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut prep_scratch = poisoned_scratch::<BE>(prep_bytes);
    let mut prepared_babies = LinearTransformationBabySteps::alloc(module, baby_steps, ct);
    module.glwe_prepare_linear_transformation_baby_steps(&mut prepared_babies, ct, keys, &mut prep_scratch.borrow());
    assert_eq!(prepared_babies.baby_steps().collect::<Vec<_>>(), baby_steps);

    let mut right_prepared = module.cnv_pvec_right_alloc(module.n(), 1, pt.size(), PrepareHint::Reuse);
    let pt_ref = <GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendRef<BE>>::to_backend_ref(pt);
    module.cnv_prepare_right(&mut right_prepared.to_backend_mut(), &pt_ref.data, &mut scratch.borrow());

    for &rot in baby_steps {
        let mut expected: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(ct);
        if rot == 0 {
            module.glwe_copy(&mut expected, ct, &mut scratch.borrow());
        } else {
            let p = module.galois_element(rot);
            let key = keys
                .get_automorphism_key(p, ct.k())
                .unwrap_or_else(|e| panic!("baby-step rotation {rot}: {e}"));
            module.glwe_automorphism(&mut expected, ct, &key, &mut scratch.borrow());
        }

        let mut expected_prepared = module.cnv_pvec_left_alloc(module.n(), rank + 1, expected.size(), PrepareHint::Reuse);
        let expected_ref = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendRef<BE>>::to_backend_ref(&expected);
        module.cnv_prepare_left(
            &mut expected_prepared.to_backend_mut(),
            &expected_ref.data,
            &mut scratch.borrow(),
        );

        let baby = prepared_babies.baby_step(rot);
        assert_eq!(
            baby.shape(),
            expected_prepared.shape(),
            "prepared baby rotation {rot} has wrong shape"
        );

        for col in 0..rank + 1 {
            let mut have: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(module.n(), 1, product_size);
            let mut want: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(module.n(), 1, product_size);
            let mut have_dft = module.vec_znx_dft_alloc(module.n(), 1, product_size);
            let mut want_dft = module.vec_znx_dft_alloc(module.n(), 1, product_size);
            let mut have_big = module.vec_znx_big_alloc(module.n(), 1, product_size);
            let mut want_big = module.vec_znx_big_alloc(module.n(), 1, product_size);
            let right_ref = right_prepared.to_backend_ref();

            module.cnv_apply_dft(
                0,
                &mut have_dft.to_backend_mut(),
                0,
                &baby.to_backend_ref(),
                col,
                &right_ref,
                0,
                1,
                &mut scratch.borrow(),
            );
            module.vec_znx_idft_apply(
                &mut have_big.to_backend_mut(),
                0,
                &have_dft.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut have),
                ct.base2k().as_usize(),
                ct.k().as_usize(),
                0,
                0,
                &have_big.to_backend_ref(),
                ct.base2k().as_usize(),
                0,
                &mut scratch.borrow(),
            );

            module.cnv_apply_dft(
                0,
                &mut want_dft.to_backend_mut(),
                0,
                &expected_prepared.to_backend_ref(),
                col,
                &right_ref,
                0,
                1,
                &mut scratch.borrow(),
            );
            module.vec_znx_idft_apply(
                &mut want_big.to_backend_mut(),
                0,
                &want_dft.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut want),
                ct.base2k().as_usize(),
                ct.k().as_usize(),
                0,
                0,
                &want_big.to_backend_ref(),
                ct.base2k().as_usize(),
                0,
                &mut scratch.borrow(),
            );

            assert_eq!(
                have, want,
                "prepared baby rotation {rot} column {col} differs from on-the-fly automorphism"
            );
        }
    }
}

/// A diagonal stored across more limbs than its `k` selects does not fit the
/// slot sized from that `k`: preparing it must panic instead of dropping the
/// limbs past the slot.
pub fn test_glwe_prepare_linear_transformation_rhs_rejects_wide_diagonal<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: params.base2k.into(),
        k: (3 * params.base2k).into(),
        rank: 0u32.into(),
    };
    let mut diagonal: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    diagonal.set_k(params.base2k.into());

    let schedule = LinearTransformationLayout {
        indexes: vec![0],
        slots: module.n() / 2,
        strategy: LinearTransformationStrategy::Bsgs { giant_step: 1 },
    };
    let mut prepared: LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>> =
        LinearTransformation::alloc_prepared(module, &schedule, &diagonal);
    let mut scratch = poisoned_scratch::<BE>(module.glwe_prepare_linear_transformation_rhs_tmp_bytes(&layout));
    let lt = LinearTransformation {
        baby_steps: vec![0],
        rotations: Default::default(),
        giant_steps: vec![LinearTransformationGiantStep {
            rot: 0,
            diagonals: vec![LinearTransformationDiagonal {
                baby: 0,
                plaintext: diagonal,
            }],
        }],
    };
    assert_refusal(
        "assertion `left == right` failed: linear transformation plaintext size does not match its effective precision\n  left: 1\n right: 3",
        &mut prepared,
        prepared_state::<BE>,
        |prepared| module.glwe_prepare_linear_transformation_rhs(prepared, &lt, &mut scratch.borrow()),
    );
}

/// Every prepared diagonal's bytes, shape and metadata.
fn prepared_state<BE: Backend>(
    lt: &LinearTransformationPrepared<BE>,
) -> Vec<(i64, i64, Vec<u8>, usize, usize, crate::layouts::GLWEPlaintextMeta)> {
    lt.giant_steps
        .iter()
        .flat_map(|step| {
            step.diagonals.iter().map(move |d| {
                (
                    step.rot,
                    d.baby,
                    BE::to_host_bytes(d.plaintext.cnv.data()),
                    d.plaintext.cnv.size(),
                    d.plaintext.k.as_usize(),
                    d.plaintext.plaintext_meta,
                )
            })
        })
        .collect()
}

/// Resident transforms summed on one giant schedule: one term is the single
/// transform bit for bit, and three decrypt to the sum of their single
/// transforms up to the giant key switches and roundings the sum saves.
pub fn test_glwe_linear_transformation_sum<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: CnvPVecAlloc<BE>
        + GLWEAdd<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + GLWELinearTransformations<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let base2k = params.base2k.min(12);
    let k = 4 * base2k + 1;
    let ct_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1u32.into(),
    };
    // One-limb integer diagonals, read at `cnv_offset = base2k`.
    let pt_infos = GLWELayout {
        k: base2k.into(),
        rank: 0u32.into(),
        ..ct_infos
    };
    let atk_infos = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: k.div_ceil(base2k).into(),
        k_aux: (base2k + module.log_n()).into(),
        rank: 1u32.into(),
        k_digit: base2k.into(),
    };
    // Babies {0, 1}, giants {0, 4}.
    let schedule = LinearTransformationLayout {
        indexes: vec![0, 1, 4, 5],
        slots: n / 2,
        strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
    };
    let plan = schedule.index();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos)
            | module.glwe_automorphism_key_prepare_tmp_bytes(&atk_infos)
            | module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
            | module.glwe_decrypt_tmp_bytes(&ct_infos)
            | module.glwe_noise_tmp_bytes(&ct_infos)
            | module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &plan, &atk_infos)
            | module.glwe_prepare_linear_transformation_rhs_tmp_bytes(&pt_infos),
    );
    let mut source_xs = Source::new([6u8; 32]);
    let mut source_xe = Source::new([7u8; 32]);
    let mut source_xa = Source::new([8u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut keys = HashMap::new();
    for rot in [1, 4] {
        let p = module.galois_element(rot);
        let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_encrypt_sk(&mut atk, p, &sk, &mut source_xe, &mut source_xa, &mut scratch.borrow());
        let mut prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_prepare(&mut prepared, &atk, &mut scratch.borrow());
        keys.insert(p, prepared);
    }

    // Each term has its own input and diagonals.
    let mut babies = Vec::new();
    let mut lts = Vec::new();
    for _ in 0..3 {
        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
        module.vec_znx_fill_uniform_source(base2k, k, &mut vec_znx_backend_mut::<BE>(&mut pt.data), 0, &mut source_xa);
        let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        module.glwe_encrypt_sk(
            &mut ct,
            &pt,
            &sk_prepared,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        let mut cache = LinearTransformationBabySteps::alloc(module, &plan.baby_steps, &ct);
        module.glwe_prepare_linear_transformation_baby_steps(&mut cache, &ct, &keys, &mut scratch.borrow());
        babies.push(cache);

        let mut lt = LinearTransformation {
            baby_steps: plan.baby_steps.clone(),
            rotations: plan.rotations,
            giant_steps: Vec::new(),
        };
        for (&rot, index) in plan.giant_steps.iter().zip(&plan.index) {
            let mut diagonals = Vec::new();
            for &baby in index {
                let mut diagonal: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&pt_infos);
                module.vec_znx_fill_uniform_source(
                    base2k,
                    base2k,
                    &mut vec_znx_backend_mut::<BE>(&mut diagonal.data),
                    0,
                    &mut source_xa,
                );
                diagonals.push(LinearTransformationDiagonal {
                    baby,
                    plaintext: diagonal,
                });
            }
            lt.giant_steps.push(LinearTransformationGiantStep { rot, diagonals });
        }
        let mut prepared: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, &schedule, &pt_infos);
        module.glwe_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow());
        lts.push(prepared);
    }
    let (babies, lts): (Vec<_>, Vec<_>) = (babies.iter().collect(), lts.iter().collect());

    let mut single: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_into(
        base2k,
        &mut single,
        babies[0],
        lts[0],
        &keys,
        &mut poisoned_scratch::<BE>(
            module.glwe_eval_linear_transformation_tmp_bytes(&ct_infos, &ct_infos, &pt_infos, &plan, &atk_infos, &atk_infos),
        )
        .borrow(),
    );
    let mut one: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_sum_into(
        base2k,
        &mut one,
        &babies[..1],
        &lts[..1],
        &keys,
        &mut poisoned_scratch::<BE>(
            module.glwe_eval_linear_transformation_sum_tmp_bytes(&ct_infos, &ct_infos, &pt_infos, &plan, &atk_infos, &atk_infos),
        )
        .borrow(),
    );
    assert_eq!(one, single, "a one-term sum is not the single linear transformation");
    assert!(one.is_canonical());

    let mut sum: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_sum_into(
        base2k,
        &mut sum,
        &babies,
        &lts,
        &keys,
        &mut poisoned_scratch::<BE>(
            module.glwe_eval_linear_transformation_sum_tmp_bytes(&ct_infos, &ct_infos, &pt_infos, &plan, &atk_infos, &atk_infos),
        )
        .borrow(),
    );
    let mut want = single;
    for (&cache, &lt) in babies[1..].iter().zip(&lts[1..]) {
        let mut term: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        module.glwe_eval_linear_transformation_into(
            base2k,
            &mut term,
            cache,
            lt,
            &keys,
            &mut poisoned_scratch::<BE>(
                module.glwe_eval_linear_transformation_tmp_bytes(&ct_infos, &ct_infos, &pt_infos, &plan, &atk_infos, &atk_infos),
            )
            .borrow(),
        );
        module.glwe_add_assign(&mut want, &term);
    }
    let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    module.glwe_decrypt(&want, &mut pt_want, &sk_prepared, &mut scratch.borrow());

    // Both sides share the inputs' noise. They differ by at most one key switch
    // per giant step for each term and for the sum, each with its rounding.
    let var_key_switch =
        atk_infos.var_noise_keyswitch(&ct_infos, &ct_infos, 0.5, 0.5, 0.0, DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE, 0.0);
    let noise_max = 0.5 * ((babies.len() + 1) as f64 * plan.giant_steps.len() as f64 * var_key_switch).log2() + 1.0;
    let noise_have = glwe_noise_checked(module, &sum, &pt_want, &sk_prepared, &mut scratch.borrow())
        .std()
        .log2();
    assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}");
}

/// A transformation evaluated with the keys of the power-of-two rotations decrypts to its
/// per-rotation evaluation up to the key switches that differ between them.
pub fn test_glwe_linear_transformation_powers_of_two<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: CnvPVecAlloc<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + GLWELinearTransformations<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let base2k = params.base2k.min(12);
    let k = 4 * base2k + 1;
    let ct_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1u32.into(),
    };
    let pt_infos = GLWELayout {
        k: base2k.into(),
        rank: 0u32.into(),
        ..ct_infos
    };
    let atk_infos = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: k.div_ceil(base2k).into(),
        k_aux: (base2k + module.log_n()).into(),
        rank: 1u32.into(),
        k_digit: base2k.into(),
    };
    // Diagonals -8..8: babies {0, 1, 2, 3} and giants {-8, -4, 0, 4}, a tree of two levels.
    let layout = |strategy| LinearTransformationLayout {
        indexes: (-8..8).collect(),
        slots: n / 2,
        strategy,
    };
    let schedules = [
        layout(LinearTransformationStrategy::Bsgs { giant_step: 4 }),
        layout(LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 4 }),
    ];
    let cyclotomic_order = poulpy_hal::layouts::CyclotomicOrder::cyclotomic_order(module);
    let mut elements: Vec<i64> = schedules
        .iter()
        .flat_map(|schedule| schedule.galois_elements(cyclotomic_order))
        .collect();
    // Two doublings and two levels against six rotations.
    assert_eq!(schedules[1].galois_elements(cyclotomic_order).len(), 4);
    assert_eq!(schedules[0].galois_elements(cyclotomic_order).len(), 6);
    elements.sort_unstable();
    elements.dedup();

    let work = module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos)
        | module.glwe_automorphism_key_prepare_tmp_bytes(&atk_infos)
        | module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
        | module.glwe_decrypt_tmp_bytes(&ct_infos)
        | module.glwe_noise_tmp_bytes(&ct_infos)
        | module.glwe_prepare_linear_transformation_rhs_tmp_bytes(&pt_infos)
        | schedules
            .iter()
            .map(|schedule| {
                let plan = schedule.index();
                module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &plan, &atk_infos)
                    | module
                        .glwe_eval_linear_transformation_tmp_bytes(&ct_infos, &ct_infos, &pt_infos, &plan, &atk_infos, &atk_infos)
            })
            .fold(0, usize::max);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(work);
    let mut source_xs = Source::new([9u8; 32]);
    let mut source_xe = Source::new([10u8; 32]);
    let mut source_xa = Source::new([11u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut keys = HashMap::new();
    for p in elements {
        let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_encrypt_sk(&mut atk, p, &sk, &mut source_xe, &mut source_xa, &mut scratch.borrow());
        let mut prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_prepare(&mut prepared, &atk, &mut scratch.borrow());
        keys.insert(p, prepared);
    }

    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    module.vec_znx_fill_uniform_source(base2k, k, &mut vec_znx_backend_mut::<BE>(&mut pt.data), 0, &mut source_xa);
    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        &sk_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    // The same diagonals under both schedules.
    let plan = schedules[0].index();
    let mut diagonals: HashMap<(i64, i64), GLWE<BE::OwnedBuf, BE::ZnxWord>> = HashMap::new();
    for (&rot, index) in plan.giant_steps.iter().zip(&plan.index) {
        for &baby in index {
            let mut diagonal: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&pt_infos);
            module.vec_znx_fill_uniform_source(
                base2k,
                base2k,
                &mut vec_znx_backend_mut::<BE>(&mut diagonal.data),
                0,
                &mut source_xa,
            );
            diagonals.insert((rot, baby), diagonal);
        }
    }
    let mut results = Vec::new();
    for schedule in &schedules {
        let plan = schedule.index();
        let mut lt = LinearTransformation {
            baby_steps: plan.baby_steps.clone(),
            rotations: plan.rotations,
            giant_steps: Vec::new(),
        };
        for (&rot, index) in plan.giant_steps.iter().zip(&plan.index) {
            let diagonals = index
                .iter()
                .map(|&baby| LinearTransformationDiagonal {
                    baby,
                    plaintext: diagonals[&(rot, baby)].clone(),
                })
                .collect();
            lt.giant_steps.push(LinearTransformationGiantStep { rot, diagonals });
        }
        let mut prepared: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, schedule, &pt_infos);
        module.glwe_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow());
        let mut cache = LinearTransformationBabySteps::alloc_for(module, &prepared, &ct);
        module.glwe_prepare_linear_transformation_baby_steps(&mut cache, &ct, &keys, &mut scratch.borrow());
        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        module.glwe_eval_linear_transformation_into(base2k, &mut res, &cache, &prepared, &keys, &mut scratch.borrow());

        // A cache carved from scratch takes its byte count less at most the alignment of its last operand, and
        // gives the same ciphertext.
        let plan = prepared.index();
        let cache_bytes = module.linear_transformation_baby_steps_bytes_of(&plan, &ct);
        let mut carved_scratch: ScratchOwned<BE> = ScratchOwned::alloc(cache_bytes + work);
        let arena = carved_scratch.borrow();
        let available = arena.available();
        let (mut carved, mut rest) = arena.take_linear_transformation_baby_steps_scratch(module, &plan, &ct);
        let taken = available - rest.available();
        assert!(
            taken <= cache_bytes && cache_bytes - taken < BE::SCRATCH_ALIGN,
            "the carve took {taken} bytes of {cache_bytes}"
        );
        assert_eq!(carved.baby_steps().collect::<Vec<_>>(), plan.baby_steps);
        module.glwe_prepare_linear_transformation_baby_steps(&mut carved, &ct, &keys, &mut rest);
        let mut from_carved: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        module.glwe_eval_linear_transformation_into(base2k, &mut from_carved, &carved, &prepared, &keys, &mut rest);
        assert_eq!(from_carved, res, "a carved baby-step cache gave another ciphertext");
        results.push(res);
    }
    assert!(results[1] != results[0], "the two schedules gave the same ciphertext");
    let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    module.glwe_decrypt(&results[0], &mut pt_want, &sk_prepared, &mut scratch.borrow());

    // Babies 1 and 2 are the same switch of the input on both sides; baby 3 differs by three
    // switches, each read by four integer diagonals of `n` uniform coefficients of `base2k`
    // bits. The giant steps differ by the three switches of each side.
    let var_key_switch =
        atk_infos.var_noise_keyswitch(&ct_infos, &ct_infos, 0.5, 0.5, 0.0, DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE, 0.0);
    let var_diagonal = n as f64 * (2.0f64).powi(2 * base2k as i32) / 12.0;
    let noise_max = 0.5 * ((3.0 * 4.0 * var_diagonal + 6.0) * var_key_switch).log2() + 3.0;
    let noise_have = glwe_noise_checked(module, &results[1], &pt_want, &sk_prepared, &mut scratch.borrow())
        .std()
        .log2();
    assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}");
}

/// Summed transforms must share one giant schedule.
pub fn test_glwe_linear_transformation_sum_rejects_mismatched_schedule<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    eval_mismatched_sum(
        params,
        module,
        vec![0, 2],
        1,
        0,
        1,
        "summed linear transformation 1 does not share the first one's giant schedule",
    );
}

/// Summed transforms must share one diagonal layout and width.
pub fn test_glwe_linear_transformation_sum_rejects_mismatched_diagonal_width<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    eval_mismatched_sum(
        params,
        module,
        vec![0, 1],
        2,
        0,
        1,
        "summed linear transformation 1 does not share the first one's diagonal layout, width and scale",
    );
}

/// Summed transforms must share one diagonal scale.
pub fn test_glwe_linear_transformation_sum_rejects_mismatched_diagonal_scale<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    eval_mismatched_sum(
        params,
        module,
        vec![0, 1],
        1,
        19,
        1,
        "summed linear transformation 1 does not share the first one's diagonal layout, width and scale",
    );
}

/// Summed transforms must share one source layout.
pub fn test_glwe_linear_transformation_sum_rejects_mismatched_source<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    eval_mismatched_sum(
        params,
        module,
        vec![0, 1],
        1,
        0,
        2,
        "summed linear transformation 1 does not share the first one's source layout",
    );
}

/// A sum needs a term.
pub fn test_glwe_linear_transformation_sum_rejects_no_terms<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    eval_without_keys(params, module, &[], &[], true, "linear transformation sum has no terms");
}

/// A sum takes as many transforms as baby caches.
pub fn test_glwe_linear_transformation_sum_rejects_unequal_lengths<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies, lt) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    eval_without_keys(
        params,
        module,
        &[&babies, &babies],
        &[&lt],
        true,
        "assertion `left == right` failed: linear transformation sum: babies and lts differ in length\n  left: 2\n right: 1",
    );
}

/// A sum's first term needs a giant step.
pub fn test_glwe_linear_transformation_sum_rejects_first_term_without_giant_steps<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies_a, mut lt_a) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (babies_b, lt_b) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    lt_a.giant_steps.clear();
    eval_without_keys(
        params,
        module,
        &[&babies_a, &babies_b],
        &[&lt_a, &lt_b],
        true,
        "linear transformation has no non-empty giant steps",
    );
}

/// A transform needs a giant step.
pub fn test_glwe_linear_transformation_rejects_no_giant_steps<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies, mut lt) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    lt.giant_steps.clear();
    eval_without_keys(
        params,
        module,
        &[&babies],
        &[&lt],
        false,
        "linear transformation has no non-empty giant steps",
    );
}

/// Every giant step of a transform needs a diagonal, checked before the
/// fallback writes `res` at the first one.
pub fn test_glwe_linear_transformation_rejects_empty_giant_step<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies, mut lt) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    lt.giant_steps[1].diagonals.clear();
    eval_without_keys(
        params,
        module,
        &[&babies],
        &[&lt],
        false,
        "giant step 1 of linear transformation 0 has no diagonals",
    );
}

/// Every giant step of every summed transform needs a diagonal, checked
/// before the fallback writes `res` at the first one.
pub fn test_glwe_linear_transformation_sum_rejects_empty_giant_step<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies_a, lt_a) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (babies_b, mut lt_b) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    lt_b.giant_steps[1].diagonals.clear();
    eval_without_keys(
        params,
        module,
        &[&babies_a, &babies_b],
        &[&lt_a, &lt_b],
        true,
        "giant step 1 of linear transformation 1 has no diagonals",
    );
}

/// Every baby rotation a summed transform reads must be in its cache, checked
/// before the fallback writes `res` at the first giant step.
pub fn test_glwe_linear_transformation_sum_rejects_missing_baby_step<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies_a, lt_a) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (babies_b, mut lt_b) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    lt_b.giant_steps[1].diagonals[0].baby = 1;
    eval_without_keys(
        params,
        module,
        &[&babies_a, &babies_b],
        &[&lt_a, &lt_b],
        true,
        "missing prepared baby-step rotation 1 of linear transformation 1",
    );
}

/// A summed transform's cache must hold the rotations it reads, checked before
/// its layout is compared with the first term's.
pub fn test_glwe_linear_transformation_sum_rejects_empty_cache<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies_a, lt_a) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (_, lt_b) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let empty = LinearTransformationBabySteps::alloc(module, &[], &limbs_layout(params, module, 1, 1));
    eval_without_keys(
        params,
        module,
        &[&babies_a, &empty],
        &[&lt_a, &lt_b],
        true,
        "missing prepared baby-step rotation 0 of linear transformation 1",
    );
}

/// Every giant step of a transform must start with a diagonal as wide as its
/// first one, checked before the fallback writes `res` at the first giant step.
pub fn test_glwe_linear_transformation_rejects_uneven_giant_step_width<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies, mut lt) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (_, mut wide) = direct_term(params, module, vec![0, 1], 2, 0, 1);
    std::mem::swap(&mut lt.giant_steps[1].diagonals, &mut wide.giant_steps[1].diagonals);
    eval_without_keys(
        params,
        module,
        &[&babies],
        &[&lt],
        false,
        "giant step 1 of linear transformation 0 is not as wide as the first diagonal",
    );
}

/// Every giant step of every summed transform must start with a diagonal as
/// wide as the first transform's first one, beyond the first diagonals the sum
/// compares, checked before the fallback writes `res` at the first giant step.
pub fn test_glwe_linear_transformation_sum_rejects_uneven_giant_step_width<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies_a, lt_a) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (babies_b, mut lt_b) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (_, mut wide) = direct_term(params, module, vec![0, 1], 2, 0, 1);
    std::mem::swap(&mut lt_b.giant_steps[1].diagonals, &mut wide.giant_steps[1].diagonals);
    eval_without_keys(
        params,
        module,
        &[&babies_a, &babies_b],
        &[&lt_a, &lt_b],
        true,
        "giant step 1 of linear transformation 1 is not as wide as the first diagonal",
    );
}

/// Sums a `Direct` transform over `[0, 1]`, with one-limb diagonals at scale 0
/// and a one-limb source, and one over `indexes`, with `pt_limbs`-limb
/// diagonals at scale `log_scale` and a `ct_limbs`-limb source.
fn eval_mismatched_sum<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
    indexes: Vec<i64>,
    pt_limbs: usize,
    log_scale: usize,
    ct_limbs: usize,
    message: &str,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies_a, lt_a) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (babies_b, lt_b) = direct_term(params, module, indexes, pt_limbs, log_scale, ct_limbs);
    eval_without_keys(params, module, &[&babies_a, &babies_b], &[&lt_a, &lt_b], true, message);
}

/// A `Direct` transform over `indexes` (giant steps at those rotations, each
/// with one diagonal at baby rotation 0), with `pt_limbs`-limb diagonals at
/// scale `log_scale`, and the cache of baby rotation 0 of a `ct_limbs`-limb source.
fn direct_term<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
    indexes: Vec<i64>,
    pt_limbs: usize,
    log_scale: usize,
    ct_limbs: usize,
) -> (LinearTransformationBabySteps<BE>, LinearTransformationPrepared<BE>)
where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
{
    let schedule = LinearTransformationLayout {
        indexes,
        slots: module.n() / 2,
        strategy: LinearTransformationStrategy::Direct,
    };
    let mut lt: LinearTransformationPrepared<BE> =
        LinearTransformation::alloc_prepared(module, &schedule, &limbs_layout(params, module, pt_limbs, 0));
    lt.set_plaintext_meta(crate::layouts::GLWEPlaintextMeta {
        scale: crate::layouts::Scale::Log(log_scale),
        ..Default::default()
    });
    let babies = LinearTransformationBabySteps::alloc(module, &[0], &limbs_layout(params, module, ct_limbs, 1));
    (babies, lt)
}

/// Evaluates the sum of the terms `(babies[t], lts[t])`, or the first alone if `!sum`, into a
/// one-limb `res`. There are no keys and no scratch, so a rejection must panic
/// before any key is read or `res` is written.
fn eval_without_keys<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformationPrepared<BE>],
    sum: bool,
    message: &str,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&limbs_layout(params, module, 1, 1));
    let mut scratch = poisoned_scratch::<BE>(0);
    // Limb offset 1: a one-limb source by a one- or two-limb diagonal gives one product limb either way.
    let cnv_offset = 2 * params.base2k;
    assert_refusal(message, &mut res, glwe_state::<BE>, |res| {
        if sum {
            module.glwe_eval_linear_transformation_sum_into(cnv_offset, res, babies, lts, &keys, &mut scratch.borrow());
        } else {
            module.glwe_eval_linear_transformation_into(cnv_offset, res, babies[0], lts[0], &keys, &mut scratch.borrow());
        }
    });
}

/// A `limbs`-limb GLWE layout of rank `rank` at the test's `base2k`.
fn limbs_layout<BE: Backend>(params: &TestParams, module: &Module<BE>, limbs: usize, rank: u32) -> GLWELayout {
    GLWELayout {
        n: module.n().into(),
        base2k: params.base2k.into(),
        k: (limbs * params.base2k).into(),
        rank: rank.into(),
    }
}

/// A transformation whose diagonals are ciphertexts, prepared by `glwe_prepare_linear_transformation_rhs`, decrypts to
/// the dot product of each giant step rotated and summed, alone or in a sum, on both schedules and key radixes. Keys
/// without a tensor key or with one too narrow for the products, a cache allocated for plaintext diagonals and summed
/// caches of other widths are refused before anything is written.
pub fn test_glwe_linear_transformation_ciphertext_diagonals<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: CnvPVecAlloc<BE>
        + CnvOps<BE>
        + GLWEAdd<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + GLWELinearTransformations<BE>
        + GLWENoise<BE>
        + GLWEPreparedTensoring<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + crate::GLWENormalize<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let size = k.div_ceil(base2k);
    let scale = 2 * base2k;
    let ct_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1u32.into(),
    };
    // Keys of one-limb digits at radix `base2k`, covering `k`.
    let atk_layout = |base2k: usize| GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: k.div_ceil(base2k).into(),
        k_aux: (base2k + module.log_n()).into(),
        rank: 1u32.into(),
        k_digit: base2k.into(),
    };
    let tsk_layout = |base2k: usize| GLWETensorKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: k.div_ceil(base2k).into(),
        k_aux: (base2k + module.log_n()).into(),
        rank: 1u32.into(),
        k_digit: k_digit_of(base2k, Dsize(1)),
    };
    let (atk_infos, tsk_infos) = (atk_layout(base2k), tsk_layout(base2k));
    let (atk_low_infos, tsk_low_infos) = (atk_layout(base2k - 1), tsk_layout(base2k - 1));
    // Diagonals 0, 1, 4 and 5: babies {0, 1} and giants {0, 4}, a giant per rotation or a tree, or one giant step.
    let layout = |strategy| LinearTransformationLayout {
        indexes: vec![0, 1, 4, 5],
        slots: n / 2,
        strategy,
    };
    let schedules = [
        layout(LinearTransformationStrategy::Bsgs { giant_step: 4 }),
        layout(LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 4 }),
        layout(LinearTransformationStrategy::Bsgs { giant_step: n / 2 }),
    ];
    let cyclotomic_order = poulpy_hal::layouts::CyclotomicOrder::cyclotomic_order(module);
    let mut elements: Vec<i64> = schedules
        .iter()
        .flat_map(|schedule| schedule.galois_elements(cyclotomic_order))
        .chain([module.galois_element(4)])
        .collect();
    elements.sort_unstable();
    elements.dedup();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos)
            | module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_low_infos)
            | module.glwe_automorphism_key_prepare_tmp_bytes(&atk_infos)
            | module.glwe_automorphism_key_prepare_tmp_bytes(&atk_low_infos)
            | module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_infos)
            | module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_low_infos)
            | module.glwe_tensor_key_prepare_tmp_bytes(&tsk_infos)
            | module.glwe_tensor_key_prepare_tmp_bytes(&tsk_low_infos)
            | module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
            | module.glwe_decrypt_tmp_bytes(&ct_infos)
            | module.glwe_noise_tmp_bytes(&ct_infos)
            | module.glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, &atk_infos)
            | module.glwe_prepare_linear_transformation_rhs_tmp_bytes(&ct_infos)
            | schedules
                .iter()
                .map(|schedule| {
                    module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &schedule.index(), &atk_infos)
                })
                .fold(0, usize::max),
    );
    let mut source_xs = Source::new([12u8; 32]);
    let mut source_xe = Source::new([13u8; 32]);
    let mut source_xa = Source::new([14u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    // The automorphism keys of `elements` and the tensor key, at the layouts given.
    let keys_at = |atk_infos: &GLWEAutomorphismKeyLayout,
                   tsk_infos: &GLWETensorKeyLayout,
                   source_xe: &mut Source,
                   source_xa: &mut Source,
                   scratch: &mut ScratchOwned<BE>| {
        let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_automorphism_key_alloc_from_infos(atk_infos);
        let mut atks: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
        for &p in &elements {
            module.glwe_automorphism_key_encrypt_sk(&mut atk, p, &sk, source_xe, source_xa, &mut scratch.borrow());
            let mut prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(atk_infos);
            module.glwe_automorphism_key_prepare(&mut prepared, &atk, &mut scratch.borrow());
            atks.insert(p, prepared);
        }
        let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(tsk_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            source_xe,
            source_xa,
            &mut crate::test_suite::noise::scratch_host_arena(scratch),
        );
        let mut tsk_prepared: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(tsk_infos);
        module.glwe_tensor_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());
        (atks, tsk_prepared)
    };
    let (atks, tsk_prepared) = keys_at(&atk_infos, &tsk_infos, &mut source_xe, &mut source_xa, &mut scratch);
    let (atks_low, tsk_low) = keys_at(&atk_low_infos, &tsk_low_infos, &mut source_xe, &mut source_xa, &mut scratch);
    let keys = TransformKeys {
        automorphism: &atks,
        tensor: &tsk_prepared,
    };
    let keys_low = TransformKeys {
        automorphism: &atks_low,
        tensor: &tsk_low,
    };

    let encrypt = |source_xe: &mut Source, source_xa: &mut Source, scratch: &mut ScratchOwned<BE>| {
        let data: Vec<i64> = (0..n).map(|_| (source_xa.next_i64() & 7) - 4).collect();
        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
        pt.encode_vec_i64(&data, TorusPrecision(scale as u32));
        let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        module.glwe_encrypt_sk(&mut ct, &pt, &sk_prepared, source_xe, source_xa, &mut scratch.borrow());
        ct
    };
    // A transform of encrypted diagonals of `schedule`, and its preparation.
    let transform = |schedule: &LinearTransformationLayout,
                     source_xe: &mut Source,
                     source_xa: &mut Source,
                     scratch: &mut ScratchOwned<BE>| {
        let plan = schedule.index();
        let mut lt = LinearTransformation {
            baby_steps: plan.baby_steps.clone(),
            rotations: plan.rotations,
            giant_steps: Vec::new(),
        };
        for (&rot, index) in plan.giant_steps.iter().zip(&plan.index) {
            let diagonals = index
                .iter()
                .map(|&baby| LinearTransformationDiagonal {
                    baby,
                    plaintext: encrypt(source_xe, source_xa, scratch),
                })
                .collect();
            lt.giant_steps.push(LinearTransformationGiantStep { rot, diagonals });
        }
        let mut prepared: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, schedule, &ct_infos);
        module.glwe_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow());
        (lt, prepared)
    };
    let res_k = ct_infos.k;
    let dot_bytes = module.glwe_mul_prepared_relinearize_sum_tmp_bytes(&ct_infos, size, size, ct_infos.k, &tsk_infos);
    // Each giant step's dot product at the conversion offset `offset`, rotated and added.
    let dot_products = |offset: usize,
                        cache: &LinearTransformationBabySteps<BE>,
                        lt: &LinearTransformationPrepared<BE>,
                        scratch: &mut ScratchOwned<BE>| {
        let mut want: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        let mut term: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        let diagonal = lt.first_diagonal_plaintext().unwrap();
        for (g, gs) in lt.giant_steps.iter().enumerate() {
            let out = if g == 0 { &mut want } else { &mut term };
            module.glwe_mul_prepared_relinearize_sum(
                offset,
                out,
                res_k,
                gs.diagonals.iter().map(|d| cache.baby_step(d.baby)),
                size,
                cache.k(),
                gs.diagonals.iter().map(|d| d.plaintext.cnv()),
                diagonal.size(),
                diagonal.k(),
                &tsk_prepared,
                &mut poisoned_scratch::<BE>(dot_bytes).borrow(),
            );
            if gs.rot != 0 {
                let key = atks.get_automorphism_key(module.galois_element(gs.rot), res_k).unwrap();
                module.glwe_automorphism_assign(out, &key, &mut scratch.borrow());
            }
            if g > 0 {
                module.glwe_add_assign(&mut want, &term);
            }
        }
        want
    };
    let a = encrypt(&mut source_xe, &mut source_xa, &mut scratch);
    let eval_bytes = |plan: &LinearTransformationPlan, tsk_infos: &GLWETensorKeyLayout, atk_infos: &GLWEAutomorphismKeyLayout| {
        module.glwe_eval_linear_transformation_tmp_bytes(&ct_infos, &ct_infos, &ct_infos, plan, tsk_infos, atk_infos)
    };
    // The destination of the evaluations with other keys and of the refused calls.
    let mut out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);

    for (s, schedule) in schedules.iter().enumerate() {
        let plan = schedule.index();
        let (lt_ct, lt) = transform(schedule, &mut source_xe, &mut source_xa, &mut scratch);
        let mut cache = LinearTransformationBabySteps::alloc_for(module, &lt, &a);
        module.glwe_prepare_linear_transformation_baby_steps(&mut cache, &a, &atks, &mut scratch.borrow());

        let mut have: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        module.glwe_eval_linear_transformation_into(
            scale,
            &mut have,
            &cache,
            &lt,
            &keys,
            &mut poisoned_scratch::<BE>(eval_bytes(&plan, &tsk_infos, &atk_infos)).borrow(),
        );
        assert!(have.is_canonical());
        let want = dot_products(scale, &cache, &lt, &mut scratch);
        if lt.giant_steps.len() == 1 {
            // The offset's low bits, below a limb, are kept as the dot product's own normalization keeps them.
            for offset in [scale, scale + 3] {
                module.glwe_eval_linear_transformation_into(
                    offset,
                    &mut have,
                    &cache,
                    &lt,
                    &keys,
                    &mut poisoned_scratch::<BE>(eval_bytes(&plan, &tsk_infos, &atk_infos)).borrow(),
                );
                assert!(
                    have.data() == dot_products(offset, &cache, &lt, &mut scratch).data(),
                    "schedule {s}: one giant step at rotation 0 is not its dot product at offset {offset}"
                );
            }
            continue;
        }
        // The two share the products; they differ by the rounding of each product before its rotation, and by the
        // key switch of each giant step.
        let var_key_switch =
            atk_infos.var_noise_keyswitch(&ct_infos, &ct_infos, 0.5, 0.5, 0.0, DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE, 0.0);
        let noise_max = 0.5 * (2.0 * plan.giant_steps.len() as f64 * var_key_switch).log2() + 1.0;
        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
        module.glwe_decrypt(&want, &mut pt_want, &sk_prepared, &mut scratch.borrow());
        let noise_have = glwe_noise_checked(module, &have, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        assert!(
            noise_have <= noise_max,
            "schedule {s}: noise_have: {noise_have} > noise_max: {noise_max}"
        );

        // Keys at another radix take the tensor products, whose tensor rounding and relinearization no longer cancel.
        let var_relinearize = |tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>| {
            let noise = tsk.noise().unwrap();
            crate::reference::noise::var_noise_glwe_mul_relinearize(
                tsk,
                noise.secret_distribution().coefficient_second_moment(n).unwrap(),
                noise.body().variance_at(tsk.k()),
                noise.masks().iter().map(|mask| mask.variance_at(tsk.k())).sum(),
                0.0,
                0,
                k,
                size * base2k,
                k,
            )
        };
        let pairs = lt.giant_steps[0].diagonals.len() as f64;
        let var_products = pairs * var_relinearize(&tsk_low).1 + var_relinearize(&tsk_prepared).0;
        let var_key_switch_low =
            atk_low_infos.var_noise_keyswitch(&ct_infos, &ct_infos, 0.5, 0.5, 0.0, DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE, 0.0);
        let giants = plan.giant_steps.len() as f64;
        let noise_max_low = 0.5 * (giants * (var_products + var_key_switch + var_key_switch_low)).log2() + 1.0;
        module.glwe_eval_linear_transformation_into(
            scale,
            &mut out,
            &cache,
            &lt,
            &keys_low,
            &mut poisoned_scratch::<BE>(eval_bytes(&plan, &tsk_low_infos, &atk_low_infos)).borrow(),
        );
        let noise_low = glwe_noise_checked(module, &out, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        assert!(
            noise_low <= noise_max_low,
            "schedule {s}, keys at base2k {}: noise_have: {noise_low} > noise_max: {noise_max_low}",
            base2k - 1
        );

        if s == 0 {
            // A sum: one term is the single transform, two the sum of the single ones.
            let sum_bytes = module
                .glwe_eval_linear_transformation_sum_tmp_bytes(&ct_infos, &ct_infos, &ct_infos, &plan, &tsk_infos, &atk_infos);
            let mut one: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
            module.glwe_eval_linear_transformation_sum_into(
                scale,
                &mut one,
                &[&cache],
                &[&lt],
                &keys,
                &mut poisoned_scratch::<BE>(sum_bytes).borrow(),
            );
            assert_eq!(one, have, "a one-term sum is not the single transformation");
            let b = encrypt(&mut source_xe, &mut source_xa, &mut scratch);
            let (_, lt_b) = transform(schedule, &mut source_xe, &mut source_xa, &mut scratch);
            let mut cache_b = LinearTransformationBabySteps::alloc_for(module, &lt_b, &b);
            module.glwe_prepare_linear_transformation_baby_steps(&mut cache_b, &b, &atks, &mut scratch.borrow());
            let mut sum: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
            module.glwe_eval_linear_transformation_sum_into(
                scale,
                &mut sum,
                &[&cache, &cache_b],
                &[&lt, &lt_b],
                &keys,
                &mut poisoned_scratch::<BE>(sum_bytes).borrow(),
            );
            let mut single_b: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
            module.glwe_eval_linear_transformation_into(
                scale,
                &mut single_b,
                &cache_b,
                &lt_b,
                &keys,
                &mut poisoned_scratch::<BE>(eval_bytes(&plan, &tsk_infos, &atk_infos)).borrow(),
            );
            module.glwe_add_assign(&mut have, &single_b);
            module.glwe_decrypt(&have, &mut pt_want, &sk_prepared, &mut scratch.borrow());
            let noise_have = glwe_noise_checked(module, &sum, &pt_want, &sk_prepared, &mut scratch.borrow())
                .std()
                .log2();
            assert!(
                noise_have <= noise_max,
                "sum: noise_have: {noise_have} > noise_max: {noise_max}"
            );

            // Keys without a tensor key.
            assert_refusal(
                "get_tensor_key: a map of automorphism keys holds no tensor key",
                &mut out,
                glwe_state::<BE>,
                |res| {
                    module.glwe_eval_linear_transformation_into(
                        scale,
                        res,
                        &cache,
                        &lt,
                        &atks,
                        &mut poisoned_scratch::<BE>(eval_bytes(&plan, &tsk_infos, &atk_infos)).borrow(),
                    )
                },
            );
            // A tensor key one digit short of the products' width.
            let narrow_tsk_infos = GLWETensorKeyLayout {
                dnum: (size - 1).into(),
                ..tsk_infos
            };
            let narrow_tsk: GLWETensorKeyPrepared<BE::OwnedBuf, BE> =
                module.alloc_tensor_key_prepared_from_infos(&narrow_tsk_infos);
            assert_refuses_uncovered_input("the products took a tensor key one digit short of their width", || {
                module.glwe_eval_linear_transformation_into(
                    scale,
                    &mut out,
                    &cache,
                    &lt,
                    &TransformKeys {
                        automorphism: &atks,
                        tensor: &narrow_tsk,
                    },
                    &mut poisoned_scratch::<BE>(eval_bytes(&plan, &narrow_tsk_infos, &atk_infos)).borrow(),
                )
            });
            // Ciphertext diagonals into a cache allocated for plaintext ones.
            let mut proxy: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(
                module,
                schedule,
                &GLWELayout {
                    rank: 0u32.into(),
                    ..ct_infos
                },
            );
            assert_refusal(
                "assertion `left == right` failed: linear transformation diagonal rank does not match prepared cache\n  left: 2\n right: 1",
                &mut proxy,
                prepared_state::<BE>,
                |proxy| {
                    module.glwe_prepare_linear_transformation_rhs(
                        proxy,
                        &lt_ct,
                        &mut poisoned_scratch::<BE>(module.glwe_prepare_linear_transformation_rhs_tmp_bytes(&ct_infos)).borrow(),
                    )
                },
            );
            // A second term whose cache records a source of another width.
            cache_b.set_k(TorusPrecision(k as u32 - 1));
            assert_refusal(
                "every baby cache must hold the rank of res and the first cache's width",
                &mut out,
                glwe_state::<BE>,
                |res| {
                    module.glwe_eval_linear_transformation_sum_into(
                        scale,
                        res,
                        &[&cache, &cache_b],
                        &[&lt, &lt_b],
                        &keys,
                        &mut poisoned_scratch::<BE>(sum_bytes).borrow(),
                    )
                },
            );
        }
    }
}
