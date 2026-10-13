use poulpy_hal::layouts::CnvPVecLToBackendMut;
use poulpy_hal::layouts::CnvPVecLToBackendRef;
use poulpy_hal::layouts::CnvPVecRToBackendMut;
use poulpy_hal::layouts::CnvPVecRToBackendRef;
use poulpy_hal::layouts::VecZnxBigToBackendMut;
use poulpy_hal::layouts::VecZnxBigToBackendRef;
use poulpy_hal::layouts::VecZnxDftToBackendMut;
use std::collections::HashMap;

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxFillUniformSource, VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, GaloisElement, HostDataMut, HostDataRef, Module, PrepareHint, ScratchOwned, VecZnx},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::layouts::GLWESecretSampling;
use crate::{
    GLWEAdd, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWELinearTransformations,
    GLWENoise, LinearTransformationBabySteps, LinearTransformationPrepared,
    encryption::DEFAULT_SIGMA_XE,
    error::{CoreError, Result},
    layouts::{
        Dsize, GGLWEInfos, GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWELayout, GLWEPlaintext, GLWESecret,
        GLWESecretPreparedFactory, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LinearTransformation,
        LinearTransformationDiagonal, LinearTransformationGiantStep, LinearTransformationLayout, LinearTransformationStrategy,
        ModuleCoreAlloc, SetK, TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedFactory,
            GLWEAutomorphismKeyPreparedToBackendRef, GLWESecretPrepared, PreparedDiagonal,
        },
    },
    noise::GGLWENoiseModel,
    test_suite::noise::glwe_noise_checked,
};

/// The stored keys, with the rotations listed in `coarse_ps` answered through a
/// coarser `dsize`: one baby-step key set carrying two different layouts.
struct MixedDsize<'a, BE: Backend> {
    atks: &'a HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    coarse: Dsize,
    coarse_ps: &'a [i64],
}

impl<BE: Backend> GetAutomorphismKey<BE> for MixedDsize<'_, BE> {
    fn lookup_automorphism_key(&self, p: i64, _k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let key = self.atks.get(&p).ok_or(CoreError::GGLWEKeyUse {
            op: "get_automorphism_key",
            detail: format!("no automorphism key for p={p}"),
        })?;
        if self.coarse_ps.contains(&p) {
            key.with_dsize(self.coarse)
        } else {
            Ok(key.to_backend_ref())
        }
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
        + Convolution<BE>
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
        + VecZnxIdftApplyTmpA<BE>
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
    // Wide enough that the keys still expose a digit when read at twice their
    // stored `dsize`, which the mixed key set below does.
    let k_in = 4 * in_base2k + 1;
    let dsize = 2;
    let dnum = k_in.div_ceil(key_base2k * dsize);

    let ct_infos = GLWELayout {
        n: n.into(),
        base2k: in_base2k.into(),
        k: k_in.into(),
        rank: rank.into(),
    };
    let atk_infos = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: key_base2k.into(),
        dnum: dnum.into(),
        k_aux: (dsize * key_base2k + module.log_n()).into(),
        rank: rank.into(),
        dsize: dsize.into(),
    };

    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    let mut source_xs = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);
    let mut source_xa = Source::new([5u8; 32]);
    let baby_steps = vec![0, 1, 3, 5];
    let product_size = ct_infos.size() + pt.size();

    let coarse_dsize = Dsize(2 * dsize as u32);
    let coarse_infos = atk_infos.gglwe_layout().at_dsize(coarse_dsize).unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos)
            | module.glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, &coarse_infos)
            | module.glwe_automorphism_key_prepare_tmp_bytes(&atk_infos)
            | module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
            | module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &atk_infos)
            | module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size())
            | module.cnv_apply_dft_tmp_bytes(0, product_size, ct_infos.size(), pt.size())
            | module.vec_znx_big_normalize_tmp_bytes(),
    );

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

    let mut atks = HashMap::new();
    for &rot in baby_steps.iter().filter(|&&rot| rot != 0) {
        let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_encrypt_sk(
            &mut atk,
            module.galois_element(rot),
            &sk,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        let mut prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_prepare(&mut prepared, &atk, &mut scratch.borrow());
        // Automorphism keys are indexed by Galois element throughout the engine.
        atks.insert(module.galois_element(rot), prepared);
    }

    // Two key sets over the same stored keys: all rotations at the stored
    // layout, then half of them read through a coarser `dsize`. The public
    // query sizes one layout at a time, so a caller holding several takes the
    // max over them; both sets must fit in exactly that budget.
    let prep_bytes = module
        .glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &atk_infos)
        .max(module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &coarse_infos));
    let coarse_sets: [Vec<i64>; 2] = [
        Vec::new(),
        baby_steps
            .iter()
            .filter(|&&rot| rot != 0)
            .skip(1)
            .map(|&rot| module.galois_element(rot))
            .collect(),
    ];

    for coarse_ps in coarse_sets {
        let keys = MixedDsize::<BE> {
            atks: &atks,
            coarse: coarse_dsize,
            coarse_ps: &coarse_ps,
        };
        let mut prep_scratch: ScratchOwned<BE> = ScratchOwned::alloc(prep_bytes);
        let mut prepared_babies = LinearTransformationBabySteps::alloc(module, &baby_steps, &ct);
        module.glwe_prepare_linear_transformation_baby_steps(&mut prepared_babies, &ct, &keys, &mut prep_scratch.borrow());
        assert_eq!(prepared_babies.baby_steps().collect::<Vec<_>>(), baby_steps);

        let mut right_prepared = module.cnv_pvec_right_alloc(module.n(), 1, pt.size(), PrepareHint::Reuse);
        let pt_ref = <GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendRef<BE>>::to_backend_ref(&pt);
        module.cnv_prepare_right(&mut right_prepared.to_backend_mut(), &pt_ref.data, &mut scratch.borrow());

        for &rot in &baby_steps {
            let mut expected: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct);
            if rot == 0 {
                module.glwe_copy(&mut expected, &ct, &mut scratch.borrow());
            } else {
                let p = module.galois_element(rot);
                let key = keys
                    .get_automorphism_key(p, ct.k())
                    .unwrap_or_else(|e| panic!("baby-step rotation {rot}: {e}"));
                module.glwe_automorphism(&mut expected, &ct, &key, &mut scratch.borrow());
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
                    &mut scratch.borrow(),
                );
                module.vec_znx_idft_apply_tmpa(&mut have_big.to_backend_mut(), 0, &mut have_dft.to_backend_mut(), 0);
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
                    &mut scratch.borrow(),
                );
                module.vec_znx_idft_apply_tmpa(&mut want_big.to_backend_mut(), 0, &mut want_dft.to_backend_mut(), 0);
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
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_prepare_linear_transformation_rhs_tmp_bytes(&layout));
    let lt = LinearTransformation {
        baby_steps: vec![0],
        giant_steps: vec![LinearTransformationGiantStep {
            rot: 0,
            diagonals: vec![LinearTransformationDiagonal {
                baby: 0,
                plaintext: diagonal,
            }],
        }],
    };
    module.glwe_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow());
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
        dsize: 1u32.into(),
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
            | module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &atk_infos)
            | module.glwe_prepare_linear_transformation_rhs_tmp_bytes(&pt_infos)
            | module.glwe_eval_linear_transformation_sum_tmp_bytes(&ct_infos, &ct_infos, &pt_infos, &atk_infos),
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
    let terms: Vec<_> = babies.iter().zip(&lts).collect();

    let mut single: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_into(base2k, &mut single, terms[0].0, terms[0].1, &keys, &mut scratch.borrow());
    let mut one: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_sum_into(base2k, &mut one, &terms[..1], &keys, &mut scratch.borrow());
    assert_eq!(one, single, "a one-term sum is not the single linear transformation");
    assert!(one.is_canonical());

    let mut sum: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_sum_into(base2k, &mut sum, &terms, &keys, &mut scratch.borrow());
    let mut want = single;
    for &(cache, lt) in &terms[1..] {
        let mut term: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
        module.glwe_eval_linear_transformation_into(base2k, &mut term, cache, lt, &keys, &mut scratch.borrow());
        module.glwe_add_assign(&mut want, &term);
    }
    let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    module.glwe_decrypt(&want, &mut pt_want, &sk_prepared, &mut scratch.borrow());

    // Both sides share the inputs' noise. They differ by at most one key switch
    // per giant step for each term and for the sum, each with its rounding.
    let var_key_switch =
        atk_infos.var_noise_keyswitch(&ct_infos, &ct_infos, 0.5, 0.5, 0.0, DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE, 0.0);
    let noise_max = 0.5 * ((terms.len() + 1) as f64 * plan.giant_steps.len() as f64 * var_key_switch).log2() + 1.0;
    let noise_have = glwe_noise_checked(module, &sum, &pt_want, &sk_prepared, &mut scratch.borrow())
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
    eval_mismatched_sum(params, module, vec![0, 2], 1, 0, 1);
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
    eval_mismatched_sum(params, module, vec![0, 1], 2, 0, 1);
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
    eval_mismatched_sum(params, module, vec![0, 1], 1, 19, 1);
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
    eval_mismatched_sum(params, module, vec![0, 1], 1, 0, 2);
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
    eval_without_keys(params, module, &[], true);
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
    eval_without_keys(params, module, &[(&babies_a, &lt_a), (&babies_b, &lt_b)], true);
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
    eval_without_keys(params, module, &[(&babies, &lt)], false);
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
    eval_without_keys(params, module, &[(&babies, &lt)], false);
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
    eval_without_keys(params, module, &[(&babies_a, &lt_a), (&babies_b, &lt_b)], true);
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
    eval_without_keys(params, module, &[(&babies_a, &lt_a), (&babies_b, &lt_b)], true);
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
    eval_without_keys(params, module, &[(&babies, &lt)], false);
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
    eval_without_keys(params, module, &[(&babies_a, &lt_a), (&babies_b, &lt_b)], true);
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
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (babies_a, lt_a) = direct_term(params, module, vec![0, 1], 1, 0, 1);
    let (babies_b, lt_b) = direct_term(params, module, indexes, pt_limbs, log_scale, ct_limbs);
    eval_without_keys(params, module, &[(&babies_a, &lt_a), (&babies_b, &lt_b)], true);
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
    lt.set_log_scale(log_scale);
    let babies = LinearTransformationBabySteps::alloc(module, &[0], &limbs_layout(params, module, ct_limbs, 1));
    (babies, lt)
}

/// Evaluates the sum of `terms`, or its first term alone if `!sum`, into a
/// one-limb `res`. There are no keys and no scratch, so a rejection must panic
/// before any key is read or `res` is written.
fn eval_without_keys<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
    terms: &[(&LinearTransformationBabySteps<BE>, &LinearTransformationPrepared<BE>)],
    sum: bool,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWELinearTransformations<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&limbs_layout(params, module, 1, 1));
    let mut scratch = ScratchOwned::<BE>::alloc(0);
    if sum {
        module.glwe_eval_linear_transformation_sum_into(0, &mut res, terms, &keys, &mut scratch.borrow());
    } else {
        let (babies, lt) = terms[0];
        module.glwe_eval_linear_transformation_into(0, &mut res, babies, lt, &keys, &mut scratch.borrow());
    }
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
