use poulpy_hal::layouts::CnvPVecLToBackendMut;
use poulpy_hal::layouts::CnvPVecLToBackendRef;
use poulpy_hal::layouts::CnvPVecRToBackendMut;
use poulpy_hal::layouts::CnvPVecRToBackendRef;
use poulpy_hal::layouts::VecZnxBigToBackendMut;
use poulpy_hal::layouts::VecZnxBigToBackendRef;
use poulpy_hal::layouts::VecZnxDftToBackendMut;
use std::cell::RefCell;
use std::collections::HashMap;

use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxBigAlloc,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxFillUniformSourceBackend, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, GaloisElement, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, VecZnx, VecZnxDftBackendMut,
    },
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};

use crate::default::linear_transformation::DiagonalProd;
use crate::layouts::GLWESecretSampling;
use crate::{
    EncryptionLayout, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEEncryptSk, GLWELinearTransformations,
    LinearTransformation, LinearTransformationBabySteps, LinearTransformationGiantStep, LinearTransformationLayout,
    LinearTransformationPrepared, LinearTransformationStrategy,
    layouts::{
        Base2K, Degree, Dsize, GGLWEInfos, GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyLayoutHelper,
        GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
        TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPrepared, PreparedDiagonal},
    },
    msb_mask_bottom_limb,
};

struct PrecisionDependentLayout {
    key: GLWEAutomorphismKeyLayout,
    threshold: TorusPrecision,
    low_dsize: Dsize,
}

impl GLWEAutomorphismKeyLayoutHelper<GLWEAutomorphismKeyLayout> for PrecisionDependentLayout {
    fn get_automorphism_key_layout_for(&self, _p: i64, k: TorusPrecision) -> crate::Result<(&GLWEAutomorphismKeyLayout, Dsize)> {
        Ok((
            &self.key,
            if k < self.threshold {
                self.low_dsize
            } else {
                self.key.effective_dsize()
            },
        ))
    }
}

/// A whole-chain representative can be dense at its high proxy precision even
/// though the same physical key is used through strided rows at a lower precision.
pub fn test_glwe_linear_transformation_bound_covers_lower_strided_key<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: CnvPVecAlloc<BE> + GLWELinearTransformations<BE>,
{
    let base2k = params.base2k;
    let rank = 1usize;
    let high_k = 16 * base2k;
    let low_k = 5 * base2k;
    let key = GLWEAutomorphismKeyLayout {
        n: module.n().into(),
        base2k: base2k.into(),
        dnum: 8usize.into(),
        k_aux: (2 * base2k + module.log_n()).into(),
        rank: rank.into(),
        dsize: 2usize.into(),
    };
    let helper = PrecisionDependentLayout {
        key,
        threshold: high_k.into(),
        low_dsize: Dsize(4),
    };
    let src_high = GLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: high_k.into(),
        rank: rank.into(),
    };
    let src_low = GLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: low_k.into(),
        rank: rank.into(),
    };
    let pt = GLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: (2 * base2k).into(),
        rank: 0usize.into(),
    };
    let layout = LinearTransformationLayout {
        indexes: vec![0, 1, 2, 3],
        slots: module.n(),
        strategy: LinearTransformationStrategy::Bsgs { giant_step: 2 },
    };
    let lt: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, &layout, &pt);

    let exact_lazy = module.glwe_eval_linear_transformation_tmp_bytes(&src_low, &src_low, &lt, &helper);
    let bound_lazy = module.glwe_eval_linear_transformation_bound_tmp_bytes(&src_high, &src_high, &pt, &helper.key);
    assert!(
        bound_lazy >= exact_lazy,
        "whole-chain lazy bound {bound_lazy} < lower-precision exact {exact_lazy}"
    );

    let dst_high = GLWELayout {
        n: module.n().into(),
        base2k: (base2k + 1).into(),
        k: high_k.into(),
        rank: rank.into(),
    };
    let dst_low = GLWELayout {
        n: module.n().into(),
        base2k: (base2k + 1).into(),
        k: low_k.into(),
        rank: rank.into(),
    };
    let exact_fallback = module.glwe_eval_linear_transformation_tmp_bytes(&dst_low, &src_low, &lt, &helper);
    let bound_fallback = module.glwe_eval_linear_transformation_bound_tmp_bytes(&dst_high, &src_high, &pt, &helper.key);
    assert!(
        bound_fallback >= exact_fallback,
        "whole-chain fallback bound {bound_fallback} < lower-precision exact {exact_fallback}"
    );
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
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let rank = 2usize;
    let in_base2k = params.base2k;
    let key_base2k = params.base2k;
    let k_in = 2 * in_base2k + 1;
    let dsize = 2;
    let dnum = k_in.div_ceil(key_base2k * dsize);

    let ct_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n.into(),
        base2k: in_base2k.into(),
        k: k_in.into(),
        rank: rank.into(),
    })
    .unwrap();
    let atk_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: key_base2k.into(),
        dnum: dnum.into(),
        k_aux: (dsize * key_base2k + module.log_n()).into(),
        rank: rank.into(),
        dsize: dsize.into(),
    })
    .unwrap();

    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    let mut source_xs = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);
    let mut source_xa = Source::new([5u8; 32]);
    let baby_steps = vec![0, 1, 3, 5];
    let product_size = ct_infos.size() + pt.size();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos)
            | module.glwe_automorphism_key_prepare_tmp_bytes(&atk_infos)
            | module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
            | module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &baby_steps, &atk_infos)
            | module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size())
            | module.cnv_apply_dft_tmp_bytes(0, product_size, ct_infos.size(), pt.size())
            | module.vec_znx_big_normalize_tmp_bytes(),
    );

    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    module.vec_znx_fill_uniform_source_backend(in_base2k, &mut vec_znx_backend_mut::<BE>(&mut pt.data), 0, &mut source_xa);
    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        &sk_prepared,
        &ct_infos,
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
            &atk_infos,
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

    let mut prepared_babies = LinearTransformationBabySteps::alloc(module, &baby_steps, &ct);
    module.glwe_prepare_linear_transformation_baby_steps(&mut prepared_babies, &ct, &atks, &mut scratch.borrow());
    assert_eq!(prepared_babies.baby_steps().collect::<Vec<_>>(), baby_steps);

    let mut right_prepared = module.cnv_pvec_right_alloc(1, pt.size());
    let pt_ref = <GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendRef<BE>>::to_backend_ref(&pt);
    module.cnv_prepare_right(
        &mut right_prepared.to_backend_mut(),
        &pt_ref.data,
        !0i64,
        &mut scratch.borrow(),
    );

    let mask = msb_mask_bottom_limb(ct.base2k().as_usize(), k_in);
    for &rot in &baby_steps {
        let mut expected: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct);
        if rot == 0 {
            module.glwe_copy(&mut expected, &ct);
        } else {
            let key = atks.get(&module.galois_element(rot)).unwrap();
            module.glwe_automorphism(&mut expected, &ct, key, &mut scratch.borrow());
        }

        let mut expected_prepared = module.cnv_pvec_left_alloc(rank + 1, expected.size());
        let expected_ref = <GLWE<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendRef<BE>>::to_backend_ref(&expected);
        module.cnv_prepare_left(
            &mut expected_prepared.to_backend_mut(),
            &expected_ref.data,
            mask,
            &mut scratch.borrow(),
        );

        let baby = prepared_babies.baby_step(rot);
        assert_eq!(
            baby.shape(),
            expected_prepared.shape(),
            "prepared baby rotation {rot} has wrong shape"
        );

        for col in 0..rank + 1 {
            let mut have: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(1, product_size);
            let mut want: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(1, product_size);
            let mut have_dft = module.vec_znx_dft_alloc(1, product_size);
            let mut want_dft = module.vec_znx_dft_alloc(1, product_size);
            let mut have_big = module.vec_znx_big_alloc(1, product_size);
            let mut want_big = module.vec_znx_big_alloc(1, product_size);
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

/// **Empty giant-step buckets are inert.** A transform carrying an empty bucket
/// at a nonzero rotation must evaluate exactly like the pruned transform and
/// must not consult the key map: with no key for that rotation, any lookup (or
/// the `automorphism_key_infos()` the planner would reach for) panics.
pub fn test_glwe_eval_linear_transformation_skips_empty_giant_steps<BE: crate::test_suite::noise::TestBackend>(
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
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let base2k = params.base2k;
    let k_in = 2 * base2k + 1;

    let ct_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_in.into(),
        rank: 1usize.into(),
    })
    .unwrap();

    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);
    let mut source_xa = Source::new([13u8; 32]);

    // Identity-only transform: one diagonal at index 0, so neither the baby
    // preparation nor the giant loop legitimately needs a key.
    let layout = LinearTransformationLayout {
        indexes: vec![0],
        slots: n,
        strategy: LinearTransformationStrategy::Direct,
    };
    let mut lt: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, &layout, &pt);

    // No keys at all: `automorphism_key_infos()` panics on an empty map, so any
    // planning step that counts the empty bucket as a live rotation aborts here,
    // sizing included.
    let keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
            | module.glwe_eval_linear_transformation_tmp_bytes(&ct_infos, &ct_infos, &lt, &keys)
            | module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size())
            | module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, lt.baby_steps(), &keys),
    );

    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);
    module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt.data), 0, &mut source_xa);
    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        &sk_prepared,
        &ct_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    {
        let pt_ref = <GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> as GLWEToBackendRef<BE>>::to_backend_ref(&pt);
        for gs in &mut lt.giant_steps {
            for d in &mut gs.diagonals {
                module.cnv_prepare_right(
                    &mut d.plaintext.cnv_mut().to_backend_mut(),
                    &pt_ref.data,
                    !0i64,
                    &mut scratch.borrow(),
                );
            }
        }
    }

    let mut babies = LinearTransformationBabySteps::alloc(module, lt.baby_steps(), &ct);
    module.glwe_prepare_linear_transformation_baby_steps(&mut babies, &ct, &keys, &mut scratch.borrow());

    let mut pruned: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_into(0, &mut pruned, &babies, &lt, &keys, &mut scratch.borrow());

    // Same transform plus an empty bucket at a nonzero rotation.
    lt.giant_steps.push(LinearTransformationGiantStep {
        rot: 3,
        diagonals: Vec::new(),
    });
    let mut with_empty: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    module.glwe_eval_linear_transformation_into(0, &mut with_empty, &babies, &lt, &keys, &mut scratch.borrow());

    assert_eq!(
        pruned, with_empty,
        "an empty giant-step bucket changed the linear transformation result"
    );
}

thread_local! {
    /// Giant rotations passed to [`SequentialOnlyDiagonal::accumulate_giant_prod`],
    /// in call order.
    static SEQUENTIAL_ONLY_CALLS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

/// Diagonal representation implementing only
/// [`DiagonalProd::accumulate_giant_prod`], so the batched call must go through
/// the trait default.
struct SequentialOnlyDiagonal;

impl LWEInfos for SequentialOnlyDiagonal {
    fn n(&self) -> Degree {
        Degree(0)
    }
    fn k(&self) -> TorusPrecision {
        TorusPrecision(0)
    }
    fn base2k(&self) -> Base2K {
        Base2K(1)
    }
    fn max_size(&self) -> usize {
        0
    }
}

impl<BE: Backend> DiagonalProd<BE> for SequentialOnlyDiagonal {
    fn accumulate_giant_prod<M>(
        _module: &M,
        _cnv_offset_hi: usize,
        _prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        _lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        _scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        SEQUENTIAL_ONLY_CALLS.with_borrow_mut(|calls| calls.push(gs.rot));
    }
}

/// `PreparedDiagonal::accumulate_giant_prods` matches the ordered single-giant
/// calls for batches of two and four, and the trait default runs in
/// result-index order.
///
/// The giants carry different baby subsets, two of them have their diagonal
/// vector reversed, and the first carries one baby twice, so an implementation
/// that zips the diagonal vectors by index or deduplicates babies differs.
pub fn test_glwe_prepared_giant_prods_match_sequential<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: CnvPVecAlloc<BE>
        + Convolution<BE>
        + VecZnxAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let base2k = params.base2k;
    let rank = 1usize;
    let cols = rank + 1;

    let ct_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: (2 * base2k).into(),
        rank: rank.into(),
    };
    let pt_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: (2 * base2k).into(),
        rank: 0usize.into(),
    };

    // Giant 0 takes babies {0, 1, 2}, giant 3 {0, 2}, giant 6 {0, 1}, giant 9
    // {0, 1, 2}: no two giants share a diagonal-vector index layout.
    let layout = LinearTransformationLayout {
        indexes: vec![0, 1, 2, 3, 5, 6, 7, 9, 10, 11],
        slots: n,
        strategy: LinearTransformationStrategy::Bsgs { giant_step: 3 },
    };
    let mut lt: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, &layout, &pt_infos);
    assert_eq!(lt.giant_steps.len(), 4);

    // One extra occurrence of baby 1 in giant 0, carved out of a throw-away
    // transform: a deduplicating batch implementation drops its contribution.
    let mut spare: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, &layout, &pt_infos);
    let mut extra = spare.giant_steps[0].diagonals.pop().unwrap();
    extra.baby = 1;
    lt.giant_steps[0].diagonals.push(extra);
    lt.giant_steps[1].diagonals.reverse();
    lt.giant_steps[3].diagonals.reverse();

    let mut source = Source::new([21u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.cnv_prepare_right_tmp_bytes(pt_infos.size(), pt_infos.size())
            | module.cnv_prepare_left_tmp_bytes(ct_infos.size(), ct_infos.size())
            | module.cnv_accumulate_dft_tmp_bytes(0, ct_infos.size() + pt_infos.size(), ct_infos.size(), pt_infos.size())
            | module.vec_znx_big_normalize_tmp_bytes(),
    );

    // Distinct random content per diagonal and per baby step: the comparison is
    // only meaningful if swapping or dropping an operand changes the result.
    let mut pt = module.vec_znx_alloc(1, pt_infos.size());
    for gs in &mut lt.giant_steps {
        for d in &mut gs.diagonals {
            module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt), 0, &mut source);
            module.cnv_prepare_right(
                &mut d.plaintext.cnv_mut().to_backend_mut(),
                &vec_znx_backend_ref::<BE>(&pt),
                !0i64,
                &mut scratch.borrow(),
            );
        }
    }

    let mut babies = LinearTransformationBabySteps::alloc(module, lt.baby_steps(), &ct_infos);
    let mut baby_data = module.vec_znx_alloc(cols, ct_infos.size());
    for (_, slot) in babies.baby_steps_mut() {
        for col in 0..cols {
            module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut baby_data), col, &mut source);
        }
        module.cnv_prepare_left(
            &mut slot.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&baby_data),
            !0i64,
            &mut scratch.borrow(),
        );
    }

    for batch in [2usize, 4] {
        let giant_steps: Vec<&LinearTransformationGiantStep<PreparedDiagonal<BE::OwnedBuf, BE>>> =
            lt.giant_steps.iter().take(batch).collect();

        for cnv_offset_hi in [0usize, 1] {
            let prod_size = ct_infos.size() + pt_infos.size() - cnv_offset_hi;
            let mut have: Vec<_> = (0..batch).map(|_| module.vec_znx_dft_alloc(cols, prod_size)).collect();
            let mut want: Vec<_> = (0..batch).map(|_| module.vec_znx_dft_alloc(cols, prod_size)).collect();
            let mut big = module.vec_znx_big_alloc(1, prod_size);

            {
                let mut prod_dfts: Vec<_> = have.iter_mut().map(|p| p.to_backend_mut()).collect();
                PreparedDiagonal::<BE::OwnedBuf, BE>::accumulate_giant_prods(
                    module,
                    cnv_offset_hi,
                    &mut prod_dfts,
                    &babies,
                    &giant_steps,
                    &mut scratch.borrow(),
                );
            }
            for (dst, gs) in want.iter_mut().zip(&giant_steps) {
                PreparedDiagonal::<BE::OwnedBuf, BE>::accumulate_giant_prod(
                    module,
                    cnv_offset_hi,
                    &mut dst.to_backend_mut(),
                    &babies,
                    gs,
                    &mut scratch.borrow(),
                );
            }

            for (lane, (have, want)) in have.iter_mut().zip(want.iter_mut()).enumerate() {
                for col in 0..cols {
                    let mut drained = [module.vec_znx_alloc(1, prod_size), module.vec_znx_alloc(1, prod_size)];
                    for (dst, src) in drained.iter_mut().zip([&mut *have, &mut *want]) {
                        module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut src.to_backend_mut(), col);
                        module.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BE>(dst),
                            base2k,
                            0,
                            0,
                            &big.to_backend_ref(),
                            base2k,
                            0,
                            &mut scratch.borrow(),
                        );
                    }
                    assert_eq!(
                        drained[0], drained[1],
                        "batched giant PROD != ordered single calls (batch={batch}, cnv_offset_hi={cnv_offset_hi}, \
                         lane={lane}, col={col})"
                    );
                }
            }
        }
    }

    // The trait default: a diagonal type defining only `accumulate_giant_prod`
    // is driven in result-index order.
    let prod_size = ct_infos.size() + pt_infos.size();
    let mut prod: Vec<_> = (0..3).map(|_| module.vec_znx_dft_alloc(cols, prod_size)).collect();
    let mut prod_dfts: Vec<_> = prod.iter_mut().map(|p| p.to_backend_mut()).collect();
    let sequential: Vec<LinearTransformationGiantStep<SequentialOnlyDiagonal>> = [3i64, 9, 5]
        .into_iter()
        .map(|rot| LinearTransformationGiantStep {
            rot,
            diagonals: Vec::new(),
        })
        .collect();
    let sequential: Vec<&LinearTransformationGiantStep<SequentialOnlyDiagonal>> = sequential.iter().collect();
    SEQUENTIAL_ONLY_CALLS.with_borrow_mut(|calls| calls.clear());
    SequentialOnlyDiagonal::accumulate_giant_prods(module, 0, &mut prod_dfts, &babies, &sequential, &mut scratch.borrow());
    SEQUENTIAL_ONLY_CALLS.with_borrow(|calls| {
        assert_eq!(
            calls.as_slice(),
            &[3, 9, 5],
            "the DiagonalProd default did not run the giant steps in result-index order"
        )
    });
}
