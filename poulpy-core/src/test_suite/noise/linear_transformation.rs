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
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxFillUniformSourceBackend, VecZnxIdftApplyTmpA,
    },
    layouts::{GaloisElement, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::layouts::GLWESecretSampling;
use crate::{
    EncryptionLayout, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEEncryptSk, GLWELinearTransformations,
    LinearTransformation, LinearTransformationBabySteps, LinearTransformationGiantStep, LinearTransformationLayout,
    LinearTransformationPrepared, LinearTransformationStrategy,
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory,
        GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPrepared},
    },
    msb_mask_bottom_limb,
};

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
            | module.glwe_eval_linear_transformation_tmp_bytes(&ct_infos, &ct_infos, &ct_infos, &atk_infos)
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
    let dsize = 2;
    let dnum = k_in.div_ceil(base2k * dsize);

    let ct_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_in.into(),
        rank: 1usize.into(),
    })
    .unwrap();
    let atk_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: dnum.into(),
        k_aux: (dsize * base2k + module.log_n()).into(),
        rank: 1usize.into(),
        dsize: dsize.into(),
    })
    .unwrap();

    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ct_infos);
    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ct_infos);
    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);
    let mut source_xa = Source::new([13u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_encrypt_sk_tmp_bytes(&ct_infos)
            | module.glwe_eval_linear_transformation_tmp_bytes(&ct_infos, &ct_infos, &ct_infos, &atk_infos)
            | module.cnv_prepare_right_tmp_bytes(pt.size(), pt.size())
            | module.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct_infos, &atk_infos),
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

    // Identity-only transform: one diagonal at index 0, so neither the baby
    // preparation nor the giant loop legitimately needs a key.
    let layout = LinearTransformationLayout {
        indexes: vec![0],
        slots: n,
        strategy: LinearTransformationStrategy::Direct,
    };
    let mut lt: LinearTransformationPrepared<BE> = LinearTransformation::alloc_prepared(module, &layout, &pt);
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

    // No keys at all: `automorphism_key_infos()` panics on an empty map, so any
    // planning step that counts the empty bucket as a live rotation aborts here.
    let keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
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
