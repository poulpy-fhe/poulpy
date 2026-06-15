use std::collections::HashMap;

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxFillUniformSourceBackend, VecZnxIdftApplyTmpA,
    },
    layouts::{
        CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, GaloisElement, HostDataMut,
        HostDataRef, Module, ScratchOwned, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut,
    },
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::{
    EncryptionLayout, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEEncryptSk, GLWELinearTransformations,
    LinearTransformationBabySteps,
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory,
        GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPrepared},
    },
    msb_mask_bottom_limb,
};

pub fn test_glwe_hoisted_baby_rotations_match_automorphism<BE: crate::test_suite::TestBackend>(
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
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
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
    let k_ksk = k_in + key_base2k * dsize;

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
        k: k_ksk.into(),
        rank: rank.into(),
        dnum: dnum.into(),
        dsize: dsize.into(),
    })
    .unwrap();

    let mut ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&ct_infos);
    let mut pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&ct_infos);
    let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&ct_infos);
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

    sk.fill_ternary_prob(0.5, &mut source_xs);
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
        let mut atk: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
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
    module.glwe_prepare_linear_transformation_lhs(
        &mut prepared_babies,
        &ct,
        k_in,
        &atks,
        atk_infos.size(),
        &mut scratch.borrow(),
    );
    assert_eq!(prepared_babies.baby_steps().collect::<Vec<_>>(), baby_steps);

    let mut right_prepared = module.cnv_pvec_right_alloc(1, pt.size());
    let pt_ref = <GLWEPlaintext<Vec<u8>> as GLWEToBackendRef<BE>>::to_backend_ref(&pt);
    module.cnv_prepare_right(
        &mut right_prepared.to_backend_mut(),
        &pt_ref.data,
        !0i64,
        &mut scratch.borrow(),
    );

    let mask = msb_mask_bottom_limb(ct.base2k().as_usize(), k_in);
    for &rot in &baby_steps {
        let mut expected: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(&ct);
        if rot == 0 {
            module.glwe_copy(&mut expected, &ct);
        } else {
            let key = atks.get(&module.galois_element(rot)).unwrap();
            module.glwe_automorphism(&mut expected, &ct, key, atk_infos.size(), &mut scratch.borrow());
        }

        let mut expected_prepared = module.cnv_pvec_left_alloc(rank + 1, expected.size());
        let expected_ref = <GLWE<BE::OwnedBuf> as GLWEToBackendRef<BE>>::to_backend_ref(&expected);
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
            let mut have: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, product_size);
            let mut want: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, product_size);
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
