use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeAssignBackend},
    layouts::{FillUniform, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::convolution::bivariate_convolution_naive,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};
use rand::Rng;
use std::f64::consts::SQRT_2;

use crate::{
    EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEMulConst, GLWEMulPlain, GLWESub, GLWETensorDecrypt, GLWETensorKeyEncryptSk,
    GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        Dsize, GGLWEInfos, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESecretTensor,
        GLWESecretTensorFactory, GLWESecretTensorPrepared, GLWESecretTensorPreparedFactory, GLWETensor, GLWETensorKey,
        GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};

pub fn test_glwe_tensoring<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensoring<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWETensorDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + GLWESecretTensorFactory<BE>
        + VecZnxNormalize<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let out_base2k: usize = base2k - 2;
    let tsk_base2k: usize = base2k;
    let k: usize = 8 * base2k + 1;
    let k_tsk = k + tsk_base2k;

    for rank in 1_usize..=3 {
        let n: usize = module.n();

        let glwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let tsk_infos = EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: n.into(),
            base2k: tsk_base2k.into(),
            k: k_tsk.into(),
            rank: rank.into(),
            dnum: k.div_ceil(tsk_base2k).into(),
            dsize: Dsize(1),
        })
        .unwrap();

        let mut a: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut b: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res_tensor: GLWETensor<Vec<u8>> = module.glwe_tensor_alloc_from_infos(&glwe_out_infos);
        let mut res_relin: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos))
                .max(module.glwe_tensor_apply_tmp_bytes(&res_tensor, &a, &b))
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into()))
                .max(module.glwe_tensor_relinearize_tmp_bytes(&res_relin, &res_tensor, &tsk_infos)),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = module.glwe_secret_tensor_alloc(rank.into());
        module.glwe_secret_tensor_prepare(&mut sk_tensor, &sk, &mut crate::test_suite::scratch_host_arena(&mut scratch));

        let mut sk_tensor_prep: GLWESecretTensorPrepared<BE::OwnedBuf, BE> =
            module.glwe_secret_tensor_prepared_alloc(rank.into());
        module.glwe_secret_tensor_prepared_prepare(&mut sk_tensor_prep, &sk_tensor);

        let mut tsk: GLWETensorKey<Vec<u8>> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            &tsk_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::scratch_host_arena(&mut scratch),
        );

        let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut scratch.borrow());

        let scale: usize = 2 * in_base2k;

        let mut data = vec![0i64; n];
        for i in data.iter_mut() {
            *i = (source_xa.next_i64() & 7) - 4;
        }

        pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));

        let mut pt_want_base2k_in: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, pt_in.size());
        bivariate_convolution_naive::<_, BE>(
            module,
            in_base2k,
            2,
            &mut pt_want_base2k_in,
            0,
            pt_in.data(),
            0,
            pt_in.data(),
            0,
            &mut scratch.borrow(),
        );

        module.glwe_encrypt_sk(
            &mut a,
            &pt_in,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        module.glwe_encrypt_sk(
            &mut b,
            &pt_in,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        for res_offset in 0..scale {
            module.glwe_tensor_apply(
                scale + res_offset,
                &mut res_tensor,
                &a,
                a.max_k().as_usize(),
                &b,
                b.max_k().as_usize(),
                &mut scratch.borrow(),
            );

            module.glwe_tensor_decrypt(&res_tensor, &mut pt_have, &sk_dft, &sk_tensor_prep, &mut scratch.borrow());
            module.vec_znx_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                out_base2k,
                res_offset as i64,
                0,
                &vec_znx_backend_ref::<BE>(&pt_want_base2k_in),
                in_base2k,
                0,
                &mut scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign_backend(
                pt_tmp.base2k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_tmp.data),
                0,
                &mut scratch.borrow(),
            );

            let noise_have: f64 = pt_tmp.stats().std().log2();
            let noise_want = -((k - scale - res_offset - module.log_n()) as f64 - ((rank - 1) as f64) / SQRT_2);

            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);

            module.glwe_tensor_relinearize(
                &mut res_relin,
                &res_tensor,
                &tsk_prep,
                (res_tensor.size() + tsk_prep.dsize().as_usize()).min(tsk_prep.size()),
                &mut scratch.borrow(),
            );
            module.glwe_decrypt(&res_relin, &mut pt_have, &sk_dft, &mut scratch.borrow());

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign_backend(
                pt_tmp.base2k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_tmp.data),
                0,
                &mut scratch.borrow(),
            );

            // We can reuse the same noise bound because the relinearization noise (which is additive)
            // is much smaller than the tensoring noise (which is multiplicative)
            let noise_have: f64 = pt_tmp.stats().std().log2();
            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);
        }
    }
}

pub fn test_glwe_tensor_square<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensoring<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + GLWESecretTensorFactory<BE>
        + VecZnxNormalize<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let out_base2k: usize = base2k - 2;
    let tsk_base2k: usize = base2k;
    let k: usize = 8 * base2k + 1;

    for rank in 1_usize..=3 {
        let n: usize = module.n();

        let glwe_in_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let tsk_infos: GLWETensorKeyLayout = GLWETensorKeyLayout {
            n: n.into(),
            base2k: tsk_base2k.into(),
            k: (k + tsk_base2k).into(),
            rank: rank.into(),
            dnum: k.div_ceil(tsk_base2k).into(),
            dsize: Dsize(1),
        };

        let mut a: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res_square: GLWETensor<Vec<u8>> = module.glwe_tensor_alloc_from_infos(&glwe_out_infos);
        let mut res_tensor: GLWETensor<Vec<u8>> = module.glwe_tensor_alloc_from_infos(&glwe_out_infos);
        let mut res_relin_square: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut res_relin_tensor: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos))
                .max(module.glwe_tensor_square_apply_tmp_bytes(&res_square, &a))
                .max(module.glwe_tensor_apply_tmp_bytes(&res_tensor, &a, &a))
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into()))
                .max(module.glwe_tensor_relinearize_tmp_bytes(&res_relin_square, &res_square, &tsk_infos))
                .max(module.glwe_tensor_relinearize_tmp_bytes(&res_relin_tensor, &res_tensor, &tsk_infos)),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let tsk_enc_infos = EncryptionLayout::new_from_default_sigma(tsk_infos).unwrap();
        let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_in_infos).unwrap();
        let mut tsk: GLWETensorKey<Vec<u8>> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            &tsk_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::scratch_host_arena(&mut scratch),
        );

        let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut scratch.borrow());

        let scale: usize = 2 * in_base2k;

        let mut data = vec![0i64; n];
        for i in data.iter_mut() {
            *i = (source_xa.next_i64() & 7) - 4;
        }
        pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));
        module.glwe_encrypt_sk(
            &mut a,
            &pt_in,
            &sk_dft,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        for res_offset in 0..scale {
            module.glwe_tensor_square_apply(
                scale + res_offset,
                &mut res_square,
                &a,
                a.max_k().as_usize(),
                &mut scratch.borrow(),
            );
            module.glwe_tensor_apply(
                scale + res_offset,
                &mut res_tensor,
                &a,
                a.max_k().as_usize(),
                &a,
                a.max_k().as_usize(),
                &mut scratch.borrow(),
            );

            assert_eq!(res_square.data().raw(), res_tensor.data().raw());

            module.glwe_tensor_relinearize(
                &mut res_relin_square,
                &res_square,
                &tsk_prep,
                (res_square.size() + tsk_prep.dsize().as_usize()).min(tsk_prep.size()),
                &mut scratch.borrow(),
            );
            module.glwe_tensor_relinearize(
                &mut res_relin_tensor,
                &res_tensor,
                &tsk_prep,
                (res_tensor.size() + tsk_prep.dsize().as_usize()).min(tsk_prep.size()),
                &mut scratch.borrow(),
            );
            assert_eq!(res_relin_square.data().raw(), res_relin_tensor.data().raw());

            // Decrypt one side to ensure the square path remains functionally valid.
            module.glwe_decrypt(&res_relin_square, &mut pt_have, &sk_dft, &mut scratch.borrow());
            module.glwe_decrypt(&res_relin_tensor, &mut pt_want, &sk_dft, &mut scratch.borrow());
            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign_backend(
                pt_tmp.base2k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_tmp.data),
                0,
                &mut scratch.borrow(),
            );
            let noise_have: f64 = pt_tmp.stats().std().log2();
            assert!(noise_have <= -20.0, "{} > -20", noise_have);
        }
    }
}

pub fn test_glwe_mul_plain<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxNormalize<BE>
        + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k;
    let out_base2k: usize = base2k - 1;
    let k: usize = 8 * base2k + 1;

    for rank in 1_usize..=3 {
        let n: usize = module.n();

        let glwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let mut a: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_a: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_b: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc(in_base2k.into(), (2 * in_base2k).into());
        let mut pt_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos)),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let scale: usize = 2 * in_base2k;

        pt_b.data_mut().fill_uniform(17, &mut source_xa);
        pt_a.data_mut().fill_uniform(17, &mut source_xa);

        let mut pt_want_base2k_in: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, pt_a.size() + pt_b.size());
        bivariate_convolution_naive(
            module,
            in_base2k,
            2,
            &mut pt_want_base2k_in,
            0,
            pt_a.data(),
            0,
            pt_b.data(),
            0,
            &mut scratch.borrow(),
        );

        module.glwe_encrypt_sk(
            &mut a,
            &pt_a,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut scratch_cnv = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(&res, &a, &pt_b));

        for res_offset in 0..scale {
            module.glwe_mul_plain(
                scale + res_offset,
                &mut res,
                &a,
                a.max_k().as_usize(),
                &pt_b,
                pt_b.max_k().as_usize(),
                &mut scratch_cnv.borrow(),
            );

            module.glwe_decrypt(&res, &mut pt_have, &sk_dft, &mut scratch.borrow());
            module.vec_znx_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                out_base2k,
                res_offset as i64,
                0,
                &vec_znx_backend_ref::<BE>(&pt_want_base2k_in),
                in_base2k,
                0,
                &mut scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign_backend(
                pt_tmp.base2k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_tmp.data),
                0,
                &mut scratch.borrow(),
            );

            let noise_have: f64 = pt_tmp.stats().std().log2();
            let noise_want = -((k - scale - res_offset - module.log_n()) as f64 - ((rank - 1) as f64) / SQRT_2);

            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);
        }
    }
}

pub fn test_glwe_mul_const<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxNormalize<BE>
        + GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k;
    let out_base2k: usize = base2k;
    let k: usize = 8 * base2k + 1;
    let b_coeff: usize = 0;

    for rank in 1_usize..=3 {
        let n: usize = module.n();

        let glwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let mut a: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_a: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_b: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc(in_base2k.into(), (2 * in_base2k).into());
        let mut pt_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos))
                .max(module.glwe_mul_const_tmp_bytes(&res, &a, &pt_b)),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let scale: usize = 2 * in_base2k;

        pt_a.data_mut().fill_uniform(17, &mut source_xa);

        let mask = (1 << in_base2k) - 1;
        for j in 0..1 {
            let r = source_xa.next_u64() & mask;
            pt_b.data_mut().at_mut(0, j)[b_coeff] = ((r << (64 - 17)) as i64) >> (64 - 17);
        }

        let mut pt_want_base2k_in: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, pt_a.size() + pt_b.size());
        bivariate_convolution_naive(
            module,
            in_base2k,
            2,
            &mut pt_want_base2k_in,
            0,
            pt_a.data(),
            0,
            pt_b.data(),
            0,
            &mut scratch.borrow(),
        );

        module.glwe_encrypt_sk(
            &mut a,
            &pt_a,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        for res_offset in 0..scale {
            module.glwe_mul_const(scale + res_offset, &mut res, &a, &pt_b, b_coeff, &mut scratch.borrow());

            module.glwe_decrypt(&res, &mut pt_have, &sk_dft, &mut scratch.borrow());
            module.vec_znx_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                out_base2k,
                res_offset as i64,
                0,
                &vec_znx_backend_ref::<BE>(&pt_want_base2k_in),
                in_base2k,
                0,
                &mut scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign_backend(
                pt_tmp.base2k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_tmp.data),
                0,
                &mut scratch.borrow(),
            );

            let noise_have: f64 = pt_tmp.stats().std().log2();
            let noise_want = -((k - scale - res_offset - module.log_n()) as f64 - ((rank - 1) as f64) / SQRT_2);

            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);
        }
    }
}
