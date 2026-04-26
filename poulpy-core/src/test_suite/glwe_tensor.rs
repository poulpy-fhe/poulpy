use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxFillUniform, VecZnxNormalize,
        VecZnxNormalizeAssign,
    },
    layouts::{DeviceBuf, FillUniform, Module, Scratch, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
    test_suite::convolution::bivariate_convolution_naive,
};
use rand::Rng;
use std::f64::consts::SQRT_2;

use crate::{
    EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEMulConst, GLWEMulPlain, GLWESub, GLWETensorDecrypt, GLWETensorKeyEncryptSk,
    GLWETensoring, ScratchTakeCore,
    layouts::{
        Dsize, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESecretTensor, GLWESecretTensorFactory,
        GLWESecretTensorPrepared, GLWESecretTensorPreparedFactory, GLWETensor, GLWETensorKey, GLWETensorKeyLayout,
        GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos, TorusPrecision, prepared::GLWESecretPrepared,
    },
};

pub fn test_glwe_tensoring<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWETensoring<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + VecZnxFillUniform
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + VecZnxNormalizeAssign<BE>
        + GLWESecretTensorFactory<BE>
        + VecZnxCopy
        + VecZnxNormalize<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
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

        let mut a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut b: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut res_tensor: GLWETensor<Vec<u8>> = GLWETensor::alloc_from_infos(&glwe_out_infos);
        let mut res_relin: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

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

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let mut sk_tensor: GLWESecretTensor<Vec<u8>> = GLWESecretTensor::alloc(module.n().into(), rank.into());
        module.glwe_secret_tensor_prepare(&mut sk_tensor, &sk, scratch.borrow());

        let mut sk_tensor_prep: GLWESecretTensorPrepared<DeviceBuf<BE>, BE> =
            module.glwe_secret_tensor_prepared_alloc(rank.into());
        module.glwe_secret_tensor_prepared_prepare(&mut sk_tensor_prep, &sk_tensor);

        let mut tsk: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(&mut tsk, &sk, &tsk_infos, &mut source_xe, &mut source_xa, scratch.borrow());

        let mut tsk_prep: GLWETensorKeyPrepared<DeviceBuf<BE>, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, scratch.borrow());

        let scale: usize = 2 * in_base2k;

        let mut data = vec![0i64; n];
        for i in data.iter_mut() {
            *i = (source_xa.next_i64() & 7) - 4;
        }

        pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));

        let mut pt_want_base2k_in = VecZnx::alloc(n, 1, pt_in.size());
        bivariate_convolution_naive(
            module,
            in_base2k,
            2,
            &mut pt_want_base2k_in,
            0,
            pt_in.data(),
            0,
            pt_in.data(),
            0,
            scratch.borrow(),
        );

        module.glwe_encrypt_sk(
            &mut a,
            &pt_in,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );
        module.glwe_encrypt_sk(
            &mut b,
            &pt_in,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );

        for res_offset in 0..scale {
            module.glwe_tensor_apply(
                scale + res_offset,
                &mut res_tensor,
                &a,
                a.max_k().as_usize(),
                &b,
                b.max_k().as_usize(),
                scratch.borrow(),
            );

            module.glwe_tensor_decrypt(&res_tensor, &mut pt_have, &sk_dft, &sk_tensor_prep, scratch.borrow());
            module.vec_znx_normalize(
                pt_want.data_mut(),
                out_base2k,
                res_offset as i64,
                0,
                &pt_want_base2k_in,
                in_base2k,
                0,
                scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(pt_tmp.base2k().as_usize(), &mut pt_tmp.data, 0, scratch.borrow());

            let noise_have: f64 = pt_tmp.stats().std().log2();
            let noise_want = -((k - scale - res_offset - module.log_n()) as f64 - ((rank - 1) as f64) / SQRT_2);

            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);

            module.glwe_tensor_relinearize(&mut res_relin, &res_tensor, &tsk_prep, tsk_prep.size(), scratch.borrow());
            module.glwe_decrypt(&res_relin, &mut pt_have, &sk_dft, scratch.borrow());

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(pt_tmp.base2k().as_usize(), &mut pt_tmp.data, 0, scratch.borrow());

            // We can reuse the same noise bound because the relinearization noise (which is additive)
            // is much smaller than the tensoring noise (which is multiplicative)
            let noise_have: f64 = pt_tmp.stats().std().log2();
            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);
        }
    }
}

pub fn test_glwe_tensor_apply_add_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let out_base2k: usize = base2k - 2;
    let k: usize = 4 * base2k + 1;

    for rank in 1_usize..=3 {
        let n: usize = module.n();

        let glwe_in_infos = GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let glwe_out_infos = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let mut a = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in_infos);
        let mut b = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in_infos);
        let mut product = GLWETensor::<Vec<u8>>::alloc_from_infos(&glwe_out_infos);
        let mut acc = GLWETensor::<Vec<u8>>::alloc_from_infos(&glwe_out_infos);

        for (i, x) in a.data_mut().raw_mut().iter_mut().enumerate() {
            *x = (i as i64 % 7) - 3;
        }
        for (i, x) in b.data_mut().raw_mut().iter_mut().enumerate() {
            *x = (i as i64 % 5) - 2;
        }
        for (i, x) in acc.data_mut().raw_mut().iter_mut().enumerate() {
            *x = (i as i64 % 3) - 1;
        }

        let initial = acc.data().raw().to_vec();
        let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&product, &a, &b));

        module.glwe_tensor_apply(
            base2k,
            &mut product,
            &a,
            a.max_k().as_usize(),
            &b,
            b.max_k().as_usize(),
            scratch.borrow(),
        );
        module.glwe_tensor_apply_add_assign(
            base2k,
            &mut acc,
            &a,
            a.max_k().as_usize(),
            &b,
            b.max_k().as_usize(),
            scratch.borrow(),
        );

        let want: Vec<i64> = initial.iter().zip(product.data().raw()).map(|(x, y)| x + y).collect();
        assert_eq!(acc.data().raw(), want.as_slice());
    }
}

pub fn test_glwe_tensor_square<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWETensoring<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + VecZnxFillUniform
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + VecZnxNormalizeAssign<BE>
        + GLWESecretTensorFactory<BE>
        + VecZnxCopy
        + VecZnxNormalize<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
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

        let mut a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut res_square: GLWETensor<Vec<u8>> = GLWETensor::alloc_from_infos(&glwe_out_infos);
        let mut res_tensor: GLWETensor<Vec<u8>> = GLWETensor::alloc_from_infos(&glwe_out_infos);
        let mut res_relin_square: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
        let mut res_relin_tensor: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

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

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let tsk_enc_infos = EncryptionLayout::new_from_default_sigma(tsk_infos).unwrap();
        let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_in_infos).unwrap();
        let mut tsk: GLWETensorKey<Vec<u8>> = GLWETensorKey::alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            &tsk_enc_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );

        let mut tsk_prep: GLWETensorKeyPrepared<DeviceBuf<BE>, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, scratch.borrow());

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
            scratch.borrow(),
        );

        for res_offset in 0..scale {
            module.glwe_tensor_square_apply(
                scale + res_offset,
                &mut res_square,
                &a,
                a.max_k().as_usize(),
                scratch.borrow(),
            );
            module.glwe_tensor_apply(
                scale + res_offset,
                &mut res_tensor,
                &a,
                a.max_k().as_usize(),
                &a,
                a.max_k().as_usize(),
                scratch.borrow(),
            );

            assert_eq!(res_square.data().raw(), res_tensor.data().raw());

            module.glwe_tensor_relinearize(
                &mut res_relin_square,
                &res_square,
                &tsk_prep,
                tsk_prep.size(),
                scratch.borrow(),
            );
            module.glwe_tensor_relinearize(
                &mut res_relin_tensor,
                &res_tensor,
                &tsk_prep,
                tsk_prep.size(),
                scratch.borrow(),
            );
            assert_eq!(res_relin_square.data().raw(), res_relin_tensor.data().raw());

            // Decrypt one side to ensure the square path remains functionally valid.
            module.glwe_decrypt(&res_relin_square, &mut pt_have, &sk_dft, scratch.borrow());
            module.glwe_decrypt(&res_relin_tensor, &mut pt_want, &sk_dft, scratch.borrow());
            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(pt_tmp.base2k().as_usize(), &mut pt_tmp.data, 0, scratch.borrow());
            let noise_have: f64 = pt_tmp.stats().std().log2();
            assert!(noise_have <= -20.0, "{} > -20", noise_have);
        }
    }
}

pub fn test_glwe_mul_plain<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + VecZnxFillUniform
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + VecZnxNormalizeAssign<BE>
        + VecZnxCopy
        + VecZnxNormalize<BE>
        + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
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

        let mut a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
        let mut pt_a: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
        let mut pt_b: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n().into(), in_base2k.into(), (2 * in_base2k).into());
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos)),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let scale: usize = 2 * in_base2k;

        pt_b.data_mut().fill_uniform(17, &mut source_xa);
        pt_a.data_mut().fill_uniform(17, &mut source_xa);

        let mut pt_want_base2k_in = VecZnx::alloc(n, 1, pt_a.size() + pt_b.size());
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
            scratch.borrow(),
        );

        module.glwe_encrypt_sk(
            &mut a,
            &pt_a,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
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
                scratch_cnv.borrow(),
            );

            module.glwe_decrypt(&res, &mut pt_have, &sk_dft, scratch.borrow());
            module.vec_znx_normalize(
                pt_want.data_mut(),
                out_base2k,
                res_offset as i64,
                0,
                &pt_want_base2k_in,
                in_base2k,
                0,
                scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(pt_tmp.base2k().as_usize(), &mut pt_tmp.data, 0, scratch.borrow());

            let noise_have: f64 = pt_tmp.stats().std().log2();
            let noise_want = -((k - scale - res_offset - module.log_n()) as f64 - ((rank - 1) as f64) / SQRT_2);

            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);
        }
    }
}

pub fn test_glwe_mul_const<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + VecZnxFillUniform
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + VecZnxNormalizeAssign<BE>
        + VecZnxCopy
        + VecZnxNormalize<BE>
        + GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k;
    let out_base2k: usize = base2k;
    let k: usize = 8 * base2k + 1;
    let b_size: usize = 3;

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

        let mut a: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
        let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
        let mut pt_a: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
        let mut pt_b: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n().into(), in_base2k.into(), (2 * in_base2k).into());
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos))
                .max(module.glwe_mul_const_tmp_bytes(&res, &a, b_size)),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n().into(), rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let scale: usize = 2 * in_base2k;

        pt_a.data_mut().fill_uniform(17, &mut source_xa);

        let mut b_const = vec![0i64; b_size];
        let mask = (1 << in_base2k) - 1;
        for (j, x) in b_const[..1].iter_mut().enumerate() {
            let r = source_xa.next_u64() & mask;
            *x = ((r << (64 - 17)) as i64) >> (64 - 17);
            pt_b.data_mut().at_mut(0, j)[0] = *x
        }

        let mut pt_want_base2k_in = VecZnx::alloc(n, 1, pt_a.size() + pt_b.size());
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
            scratch.borrow(),
        );

        module.glwe_encrypt_sk(
            &mut a,
            &pt_a,
            &sk_dft,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );

        for res_offset in 0..scale {
            module.glwe_mul_const(scale + res_offset, &mut res, &a, &b_const, scratch.borrow());

            module.glwe_decrypt(&res, &mut pt_have, &sk_dft, scratch.borrow());
            module.vec_znx_normalize(
                pt_want.data_mut(),
                out_base2k,
                res_offset as i64,
                0,
                &pt_want_base2k_in,
                in_base2k,
                0,
                scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(pt_tmp.base2k().as_usize(), &mut pt_tmp.data, 0, scratch.borrow());

            let noise_have: f64 = pt_tmp.stats().std().log2();
            let noise_want = -((k - scale - res_offset - module.log_n()) as f64 - ((rank - 1) as f64) / SQRT_2);

            assert!(noise_have - noise_want <= 0.5, "{} > {}", noise_have, noise_want);
        }
    }
}
