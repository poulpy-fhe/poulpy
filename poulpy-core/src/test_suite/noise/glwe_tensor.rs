use crate::default::keyswitching::glwe::bound_for;
use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeAssignBackend},
    layouts::{FillUniform, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::convolution::bivariate_convolution_naive,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};
use rand::Rng;
use std::f64::consts::SQRT_2;

use crate::layouts::GLWESecretSampling;
use crate::{
    EncryptionInfos, EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEMulConst, GLWEMulPlain, GLWESub, GLWETensorDecrypt,
    GLWETensorKeyEncryptSk, GLWETensoring,
    api::{
        GLWEBytesOf, TensorApplyRelinearizeItem, TensorPreparedRightRelinearizeAssignItem, TensorSquareApplyRelinearizeItem,
        TensorSquareRelinearizeAssignItem,
    },
    default::operations::glwe_prepare_right,
    layouts::{
        BackendGLWE, Dnum, Dsize, GGLWEActiveUse, GGLWEBind, GGLWEInfos, GGLWEUse, GLWE, GLWELayout, GLWEPlaintext, GLWESecret,
        GLWESecretPreparedFactory, GLWESecretTensor, GLWESecretTensorFactory, GLWESecretTensorPrepared,
        GLWESecretTensorPreparedFactory, GLWETensor, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPrepared,
        GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, TorusPrecision, WithEffectiveDsize,
        prepared::{GLWESecretPrepared, GLWETensorKeyPreparedBound, GLWETensorKeyPreparedToBackendRef},
    },
    log2_std_noise_glwe_tensor,
};

/// Slack allowed above [`log2_std_noise_glwe_tensor`] for the measured
/// tensoring noise, in bits.
///
/// The model is an upper estimate but the realised noise depends on the secret
/// draw far more than the sampled variance alone would suggest: over 32 secret
/// seeds x 32 convolution offsets x ranks 1..3, `noise_have - noise_want` had
/// mean -0.5 / standard deviation 0.3 and peaked at +1.1 on the FFT64 reference
/// backend (`n = 256`, `base2k = 17`), and mean -0.1 / standard deviation 0.7
/// peaking at +1.9 on the NTT4x30 one (`base2k = 52`). Two bits keeps every
/// measured draw inside the bound while still catching a noise regression of
/// 4x or more.
const TENSOR_NOISE_MARGIN: f64 = 2.0;

fn active_use<K: GGLWEInfos>(key: &K, k: TorusPrecision) -> GGLWEActiveUse {
    match key.bind_covering_for(k).unwrap() {
        GGLWEUse::Active(use_) => use_,
        GGLWEUse::Empty => panic!("tensor batch test requires a positive-precision use"),
    }
}

fn prepared_bounds<'a, BE: poulpy_hal::layouts::Backend, K: GLWETensorKeyPreparedToBackendRef<BE>>(
    key: &'a K,
    uses: &[GGLWEActiveUse],
) -> Vec<GLWETensorKeyPreparedBound<'a, BE>> {
    uses.iter()
        .map(|use_| GLWETensorKeyPreparedBound::new(GLWETensorKeyPreparedToBackendRef::to_backend_ref(key), *use_).unwrap())
        .collect()
}

pub fn test_glwe_tensoring<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
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
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let out_base2k: usize = base2k - 2;
    let tsk_base2k: usize = base2k;
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

        let tsk_infos = EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: n.into(),
            base2k: tsk_base2k.into(),
            dnum: k.div_ceil(tsk_base2k).into(),
            k_aux: (tsk_base2k + module.log_n()).into(),
            rank: rank.into(),
            dsize: Dsize(1),
        })
        .unwrap();

        let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut b: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res_tensor: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&glwe_out_infos);
        let mut res_relin: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_have: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos))
                .max(module.glwe_tensor_apply_tmp_bytes(&res_tensor, &a, &b))
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into()))
                .max(module.glwe_tensor_relinearize_tmp_bytes(&res_relin, &res_tensor, &tsk_infos)),
        );

        // Distinct seeds: with a shared one the secret, the mask and the error
        // are all drawn from the same byte stream, correlating `s` with `a` and
        // `e` and making the measured tensoring noise unrepresentative.
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([2u8; 32]);

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let mut sk_tensor: GLWESecretTensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_tensor_alloc(rank.into());
        module.glwe_secret_tensor_prepare(
            &mut sk_tensor,
            &sk,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );

        let mut sk_tensor_prep: GLWESecretTensorPrepared<BE::OwnedBuf, BE> =
            module.glwe_secret_tensor_prepared_alloc(rank.into());
        module.glwe_secret_tensor_prepared_prepare(&mut sk_tensor_prep, &sk_tensor);

        let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            &tsk_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );

        let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut scratch.borrow());

        let scale: usize = 2 * in_base2k;

        let mut data = vec![0i64; n];
        for i in data.iter_mut() {
            *i = (source_xa.next_i64() & 7) - 4;
        }

        pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));

        // Tensoring rescales by `2^cnv_offset` (it drops that many low bits of
        // the product), so `res_offset` shifts the noise one bit at a time.
        // `var_xs = 0.5` is `E[s^2]` of the ternary-prob-0.5 secret above.
        let noise = glwe_in_infos.noise_infos();
        let noise_want = |res_offset: usize| -> f64 {
            log2_std_noise_glwe_tensor(
                n as f64,
                rank as f64,
                0.5,
                noise.sigma,
                noise.k,
                noise.sigma,
                noise.k,
                scale + res_offset,
            )
        };

        let mut pt_want_base2k_in: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(1, pt_in.size());
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
            module.glwe_tensor_apply(scale + res_offset, &mut res_tensor, &a, &b, &mut scratch.borrow());

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

            assert!(
                noise_have - noise_want(res_offset) <= TENSOR_NOISE_MARGIN,
                "{} > {}",
                noise_have,
                noise_want(res_offset)
            );

            module.glwe_tensor_relinearize(&mut res_relin, &res_tensor, &tsk_prep, &mut scratch.borrow());
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
            assert!(
                noise_have - noise_want(res_offset) <= TENSOR_NOISE_MARGIN,
                "{} > {}",
                noise_have,
                noise_want(res_offset)
            );
        }
    }
}

pub fn test_glwe_tensor_square<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
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
            dnum: k.div_ceil(tsk_base2k).into(),
            k_aux: (tsk_base2k + module.log_n()).into(),
            rank: rank.into(),
            dsize: Dsize(1),
        };

        let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res_square: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&glwe_out_infos);
        let mut res_tensor: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&glwe_out_infos);
        let mut res_relin_square: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut res_relin_tensor: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_have: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

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

        // Distinct seeds: with a shared one the secret, the mask and the error
        // are all drawn from the same byte stream, correlating `s` with `a` and
        // `e` and making the measured tensoring noise unrepresentative.
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([2u8; 32]);

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let tsk_enc_infos = EncryptionLayout::new_from_default_sigma(tsk_infos).unwrap();
        let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_in_infos).unwrap();
        let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            &tsk_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
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
            module.glwe_tensor_square_apply(scale + res_offset, &mut res_square, &a, &mut scratch.borrow());
            module.glwe_tensor_apply(scale + res_offset, &mut res_tensor, &a, &a, &mut scratch.borrow());

            module.glwe_tensor_relinearize(&mut res_relin_square, &res_square, &tsk_prep, &mut scratch.borrow());
            module.glwe_tensor_relinearize(&mut res_relin_tensor, &res_tensor, &tsk_prep, &mut scratch.borrow());

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

pub fn test_glwe_mul_plain<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
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

        let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_a: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_b: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_plaintext_alloc(in_base2k.into(), (2 * in_base2k).into());
        let mut pt_have: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos)),
        );

        // Distinct seeds: with a shared one the secret, the mask and the error
        // are all drawn from the same byte stream, correlating `s` with `a` and
        // `e` and making the measured tensoring noise unrepresentative.
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([2u8; 32]);

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let scale: usize = 2 * in_base2k;

        pt_b.data_mut().fill_uniform(17, &mut source_xa);
        pt_a.data_mut().fill_uniform(17, &mut source_xa);

        let mut pt_want_base2k_in: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(1, pt_a.size() + pt_b.size());
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
            module.glwe_mul_plain(scale + res_offset, &mut res, &a, &pt_b, &mut scratch_cnv.borrow());

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

pub fn test_glwe_mul_const<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
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

        let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_a: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_b: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_plaintext_alloc(in_base2k.into(), (2 * in_base2k).into());
        let mut pt_have: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_tmp: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                .max((module).glwe_decrypt_tmp_bytes(&glwe_out_infos))
                .max(module.glwe_mul_const_tmp_bytes(&res, &a, &pt_b)),
        );

        // Distinct seeds: with a shared one the secret, the mask and the error
        // are all drawn from the same byte stream, correlating `s` with `a` and
        // `e` and making the measured tensoring noise unrepresentative.
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([2u8; 32]);

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let scale: usize = 2 * in_base2k;

        pt_a.data_mut().fill_uniform(17, &mut source_xa);

        let mask = (1 << in_base2k) - 1;
        for j in 0..1 {
            let r = source_xa.next_u64() & mask;
            pt_b.data_mut().at_mut(0, j)[b_coeff] = ((r << (64 - 17)) as i64) >> (64 - 17);
        }

        let mut pt_want_base2k_in: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(1, pt_a.size() + pt_b.size());
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

/// Runs `op` over a zeroed and a poisoned arena and requires both results to
/// equal `want`.
fn assert_fused<BE, F>(scratch: &mut ScratchOwned<BE>, seed: &BackendGLWE<BE>, want: &BackendGLWE<BE>, label: &str, mut op: F)
where
    BE: poulpy_hal::layouts::Backend,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut + Clone,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    F: FnMut(&mut BackendGLWE<BE>, &mut poulpy_hal::layouts::ScratchArena<'_, BE>),
{
    // 0x00 then 0xFF: as `i64` the poison is -1, so any limb read before being
    // written shows up in the comparison.
    for fill in [0x00u8, 0xFFu8] {
        <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut scratch.data).fill(fill);
        let mut have = seed.clone();
        op(&mut have, &mut scratch.borrow());
        assert_eq!(want, &have, "{label} (arena filled with {fill:#04x})");
    }
}

/// The three fused apply+relinearize composites equal the explicit
/// materialized composition byte-for-byte, over an intermediate wider than the
/// result and a non-limb-aligned `cnv_offset`.
///
/// Each arena is sized at exactly the documented composite bound
/// `glwe_tensor_bytes_of_from_infos(tensor) + max(apply, relinearize)`, which is
/// what `ckks_mul_tmp_bytes` returns, so a composite exceeding it fails here.
/// Prepared-right is given the *ordinary* apply bound for the equivalent
/// unprepared layouts, pinning `P <= A`.
pub fn test_glwe_tensor_fused_relinearize<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensoring<BE> + GLWETensorKeyPreparedFactory<BE> + GLWEBytesOf<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();

    for rank in 1_usize..=2 {
        let layout = |limbs: usize| GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: (limbs * base2k).into(),
            rank: rank.into(),
        };

        // Operands and intermediate five limbs wide, destination three: the
        // composite cannot infer `tensor_infos` from `res`.
        let ab_infos = layout(5);
        let res_infos = layout(3);
        let tensor_infos = layout(5);

        let tsk_infos = GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: 5_usize.into(),
            k_aux: (base2k + module.log_n()).into(),
            rank: rank.into(),
            dsize: Dsize(1),
        };

        let mut source: Source = Source::new([rank as u8; 32]);

        let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ab_infos);
        a.fill_uniform(base2k, &mut source);
        let mut b: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ab_infos);
        b.fill_uniform(base2k, &mut source);
        let mut seed: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&res_infos);
        seed.fill_uniform(base2k, &mut source);

        let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        tsk.fill_uniform(base2k, &mut source);

        let tensor_bytes: usize = module.glwe_tensor_bytes_of_from_infos(&tensor_infos);
        let apply_bytes: usize = module.glwe_tensor_apply_tmp_bytes(&tensor_infos, &ab_infos, &ab_infos);
        let square_bytes: usize = module.glwe_tensor_square_apply_tmp_bytes(&tensor_infos, &res_infos);
        let prepared_bytes: usize = module.glwe_tensor_apply_tmp_bytes(&tensor_infos, &res_infos, &ab_infos);
        let relin_bytes: usize = module.glwe_tensor_relinearize_tmp_bytes(&res_infos, &tensor_infos, &tsk_infos);

        let mut apply_scratch: ScratchOwned<BE> = ScratchOwned::alloc(tensor_bytes + apply_bytes.max(relin_bytes));
        let mut square_scratch: ScratchOwned<BE> = ScratchOwned::alloc(tensor_bytes + square_bytes.max(relin_bytes));
        let mut prepared_scratch: ScratchOwned<BE> = ScratchOwned::alloc(tensor_bytes + prepared_bytes.max(relin_bytes));
        let mut setup: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .prepare_tensor_key_tmp_bytes(&tsk_infos)
                .max(apply_bytes)
                .max(square_bytes)
                .max(prepared_bytes)
                .max(relin_bytes),
        );

        let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut setup.borrow());

        let b_size: usize = ab_infos.size();
        let mut b_prep = module.cnv_pvec_right_alloc(rank + 1, b_size);
        glwe_prepare_right(module, &mut b_prep, &b, ab_infos.k().as_usize(), &mut setup.borrow());

        // 0 aligns on a limb, `base2k` on the next, `base2k + 7` on neither.
        for cnv_offset in [0_usize, base2k, base2k + 7] {
            let mut tensor: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&tensor_infos);

            let mut want = seed.clone();
            module.glwe_tensor_apply(cnv_offset, &mut tensor, &a, &b, &mut setup.borrow());
            module.glwe_tensor_relinearize(&mut want, &tensor, &tsk_prep, &mut setup.borrow());
            assert_fused(
                &mut apply_scratch,
                &seed,
                &want,
                &format!("apply_relinearize (rank={rank}, cnv_offset={cnv_offset})"),
                |res, s| module.glwe_tensor_apply_relinearize(cnv_offset, res, &tensor_infos, &a, &b, &tsk_prep, s),
            );

            let mut want = seed.clone();
            module.glwe_tensor_square_apply(cnv_offset, &mut tensor, &want, &mut setup.borrow());
            module.glwe_tensor_relinearize(&mut want, &tensor, &tsk_prep, &mut setup.borrow());
            assert_fused(
                &mut square_scratch,
                &seed,
                &want,
                &format!("square_relinearize_assign (rank={rank}, cnv_offset={cnv_offset})"),
                |res, s| module.glwe_tensor_square_relinearize_assign(cnv_offset, res, &tensor_infos, &tsk_prep, s),
            );

            let mut want = seed.clone();
            module.glwe_tensor_apply_prepared_right(cnv_offset, &mut tensor, &want, &b_prep, b_size, &mut setup.borrow());
            module.glwe_tensor_relinearize(&mut want, &tensor, &tsk_prep, &mut setup.borrow());
            assert_fused(
                &mut prepared_scratch,
                &seed,
                &want,
                &format!("apply_prepared_right_relinearize_assign (rank={rank}, cnv_offset={cnv_offset})"),
                |res, s| {
                    module.glwe_tensor_apply_prepared_right_relinearize_assign(
                        cnv_offset,
                        res,
                        &tensor_infos,
                        &b_prep,
                        b_size,
                        &tsk_prep,
                        s,
                    )
                },
            );
        }
    }
}

/// The four fused-composite batches equal the ordered scalar composites, item
/// for item, at lengths 0 through 4.
///
/// Each item carries its own `cnv_offset` and tensor layout, read-only operands
/// and prepared operands repeat across items, and every batch runs on an arena
/// of exactly the bytes its `*_batch_tmp_bytes` advertises, once zeroed and once
/// poisoned.
pub fn test_glwe_tensor_fused_relinearize_batch<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensoring<BE> + GLWETensorKeyPreparedFactory<BE> + GLWEBytesOf<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let rank: usize = 1;

    let layout = |limbs: usize| GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: (limbs * base2k).into(),
        rank: rank.into(),
    };
    let ab_infos = layout(5);
    let res_infos = layout(3);
    // Two intermediate widths, so items in one batch differ in tensor layout.
    let tensor_infos = [layout(5), layout(4)];
    // 0 aligns on a limb, `base2k` on the next, `base2k + 7` on neither.
    let offsets = [0_usize, base2k, base2k + 7, 0];

    let tsk_infos = GLWETensorKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: 5_usize.into(),
        k_aux: (base2k + module.log_n()).into(),
        rank: rank.into(),
        dsize: Dsize(1),
    };

    let mut source: Source = Source::new([11u8; 32]);
    let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ab_infos);
    a.fill_uniform(base2k, &mut source);
    let mut b: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&ab_infos);
    b.fill_uniform(base2k, &mut source);
    let mut seed: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&res_infos);
    seed.fill_uniform(base2k, &mut source);
    let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    tsk.fill_uniform(base2k, &mut source);

    // The scalar reference runs on one arena covering every composite at either
    // intermediate width.
    let mut setup_bytes: usize = module.prepare_tensor_key_tmp_bytes(&tsk_infos);
    for tensor in tensor_infos.iter() {
        let relin = module.glwe_tensor_relinearize_tmp_bytes(&res_infos, tensor, &tsk_infos);
        let stage = module
            .glwe_tensor_apply_tmp_bytes(tensor, &ab_infos, &ab_infos)
            .max(module.glwe_tensor_square_apply_tmp_bytes(tensor, &ab_infos))
            .max(module.glwe_tensor_square_apply_tmp_bytes(tensor, &res_infos))
            .max(module.glwe_tensor_apply_tmp_bytes(tensor, &res_infos, &ab_infos))
            .max(relin);
        setup_bytes = setup_bytes.max(module.glwe_tensor_bytes_of_from_infos(tensor) + stage);
    }
    let mut setup: ScratchOwned<BE> = ScratchOwned::alloc(setup_bytes);
    let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut setup.borrow());

    let b_size: usize = ab_infos.size();
    let mut b_prep = module.cnv_pvec_right_alloc(rank + 1, b_size);
    glwe_prepare_right(module, &mut b_prep, &b, ab_infos.k().as_usize(), &mut setup.borrow());

    let alloc_dst = |len: usize| -> Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> {
        (0..len)
            .map(|_| {
                let mut ct = module.glwe_alloc_from_infos(&res_infos);
                ct.fill_uniform(base2k, &mut Source::new([13u8; 32]));
                ct
            })
            .collect()
    };
    let infos_of = |i: usize| &tensor_infos[i % 2];

    for len in 0..=4usize {
        // ── apply_relinearize: `a` repeats across every item.
        let mut want = alloc_dst(len);
        for (i, dst) in want.iter_mut().enumerate() {
            module.glwe_tensor_apply_relinearize(offsets[i], dst, infos_of(i), &a, &b, &tsk_prep, &mut setup.borrow());
        }
        let query: Vec<TensorApplyRelinearizeItem<&GLWE<_, _>, _, _, _>> = want
            .iter()
            .enumerate()
            .map(|(i, dst)| TensorApplyRelinearizeItem {
                cnv_offset: offsets[i],
                res: dst,
                tensor_infos: infos_of(i),
                a: &a,
                b: &b,
            })
            .collect();
        let uses: Vec<_> = query
            .iter()
            .map(|item| active_use(&tsk_infos, item.tensor_infos.k()))
            .collect();
        let bytes = module.glwe_tensor_apply_relinearize_batch_tmp_bytes(&query, &uses);
        drop(query);
        for fill in [0x00u8, 0xFFu8] {
            let mut exact: ScratchOwned<BE> = ScratchOwned::alloc(bytes.max(1));
            <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut exact.data).fill(fill);
            let mut have = alloc_dst(len);
            let mut items: Vec<TensorApplyRelinearizeItem<&mut GLWE<_, _>, _, _, _>> = have
                .iter_mut()
                .enumerate()
                .map(|(i, dst)| TensorApplyRelinearizeItem {
                    cnv_offset: offsets[i],
                    res: dst,
                    tensor_infos: infos_of(i),
                    a: &a,
                    b: &b,
                })
                .collect();
            let bounds = prepared_bounds(&tsk_prep, &uses);
            module.glwe_tensor_apply_relinearize_batch(&mut items, &bounds, &mut exact.borrow());
            drop(items);
            assert_eq!(want, have, "apply_relinearize_batch len={len} fill={fill:#04x}");
        }

        // ── square_apply_relinearize
        let mut want = alloc_dst(len);
        for (i, dst) in want.iter_mut().enumerate() {
            module.glwe_tensor_square_apply_relinearize(offsets[i], dst, infos_of(i), &a, &tsk_prep, &mut setup.borrow());
        }
        let query: Vec<TensorSquareApplyRelinearizeItem<&GLWE<_, _>, _, _>> = want
            .iter()
            .enumerate()
            .map(|(i, dst)| TensorSquareApplyRelinearizeItem {
                cnv_offset: offsets[i],
                res: dst,
                tensor_infos: infos_of(i),
                a: &a,
            })
            .collect();
        let uses: Vec<_> = query
            .iter()
            .map(|item| active_use(&tsk_infos, item.tensor_infos.k()))
            .collect();
        let bytes = module.glwe_tensor_square_apply_relinearize_batch_tmp_bytes(&query, &uses);
        drop(query);
        for fill in [0x00u8, 0xFFu8] {
            let mut exact: ScratchOwned<BE> = ScratchOwned::alloc(bytes.max(1));
            <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut exact.data).fill(fill);
            let mut have = alloc_dst(len);
            let mut items: Vec<TensorSquareApplyRelinearizeItem<&mut GLWE<_, _>, _, _>> = have
                .iter_mut()
                .enumerate()
                .map(|(i, dst)| TensorSquareApplyRelinearizeItem {
                    cnv_offset: offsets[i],
                    res: dst,
                    tensor_infos: infos_of(i),
                    a: &a,
                })
                .collect();
            let bounds = prepared_bounds(&tsk_prep, &uses);
            module.glwe_tensor_square_apply_relinearize_batch(&mut items, &bounds, &mut exact.borrow());
            drop(items);
            assert_eq!(want, have, "square_apply_relinearize_batch len={len} fill={fill:#04x}");
        }

        // ── square_relinearize_assign: each item aliases its own destination.
        let mut want = alloc_dst(len);
        for (i, dst) in want.iter_mut().enumerate() {
            module.glwe_tensor_square_relinearize_assign(offsets[i], dst, infos_of(i), &tsk_prep, &mut setup.borrow());
        }
        let query: Vec<TensorSquareRelinearizeAssignItem<&GLWE<_, _>, _>> = want
            .iter()
            .enumerate()
            .map(|(i, dst)| TensorSquareRelinearizeAssignItem {
                cnv_offset: offsets[i],
                res: dst,
                tensor_infos: infos_of(i),
            })
            .collect();
        let uses: Vec<_> = query
            .iter()
            .map(|item| active_use(&tsk_infos, item.tensor_infos.k()))
            .collect();
        let bytes = module.glwe_tensor_square_relinearize_assign_batch_tmp_bytes(&query, &uses);
        drop(query);
        for fill in [0x00u8, 0xFFu8] {
            let mut exact: ScratchOwned<BE> = ScratchOwned::alloc(bytes.max(1));
            <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut exact.data).fill(fill);
            let mut have = alloc_dst(len);
            let mut items: Vec<TensorSquareRelinearizeAssignItem<&mut GLWE<_, _>, _>> = have
                .iter_mut()
                .enumerate()
                .map(|(i, dst)| TensorSquareRelinearizeAssignItem {
                    cnv_offset: offsets[i],
                    res: dst,
                    tensor_infos: infos_of(i),
                })
                .collect();
            let bounds = prepared_bounds(&tsk_prep, &uses);
            module.glwe_tensor_square_relinearize_assign_batch(&mut items, &bounds, &mut exact.borrow());
            drop(items);
            assert_eq!(want, have, "square_relinearize_assign_batch len={len} fill={fill:#04x}");
        }

        // ── apply_prepared_right_relinearize_assign: one prepared operand,
        // repeated across every item.
        let mut want = alloc_dst(len);
        for (i, dst) in want.iter_mut().enumerate() {
            module.glwe_tensor_apply_prepared_right_relinearize_assign(
                offsets[i],
                dst,
                infos_of(i),
                &b_prep,
                b_size,
                &tsk_prep,
                &mut setup.borrow(),
            );
        }
        let query: Vec<TensorPreparedRightRelinearizeAssignItem<&GLWE<_, _>, _, _>> = want
            .iter()
            .enumerate()
            .map(|(i, dst)| TensorPreparedRightRelinearizeAssignItem {
                cnv_offset: offsets[i],
                res: dst,
                tensor_infos: infos_of(i),
                prepared_right: &b_prep,
                prepared_right_size: b_size,
            })
            .collect();
        let uses: Vec<_> = query
            .iter()
            .map(|item| active_use(&tsk_infos, item.tensor_infos.k()))
            .collect();
        let bytes = module.glwe_tensor_apply_prepared_right_relinearize_assign_batch_tmp_bytes(&query, &uses);
        drop(query);
        for fill in [0x00u8, 0xFFu8] {
            let mut exact: ScratchOwned<BE> = ScratchOwned::alloc(bytes.max(1));
            <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut exact.data).fill(fill);
            let mut have = alloc_dst(len);
            let mut items: Vec<TensorPreparedRightRelinearizeAssignItem<&mut GLWE<_, _>, _, _>> = have
                .iter_mut()
                .enumerate()
                .map(|(i, dst)| TensorPreparedRightRelinearizeAssignItem {
                    cnv_offset: offsets[i],
                    res: dst,
                    tensor_infos: infos_of(i),
                    prepared_right: &b_prep,
                    prepared_right_size: b_size,
                })
                .collect();
            let bounds = prepared_bounds(&tsk_prep, &uses);
            module.glwe_tensor_apply_prepared_right_relinearize_assign_batch(&mut items, &bounds, &mut exact.borrow());
            drop(items);
            assert_eq!(
                want, have,
                "apply_prepared_right_relinearize_assign_batch len={len} fill={fill:#04x}"
            );
        }
    }

    // A malformed late lane must be rejected by the frontier-wide preflight,
    // before the valid first lane can write its destination.
    let invalid_offsets = [0, usize::MAX];
    let uses: Vec<_> = (0..2).map(|i| active_use(&tsk_infos, infos_of(i).k())).collect();
    let bounds = prepared_bounds(&tsk_prep, &uses);
    let mut have = alloc_dst(2);
    let untouched = alloc_dst(2);
    let mut items: Vec<TensorApplyRelinearizeItem<&mut GLWE<_, _>, _, _, _>> = have
        .iter_mut()
        .enumerate()
        .map(|(i, dst)| TensorApplyRelinearizeItem {
            cnv_offset: invalid_offsets[i],
            res: dst,
            tensor_infos: infos_of(i),
            a: &a,
            b: &b,
        })
        .collect();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.glwe_tensor_apply_relinearize_batch(&mut items, &bounds, &mut setup.borrow());
    }));
    assert!(panic.is_err(), "batch accepted an out-of-range late conversion offset");
    drop(items);
    assert_eq!(untouched, have, "a late invalid tensor lane wrote an earlier destination");

    let query: Vec<TensorApplyRelinearizeItem<&GLWE<_, _>, _, _, _>> = have
        .iter()
        .enumerate()
        .map(|(i, dst)| TensorApplyRelinearizeItem {
            cnv_offset: invalid_offsets[i],
            res: dst,
            tensor_infos: infos_of(i),
            a: &a,
            b: &b,
        })
        .collect();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.glwe_tensor_apply_relinearize_batch_tmp_bytes(&query, &uses)
    }));
    assert!(panic.is_err(), "batch scratch query accepted an out-of-range late offset");
}

/// Relinearization honours the effective `dsize` carried on the key.
///
/// The selected product's own oracle (bit-identical to a dense copy of the same
/// rows) is [`test_gglwe_product_dft_selected`](crate::test_suite::parity::test_gglwe_product_dft_selected);
/// here the point is that `glwe_tensor_relinearize` routes through it, so the
/// coarsened result must differ from the native one and still decrypt.
pub fn test_glwe_tensor_relinearize_selected<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
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
{
    let base2k: usize = params.base2k;
    let k: usize = 8 * base2k + 1;
    let rank: usize = 1;
    let n: usize = module.n();
    let effective_dsize: Dsize = Dsize(2);

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: rank.into(),
    })
    .unwrap();

    // `dnum` exceeds what `k` needs, which is what leaves room for a coarser
    // decomposition to be carved out of the stored rows.
    let tsk_infos = EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: Dnum(12),
        k_aux: (base2k + module.log_n()).into(),
        rank: rank.into(),
        dsize: Dsize(1),
    })
    .unwrap();

    let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    let mut res_tensor: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let mut res_native: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    let mut res_selected: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    let mut pt_have: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
            .max(module.glwe_decrypt_tmp_bytes(&glwe_infos))
            .max(module.glwe_tensor_apply_tmp_bytes(&res_tensor, &a, &a))
            .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into()))
            .max(module.glwe_tensor_relinearize_tmp_bytes(&res_native, &res_tensor, &tsk_infos))
            .max(module.glwe_tensor_relinearize_tmp_bytes(&res_selected, &res_tensor, &tsk_infos.with_dsize(effective_dsize))),
    );

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_dft, &sk);

    let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    module.glwe_tensor_key_encrypt_sk(
        &mut tsk,
        &sk,
        &tsk_infos,
        &mut source_xe,
        &mut source_xa,
        &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
    );

    let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut scratch.borrow());

    let scale: usize = 2 * base2k;
    let mut data = vec![0i64; n];
    for i in data.iter_mut() {
        *i = (source_xa.next_i64() & 7) - 4;
    }
    pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));

    module.glwe_encrypt_sk(
        &mut a,
        &pt_in,
        &sk_dft,
        &glwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    module.glwe_tensor_apply(scale, &mut res_tensor, &a, &a, &mut scratch.borrow());

    module.glwe_tensor_relinearize(&mut res_native, &res_tensor, &tsk_prep, &mut scratch.borrow());
    module.glwe_tensor_relinearize(
        &mut res_selected,
        &res_tensor,
        &tsk_prep.with_dsize(effective_dsize),
        &mut scratch.borrow(),
    );

    assert_ne!(
        res_native.data.raw(),
        res_selected.data.raw(),
        "the effective dsize was ignored: the coarsened relinearization reproduced the native one"
    );

    // The message survives the coarser decomposition; only the key-switching
    // noise grows, and it stays far under the tensoring noise it is added to.
    let mut want = vec![0i64; n];
    module.glwe_decrypt(&res_native, &mut pt_have, &sk_dft, &mut scratch.borrow());
    pt_have.decode_vec_i64(&mut want, TorusPrecision(scale as u32));
    let mut have = vec![0i64; n];
    module.glwe_decrypt(&res_selected, &mut pt_have, &sk_dft, &mut scratch.borrow());
    pt_have.decode_vec_i64(&mut have, TorusPrecision(scale as u32));
    assert_eq!(have, want, "coarsened relinearization lost the message");
}

/// **Cross-radix relinearization: the bound and the operand width agree.**
///
/// The tensor operand is stored at one `base2k` and the key at another, with a
/// precision that is not a whole number of key limbs. The operand fed to the
/// product then carries `ceil(a.k() / key_base2k)` limbs, which is what the
/// bound resolves; deriving it from the operand's *storage* width instead gives
/// a different limb count and the product rejects it outright.
///
/// The check is the width identity plus completion: under the defect the
/// product is handed one limb too many and rejects the call outright.
pub fn test_glwe_tensor_relinearize_cross_radix<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
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
{
    let rank: usize = 1;
    let n: usize = module.n();

    // Keep both radices inside the envelope selected by each backend suite. In
    // particular, FFT64's configured radix is 17; the old hard-coded radix 30
    // overflowed its i64 BIG accumulator before exercising this regression.
    let hi: usize = params.base2k;
    let lo: usize = hi.checked_sub(1).expect("cross-radix test requires base2k >= 2");

    // (operand radix, key radix, operand precision), chosen so that the storage
    // width and the exact precision round to different key-limb counts.
    for (a_base2k, key_base2k) in [(hi, lo), (lo, hi)] {
        let k: usize = a_base2k
            .checked_mul(4)
            .and_then(|v| v.checked_add(1))
            .expect("cross-radix test precision overflows usize");
        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: a_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();
        assert_ne!(
            (glwe_infos.size() * a_base2k).div_ceil(key_base2k),
            k.div_ceil(key_base2k),
            "the case must separate the storage width from the exact precision"
        );

        let key_infos = |base2k: usize| {
            EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                dnum: Dnum(k.div_ceil(base2k) as u32),
                k_aux: (base2k + module.log_n()).into(),
                rank: rank.into(),
                dsize: Dsize(1),
            })
            .unwrap()
        };
        let cross_infos = key_infos(key_base2k);

        let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        let mut res_tensor: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&glwe_infos);
        let mut res_cross: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
                .max(module.glwe_decrypt_tmp_bytes(&glwe_infos))
                .max(module.glwe_tensor_apply_tmp_bytes(&res_tensor, &a, &a))
                .max(module.glwe_secret_tensor_prepare_tmp_bytes(rank.into()))
                .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&cross_infos))
                .max(module.glwe_tensor_relinearize_tmp_bytes(&res_cross, &res_tensor, &cross_infos)),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([2u8; 32]);

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let scale: usize = 2 * a_base2k;
        let mut data = vec![0i64; n];
        for i in data.iter_mut() {
            *i = (source_xa.next_i64() & 3) - 2;
        }
        pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));
        module.glwe_encrypt_sk(
            &mut a,
            &pt_in,
            &sk_dft,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        module.glwe_tensor_apply(scale, &mut res_tensor, &a, &a, &mut scratch.borrow());

        // The bound and the operand width must agree on the same number, and it
        // must be the one the exact precision gives.
        let use_ = bound_for(&cross_infos, glwe_infos.k());
        assert_eq!(
            use_.active().expect("positive precision").input_size(),
            k.div_ceil(key_base2k),
            "the bound reads a different limb count than the exact precision implies"
        );

        let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&cross_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            &cross_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );
        let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&cross_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut scratch.borrow());
        // Reaching the end is the regression: deriving the operand width from
        // the storage precision hands the product a limb count its bound does
        // not accept, and it rejects the call before doing any work.
        module.glwe_tensor_relinearize(&mut res_cross, &res_tensor, &tsk_prep, &mut scratch.borrow());
    }
}
