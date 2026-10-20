use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSource, VecZnxNormalize,
        VecZnxNormalizeAssign, VecZnxSwitchRing,
    },
    layouts::{Backend, HostDataRef, Module, PrepareHint, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::convolution::bivariate_convolution_naive,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};
use rand::Rng;
use std::f64::consts::SQRT_2;

use crate::layouts::GLWESecretSampling;
use crate::test_suite::noise::glwe_decrypt_checked;
use crate::{
    GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEMaskFill, GLWEMulConst, GLWEMulPlain, GLWEMulRelinearizeSum,
    GLWENormalize, GLWEPreparedTensoring, GLWESub, GLWETensorDecrypt, GLWETensorKeyEncryptSk, GLWETensoring, glwe_prepare_left,
    glwe_prepare_left_tmp_bytes, glwe_prepare_right, glwe_prepare_right_tmp_bytes,
    layouts::{
        Dnum, Dsize, GLWE, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWESecret, GLWESecretPreparedFactory,
        GLWESecretTensor, GLWESecretTensorFactory, GLWESecretTensorPrepared, GLWESecretTensorPreparedFactory, GLWETensor,
        GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        TorusPrecision, prepared::GLWESecretPrepared,
    },
    log2_std_noise_glwe_tensor, log2_std_noise_glwe_tensor_relinearized,
};

/// Slack allowed above [`log2_std_noise_glwe_tensor`] for the measured
/// tensoring noise, in bits.
///
/// Realised noise varies with the secret draw. This test allows a factor of
/// four above the model's estimated standard deviation (two bits on the log2
/// scale). The same margin applies to every implementation under test.
const TENSOR_NOISE_MARGIN: f64 = 2.0;

pub(crate) fn assert_canonical(a: &VecZnx<impl HostDataRef, i64>, base2k: usize, k: usize) {
    let active_size = k.div_ceil(base2k);
    let half = 1i128 << (base2k - 1);
    for col in 0..a.cols() {
        for limb in 0..active_size {
            assert!(a.at(col, limb).iter().all(|&digit| (-half..half).contains(&(digit as i128))));
        }
        for limb in active_size..a.size() {
            assert!(a.at(col, limb).iter().all(|&digit| digit == 0));
        }
        let padding = (base2k - k % base2k) % base2k;
        if active_size != 0 && padding != 0 {
            let mask = (1u64 << padding) - 1;
            assert!(a.at(col, active_size - 1).iter().all(|&digit| digit as u64 & mask == 0));
        }
    }
}

pub fn test_glwe_tensoring<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensoring<BE>
        + GLWEMulRelinearizeSum<BE>
        + GLWEPreparedTensoring<BE>
        + CnvPVecAlloc<BE>
        + Convolution<BE>
        + GLWENormalize<BE>
        + GLWEAdd<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWETensorDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub<BE>
        + VecZnxNormalizeAssign<BE>
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

        let glwe_in_infos = GLWELayout {
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

        let tsk_infos = GLWETensorKeyLayout {
            n: n.into(),
            base2k: tsk_base2k.into(),
            dnum: k.div_ceil(tsk_base2k).into(),
            k_aux: (tsk_base2k + module.log_n()).into(),
            rank: rank.into(),
            dsize: Dsize(1),
        };

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
                .max(module.glwe_tensor_apply_prepared_tmp_bytes(
                    &res_tensor,
                    k.div_ceil(in_base2k),
                    k.div_ceil(in_base2k),
                    in_base2k.into(),
                ))
                .max(glwe_prepare_left_tmp_bytes(module, k.div_ceil(in_base2k), &a))
                .max(glwe_prepare_right_tmp_bytes(module, k.div_ceil(in_base2k), &b))
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
        let noise_k = glwe_in_infos.k().as_usize();
        let noise_want = |res_offset: usize| -> f64 {
            log2_std_noise_glwe_tensor(
                n as f64,
                rank as f64,
                0.5,
                crate::DEFAULT_SIGMA_XE,
                noise_k,
                crate::DEFAULT_SIGMA_XE,
                noise_k,
                scale + res_offset,
                k,
                out_base2k,
            )
        };
        let noise_want_relinearized = |res_offset: usize| -> f64 {
            log2_std_noise_glwe_tensor_relinearized(
                n as f64,
                rank as f64,
                0.5,
                crate::DEFAULT_SIGMA_XE,
                noise_k,
                crate::DEFAULT_SIGMA_XE,
                noise_k,
                scale + res_offset,
                k,
                out_base2k,
                k,
                out_base2k,
            )
        };

        let mut pt_want_base2k_in: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(module.n(), 1, pt_in.size());
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

        module.glwe_encrypt_sk(&mut a, &pt_in, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());
        module.glwe_encrypt_sk(&mut b, &pt_in, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());

        // The same product of operands prepared once, under either hint.
        let size: usize = k.div_ceil(in_base2k);
        let mut a_prep = module.cnv_pvec_left_alloc(n, rank + 1, size, PrepareHint::OneShot);
        glwe_prepare_left(module, &mut a_prep, &a, k, &mut scratch.borrow());
        let mut b_prep = module.cnv_pvec_right_alloc(n, rank + 1, size, PrepareHint::Reuse);
        glwe_prepare_right(module, &mut b_prep, &b, k, &mut scratch.borrow());
        let mut res_prepared: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&glwe_out_infos);

        for res_offset in 0..scale {
            module.glwe_tensor_apply(scale + res_offset, &mut res_tensor, &a, &b, &mut scratch.borrow());
            assert_canonical(res_tensor.data(), out_base2k, k);
            module.glwe_tensor_apply_prepared(
                scale + res_offset,
                &mut res_prepared,
                &a_prep,
                size,
                &b_prep,
                size,
                in_base2k.into(),
                &mut scratch.borrow(),
            );
            assert!(
                res_prepared.data() == res_tensor.data(),
                "prepared tensor product, rank {rank}, offset {res_offset}: not the product"
            );

            module.glwe_tensor_decrypt(&res_tensor, &mut pt_have, &sk_dft, &sk_tensor_prep, &mut scratch.borrow());
            module.vec_znx_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                out_base2k,
                k,
                res_offset as i64,
                0,
                &vec_znx_backend_ref::<BE>(&pt_want_base2k_in),
                in_base2k,
                0,
                &mut scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(
                pt_tmp.base2k().as_usize(),
                pt_tmp.data.size() * pt_tmp.base2k().as_usize(),
                0,
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
            glwe_decrypt_checked(module, &res_relin, &mut pt_have, &sk_dft, &mut scratch.borrow());

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(
                pt_tmp.base2k().as_usize(),
                pt_tmp.data.size() * pt_tmp.base2k().as_usize(),
                0,
                &mut vec_znx_backend_mut::<BE>(&mut pt_tmp.data),
                0,
                &mut scratch.borrow(),
            );

            let noise_have: f64 = pt_tmp.stats().std().log2();
            assert!(
                noise_have - noise_want_relinearized(res_offset) <= TENSOR_NOISE_MARGIN,
                "{} > {}",
                noise_have,
                noise_want_relinearized(res_offset)
            );
        }
    }

    one_pass_product_noise(params, module);
}

/// Noise of the one-pass product and relinearization, held to the bound of the two-step path.
///
/// Rank 1 with one radix for the operands, the result and the tensor key, which the one-pass path requires.
/// Covers the operand roles: a product, a squaring, a product whose left operand is the destination, and a sum of
/// three products over operands of two plaintexts, held to three times the bound on exactly its scratch. One pair,
/// alone or beside a pair of zeros, is the product bit for bit, and pairs prepared once give the sum bit for bit.
fn one_pass_product_noise<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensoring<BE>
        + GLWEMulRelinearizeSum<BE>
        + GLWEPreparedTensoring<BE>
        + CnvPVecAlloc<BE>
        + Convolution<BE>
        + GLWENormalize<BE>
        + GLWEAdd<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalize<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 8 * base2k + 1;
    let n: usize = module.n();
    let size = k.div_ceil(base2k);

    let glwe_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1u32.into(),
    };
    let tsk_infos = GLWETensorKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: k.div_ceil(base2k).into(),
        k_aux: (base2k + module.log_n()).into(),
        rank: 1u32.into(),
        dsize: Dsize(1),
    };
    // A result wider than the operands grows the key-switch output, up to the auxiliary limbs of a key of dsize 4.
    let wide_infos = GLWELayout {
        k: (k + 4 * base2k).into(),
        ..glwe_infos
    };
    let wide_tsk_infos = GLWETensorKeyLayout {
        dnum: k.div_ceil(4 * base2k).into(),
        k_aux: (4 * base2k + module.log_n()).into(),
        dsize: Dsize(4),
        ..tsk_infos
    };

    let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    let mut b: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    let mut pt_have: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    let mut pt_tmp: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
            .max(module.glwe_decrypt_tmp_bytes(&glwe_infos))
            .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_infos))
            .max(module.prepare_tensor_key_tmp_bytes(&tsk_infos))
            .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&wide_tsk_infos))
            .max(module.prepare_tensor_key_tmp_bytes(&wide_tsk_infos))
            .max(module.glwe_mul_relinearize_tmp_bytes(&glwe_infos, size, size, TorusPrecision(k as u32), &tsk_infos))
            .max(glwe_prepare_left_tmp_bytes(module, size, &glwe_infos))
            .max(glwe_prepare_right_tmp_bytes(module, size, &glwe_infos)),
    );

    let mut source_xs: Source = Source::new([3u8; 32]);
    let mut source_xe: Source = Source::new([4u8; 32]);
    let mut source_xa: Source = Source::new([5u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(1u32.into());
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_dft, &sk);

    let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    module.glwe_tensor_key_encrypt_sk(
        &mut tsk,
        &sk,
        &mut source_xe,
        &mut source_xa,
        &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
    );
    let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut scratch.borrow());
    let mut wide_tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&wide_tsk_infos);
    module.glwe_tensor_key_encrypt_sk(
        &mut wide_tsk,
        &sk,
        &mut source_xe,
        &mut source_xa,
        &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
    );
    let mut wide_tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&wide_tsk_infos);
    module.prepare_tensor_key(&mut wide_tsk_prep, &wide_tsk, &mut scratch.borrow());

    let scale: usize = 2 * base2k;
    let mut data = vec![0i64; n];
    for i in data.iter_mut() {
        *i = (source_xa.next_i64() & 7) - 4;
    }
    pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));

    let noise_want = |res_offset: usize| -> f64 {
        log2_std_noise_glwe_tensor_relinearized(
            n as f64,
            1.0,
            0.5,
            crate::DEFAULT_SIGMA_XE,
            k,
            crate::DEFAULT_SIGMA_XE,
            k,
            scale + res_offset,
            k,
            base2k,
            k,
            base2k,
        )
    };

    let mut pt_square: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(module.n(), 1, pt_in.size());
    bivariate_convolution_naive::<_, BE>(
        module,
        base2k,
        2,
        &mut pt_square,
        0,
        pt_in.data(),
        0,
        pt_in.data(),
        0,
        &mut scratch.borrow(),
    );

    // Both operands encrypt the same plaintext, so every single product has the same expected value.
    for ct in [&mut a, &mut b] {
        module.glwe_encrypt_sk(ct, &pt_in, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());
    }

    // `c` encrypts another plaintext, so that a sum whose later pairs multiplied other operands would decrypt to
    // another value: `a * b + b * c + c * a` decrypts to `pt_in * (pt_in + 2 * pt_c)`.
    let data_c: Vec<i64> = (0..n).map(|_| (source_xa.next_i64() & 7) - 4).collect();
    let mut pt_c: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    pt_c.encode_vec_i64(&data_c, TorusPrecision(scale as u32));
    let mut c: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    module.glwe_encrypt_sk(&mut c, &pt_c, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());
    let data_mix: Vec<i64> = data.iter().zip(&data_c).map(|(x, y)| x + 2 * y).collect();
    let mut pt_mix: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    pt_mix.encode_vec_i64(&data_mix, TorusPrecision(scale as u32));
    let mut pt_sum: VecZnx<BE::OwnedBuf, BE::ZnxWord> = module.vec_znx_alloc(module.n(), 1, pt_in.size());
    bivariate_convolution_naive::<_, BE>(
        module,
        base2k,
        2,
        &mut pt_sum,
        0,
        pt_in.data(),
        0,
        pt_mix.data(),
        0,
        &mut scratch.borrow(),
    );

    // The sums run on exactly their scratch, whatever the number of pairs.
    let mut sum_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_relinearize_sum_tmp_bytes(
        &glwe_infos,
        size,
        size,
        TorusPrecision(k as u32),
        &tsk_infos,
    ));
    let zero: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);

    // The same pairs prepared once, under both hints on each side, `s = a + c` among them: its canonical flag is
    // clear and its digits carry, so both forms normalize it first.
    let mut s: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    module.glwe_add_into(&mut s, &a, &c);
    assert!(!s.is_canonical());
    let (mixed_a, mixed_b) = ([&a, &s, &c], [&b, &c, &s]);
    let prepared: Vec<_> = mixed_a
        .iter()
        .zip(&mixed_b)
        .enumerate()
        .map(|(i, (l, r))| {
            let (l_hint, r_hint) = crate::test_suite::prepare_hints(i);
            let mut l_prep = module.cnv_pvec_left_alloc(n, 2, size, l_hint);
            glwe_prepare_left(module, &mut l_prep, *l, k, &mut scratch.borrow());
            let mut r_prep = module.cnv_pvec_right_alloc(n, 2, size, r_hint);
            glwe_prepare_right(module, &mut r_prep, *r, k, &mut scratch.borrow());
            (l_prep, r_prep)
        })
        .collect();
    let (prepared_a, prepared_b): (Vec<_>, Vec<_>) = prepared.iter().map(|(l, r)| (l, r)).unzip();
    let mut prepared_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_prepared_relinearize_sum_tmp_bytes(
        &glwe_infos,
        size,
        size,
        TorusPrecision(k as u32),
        &tsk_infos,
    ));

    let res_k = TorusPrecision(k as u32);
    for res_offset in [0, 1, base2k - 1] {
        let mut want: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_mul_relinearize_sum(
            scale + res_offset,
            &mut want,
            res_k,
            &mixed_a,
            &mixed_b,
            &tsk_prep,
            &mut sum_scratch.borrow(),
        );
        let mut have: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_mul_prepared_relinearize_sum(
            scale + res_offset,
            &mut have,
            res_k,
            &prepared_a,
            &prepared_b,
            size,
            glwe_infos.k,
            size,
            glwe_infos.k,
            &tsk_prep,
            &mut prepared_scratch.borrow(),
        );
        assert!(
            have.data() == want.data() && have.is_canonical(),
            "one-pass sum of prepared pairs, offset {res_offset}: not the sum"
        );

        for role in ["product", "square", "left is the destination", "sum of three products"] {
            let sum_role = role == "sum of three products";
            module.vec_znx_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                base2k,
                k,
                res_offset as i64,
                0,
                &vec_znx_backend_ref::<BE>(if sum_role { &pt_sum } else { &pt_square }),
                base2k,
                0,
                &mut scratch.borrow(),
            );
            let cnv_offset = scale + res_offset;
            if role == "left is the destination" {
                module.glwe_copy(&mut res, &a, &mut scratch.borrow());
            }
            match role {
                "square" => module.glwe_square_relinearize(cnv_offset, &mut res, res_k, &a, &tsk_prep, &mut scratch.borrow()),
                "left is the destination" => {
                    module.glwe_mul_relinearize_assign(cnv_offset, &mut res, res_k, &b, &tsk_prep, &mut scratch.borrow())
                }
                "sum of three products" => module.glwe_mul_relinearize_sum(
                    cnv_offset,
                    &mut res,
                    res_k,
                    &[&a, &b, &c],
                    &[&b, &c, &a],
                    &tsk_prep,
                    &mut sum_scratch.borrow(),
                ),
                _ => {
                    module.glwe_mul_relinearize(cnv_offset, &mut res, res_k, &a, &b, &tsk_prep, &mut scratch.borrow());
                    let (sum_a, sum_b) = ([&a, &zero], [&b, &zero]);
                    for len in 1..=2 {
                        // A fresh result each time, so that a sum which left it unwritten fails.
                        let mut sum: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
                        module.glwe_mul_relinearize_sum(
                            cnv_offset,
                            &mut sum,
                            res_k,
                            &sum_a[..len],
                            &sum_b[..len],
                            &tsk_prep,
                            &mut sum_scratch.borrow(),
                        );
                        assert!(
                            sum.data() == res.data(),
                            "one-pass sum of {len} pairs, offset {res_offset}: not the product"
                        );
                    }
                }
            }
            assert_canonical(res.data(), base2k, k);
            glwe_decrypt_checked(module, &res, &mut pt_have, &sk_dft, &mut scratch.borrow());

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(
                base2k,
                pt_tmp.data.size() * base2k,
                0,
                &mut vec_znx_backend_mut::<BE>(&mut pt_tmp.data),
                0,
                &mut scratch.borrow(),
            );
            let noise_have: f64 = pt_tmp.stats().std().log2();
            // A sum of three products is held to three times the bound of one.
            let noise_bound = noise_want(res_offset) + if sum_role { 3f64.log2() } else { 0.0 };
            assert!(
                noise_have - noise_bound <= TENSOR_NOISE_MARGIN,
                "one-pass {role}, offset {res_offset}: {noise_have} > {noise_bound}"
            );
        }
    }

    let mut wide: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&wide_infos);
    let mut wide_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_relinearize_tmp_bytes(
        &wide_infos,
        size,
        size,
        TorusPrecision(k as u32),
        &wide_tsk_infos,
    ));
    module.glwe_mul_relinearize(
        scale,
        &mut wide,
        wide_infos.k,
        &a,
        &b,
        &wide_tsk_prep,
        &mut wide_scratch.borrow(),
    );
    assert_canonical(wide.data(), base2k, wide_infos.k.as_usize());
}

/// A sum needs a pair.
pub fn test_glwe_mul_relinearize_sum_rejects_no_pairs<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEMulRelinearizeSum<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    rejected_sum(params, module, SumCase::NoPairs);
}

/// Every pair of a sum shares the first one's layouts, checked before `res` is written.
pub fn test_glwe_mul_relinearize_sum_rejects_mismatched_layouts<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEMulRelinearizeSum<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    rejected_sum(params, module, SumCase::NarrowPair);
}

/// A sum takes as many right operands as left ones, checked before `res` is written.
pub fn test_glwe_mul_relinearize_sum_rejects_unequal_lengths<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEMulRelinearizeSum<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    rejected_sum(params, module, SumCase::UnequalLengths);
}

enum SumCase {
    NoPairs,
    /// A second pair whose left operand is one limb narrower than the first's.
    NarrowPair,
    /// Two left operands and one right one.
    UnequalLengths,
}

/// Runs a sum without scratch and with an unfilled key.
fn rejected_sum<BE: Backend>(params: &TestParams, module: &Module<BE>, case: SumCase)
where
    Module<BE>: GLWEMulRelinearizeSum<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k = params.base2k;
    let infos = GLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: (2 * base2k).into(),
        rank: 1u32.into(),
    };
    let tsk: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&GLWETensorKeyLayout {
        n: infos.n,
        base2k: infos.base2k,
        dnum: Dnum(2),
        k_aux: (base2k + 1).into(),
        rank: infos.rank,
        dsize: Dsize(1),
    });
    let a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    let narrow: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&GLWELayout {
        k: base2k.into(),
        ..infos
    });
    let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    let (left, right) = match case {
        SumCase::NoPairs => (vec![], vec![]),
        SumCase::NarrowPair => (vec![&a, &narrow], vec![&a, &a]),
        SumCase::UnequalLengths => (vec![&a, &a], vec![&a]),
    };
    module.glwe_mul_relinearize_sum(
        base2k,
        &mut res,
        infos.k,
        &left,
        &right,
        &tsk,
        &mut ScratchOwned::alloc(0).borrow(),
    );
}

/// A sum of prepared pairs needs a pair.
pub fn test_glwe_mul_prepared_relinearize_sum_rejects_no_pairs<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEPreparedTensoring<BE> + GLWETensorKeyPreparedFactory<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    rejected_prepared_sum(params, module, PreparedSumCase::NoPairs);
}

/// Every prepared pair of a sum is of the degree of `res`, of rank 1 and of the given sizes, checked before anything
/// else.
pub fn test_glwe_mul_prepared_relinearize_sum_rejects_mismatched_pairs<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEPreparedTensoring<BE> + GLWETensorKeyPreparedFactory<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    rejected_prepared_sum(params, module, PreparedSumCase::NarrowPair);
}

/// The given sizes of a prepared sum are the limb counts of the given widths, checked before anything else.
pub fn test_glwe_mul_prepared_relinearize_sum_rejects_size_of_another_width<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEPreparedTensoring<BE> + GLWETensorKeyPreparedFactory<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    rejected_prepared_sum(params, module, PreparedSumCase::WiderLeftWidth);
}

/// A prepared sum takes as many right operands as left ones, checked before anything else.
pub fn test_glwe_mul_prepared_relinearize_sum_rejects_unequal_lengths<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEPreparedTensoring<BE> + GLWETensorKeyPreparedFactory<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    rejected_prepared_sum(params, module, PreparedSumCase::UnequalLengths);
}

enum PreparedSumCase {
    NoPairs,
    /// A second pair whose left operand is one limb narrower than the given size.
    NarrowPair,
    /// One pair, its left width one limb wider than the given size.
    WiderLeftWidth,
    /// Two left operands and one right one.
    UnequalLengths,
}

/// Runs a sum of prepared pairs without scratch and with an unfilled key.
fn rejected_prepared_sum<BE: Backend>(params: &TestParams, module: &Module<BE>, case: PreparedSumCase)
where
    Module<BE>: GLWEPreparedTensoring<BE> + GLWETensorKeyPreparedFactory<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k = params.base2k;
    let n = module.n();
    let infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: (2 * base2k).into(),
        rank: 1u32.into(),
    };
    let tsk: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&GLWETensorKeyLayout {
        n: infos.n,
        base2k: infos.base2k,
        dnum: Dnum(2),
        k_aux: (base2k + 1).into(),
        rank: infos.rank,
        dsize: Dsize(1),
    });
    let a = module.cnv_pvec_left_alloc(n, 2, 2, PrepareHint::Reuse);
    let narrow = module.cnv_pvec_left_alloc(n, 2, 1, PrepareHint::Reuse);
    let b = module.cnv_pvec_right_alloc(n, 2, 2, PrepareHint::Reuse);
    let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    let (left, right, a_k) = match case {
        PreparedSumCase::NoPairs => (vec![], vec![], infos.k),
        PreparedSumCase::NarrowPair => (vec![&a, &narrow], vec![&b, &b], infos.k),
        PreparedSumCase::WiderLeftWidth => (vec![&a], vec![&b], (3 * base2k).into()),
        PreparedSumCase::UnequalLengths => (vec![&a, &a], vec![&b], infos.k),
    };
    module.glwe_mul_prepared_relinearize_sum(
        base2k,
        &mut res,
        infos.k,
        &left,
        &right,
        2,
        a_k,
        2,
        infos.k,
        &tsk,
        &mut ScratchOwned::alloc(0).borrow(),
    );
}

/// The prepared operands of a tensor product are of the degree and rank of `res` and of the given sizes, checked
/// before the scratch: the product runs without scratch on a right operand one limb narrower than the given size.
pub fn test_glwe_tensor_apply_prepared_rejects_mismatched_operands<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEPreparedTensoring<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k = params.base2k;
    let n = module.n();
    let infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: (2 * base2k).into(),
        rank: 1u32.into(),
    };
    let a = module.cnv_pvec_left_alloc(n, 2, 2, PrepareHint::Reuse);
    let narrow = module.cnv_pvec_right_alloc(n, 2, 1, PrepareHint::Reuse);
    let mut res: GLWETensor<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_alloc_from_infos(&infos);
    module.glwe_tensor_apply_prepared(
        base2k,
        &mut res,
        &a,
        2,
        &narrow,
        2,
        infos.base2k,
        &mut ScratchOwned::alloc(0).borrow(),
    );
}

/// A prepared operand holds its ciphertext at the given width, checked before the scratch: the left operand is
/// prepared without scratch into one limb more than `ceil(k / base2k)`.
pub fn test_glwe_prepare_left_rejects_size_of_another_width<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: CnvPVecAlloc<BE> + Convolution<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k = params.base2k;
    let n = module.n();
    let infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: (2 * base2k).into(),
        rank: 1u32.into(),
    };
    let a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    let mut a_prep = module.cnv_pvec_left_alloc(n, 2, 3, PrepareHint::Reuse);
    glwe_prepare_left(
        module,
        &mut a_prep,
        &a,
        infos.k.as_usize(),
        &mut ScratchOwned::alloc(0).borrow(),
    );
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
        + VecZnxNormalizeAssign<BE>
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
        let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
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
        module.glwe_encrypt_sk(&mut a, &pt_in, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());

        for res_offset in 0..scale {
            module.glwe_tensor_square_apply(scale + res_offset, &mut res_square, &a, &mut scratch.borrow());
            module.glwe_tensor_apply(scale + res_offset, &mut res_tensor, &a, &a, &mut scratch.borrow());
            assert_canonical(res_square.data(), out_base2k, k);
            assert_canonical(res_tensor.data(), out_base2k, k);

            module.glwe_tensor_relinearize(&mut res_relin_square, &res_square, &tsk_prep, &mut scratch.borrow());
            module.glwe_tensor_relinearize(&mut res_relin_tensor, &res_tensor, &tsk_prep, &mut scratch.borrow());

            // Decrypt one side to ensure the square path remains functionally valid.
            glwe_decrypt_checked(module, &res_relin_square, &mut pt_have, &sk_dft, &mut scratch.borrow());
            glwe_decrypt_checked(module, &res_relin_tensor, &mut pt_want, &sk_dft, &mut scratch.borrow());
            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(
                pt_tmp.base2k().as_usize(),
                pt_tmp.data.size() * pt_tmp.base2k().as_usize(),
                0,
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
        + VecZnxNormalizeAssign<BE>
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

        let glwe_in_infos = GLWELayout {
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

        let (b_size, a_size) = (pt_b.size(), pt_a.size());
        module.vec_znx_fill_uniform_source(
            17,
            b_size * 17,
            &mut vec_znx_backend_mut::<BE>(pt_b.data_mut()),
            0,
            &mut source_xa,
        );
        module.vec_znx_fill_uniform_source(
            17,
            a_size * 17,
            &mut vec_znx_backend_mut::<BE>(pt_a.data_mut()),
            0,
            &mut source_xa,
        );

        let mut pt_want_base2k_in: VecZnx<BE::OwnedBuf, BE::ZnxWord> =
            module.vec_znx_alloc(module.n(), 1, pt_a.size() + pt_b.size());
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

        module.glwe_encrypt_sk(&mut a, &pt_a, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());

        let mut scratch_cnv = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(&res, &a, &pt_b));

        for res_offset in 0..scale {
            module.glwe_mul_plain(scale + res_offset, &mut res, &a, &pt_b, &mut scratch_cnv.borrow());

            glwe_decrypt_checked(module, &res, &mut pt_have, &sk_dft, &mut scratch.borrow());
            module.vec_znx_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                out_base2k,
                k,
                res_offset as i64,
                0,
                &vec_znx_backend_ref::<BE>(&pt_want_base2k_in),
                in_base2k,
                0,
                &mut scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(
                pt_tmp.base2k().as_usize(),
                pt_tmp.data.size() * pt_tmp.base2k().as_usize(),
                0,
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

/// `glwe_mul_plain` and `glwe_mul_plain_assign` with a plaintext of degree
/// `n/2` and `n/4`, never below the backend floor, are bit-identical to the same
/// product with the plaintext embedded at the module degree by `switch_ring`.
pub fn test_glwe_mul_plain_plaintext_degree<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEMulPlain<BE> + VecZnxSwitchRing<BE> + GLWEMaskFill<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n: usize = params.n;
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k;
    let mut source: Source = Source::new([3u8; 32]);
    for rank in 1_usize..=2 {
        let layout: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };
        let mut a: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        module.fill_glwe_from_source(&mut a, &mut source);
        for b_n in [n / 2, n / 4].into_iter().filter(|&d| d >= BE::MIN_DEGREE) {
            let mut pt_small: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
                    n: b_n.into(),
                    base2k: base2k.into(),
                    k: (2 * base2k).into(),
                });
            module.fill_glwe_from_source(&mut pt_small, &mut source);
            let mut pt_dense: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: (2 * base2k).into(),
                });
            module.vec_znx_switch_ring(
                &mut vec_znx_backend_mut::<BE>(pt_dense.data_mut()),
                0,
                &vec_znx_backend_ref::<BE>(pt_small.data()),
                0,
            );
            for cnv_offset in [0usize, base2k] {
                let mut res_small: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
                let mut res_dense: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
                // The assign form multiplies `res` in place, so its budget is
                // the product one taken at `res` for both ciphertext operands.
                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    module
                        .glwe_mul_plain_tmp_bytes(&res_dense, &a, &pt_dense)
                        .max(module.glwe_mul_plain_tmp_bytes(&res_dense, &res_dense, &pt_dense)),
                );
                module.glwe_mul_plain(cnv_offset, &mut res_small, &a, &pt_small, &mut scratch.borrow());
                module.glwe_mul_plain(cnv_offset, &mut res_dense, &a, &pt_dense, &mut scratch.borrow());
                assert_eq!(
                    res_small.data().raw(),
                    res_dense.data().raw(),
                    "glwe_mul_plain: b.n()={b_n} cnv_offset={cnv_offset} rank={rank}"
                );

                let mut assign_small: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
                let mut assign_dense: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
                assign_small.data_mut().raw_mut().copy_from_slice(a.data().raw());
                assign_dense.data_mut().raw_mut().copy_from_slice(a.data().raw());
                module.glwe_mul_plain_assign(cnv_offset, &mut assign_small, &pt_small, &mut scratch.borrow());
                module.glwe_mul_plain_assign(cnv_offset, &mut assign_dense, &pt_dense, &mut scratch.borrow());
                assert_eq!(
                    assign_small.data().raw(),
                    assign_dense.data().raw(),
                    "glwe_mul_plain_assign: b.n()={b_n} cnv_offset={cnv_offset} rank={rank}"
                );
            }
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
        + VecZnxNormalizeAssign<BE>
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

        let glwe_in_infos = GLWELayout {
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

        let a_size = pt_a.size();
        module.vec_znx_fill_uniform_source(
            17,
            a_size * 17,
            &mut vec_znx_backend_mut::<BE>(pt_a.data_mut()),
            0,
            &mut source_xa,
        );

        let mask = (1 << in_base2k) - 1;
        for j in 0..1 {
            let r = source_xa.next_u64() & mask;
            pt_b.data_mut().at_mut(0, j)[b_coeff] = ((r << (64 - 17)) as i64) >> (64 - 17);
        }

        let mut pt_want_base2k_in: VecZnx<BE::OwnedBuf, BE::ZnxWord> =
            module.vec_znx_alloc(module.n(), 1, pt_a.size() + pt_b.size());
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

        module.glwe_encrypt_sk(&mut a, &pt_a, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());

        for res_offset in 0..scale {
            module.glwe_mul_const(scale + res_offset, &mut res, &a, &pt_b, b_coeff, &mut scratch.borrow());

            glwe_decrypt_checked(module, &res, &mut pt_have, &sk_dft, &mut scratch.borrow());
            module.vec_znx_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                out_base2k,
                k,
                res_offset as i64,
                0,
                &vec_znx_backend_ref::<BE>(&pt_want_base2k_in),
                in_base2k,
                0,
                &mut scratch.borrow(),
            );

            module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);
            module.vec_znx_normalize_assign(
                pt_tmp.base2k().as_usize(),
                pt_tmp.data.size() * pt_tmp.base2k().as_usize(),
                0,
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
///
/// The relinearization runs on exactly the scratch its query returns, at rank 1
/// and 2, also with a key at a quarter of the operand's radix: the call converts
/// the operand to the key's radix a second time while it holds the key-switched
/// sum, which the query must count.
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
        + VecZnxNormalizeAssign<BE>
        + GLWESecretTensorFactory<BE>
        + VecZnxNormalize<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n: usize = module.n();

    // Keep both radices inside the envelope selected by the caller. A fixed
    // radix could exceed the accumulator range before exercising this regression.
    let hi: usize = params.base2k;
    let lo: usize = hi.checked_sub(1).expect("cross-radix test requires base2k >= 2");

    // (operand radix, key radix, operand precision), chosen so that the storage
    // width and the exact precision round to different key-limb counts.
    let radixes = [(hi, lo), (lo, hi), (hi, (hi / 4).max(1))];
    for (rank, (a_base2k, key_base2k)) in [1usize, 2].into_iter().flat_map(|rank| radixes.map(|radix| (rank, radix))) {
        let k: usize = a_base2k
            .checked_mul(4)
            .and_then(|v| v.checked_add(1))
            .expect("cross-radix test precision overflows usize");
        let glwe_infos = GLWELayout {
            n: n.into(),
            base2k: a_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };
        assert_ne!(
            (glwe_infos.size() * a_base2k).div_ceil(key_base2k),
            k.div_ceil(key_base2k),
            "the case must separate the storage width from the exact precision"
        );

        let key_infos = |base2k: usize| GLWETensorKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: Dnum(k.div_ceil(base2k) as u32),
            k_aux: (base2k + module.log_n()).into(),
            rank: rank.into(),
            dsize: Dsize(1),
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
        module.glwe_encrypt_sk(&mut a, &pt_in, &sk_dft, &mut source_xe, &mut source_xa, &mut scratch.borrow());
        module.glwe_tensor_apply(scale, &mut res_tensor, &a, &a, &mut scratch.borrow());
        assert_canonical(res_tensor.data(), a_base2k, k);

        let mut tsk: GLWETensorKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_tensor_key_alloc_from_infos(&cross_infos);
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );
        let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&cross_infos);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut scratch.borrow());
        // Reaching the end is the regression: deriving the operand width from
        // the storage precision hands the product a limb count its bound does
        // not accept, and it rejects the call before doing any work; a query
        // short of what the call takes runs out of this exact scratch.
        let mut exact =
            ScratchOwned::<BE>::alloc(module.glwe_tensor_relinearize_tmp_bytes(&res_cross, &res_tensor, &cross_infos));
        module.glwe_tensor_relinearize(&mut res_cross, &res_tensor, &tsk_prep, &mut exact.borrow());
    }
}
