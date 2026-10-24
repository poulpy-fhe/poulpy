//! Plaintexts of `m < N/2` slots: every consumer reads one as its `m` slots repeated over the ciphertext's.

use std::collections::HashMap;

use poulpy_core::layouts::{Diagonals, Evaluate, LWEInfos, LinearTransformationStrategy};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, Module, ScratchArena, Standard},
};

use crate::{
    CKKSInfos, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSSubOps,
        LinearTransformation, LinearTransformationPrepared,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision, ckks_encrypt,
            gen_atk, gen_sk, gen_sk_with_raw, quantize, test_vector_1, test_vector_2, upload_pt, want_add, want_mul,
        },
        reference_encoder::ReferenceEncoder,
    },
};

/// The slot count of these tests, an eighth of the degree, so the plaintexts' degree `N / 4` clears every backend floor
/// at the suite's `N = 256`.
fn test_slots(params: &CKKSTestParams) -> usize {
    params.n / 8
}

/// `ct + pt`, `ct - pt`, `ct * pt` and `ct *= pt` for `pt` of `m < N/2` slots decrypt to the expected values within the
/// suite precision; a plaintext degree that does not embed is an error.
pub fn test_plaintext_slots_add_sub_mul<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = test_slots(&params);
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let encrypt = |scratch: &mut ScratchArena<'_, BE>| {
        ckks_encrypt(&params, module, host_module, &encoder, &sk, params.k, &re1, &im1, scratch)
    };
    let ct = encrypt(&mut scratch.borrow());
    let prec = params.prec();
    let mut host_pt = host_module.ckks_pt_vec_alloc(m, params.base2k.into(), prec.k());
    host_pt.set_meta(prec.meta());
    encoder.encode_reim(&mut host_pt, &re2, &im2).unwrap();
    let pt = upload_pt(module, &host_pt);
    let (q_re2, q_im2) = (quantize(&re2, prec.log_delta()), quantize(&im2, prec.log_delta()));
    let neg_re2: Vec<F> = q_re2.iter().map(|v| F::zero() - *v).collect();
    let neg_im2: Vec<F> = q_im2.iter().map(|v| F::zero() - *v).collect();
    let check = |label: &str, res: &_, (want_re, want_im): (Vec<F>, Vec<F>), scratch: &mut ScratchArena<'_, BE>| {
        assert_decrypt_precision(label, &params, module, &encoder, res, &sk, &want_re, &want_im, scratch)
    };

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_add_pt_vec_into(&mut res, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    check("add_pt", &res, want_add(&re1, &im1, &q_re2, &q_im2), &mut scratch.borrow());
    module
        .ckks_sub_pt_vec_into(&mut res, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    check(
        "sub_pt",
        &res,
        want_add(&re1, &im1, &neg_re2, &neg_im2),
        &mut scratch.borrow(),
    );
    // The plaintext is prepared at its own degree and read through the convolution's embedding slot.
    module
        .ckks_mul_pt_vec_into(&mut res, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    check("mul_pt", &res, want_mul(&re1, &im1, &q_re2, &q_im2), &mut scratch.borrow());
    let mut assigned = encrypt(&mut scratch.borrow());
    module
        .ckks_mul_pt_vec_assign(&mut assigned, &pt, &mut scratch.borrow())
        .unwrap();
    check(
        "mul_pt_assign",
        &assigned,
        want_mul(&re1, &im1, &q_re2, &q_im2),
        &mut scratch.borrow(),
    );

    // A degree that is not a power-of-two divisor is an error, on the add and on the mul.
    let mut odd = host_module.ckks_pt_coeffs_alloc(3, params.base2k.into(), prec.k());
    odd.set_meta(prec.meta());
    let odd = upload_pt(module, &odd);
    assert!(
        module
            .ckks_add_pt_vec_into(&mut res, &ct, &odd, &mut scratch.borrow())
            .is_err(),
        "ckks_add_pt_vec_into accepted a plaintext of degree 3"
    );
    assert!(
        module
            .ckks_mul_pt_vec_into(&mut res, &ct, &odd, &mut scratch.borrow())
            .is_err(),
        "ckks_mul_pt_vec_into accepted a plaintext of degree 3"
    );
    // A degree below the backend floor is an error too.
    let mut tiny = host_module.ckks_pt_coeffs_alloc(BE::MIN_DEGREE / 2, params.base2k.into(), prec.k());
    tiny.set_meta(prec.meta());
    let tiny = upload_pt(module, &tiny);
    assert!(
        module
            .ckks_add_pt_vec_into(&mut res, &ct, &tiny, &mut scratch.borrow())
            .is_err(),
        "ckks_add_pt_vec_into accepted a plaintext below the backend floor"
    );
}

/// A six-diagonal complex matrix on `m` slots (the shape `linear_transformation.rs` uses).
fn matrix<F: TestScalar>(m: usize) -> ComplexDiagonals<F> {
    let mut re = Diagonals::<F>::new(m);
    let mut im = Diagonals::<F>::new(m);
    for i in 0..6usize {
        re.set(
            i as i64,
            (0..m)
                .map(|j| F::from_f64(0.25 * (i as f64 + 1.0) / (1.0 + (j % 8) as f64)).unwrap())
                .collect(),
        );
        im.set(
            i as i64,
            (0..m)
                .map(|j| F::from_f64(0.125 * (i as f64 + 1.0) / (1.0 + ((j + 3) % 8) as f64)).unwrap())
                .collect(),
        );
    }
    ComplexDiagonals::new(re, im)
}

/// Encodes the matrix into a transformation whose diagonals are plaintexts of its slots, through the closure shape the
/// production encoder uses.
fn encode_lt<BE, F>(
    module: &Module<BE>,
    params: &CKKSTestParams,
    b: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
    scratch: &mut ScratchArena<'_, BE>,
) -> LinearTransformation<CKKSPlaintextOwned<BE>>
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingHostOps<BE, F>,
    F: TestScalar,
{
    let prec = params.prec();
    b.build_transform(strategy, |pre_re, pre_im| {
        let mut pt = module.ckks_pt_vec_alloc(pre_re.len(), params.base2k.into(), prec.k());
        pt.set_meta_checked(prec.meta()).unwrap();
        module.ckks_encode_reim_into(&mut pt, pre_re, pre_im, scratch).unwrap();
        pt
    })
}

/// `dec(lt(enc(a), B)) ≈ B·a` for a map of `m < N/2` slots, whose diagonals are plaintexts of `m` slots, on the prepared
/// path (the prepared slots take their degree) and on the streamed path.
pub fn test_linear_transformation_diagonal_slots<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingHostOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = test_slots(&params);
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (a_re, a_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let key_params = CKKSTestParams {
        dsize: params.dsize.max(4),
        ..params
    };
    let mut scratch = alloc_scratch(&key_params, module);
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: 2 };
    let b = matrix::<F>(m);
    let lt = encode_lt(module, &params, &b, strategy, &mut scratch.borrow());
    let first = lt.first_diagonal_plaintext().unwrap();
    assert_eq!(first.n().as_usize(), 2 * m, "diagonal is not a plaintext of the map's slots");

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in lt.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&key_params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = b.evaluate((a_re.as_slice(), a_im.as_slice()), strategy);

    // Prepared path: the slots are allocated and prepared at the diagonals' degree under the one module.
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
    assert_eq!(
        prepared.first_diagonal_plaintext().unwrap().n().as_usize(),
        2 * m,
        "prepared slot is not at the diagonals' degree"
    );
    module
        .ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow())
        .unwrap();
    let mut ct_out = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_out, &ct, &prepared, &atks, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "lt_prepared",
        &params,
        module,
        &encoder,
        &ct_out,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    // Streamed path: the same transformation with its plaintext diagonals,
    // prepared on the fly at their degree.
    let mut ct_streamed = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_streamed, &ct, &lt, &atks, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "lt_streamed",
        &params,
        module,
        &encoder,
        &ct_streamed,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// The production diagonal encoder emits plaintexts of the map's slots: degree `N / 4` for a map of `N / 8` slots, `N`
/// for one of every slot.
pub fn test_diagonal_encoder_slots<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut scratch = alloc_scratch(&params, module);
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: 2 };
    for (slots, want_n) in [(test_slots(&params), params.n / 4), (params.n / 2, params.n)] {
        let lt = crate::encoding::ckks_encode_linear_transformation_from_diagonals::<BE, F>(
            module,
            params.base2k.into(),
            params.prec().into(),
            &matrix::<F>(slots),
            strategy,
            false,
            &mut scratch.borrow(),
        )
        .unwrap();
        assert_eq!(lt.first_diagonal_plaintext().unwrap().n().as_usize(), want_n, "slots {slots}");
    }
}
