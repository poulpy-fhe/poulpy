//! Plaintexts below the ring degree: a plaintext holding `slots < N/2` slots is
//! stored at its own degree. Every consumer must give the result the degree-`N`
//! plaintext of the same values gives.

use std::collections::HashMap;

use poulpy_core::layouts::{
    Diagonals, Evaluate, GLWEInfos, GLWEPlaintextInfos, LWEInfos, LinearTransformationStrategy, SetGLWEPlaintextInfos,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow, VecZnxSwitchRing},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, Module, ScratchArena, Standard},
};

use crate::{
    CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSSubOps,
        LinearTransformation, LinearTransformationPrepared, ckks_linear_transformation_output_layout,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals, slots_degree},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision,
            ckks_decrypt_decode, ckks_encrypt, gen_atk, gen_sk, gen_sk_with_raw, quantize, test_vector_1, test_vector_2,
            upload_pt, want_add, want_mul,
        },
        reference_encoder::ReferenceEncoder,
    },
};

/// The default slot count of these tests: a quarter of the degree-`N` count,
/// so the plaintext is a degree-`N/4` object (`2 * slots`), above every backend
/// floor at the suite's `N = 256`. The embedding test additionally runs two
/// slots, where `2 * slots` falls below every floor and the degree is clamped.
fn test_slots(params: &CKKSTestParams) -> usize {
    params.n / 8
}

/// Encodes `re`, `im` twice: into a degree-`N` plaintext and into one at the
/// degree of its slots, and uploads both.
fn encode_at_both_degrees<BE, F, E>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    params: &CKKSTestParams,
    encoder: &ReferenceEncoder<E>,
    re: &[F],
    im: &[F],
) -> (CKKSPlaintextOwned<BE>, CKKSPlaintextOwned<BE>)
where
    BE: TestContextBackend<Ring = Standard>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let prec = params.prec();
    let mut full = host_module.ckks_pt_vec_alloc(params.base2k.into(), prec.k());
    full.set_meta(prec.meta());
    encoder.encode_reim(&mut full, re, im).unwrap();
    // The host module's floor is below a vector backend's, so the degree
    // is derived for `BE` and the host plaintext is allocated at it directly.
    let n = (2 * re.len()).max(BE::MIN_DEGREE).min(params.n);
    let mut small = host_module.ckks_pt_coeffs_alloc(n, params.base2k.into(), prec.k());
    small.set_meta(prec.meta());
    encoder.encode_reim(&mut small, re, im).unwrap();
    assert_eq!(slots_degree::<BE, _>(module, re.len()).as_usize(), n, "plaintext degree");
    (upload_pt(module, &full), upload_pt(module, &small))
}

/// `switch_ring` of the plaintext at the degree of its slots decodes to the
/// degree-`N` plaintext's slots, at the suite's slot count and at two slots. Two
/// slots puts `2 * m` below every backend floor, so the degree is clamped to the
/// floor and the encoder writes a gap inside the polynomial; the identity must
/// hold there too.
pub fn test_plaintext_degree_embedding<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + VecZnxSwitchRing<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    for m in [test_slots(&params), 2] {
        let encoder = ReferenceEncoder::<E>::new(m).unwrap();
        let (re, im) = test_vector_1::<F>(m);
        let (full, small) = encode_at_both_degrees(host_module, module, &params, &encoder, &re, &im);
        let mut embedded = module.ckks_plaintext_alloc_from_infos(&full);
        {
            let mut res = crate::GLWEToBackendMut::<BE>::to_backend_mut(&mut embedded);
            let src = crate::GLWEToBackendRef::<BE>::to_backend_ref(&small);
            module.vec_znx_switch_ring(res.data_mut(), 0, src.data(), 0);
        }
        let (mut want_re, mut want_im) = (vec![F::zero(); m], vec![F::zero(); m]);
        let (mut have_re, mut have_im) = (vec![F::zero(); m], vec![F::zero(); m]);
        encoder
            .decode_reim(&full.to_host_owned::<BE>(), &mut want_re, &mut want_im)
            .unwrap();
        encoder
            .decode_reim(&embedded.to_host_owned::<BE>(), &mut have_re, &mut have_im)
            .unwrap();
        assert_eq!(
            (want_re, want_im),
            (have_re, have_im),
            "switch_ring(small) != full at {m} slots"
        );
    }
}

/// `ct + pt`, `ct - pt` and `ct * pt` for `pt` at the degree of its slots decrypt
/// to exactly what the same operations with the degree-`N` plaintext decrypt to, and to the expected
/// values within the suite precision; a degree that does not embed is an error.
pub fn test_plaintext_degree_add_sub_mul<BE, F, E>(
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
    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let (full, small) = encode_at_both_degrees(host_module, module, &params, &encoder, &re2, &im2);
    let log_delta = params.prec().log_delta();
    let (q_re2, q_im2) = (quantize(&re2, log_delta), quantize(&im2, log_delta));

    // add
    let (want_re, want_im) = want_add(&re1, &im1, &q_re2, &q_im2);
    let mut with_full = alloc_ct(&params, module, params.k);
    let mut with_small = alloc_ct(&params, module, params.k);
    module
        .ckks_add_pt_vec_into(&mut with_full, &ct, &full, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_add_pt_vec_into(&mut with_small, &ct, &small, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "small_add_pt",
        &params,
        module,
        &encoder,
        &with_small,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    let dec_full: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_full, &sk, &mut scratch.borrow());
    let dec_small: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_small, &sk, &mut scratch.borrow());
    assert_eq!(dec_full, dec_small, "add: plaintext at its degree != degree-N plaintext");

    // sub
    let neg_re2: Vec<F> = q_re2.iter().map(|v| F::zero() - *v).collect();
    let neg_im2: Vec<F> = q_im2.iter().map(|v| F::zero() - *v).collect();
    let (want_re, want_im) = want_add(&re1, &im1, &neg_re2, &neg_im2);
    let mut with_full = alloc_ct(&params, module, params.k);
    let mut with_small = alloc_ct(&params, module, params.k);
    module
        .ckks_sub_pt_vec_into(&mut with_full, &ct, &full, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_sub_pt_vec_into(&mut with_small, &ct, &small, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "small_sub_pt",
        &params,
        module,
        &encoder,
        &with_small,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    let dec_full: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_full, &sk, &mut scratch.borrow());
    let dec_small: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_small, &sk, &mut scratch.borrow());
    assert_eq!(dec_full, dec_small, "sub: plaintext at its degree != degree-N plaintext");

    // mul: the plaintext is prepared at its own degree and read through
    // the sparse right slot of the convolution.
    let (want_re, want_im) = want_mul(&re1, &im1, &q_re2, &q_im2);
    let mut with_full = alloc_ct(&params, module, params.k);
    let mut with_small = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut with_full, &ct, &full, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_mul_pt_vec_into(&mut with_small, &ct, &small, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "small_mul_pt",
        &params,
        module,
        &encoder,
        &with_small,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    let dec_full: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_full, &sk, &mut scratch.borrow());
    let dec_small: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_small, &sk, &mut scratch.borrow());
    assert_eq!(dec_full, dec_small, "mul: plaintext at its degree != degree-N plaintext");

    // mul assign, on a fresh encryption of the same values.
    let mut assigned = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    module
        .ckks_mul_pt_vec_assign(&mut assigned, &small, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "small_mul_pt_assign",
        &params,
        module,
        &encoder,
        &assigned,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    // A degree that is not a power-of-two divisor is an error, on the add and on the mul.
    let mut odd = host_module.ckks_pt_coeffs_alloc(3, params.base2k.into(), params.prec().k());
    odd.set_meta(params.prec().meta());
    let odd = upload_pt(module, &odd);
    assert!(
        module
            .ckks_add_pt_vec_into(&mut with_full, &ct, &odd, &mut scratch.borrow())
            .is_err(),
        "ckks_add_pt_vec_into accepted a plaintext of degree 3"
    );
    assert!(
        module
            .ckks_mul_pt_vec_into(&mut with_full, &ct, &odd, &mut scratch.borrow())
            .is_err(),
        "ckks_mul_pt_vec_into accepted a plaintext of degree 3"
    );
    // A degree below the backend floor is an error too.
    let mut tiny = host_module.ckks_pt_coeffs_alloc(BE::MIN_DEGREE / 2, params.base2k.into(), params.prec().k());
    tiny.set_meta(params.prec().meta());
    let tiny = upload_pt(module, &tiny);
    assert!(
        module
            .ckks_add_pt_vec_into(&mut with_full, &ct, &tiny, &mut scratch.borrow())
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

/// Encodes the matrix into a transformation whose diagonals are plaintexts at the
/// degree of their period, through the same closure shape the production encoder uses.
fn encode_lt_at_period_degree<BE, F>(
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
        let mut pt = module.ckks_plaintext_alloc(slots_degree::<BE, _>(module, pre_re.len()), params.base2k.into(), prec.k());
        pt.set_meta_checked(prec.meta()).unwrap();
        module.ckks_encode_reim_into(&mut pt, pre_re, pre_im, scratch).unwrap();
        pt
    })
}

/// `dec(lt(enc(a), B)) ≈ B·a` with diagonals at the degree of their period on the
/// prepared path (the prepared slots carry that degree) and on the streamed path.
pub fn test_linear_transformation_diagonal_degree<BE, F, E>(
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
    let lt = encode_lt_at_period_degree(module, &params, &b, strategy, &mut scratch.borrow());
    let first = lt.first_diagonal_plaintext().unwrap();
    assert_eq!(first.n().as_usize(), 2 * m, "diagonal is not at the degree of its period");

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

    // Prepared path: the slots are allocated and prepared at the diagonals' degree
    // under the one module.
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
    assert_eq!(
        prepared.first_diagonal_plaintext().unwrap().n().as_usize(),
        2 * m,
        "prepared slot is not at the diagonals' degree"
    );
    module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow());
    let mut ct_out = alloc_ct(&params, module, params.k);
    // A stale label on `dst` is overwritten: the dense source makes the result dense.
    ct_out.set_log_sparsity(1);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_out, &ct, &prepared, &atks, &mut scratch.borrow())
        .unwrap();
    let src_layout = CKKSLayout {
        glwe_layout: ct.glwe_layout(),
        meta: ct.meta(),
    };
    let predicted = ckks_linear_transformation_output_layout(&prepared, src_layout, ct_out.k());
    assert_eq!((ct_out.log_sparsity(), predicted.log_sparsity()), (0, 0), "lt_label");
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

/// The production diagonal encoder emits diagonals at the degree of their period:
/// below `N` for a sparse matrix, at `N` for a full one.
pub fn test_diagonal_encoder_degree<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut scratch = alloc_scratch(&params, module);
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: 2 };
    for (slots, want_n) in [(test_slots(&params), params.n / 4), (params.n / 2, params.n)] {
        let lt = crate::reference::ckks_encode_linear_transformation_from_diagonals::<BE, F>(
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
