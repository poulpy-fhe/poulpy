//! Compact plaintexts: a plaintext holding `slots < N/2` slots stored at its own
//! degree. Every consumer must give the result the dense plaintext of the same
//! values gives.

use std::collections::HashMap;

use poulpy_core::layouts::{Diagonals, Evaluate, LWEInfos, LinearTransformationStrategy};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow, VecZnxSwitchRing},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, Module, ScratchArena},
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
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision,
            ckks_decrypt_decode, ckks_encrypt, gen_atk, gen_sk, gen_sk_with_raw, quantize, test_vector_1, test_vector_2,
            upload_pt, want_add, want_mul,
        },
        reference_encoder::ReferenceEncoder,
    },
};

/// The default slot count of the compact tests: a quarter of the dense count,
/// so the plaintext is a degree-`N/4` object (`2 * slots`), above every backend
/// floor at the suite's `N = 256`. The embedding test additionally runs two
/// slots, where `2 * slots` falls below every floor and the degree is clamped.
fn compact_slots(params: &CKKSTestParams) -> usize {
    params.n / 8
}

/// Encodes `re`, `im` twice: into a dense (degree-`N`) plaintext and into the
/// compact one, and uploads both.
fn encode_dense_and_compact<BE, F, E>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    params: &CKKSTestParams,
    encoder: &ReferenceEncoder<E>,
    re: &[F],
    im: &[F],
) -> (CKKSPlaintextOwned<BE>, CKKSPlaintextOwned<BE>)
where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    Module<HostBytesBackend>: CKKSModuleAlloc<HostBytesBackend>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let prec = params.prec();
    let mut dense = host_module.ckks_pt_vec_alloc(params.base2k.into(), prec.k());
    dense.set_meta(prec.meta());
    encoder.encode_reim(&mut dense, re, im).unwrap();
    // The host module's floor is below a vector backend's, so the compact degree
    // is derived for `BE` and the host plaintext is allocated at it directly.
    let n = (2 * re.len()).max(BE::MIN_DEGREE).min(params.n);
    let mut compact = host_module.ckks_pt_coeffs_alloc(n, params.base2k.into(), prec.k());
    compact.set_meta(prec.meta());
    encoder.encode_reim(&mut compact, re, im).unwrap();
    assert_eq!(
        module
            .ckks_pt_vec_alloc_compact(re.len(), params.base2k.into(), prec.k())
            .n()
            .as_usize(),
        n,
        "compact plaintext degree"
    );
    (upload_pt(module, &dense), upload_pt(module, &compact))
}

/// `switch_ring` of the compact plaintext decodes to the dense plaintext's
/// slots, at the suite's compact slot count and at two slots. Two slots puts
/// `2 * m` below every backend floor, so the degree is clamped to the floor and
/// the encoder writes a gap inside the compact polynomial; the identity must
/// hold there too.
pub fn test_compact_plaintext_embedding<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + VecZnxSwitchRing<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    for m in [compact_slots(&params), 2] {
        let encoder = ReferenceEncoder::<E>::new(m).unwrap();
        let (re, im) = test_vector_1::<F>(m);
        let (dense, compact) = encode_dense_and_compact(host_module, module, &params, &encoder, &re, &im);
        let mut embedded = module.ckks_plaintext_alloc_from_infos(&dense);
        {
            let mut res = crate::GLWEToBackendMut::<BE>::to_backend_mut(&mut embedded);
            let src = crate::GLWEToBackendRef::<BE>::to_backend_ref(&compact);
            module.vec_znx_switch_ring(res.data_mut(), 0, src.data(), 0);
        }
        let (mut want_re, mut want_im) = (vec![F::zero(); m], vec![F::zero(); m]);
        let (mut have_re, mut have_im) = (vec![F::zero(); m], vec![F::zero(); m]);
        encoder
            .decode_reim(&dense.to_host_owned::<BE>(), &mut want_re, &mut want_im)
            .unwrap();
        encoder
            .decode_reim(&embedded.to_host_owned::<BE>(), &mut have_re, &mut have_im)
            .unwrap();
        assert_eq!(
            (want_re, want_im),
            (have_re, have_im),
            "switch_ring(compact) != dense at {m} slots"
        );
    }
}

/// `ct + compact`, `ct - compact` and `ct * compact` decrypt to exactly what the
/// same operations with the dense plaintext decrypt to, and to the expected
/// values within the suite precision; a degree that does not embed is an error.
pub fn test_compact_plaintext_add_sub_mul<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = compact_slots(&params);
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
    let (dense, compact) = encode_dense_and_compact(host_module, module, &params, &encoder, &re2, &im2);
    let log_delta = params.prec().log_delta();
    let (q_re2, q_im2) = (quantize(&re2, log_delta), quantize(&im2, log_delta));

    // add
    let (want_re, want_im) = want_add(&re1, &im1, &q_re2, &q_im2);
    let mut with_dense = alloc_ct(&params, module, params.k);
    let mut with_compact = alloc_ct(&params, module, params.k);
    module
        .ckks_add_pt_vec_into(&mut with_dense, &ct, &dense, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_add_pt_vec_into(&mut with_compact, &ct, &compact, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "compact_add_pt",
        &params,
        module,
        &encoder,
        &with_compact,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    let dec_dense: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_dense, &sk, &mut scratch.borrow());
    let dec_compact: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_compact, &sk, &mut scratch.borrow());
    assert_eq!(dec_dense, dec_compact, "add: compact plaintext != dense plaintext");

    // sub
    let neg_re2: Vec<F> = q_re2.iter().map(|v| F::zero() - *v).collect();
    let neg_im2: Vec<F> = q_im2.iter().map(|v| F::zero() - *v).collect();
    let (want_re, want_im) = want_add(&re1, &im1, &neg_re2, &neg_im2);
    let mut with_dense = alloc_ct(&params, module, params.k);
    let mut with_compact = alloc_ct(&params, module, params.k);
    module
        .ckks_sub_pt_vec_into(&mut with_dense, &ct, &dense, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_sub_pt_vec_into(&mut with_compact, &ct, &compact, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "compact_sub_pt",
        &params,
        module,
        &encoder,
        &with_compact,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    let dec_dense: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_dense, &sk, &mut scratch.borrow());
    let dec_compact: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_compact, &sk, &mut scratch.borrow());
    assert_eq!(dec_dense, dec_compact, "sub: compact plaintext != dense plaintext");

    // mul: the compact plaintext is prepared at its own degree and read through
    // the sparse right slot of the convolution.
    let (want_re, want_im) = want_mul(&re1, &im1, &q_re2, &q_im2);
    let mut with_dense = alloc_ct(&params, module, params.k);
    let mut with_compact = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut with_dense, &ct, &dense, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_mul_pt_vec_into(&mut with_compact, &ct, &compact, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "compact_mul_pt",
        &params,
        module,
        &encoder,
        &with_compact,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    let dec_dense: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_dense, &sk, &mut scratch.borrow());
    let dec_compact: (Vec<F>, Vec<F>) = ckks_decrypt_decode(&params, module, &encoder, &with_compact, &sk, &mut scratch.borrow());
    assert_eq!(dec_dense, dec_compact, "mul: compact plaintext != dense plaintext");

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
        .ckks_mul_pt_vec_assign(&mut assigned, &compact, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "compact_mul_pt_assign",
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
            .ckks_add_pt_vec_into(&mut with_dense, &ct, &odd, &mut scratch.borrow())
            .is_err(),
        "ckks_add_pt_vec_into accepted a plaintext of degree 3"
    );
    assert!(
        module
            .ckks_mul_pt_vec_into(&mut with_dense, &ct, &odd, &mut scratch.borrow())
            .is_err(),
        "ckks_mul_pt_vec_into accepted a plaintext of degree 3"
    );
    // A degree below the backend floor is an error too.
    let mut tiny = host_module.ckks_pt_coeffs_alloc(BE::MIN_DEGREE / 2, params.base2k.into(), params.prec().k());
    tiny.set_meta(params.prec().meta());
    let tiny = upload_pt(module, &tiny);
    assert!(
        module
            .ckks_add_pt_vec_into(&mut with_dense, &ct, &tiny, &mut scratch.borrow())
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

/// Encodes the matrix into a transformation whose diagonals are compact
/// plaintexts, through the same closure shape the production encoder uses.
fn encode_compact_lt<BE, F>(
    module: &Module<BE>,
    params: &CKKSTestParams,
    b: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
    scratch: &mut ScratchArena<'_, BE>,
) -> LinearTransformation<CKKSPlaintextOwned<BE>>
where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingHostOps<BE, F>,
    F: TestScalar,
{
    let prec = params.prec();
    b.build_transform(strategy, |pre_re, pre_im| {
        let mut pt = module.ckks_pt_vec_alloc_compact(pre_re.len(), params.base2k.into(), prec.k());
        pt.set_meta_checked(prec.meta()).unwrap();
        module.ckks_encode_reim_into(&mut pt, pre_re, pre_im, scratch).unwrap();
        pt
    })
}

/// `dec(lt(enc(a), B)) ≈ B·a` with compact diagonals on the prepared path (the
/// prepared slots carry the compact degree) and on the streamed path.
pub fn test_compact_linear_transformation<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingHostOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = compact_slots(&params);
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
    let lt = encode_compact_lt(module, &params, &b, strategy, &mut scratch.borrow());
    let first = lt.first_diagonal_plaintext().unwrap();
    assert_eq!(first.n().as_usize(), 2 * m, "diagonal is not compact");

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

    // Prepared path: the slots are allocated and prepared at the compact degree
    // under the one module.
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
    assert_eq!(
        prepared.first_diagonal_plaintext().unwrap().n().as_usize(),
        2 * m,
        "prepared slot is not compact"
    );
    module
        .ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow())
        .unwrap();
    let mut ct_out = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_out, &ct, &prepared, &atks, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "compact_lt_prepared",
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
    // prepared on the fly at the compact degree.
    let mut ct_streamed = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_streamed, &ct, &lt, &atks, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "compact_lt_streamed",
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

/// The production diagonal encoder emits compact diagonals for a sparse matrix
/// and dense ones for a full one.
pub fn test_compact_diagonal_encoder<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let mut scratch = alloc_scratch(&params, module);
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: 2 };
    for (slots, want_n) in [(compact_slots(&params), params.n / 4), (params.n / 2, params.n)] {
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
