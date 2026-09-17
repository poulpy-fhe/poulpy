//! Compact plaintexts: a plaintext holding `slots < N/2` slots stored at its own
//! degree. Every consumer must give the result the dense plaintext of the same
//! values gives.

use poulpy_core::layouts::LWEInfos;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow, VecZnxSwitchRing},
    layouts::{Backend, HostBytesBackend, Module},
};

use crate::{
    CKKSInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps, CKKSSubOps},
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision,
            ckks_decrypt_decode, ckks_encrypt, gen_sk, quantize, test_vector_1, test_vector_2, upload_pt, want_add, want_mul,
        },
        reference_encoder::ReferenceEncoder,
    },
};

/// The slot count every compact test uses: a quarter of the dense count, so
/// the plaintext is a degree-`N/4` object (`2 * slots`), above every backend
/// floor at the suite's `N = 256`.
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
    BE: TestContextBackend,
    Module<HostBytesBackend>: CKKSModuleAlloc<HostBytesBackend>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let prec = params.prec();
    let mut dense = host_module.ckks_pt_vec_alloc(params.base2k.into(), prec.k());
    dense.set_meta(prec.meta());
    encoder.encode_reim(&mut dense, re, im).unwrap();
    let mut compact = host_module.ckks_pt_vec_alloc_compact(re.len(), params.base2k.into(), prec.k());
    compact.set_meta(prec.meta());
    encoder.encode_reim(&mut compact, re, im).unwrap();
    assert_eq!(compact.n().as_usize(), 2 * re.len(), "compact plaintext degree");
    (upload_pt(module, &dense), upload_pt(module, &compact))
}

/// `switch_ring` of the compact plaintext decodes to the dense plaintext's slots.
pub fn test_compact_plaintext_embedding<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> <BE as Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + VecZnxSwitchRing<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = compact_slots(&params);
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
    assert_eq!((want_re, want_im), (have_re, have_im), "switch_ring(compact) != dense");
}

/// `ct + compact`, `ct - compact` and `ct * compact` decrypt to exactly what the
/// same operations with the dense plaintext decrypt to, and to the expected
/// values within the suite precision; a degree that does not embed is an error.
pub fn test_compact_plaintext_add_sub_mul<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
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
