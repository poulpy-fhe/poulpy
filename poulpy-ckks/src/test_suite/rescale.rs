//! Rescale-family tests (currently: `ckks_increase_log_delta`).

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module},
};

use poulpy_core::layouts::LWEInfos;

use crate::{
    CKKSInfos,
    api::CKKSRescaleOps,
    encoding::reim::Encoder,
    layouts::CKKSCiphertext,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_scratch, assert_precision,
            ckks_decrypt_decode, ckks_encrypt, gen_sk, test_vector_1,
        },
    },
};

/// `ckks_set_log_delta(ct, target)` must move `log_delta` to `target` (up or
/// down), preserve `log_budget` and the decoded message, and grow/compact the
/// storage accordingly.
pub fn test_set_log_delta<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSRescaleOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertext<BE::OwnedBuf>: crate::SetCKKSInfos,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new::<F>(m).unwrap();
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (re, im) = test_vector_1::<F>(m);

    // `bits` is the scale delta exercised in each direction; keep it small enough
    // that the raised `log_delta` still fits the i128 decode codec on every
    // backend. Any positive `bits` crosses a limb boundary here (encryption fills
    // `effective_k = max_k`), so this exercises the realloc/compact paths.
    let bits = 8;

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    let (d0, b0) = (ct.log_delta(), ct.log_budget());

    // Increase: log_delta raised, log_budget preserved, effective_k grown.
    module.ckks_set_log_delta(&mut ct, d0 + bits).unwrap();
    assert_eq!(ct.log_delta(), d0 + bits, "set_log_delta up: log_delta");
    assert_eq!(ct.log_budget(), b0, "set_log_delta up: log_budget preserved");
    assert!(
        ct.max_k().as_usize() >= ct.effective_k(),
        "set_log_delta up: storage holds effective_k"
    );
    let (re_up, im_up) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    assert_precision("set_log_delta up re", &re_up, &re, params.prec.log_delta, params.n);
    assert_precision("set_log_delta up im", &im_up, &im, params.prec.log_delta, params.n);

    // Decrease back to the original scale: log_budget preserved, storage compacted,
    // and the message recovered (dropping only the zero low-order padding bits).
    module.ckks_set_log_delta(&mut ct, d0).unwrap();
    assert_eq!(ct.log_delta(), d0, "set_log_delta down: log_delta");
    assert_eq!(ct.log_budget(), b0, "set_log_delta down: log_budget preserved");
    let (re_dn, im_dn) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    assert_precision("set_log_delta down re", &re_dn, &re, params.prec.log_delta, params.n);
    assert_precision("set_log_delta down im", &im_dn, &im, params.prec.log_delta, params.n);
}
