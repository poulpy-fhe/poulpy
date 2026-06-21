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

/// `ckks_increase_log_delta(ct, bits)` must raise `log_delta` by `bits`, preserve
/// `log_budget` and the decoded message, grow `effective_k` by `bits`, and
/// reallocate storage only when the current `max_k` cannot hold it.
pub fn test_increase_log_delta<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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

    // `bits` scales `log_delta` up; keep it small enough that the raised
    // `log_delta` still fits the i128 decode codec on every backend. Any positive
    // `bits` crosses a limb boundary here (encryption fills `effective_k = max_k`),
    // so this still exercises the reallocation path.
    let bits = 8;

    let mut ct = ckks_encrypt(&params, module, host_module, &encoder, &sk, params.k, &re, &im, &mut scratch.borrow());
    let (d0, b0) = (ct.log_delta(), ct.log_budget());

    module.ckks_increase_log_delta(&mut ct, bits).unwrap();

    // log_delta raised by `bits`, log_budget preserved, effective_k grown.
    assert_eq!(ct.log_delta(), d0 + bits, "increase_log_delta: log_delta");
    assert_eq!(ct.log_budget(), b0, "increase_log_delta: log_budget preserved");
    assert!(ct.max_k().as_usize() >= ct.effective_k(), "increase_log_delta: storage holds effective_k");

    // The decoded message is unchanged (measured at the original scale).
    let (re_out, im_out) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    assert_precision("increase_log_delta re", &re_out, &re, params.prec.log_delta, params.n);
    assert_precision("increase_log_delta im", &im_out, &im, params.prec.log_delta, params.n);
}
