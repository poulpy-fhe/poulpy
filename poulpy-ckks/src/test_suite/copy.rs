//! Copy tests.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_copy_aligned`] | CKKS copy into an equally-sized output |
//! | [`test_copy_smaller_output`] | CKKS copy into a smaller output, forcing the left-shift path |

use crate::api::CKKSCopyOps;

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision,
    assert_unary_output_meta, ckks_encrypt, gen_sk, test_vector_1,
};
use anyhow::Result;
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

pub fn test_copy_aligned<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
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
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_copy(&mut ct_res, &ct, &mut scratch.borrow())?;
    assert_unary_output_meta("copy", &ct_res, &ct);
    assert_decrypt_precision(
        "copy",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    Ok(())
}

pub fn test_copy_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
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
    let mut ct_res = alloc_ct(&params, module, ct.k().as_usize() - 1);
    module.ckks_copy(&mut ct_res, &ct, &mut scratch.borrow())?;
    assert_unary_output_meta("copy smaller_output", &ct_res, &ct);
    assert_decrypt_precision(
        "copy smaller_output",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    Ok(())
}
