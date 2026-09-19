//! The derived composites at the backend floor, `n = 8`: each one must run
//! inside the scratch its `_tmp_bytes` advertises, at the smallest degree the
//! reference backends serve.
//!
//! At `n = 8` every limb is already a multiple of the 64-byte scratch
//! alignment, so the rounding of a temporary shorter than one alignment unit
//! is not probed here; it needs a degree below the floor or a backend whose
//! alignment exceeds a limb.

use poulpy_hal::{
    layouts::Module,
    test_suite::{
        TestParams,
        convolution::{test_convolution_add, test_convolution_by_const_add},
        derived::{
            test_cnv_apply_dft_add_derived, test_cnv_by_const_apply_add_derived, test_vec_znx_idft_normalize_consume_derived,
            test_vec_znx_lsh_add_derived, test_vec_znx_lsh_sub_derived, test_vec_znx_rsh_add_derived,
            test_vec_znx_rsh_sub_derived, test_vmp_apply_dft_derived, test_vmp_apply_dft_to_dft_add_derived,
        },
    },
};

use crate::{FFT64Ref, NTT4x30Ref};

const PARAMS: TestParams = TestParams {
    size: 8,
    base2k: 17,
    n: 8,
};

fn run<BE>(module: &Module<BE>)
where
    BE: poulpy_hal::test_suite::TestBackend
        + poulpy_hal::oep::HalVecZnxImpl
        + poulpy_hal::oep::HalVmpImpl
        + poulpy_hal::oep::HalVecZnxDftImpl
        + poulpy_hal::oep::HalConvolutionImpl,
{
    test_vec_znx_lsh_add_derived(&PARAMS, module);
    test_vec_znx_lsh_sub_derived(&PARAMS, module);
    test_vec_znx_rsh_add_derived(&PARAMS, module);
    test_vec_znx_rsh_sub_derived(&PARAMS, module);
    test_vec_znx_idft_normalize_consume_derived(&PARAMS, module);
    test_cnv_apply_dft_add_derived(&PARAMS, module);
    test_cnv_by_const_apply_add_derived(&PARAMS, module);
}

/// The VMP composites.
fn run_vmp<BE>(module: &Module<BE>)
where
    BE: poulpy_hal::test_suite::TestBackend + poulpy_hal::oep::HalVmpImpl,
{
    test_vmp_apply_dft_derived(&PARAMS, module);
    test_vmp_apply_dft_to_dft_add_derived(&PARAMS, module);
}

#[test]
fn derived_scratch_fft64_ref_n8() {
    let module = Module::<FFT64Ref>::new(8);
    run(&module);
    run_vmp(&module);
    test_convolution_add(&module, 8, 17);
    test_convolution_by_const_add(&module, 8, 17);
}

#[test]
fn derived_scratch_ntt4x30_ref_n8() {
    let module = Module::<NTT4x30Ref>::new(8);
    run(&module);
    run_vmp(&module);
    test_convolution_add(&module, 8, 17);
    test_convolution_by_const_add(&module, 8, 17);
}
