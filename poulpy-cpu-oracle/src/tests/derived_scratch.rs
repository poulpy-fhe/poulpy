//! The derived composites at `n = 4`, where a one-limb temporary is 32 bytes
//! and the arena realigns the nested take to 64. The `_tmp_bytes` formulas
//! must round their temporaries, or the scratch they advertise is short.

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

use crate::{FFT64Oracle, NTT4x30Oracle};

const PARAMS: TestParams = TestParams { size: 4, base2k: 17 };

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

fn run_vmp<BE>(module: &Module<BE>)
where
    BE: poulpy_hal::test_suite::TestBackend + poulpy_hal::oep::HalVmpImpl,
{
    test_vmp_apply_dft_derived(&PARAMS, module);
    test_vmp_apply_dft_to_dft_add_derived(&PARAMS, module);
}

#[test]
fn derived_scratch_fft64_ref_n4() {
    let module = Module::<FFT64Oracle>::new(4);
    run(&module);
    run_vmp(&module);
    test_convolution_add(&module, 17);
    test_convolution_by_const_add(&module, 17);
}

#[test]
fn derived_scratch_ntt4x30_ref_n4() {
    let module = Module::<NTT4x30Oracle>::new(4);
    run(&module);
    run_vmp(&module);
    test_convolution_add(&module, 17);
    test_convolution_by_const_add(&module, 17);
}
