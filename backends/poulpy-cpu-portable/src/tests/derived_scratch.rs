//! The derived composites at the backend floor, `n = 8`: each one must run
//! inside the scratch its `_tmp_bytes` advertises, at the smallest degree the
//! portable backends serve.
//!
//! At `n = 8` every limb is already a multiple of the 64-byte scratch
//! alignment, so the rounding of a temporary shorter than one alignment unit
//! is not probed here; it needs a degree below the floor or a backend whose
//! alignment exceeds a limb.

use poulpy_hal::{
    layouts::{Module, Ring, Standard},
    oep::{HalCnvImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
    test_suite::{
        TestBackend, TestParams,
        convolution::{test_convolution_add, test_convolution_by_const_add},
        derived::{
            test_cnv_apply_dft_add_derived, test_cnv_by_const_apply_add_derived, test_vec_znx_idft_normalize_derived,
            test_vec_znx_lsh_add_derived, test_vec_znx_lsh_sub_derived, test_vec_znx_rsh_add_derived,
            test_vec_znx_rsh_sub_derived, test_vmp_apply_dft_derived, test_vmp_apply_dft_to_dft_add_derived,
        },
    },
};

use crate::{FFT64Portable, NTT4x30Portable};

const PARAMS: TestParams = TestParams {
    size: 8,
    base2k: 17,
    n: 8,
};

fn run<BE, Rg: Ring>(module: &Module<BE>)
where
    BE: TestBackend<Rg>
        + HalVecZnxImpl<Rg>
        + HalVmpImpl<Rg>
        + HalVecZnxDftImpl<Rg>
        + HalCnvImpl<Rg>
        + HalVecZnxBigImpl<Standard>
        + HalVmpImpl<Standard>
        + HalVecZnxDftImpl<Standard>
        + HalCnvImpl<Standard>,
{
    test_vec_znx_lsh_add_derived(&PARAMS, module);
    test_vec_znx_lsh_sub_derived(&PARAMS, module);
    test_vec_znx_rsh_add_derived(&PARAMS, module);
    test_vec_znx_rsh_sub_derived(&PARAMS, module);
    test_vec_znx_idft_normalize_derived(&PARAMS, module);
    test_cnv_apply_dft_add_derived(&PARAMS, module);
    test_cnv_by_const_apply_add_derived(&PARAMS, module);
}

/// The VMP composites.
fn run_vmp<BE, Rg: Ring>(module: &Module<BE>)
where
    BE: TestBackend<Rg> + HalVmpImpl<Rg> + HalVecZnxImpl<Rg> + HalVmpImpl<Standard> + HalVecZnxDftImpl<Standard>,
{
    test_vmp_apply_dft_derived(&PARAMS, module);
    test_vmp_apply_dft_to_dft_add_derived(&PARAMS, module);
}

#[test]
fn derived_scratch_fft64_portable_n8() {
    let module = Module::<FFT64Portable>::new(8);
    run::<_, Standard>(&module);
    run_vmp::<_, Standard>(&module);
    test_convolution_add::<_, _, Standard>(&module, 8, 17);
    test_convolution_by_const_add::<_, _, Standard>(&module, 8, 17);
}

#[test]
fn derived_scratch_ntt4x30_portable_n8() {
    let module = Module::<NTT4x30Portable>::new(8);
    run::<_, Standard>(&module);
    run_vmp::<_, Standard>(&module);
    test_convolution_add::<_, _, Standard>(&module, 8, 17);
    test_convolution_by_const_add::<_, _, Standard>(&module, 8, 17);
}
