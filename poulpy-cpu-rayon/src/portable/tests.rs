//! Conformance of the portable Rayon variants against their serial bases.

use poulpy_hal::{
    backend_test_suite, cross_backend_test_suite,
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_add, test_convolution_by_const_add, test_convolution_pairwise, test_convolution_sum,
    },
};

use super::FFT64PortableRayon;

// The size crosses the parallel-work floors of the overrides that have them.
cross_backend_test_suite! {
    mod vec_znx_dft_fft64,
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64PortableRayon,
    params = TestParams { size: 1<<14, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
    }
}

cross_backend_test_suite! {
    mod vmp_fft64,
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64PortableRayon,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
    }
}

backend_test_suite! {
    mod window_fft64,
    backend = crate::FFT64PortableRayon,
    params = TestParams { size: 1 << 8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
        test_vec_znx_dft_step_zero_rejected => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_step_zero_rejected,
        test_convolution_prepare_shape_rejected => poulpy_hal::test_suite::convolution::test_convolution_prepare_shape_rejected,
        test_vmp_apply_dft_to_dft_shape_rejected => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_shape_rejected,
        test_vec_znx_sparse_add_sub => poulpy_hal::test_suite::sparse::test_vec_znx_sparse_add_sub,
        test_vec_znx_big_sparse_add_sub => poulpy_hal::test_suite::sparse::test_vec_znx_big_sparse_add_sub,
        test_convolution_sparse => poulpy_hal::test_suite::convolution::test_convolution_sparse,
        test_convolution_by_const_degree_rejected => poulpy_hal::test_suite::convolution::test_convolution_by_const_degree_rejected,
    }
}

#[test]
fn test_convolution_fft64() {
    let module = Module::<FFT64PortableRayon>::new(1 << 8);
    test_convolution(&module, module.n(), 12);
    test_convolution_add(&module, module.n(), 12);
    test_convolution_pairwise(&module, module.n(), 12);
    test_convolution_sum(&module, module.n(), 12);
    test_convolution_by_const_add(&module, module.n(), 12);
}

poulpy_cpu_portable::conjugate_invariant_test_suite!(
    ci_fft64,
    crate::FFT64CIPortableRayon,
    crate::FFT64PortableRayon,
    reference = poulpy_cpu_portable::FFT64CIPortable
);
