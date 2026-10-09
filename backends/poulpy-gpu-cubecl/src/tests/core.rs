//! The upstream `poulpy-core` suites, with `NTT4x30Portable` as the comparison backend.

use poulpy_hal::layouts::{ConjugateInvariant, Module};

type Gpu = crate::NTT4x30Gpu<super::Runtime, poulpy_hal::layouts::Standard>;
type GpuCI = crate::NTT4x30Gpu<super::Runtime, ConjugateInvariant>;

poulpy_core::core_parity_test_suite! {
    mod core_parity,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    // computes at the module degree, no sweep
    params = TestParams { size: 1 << 8, base2k: 52, n: 1 << 8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
        gglwe_keyswitch => poulpy_core::test_suite::parity::test_gglwe_keyswitch_parity,
        glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
        glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
        glwe_copy_zero => poulpy_core::test_suite::parity::test_glwe_copy_zero_parity,
        glwe_shift => poulpy_core::test_suite::parity::test_glwe_shift_parity,
        glwe_multiplication => poulpy_core::test_suite::parity::test_glwe_multiplication_parity,
        ggsw_rotate => poulpy_core::test_suite::parity::test_ggsw_rotate_parity,
        gadget_external_product => poulpy_core::test_suite::parity::test_gadget_external_product_parity,
        gadget_conversion => poulpy_core::test_suite::parity::test_gadget_conversion_parity,
        lwe_conversion => poulpy_core::test_suite::parity::test_lwe_conversion_parity,
        gglwe_product_digits_strided => poulpy_core::test_suite::parity::test_gglwe_product_digits_strided_parity,
        polynomial_evaluation => poulpy_core::test_suite::parity::test_polynomial_evaluation_parity,
        trace_packing => poulpy_core::test_suite::parity::test_trace_packing_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        sampling => poulpy_core::test_suite::sampling::test_sampling_contract,
        preparation => poulpy_core::test_suite::parity::test_preparation_contract,
        glwe_add => poulpy_core::test_suite::parity::test_glwe_add_parity,
        glwe_sub => poulpy_core::test_suite::parity::test_glwe_sub_parity,
        glwe_negate => poulpy_core::test_suite::parity::test_glwe_negate_parity,
        glwe_normalize => poulpy_core::test_suite::parity::test_glwe_normalize_parity,
        glwe_rotate => poulpy_core::test_suite::parity::test_glwe_rotate_parity,
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu
);

poulpy_hal::backend_test_suite! {
    mod sampling,
    backend = super::Gpu,
    params = TestParams { size: 1 << 12, base2k: 50, n: 1 << 12 },
    tests = {
        test_vec_znx_add_noise => poulpy_core::test_suite::sampling::test_vec_znx_add_noise,
        test_vec_znx_big_add_noise => poulpy_core::test_suite::sampling::test_vec_znx_big_add_noise,
    }
}

/// Rank-one tensor on enough limbs for the convolution kernel to fold its accumulators several times per output.
#[test]
fn test_glwe_tensor_many_limbs() {
    use poulpy_core::{
        layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision},
        test_suite::parity::test_glwe_tensor_parity_for_layout,
    };
    let layout = GLWELayout {
        n: Degree(1 << 10),
        base2k: Base2K(52),
        k: TorusPrecision(52 * 29 + 1),
        rank: Rank(1),
    };
    let comparison = Module::<poulpy_cpu_portable::NTT4x30Portable>::new(u64::from(layout.n.0));
    let tested = Module::<Gpu>::new(u64::from(layout.n.0));
    test_glwe_tensor_parity_for_layout(&layout, &[0, 51], &comparison, &tested);
}

// `conjugate_invariant_core_test_suite!` of `poulpy-cpu-portable` without its
// coarsened-automorphism check, which reads backend allocations as host slices.
#[test]
fn conjugate_invariant_core_parity() {
    let n = 256;
    let reference = Module::<poulpy_cpu_portable::NTT4x30CIPortable>::new(n as u64);
    let module = Module::<GpuCI>::new(n as u64);
    let params = poulpy_hal::test_suite::TestParams { size: n, n, base2k: 10 };
    let shapes = poulpy_core::test_suite::parity::ParityShapes {
        ranks: vec![1, 2],
        dsizes: Some(vec![1, 2]),
        unaligned_digits: true,
    };
    poulpy_core::test_suite::parity::test_glwe_keyswitch_parity(&params, &shapes, &reference, &module);
    poulpy_core::test_suite::parity::test_glwe_external_product_parity(&params, &shapes, &reference, &module);
    poulpy_core::test_suite::parity::test_glwe_automorphism_parity(&params, &shapes, &reference, &module);
}
