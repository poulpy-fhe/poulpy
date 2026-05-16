use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{test_convolution, test_convolution_by_const, test_convolution_pairwise},
};

use crate::NTT126Ifma;

#[cfg(test)]
mod ntt126_ifma_tests {
    use crate::ntt126_ifma::{
        primes::{PrimeSetNtt126Ifma, Primes42},
        reference::arithmetic::b_ntt126_ifma_to_znx128_ref,
        vec_znx_dft::simd_b_ntt126_ifma_to_znx128,
    };
    use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

    cross_backend_test_suite! {
        mod vec_znx,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vec_znx_add_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_into_backend_matches_reference,
            test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
            test_vec_znx_extract_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_extract_coeff_backend,
            test_vec_znx_normalize_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_coeff_backend,
            test_vec_znx_normalize_coeff_assign_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_coeff_assign_backend,
            test_vec_znx_lsh_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_coeff_backend,
            test_vec_znx_lsh_add_coeff_into_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_add_coeff_into_backend,
            test_vec_znx_rsh_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_coeff_backend,
            test_vec_znx_rsh_add_coeff_into_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_add_coeff_into_backend,
            test_vec_znx_rsh_sub_coeff_into_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_sub_coeff_into_backend,
            test_vec_znx_add_scalar_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_into,
            test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
            test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
            test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
            test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
            test_vec_znx_sub_scalar => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar,
            test_vec_znx_sub_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar_assign,
            test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
            test_vec_znx_rsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_assign,
            test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
            test_vec_znx_lsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_assign,
            test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
            test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
            test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
            test_vec_znx_rotate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_assign,
            test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
            test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
            test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
            test_vec_znx_mul_xp_minus_one_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_assign,
            test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
            test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
            test_vec_znx_merge_rings => poulpy_hal::test_suite::vec_znx::test_vec_znx_merge_rings,
            test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
            test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
            test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        }
    }

    cross_backend_test_suite! {
        mod vec_znx_dft,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vec_znx_dft_add_into => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_into,
            test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
            test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
            test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
            test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
        }
    }

    cross_backend_test_suite! {
        mod svp,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
            test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
        }
    }

    cross_backend_test_suite! {
        mod vmp,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
            test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
        }
    }

    cross_backend_test_suite! {
        mod vec_znx_big,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vec_znx_big_add_into => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_into,
            test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
            test_vec_znx_big_add_small_into => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_into,
            test_vec_znx_big_add_small_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_assign,
            test_vec_znx_big_sub => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub,
            test_vec_znx_big_sub_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_assign,
            test_vec_znx_big_negate => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate,
            test_vec_znx_big_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate_assign,
            test_vec_znx_big_normalize => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize,
            test_vec_znx_big_sub_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_assign,
            test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
            test_vec_znx_big_sub_small_a_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_assign,
            test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
            test_vec_znx_big_sub_small_b_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_assign,
            test_vec_znx_big_automorphism => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism,
            test_vec_znx_big_automorphism_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism_assign,
        }
    }

    backend_test_suite! {
        mod sampling,
        backend = crate::NTT126Ifma,
        params = TestParams { size: 1<<12, base2k: 50 },
        tests = {
            test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
            test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
            test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
        }
    }

    // NTT size-range coverage.
    //
    // The IFMA NTT runs outer levels breadth-first while `nn > NTT_BLOCK`,
    // then switches to block-local depth-first for the inner levels.  For
    // `n <= NTT_BLOCK` no breadth-first pass runs at all.  These suites
    // exercise both regimes (block-local only for small `n`; mixed for
    // larger `n`) and the transition sizes, confirming bit-exact agreement
    // with the reference backend.

    // n = 1024: only block-local inner levels run.
    cross_backend_test_suite! {
        mod ntt_n1024,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<10, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    // n = 8192: exercises multiple breadth-first outer levels.
    cross_backend_test_suite! {
        mod ntt_n8192,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<13, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    // n = 16384: large size where the working set exceeds L1.
    cross_backend_test_suite! {
        mod ntt_n16384,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<14, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    // n = 32768: large size where the working set exceeds L2 on typical cores.
    cross_backend_test_suite! {
        mod ntt_n32768,
        backend_ref =  poulpy_cpu_ref::NTT120Ref,
        backend_test = crate::NTT126Ifma,
        params = TestParams { size: 1<<15, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    #[test]
    fn test_b_to_znx128_ifma_asm_edges_vs_ref() {
        const Q: [u64; 3] = <Primes42 as PrimeSetNtt126Ifma>::Q;
        let big_q = Q[0] as u128 * Q[1] as u128 * Q[2] as u128;
        let values = [
            0u128,
            1,
            Q[0] as u128 - 1,
            Q[1] as u128 - 1,
            Q[2] as u128 - 1,
            big_q / 2 - 1,
            big_q / 2,
            big_q / 2 + 1,
            big_q - 2,
            big_q - 1,
            123_456_789,
            (1u128 << 63) - 1,
            (1u128 << 64) + 17,
            (1u128 << 95) + 0x12345,
            (1u128 << 120) + 0x6789,
            big_q / 3,
            (2 * big_q) / 3,
            big_q - 123_456_789,
            42,
        ];

        fn fill_b_format(dst: &mut [u64], values: &[u128], q: &[u64; 3]) {
            for (i, &value) in values.iter().enumerate() {
                for k in 0..3 {
                    let residue = (value % q[k] as u128) as u64;
                    dst[4 * i + k] = if (i + k).is_multiple_of(2) { residue } else { residue + q[k] };
                }
                dst[4 * i + 3] = 0;
            }
        }

        fn assert_matches_ref(n: usize, b: &[u64]) {
            let mut got = vec![0i128; n];
            let mut expected = vec![0i128; n];
            unsafe { simd_b_ntt126_ifma_to_znx128(n, &mut got, b) };
            b_ntt126_ifma_to_znx128_ref(n, &mut expected, b);
            assert_eq!(got, expected);
        }

        let n = values.len();
        let mut b = vec![0u64; 4 * n];
        fill_b_format(&mut b, &values, &Q);
        assert_matches_ref(n, &b);

        let mut skewed = vec![0u64; 4 * n + 1];
        fill_b_format(&mut skewed[1..], &values, &Q);
        assert_matches_ref(n, &skewed[1..]);
    }
}

#[test]
fn test_convolution_by_const_ntt126_ifma() {
    let module: Module<NTT126Ifma> = Module::<NTT126Ifma>::new(8);
    test_convolution_by_const(&module, 12);
}

#[test]
fn test_convolution_ntt126_ifma() {
    let module: Module<NTT126Ifma> = Module::<NTT126Ifma>::new(8);
    test_convolution(&module, 12);
}

#[test]
fn test_convolution_pairwise_ntt126_ifma() {
    let module: Module<NTT126Ifma> = Module::<NTT126Ifma>::new(8);
    test_convolution_pairwise(&module, 12);
}

#[test]
#[should_panic(expected = "NTT126Ifma requires n >= 8")]
fn test_ntt126_ifma_rejects_too_small_ring() {
    let _ = Module::<NTT126Ifma>::new(4);
}
