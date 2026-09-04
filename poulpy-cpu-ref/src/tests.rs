use poulpy_hal::{
    api::VecZnxCanonicalize,
    layouts::{Module, VecZnxToBackendMut, ZnxView, ZnxViewMut, vec_znx_backend_mut_with_size},
    test_suite::convolution::{
        test_convolution, test_convolution_accumulate, test_convolution_accumulate_fused, test_convolution_by_const,
        test_convolution_by_const_add, test_convolution_pairwise,
    },
};

use crate::{FFT64Ref, NTT4x30Ref};

#[test]
fn test_vec_znx_canonicalize_matches_shift_roundtrip() {
    use crate::reference::vec_znx::{vec_znx_lsh_assign, vec_znx_rsh_assign};

    let module = Module::<FFT64Ref>::new(8);
    let (cols, size) = (3, 4);
    for base2k in [3, 17, 30, 50, 63] {
        let capacity = size * base2k;
        let mut precisions = vec![0, 1, base2k - 1, base2k, base2k + 1, 2 * base2k - 2, capacity - 1, capacity];
        precisions.sort_unstable();
        precisions.dedup();

        for k in precisions {
            let mut actual = module.vec_znx_alloc(cols, size);
            let mut state = (base2k as u64) << 32 | k as u64;
            for value in actual.raw_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *value = (state as i64) >> 40;
            }
            let mut expected = actual.clone();

            let mut actual_view = <_ as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut actual);
            module.vec_znx_canonicalize(base2k, k, &mut actual_view);

            let active_size = k.div_ceil(base2k);
            let mut expected_view = <_ as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut expected);
            for col in 0..cols {
                for limb in active_size..size {
                    expected_view.at_mut(col, limb).fill(0);
                }
            }
            let padding = (base2k - k % base2k) % base2k;
            if active_size != 0 && padding != 0 {
                let mut active = vec_znx_backend_mut_with_size::<FFT64Ref>(expected_view, active_size);
                let mut tmp = vec![0i64; 2 * module.n()];
                for col in 0..cols {
                    vec_znx_rsh_assign::<FFT64Ref>(base2k, padding, &mut active, col, &mut tmp);
                    vec_znx_lsh_assign::<FFT64Ref>(base2k, padding, &mut active, col, &mut tmp[..module.n()]);
                }
            }

            assert_eq!(actual.data(), expected.data(), "base2k={base2k}, k={k}");
        }
    }

    let mut value = module.vec_znx_alloc(1, 2);
    value.at_mut(0, 1)[0] = i64::MAX;
    let mut value_view = <_ as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut value);
    module.vec_znx_canonicalize(64, 65, &mut value_view);
    assert_eq!(value_view.at(0, 1)[0], i64::MIN);
    assert_eq!(value_view.at(0, 0)[0], 1);
}

#[cfg(feature = "enable-ckks")]
mod ckks_tests;
#[cfg(feature = "enable-core")]
mod delegating_backend;

#[test]
fn test_convolution_by_const_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution_by_const(&module, 17);
    test_convolution_by_const_add(&module, 17);
}

#[test]
fn test_convolution_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution(&module, 17);
}

#[test]
fn test_convolution_pairwise_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution_pairwise(&module, 17);
}

#[test]
fn test_convolution_accumulate_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution_accumulate(&module, 17);
}

#[test]
fn test_convolution_accumulate_fused_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution_accumulate_fused(&module, 17);
}

#[test]
fn test_convolution_by_const_ntt4x30_ref() {
    let module: Module<NTT4x30Ref> = Module::<NTT4x30Ref>::new(8);
    test_convolution_by_const(&module, 50);
    test_convolution_by_const_add(&module, 50);
}

#[test]
fn test_convolution_ntt4x30_ref() {
    let module: Module<NTT4x30Ref> = Module::<NTT4x30Ref>::new(8);
    test_convolution(&module, 50);
}

#[test]
fn test_convolution_pairwise_ntt4x30_ref() {
    let module: Module<NTT4x30Ref> = Module::<NTT4x30Ref>::new(8);
    test_convolution_pairwise(&module, 50);
}

#[test]
fn test_convolution_accumulate_ntt4x30_ref() {
    let module: Module<NTT4x30Ref> = Module::<NTT4x30Ref>::new(8);
    test_convolution_accumulate(&module, 50);
}

#[test]
fn test_convolution_accumulate_fused_ntt4x30_ref() {
    let module: Module<NTT4x30Ref> = Module::<NTT4x30Ref>::new(8);
    test_convolution_accumulate_fused(&module, 50);
}

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vec_znx_zero_backend_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_zero_backend_matches_wrapper,
        test_vec_znx_add_into_backend_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_into_backend_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_vec_znx_add_assign_backend_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign_backend_matches_wrapper,
        test_vec_znx_add_const_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_const_into,
        test_vec_znx_add_const_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_const_assign,
        test_vec_znx_extract_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_extract_coeff_backend,
        test_vec_znx_normalize_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_coeff_backend,
        test_vec_znx_normalize_coeff_assign_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_coeff_assign_backend,
        test_vec_znx_lsh_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_coeff_backend,
        test_vec_znx_lsh_add_coeff_into_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_add_coeff_into_backend,
        test_vec_znx_lsh_add_coeff_to_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_add_coeff_to_coeff_backend,
        test_vec_znx_lsh_sub_coeff_to_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_sub_coeff_to_coeff_backend,
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
        test_vec_znx_negate_backend_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_backend_matches_wrapper,
        test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
        test_vec_znx_negate_assign_backend_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign_backend_matches_wrapper,
        test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
        test_vec_znx_rotate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_assign,
        test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
        test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
        test_scalar_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_scalar_znx_automorphism,
        test_scalar_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_scalar_znx_automorphism_assign,
        test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
        test_vec_znx_mul_xp_minus_one_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_assign,
        test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
        test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
        test_vec_znx_canonicalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_canonicalize,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_switch_ring_backend_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring_backend_matches_wrapper,
        test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        test_vec_znx_copy_backend_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_backend_matches_wrapper,
        test_vec_znx_copy_range_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_range_backend,
    }
}
cross_backend_test_suite! {
    mod svp,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}
cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vec_znx_big_add_into => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_into,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
        test_vec_znx_big_seed_add_normal_matches_source_wrapper => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_seed_add_normal_matches_source_wrapper,
        test_vec_znx_big_add_small_into => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_into,
        test_vec_znx_big_add_small_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_assign,
        test_vec_znx_big_sub => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub,
        test_vec_znx_big_sub_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_assign,
        test_vec_znx_big_automorphism => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism,
        test_vec_znx_big_automorphism_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism_assign,
        test_vec_znx_big_negate => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate,
        test_vec_znx_big_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate_assign,
        test_vec_znx_big_normalize => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize,
        test_vec_znx_big_sub_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_assign,
        test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
        test_vec_znx_big_sub_small_a_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_assign,
        test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
        test_vec_znx_big_sub_small_b_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_assign,
    }
}
cross_backend_test_suite! {
    mod vec_znx_dft,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vec_znx_dft_add_into => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_into,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_copy,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
    }
}
cross_backend_test_suite! {
    mod vec_znx_dft_automorphism,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
    }
}
cross_backend_test_suite! {
    mod vmp,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT4x30Ref,
    params = TestParams { size: 1<<12, base2k: 12 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_vec_znx_seed_sampling_matches_source_wrappers => poulpy_hal::test_suite::vec_znx::test_vec_znx_seed_sampling_matches_source_wrappers,
        test_scalar_znx_binary_hw_has_exact_weight => poulpy_hal::test_suite::vec_znx::test_scalar_znx_binary_hw_has_exact_weight,
        test_scalar_znx_secret_seed_sampling_matches_source_wrappers => poulpy_hal::test_suite::vec_znx::test_scalar_znx_secret_seed_sampling_matches_source_wrappers,
        test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
        test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
    }
}

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod fft64,
    backend = crate::FFT64Ref,
    params = TestParams { size: 1<<8, base2k: 17 },
);

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod ntt4x30,
    backend = crate::NTT4x30Ref,
    params = TestParams { size: 1<<8, base2k: 52 },
);

#[test]
fn test_vec_znx_rsh_assign_multi_limb_matches_rsh() {
    use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRshAssignBackend, VecZnxRshBackend, VecZnxRshTmpBytes};
    use poulpy_hal::layouts::{FillUniform, HostBytesBackend, ScratchOwned, VecZnx};
    use poulpy_hal::source::Source;
    use poulpy_hal::test_suite::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};

    let n = 8usize;
    let module: Module<NTT4x30Ref> = Module::<NTT4x30Ref>::new(n as u64);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(n as u64);
    let mut scratch: ScratchOwned<NTT4x30Ref> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes());
    let base2k = 52usize;
    let mut source = Source::new([3u8; 32]);

    // shifts spanning >= 2 limbs previously corrupted the in-place variant
    for size in [2usize, 3, 4] {
        for k in [60usize, 90, 105, 116] {
            if k / base2k + 1 > size {
                continue;
            }
            let mut a: VecZnx<Vec<u8>, i64> = module_host.vec_znx_alloc(1, size);
            a.fill_uniform(base2k, &mut source);
            let a_be = upload_vec_znx::<NTT4x30Ref>(&a);
            let mut want_be = upload_vec_znx::<NTT4x30Ref>(&module_host.vec_znx_alloc(1, size));
            module.vec_znx_rsh_backend(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Ref>(&mut want_be),
                0,
                &vec_znx_backend_ref::<NTT4x30Ref>(&a_be),
                0,
                &mut scratch.borrow(),
            );
            let mut got_be = upload_vec_znx::<NTT4x30Ref>(&a);
            module.vec_znx_rsh_assign_backend(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Ref>(&mut got_be),
                0,
                &mut scratch.borrow(),
            );
            assert_eq!(
                download_vec_znx::<NTT4x30Ref>(&got_be),
                download_vec_znx::<NTT4x30Ref>(&want_be),
                "vec_znx_rsh_assign mismatch for size={size} k={k}"
            );
        }
    }
}

/// Compile-time regression check: container equality is byte equality, so the
/// DFT/big-family containers implement `Eq` even when the logical word is
/// `f64` (a derived `Eq` used to demand `W: Eq` and silently vanish here).
#[allow(dead_code)]
fn assert_f64_word_containers_are_eq() {
    fn requires_eq<T: Eq>() {}
    requires_eq::<poulpy_hal::layouts::VecZnxDftOwned<crate::FFT64Ref>>();
    requires_eq::<poulpy_hal::layouts::VecZnxBigOwned<crate::FFT64Ref>>();
    requires_eq::<poulpy_hal::layouts::SvpPPolOwned<crate::FFT64Ref>>();
    requires_eq::<poulpy_hal::layouts::VmpPMatOwned<crate::FFT64Ref>>();
}

#[cfg(feature = "enable-core")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64, backend = crate::FFT64Ref);

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_fft64_ref() {
    poulpy_core::test_suite::parity::test_gglwe_product_dft_selected(&Module::<FFT64Ref>::new(64), 12);
}

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_ntt4x30_ref() {
    poulpy_core::test_suite::parity::test_gglwe_product_dft_selected(&Module::<NTT4x30Ref>::new(64), 12);
}

// Cross-family parity: the NTT backend is exact, so at a radix small enough
// for FFT64 products to round exactly the two families must agree
// byte-for-byte. This catches a family-specific limb-window bug that a
// same-family parity suite cannot see.
#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_cross_family,
    backend_ref = crate::NTT4x30Ref,
    backend_test = crate::FFT64Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
        gglwe_keyswitch => poulpy_core::test_suite::parity::test_gglwe_keyswitch_parity,
        glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
        glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
        glwe_add => poulpy_core::test_suite::parity::test_glwe_add_parity,
        glwe_sub => poulpy_core::test_suite::parity::test_glwe_sub_parity,
        glwe_negate => poulpy_core::test_suite::parity::test_glwe_negate_parity,
        glwe_normalize => poulpy_core::test_suite::parity::test_glwe_normalize_parity,
        glwe_rotate => poulpy_core::test_suite::parity::test_glwe_rotate_parity,
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}
