use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_accumulate, test_convolution_accumulate_fused, test_convolution_by_const,
        test_convolution_by_const_add, test_convolution_pairwise,
    },
};

use crate::{FFT64Ref, NTT4x30Ref};

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
        test_vec_znx_zero_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_zero_matches_wrapper,
        test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_vec_znx_add_assign_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign_matches_wrapper,
        test_vec_znx_extract_coeff => poulpy_hal::test_suite::vec_znx::test_vec_znx_extract_coeff,
        test_vec_znx_lsh_add_coeff_to_coeff => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_add_coeff_to_coeff,
        test_vec_znx_lsh_sub_coeff_to_coeff => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_sub_coeff_to_coeff,
        test_vec_znx_rsh_coeff => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_coeff,
        test_vec_znx_rsh_add_coeff => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_add_coeff,
        test_vec_znx_rsh_sub_coeff => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_sub_coeff,
        test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
        test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
        test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
        test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
        test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
        test_vec_znx_rsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_assign,
        test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
        test_vec_znx_lsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_assign,
        test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
        test_vec_znx_negate_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_matches_wrapper,
        test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
        test_vec_znx_negate_assign_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign_matches_wrapper,
        test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
        test_vec_znx_rotate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_assign,
        test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
        test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
        test_scalar_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_scalar_znx_automorphism,
        test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
        test_vec_znx_mul_xp_minus_one_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_assign,
        test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
        test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_switch_ring_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring_matches_wrapper,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        test_vec_znx_copy_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_matches_wrapper,
        test_vec_znx_copy_range => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_range,
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
        test_vec_znx_big_add => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
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
        test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
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
        test_word_compat_prepare_hint_sizes => poulpy_hal::test_suite::word_compat::test_word_compat_prepare_hint_sizes,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT4x30Ref,
    params = TestParams { size: 1<<12, base2k: 12 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_scalar_znx_binary_hw_has_exact_weight => poulpy_hal::test_suite::vec_znx::test_scalar_znx_binary_hw_has_exact_weight,
        test_scalar_znx_secret_sampling => poulpy_hal::test_suite::vec_znx::test_scalar_znx_secret_sampling,
        test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
        test_vec_znx_big_add_normal => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_normal,
    }
}

backend_test_suite! {
    mod window_fft64,
    backend = crate::FFT64Ref,
    params = TestParams { size: 1 << 8, base2k: 12 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
    }
}

backend_test_suite! {
    mod window_ntt4x30,
    backend = crate::NTT4x30Ref,
    params = TestParams { size: 1 << 8, base2k: 12 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
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
    use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRsh, VecZnxRshAssign, VecZnxRshTmpBytes};
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
            module.vec_znx_rsh(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Ref>(&mut want_be),
                0,
                &vec_znx_backend_ref::<NTT4x30Ref>(&a_be),
                0,
                &mut scratch.borrow(),
            );
            let mut got_be = upload_vec_znx::<NTT4x30Ref>(&a);
            module.vec_znx_rsh_assign(
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

fn vec_znx_big_normalize_limb_bounds<BE>(module: &Module<BE>)
where
    BE: poulpy_hal::test_suite::TestBackend,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    Module<BE>: poulpy_hal::api::VecZnxBigAlloc<BE>
        + poulpy_hal::api::VecZnxBigFromSmall<BE>
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes,
    poulpy_hal::layouts::ScratchOwned<BE>: poulpy_hal::api::ScratchOwnedAlloc<BE>,
{
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes},
        layouts::{
            FillUniform, HostBytesBackend, ScratchOwned, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
            VecZnxToBackendMut, VecZnxToBackendRef, ZnxView,
        },
        source::Source,
        test_suite::upload_vec_znx,
    };
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut source = Source::new([2u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
    for a_base2k in 1..=51usize {
        for res_base2k in 1..=51usize {
            for offset in [-(a_base2k as i64), -3, -1, 0, 1, 3, a_base2k as i64] {
                for a_size in 1..=3usize {
                    for res_size in 1..=3usize {
                        let mut a = module_host.vec_znx_alloc(1, a_size);
                        a.fill_uniform(63, &mut source);
                        let uploaded = upload_vec_znx::<BE>(&a);
                        let mut big = module.vec_znx_big_alloc(1, a_size);
                        module.vec_znx_big_from_small(
                            &mut big.to_backend_mut(),
                            0,
                            &<VecZnx<BE::OwnedBuf, i64> as VecZnxToBackendRef<BE>>::to_backend_ref(&uploaded),
                            0,
                        );
                        let mut res = module.vec_znx_alloc(1, res_size);
                        module.vec_znx_big_normalize(
                            &mut <VecZnx<BE::OwnedBuf, i64> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut res),
                            res_base2k,
                            res_size * res_base2k,
                            offset,
                            0,
                            &big.to_backend_ref(),
                            a_base2k,
                            0,
                            &mut scratch.arena(),
                        );
                        let bound: i64 = 1 << (res_base2k - 1);
                        for j in 0..res_size {
                            assert!(
                                res.at(0, j).iter().all(|x| (-bound..bound).contains(x)),
                                "a_base2k={a_base2k} res_base2k={res_base2k} offset={offset} a_size={a_size} res_size={res_size} limb={j}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_vec_znx_big_normalize_limb_bounds_fft64_ref() {
    vec_znx_big_normalize_limb_bounds(&Module::<FFT64Ref>::new(8));
}

#[test]
fn test_vec_znx_big_normalize_limb_bounds_ntt4x30_ref() {
    vec_znx_big_normalize_limb_bounds(&Module::<NTT4x30Ref>::new(8));
}

const NORMALIZE_IDFT_BOUND: i128 = (1_073_479_681i128 * 1_071_513_601 * 1_070_727_169 * 1_068_236_801 - 1) / 2;

#[test]
fn test_vec_znx_big_normalize_input_bound_integer() {
    use crate::reference::{
        ntt4x30::ntt4x30_vec_znx_big_normalize,
        vec_znx::{normalize_integer_oracle, vec_znx_normalize},
    };
    use poulpy_hal::layouts::{VecZnx, VecZnxBig, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut};

    const N: usize = 8;
    let values = [
        -(1i128 << 126),
        1i128 << 126,
        -NORMALIZE_IDFT_BOUND,
        NORMALIZE_IDFT_BOUND,
        -(1i128 << 62),
        1i128 << 62,
        -1,
        1,
    ];
    let mut carry = [0i128; 3 * N];
    let mut random = 0x123456789abcdef0u128;
    for a_base2k in 1..=62 {
        for res_base2k in 1..=62 {
            for a_size in 0..=8 {
                let mut input = VecZnxBig::<Vec<u8>, i128, NTT4x30Ref>::from_data(vec![0; 16 * N * a_size], N, 1, a_size);
                for j in 0..a_size {
                    for i in 0..N {
                        random ^= random << 17;
                        random ^= random >> 29;
                        random ^= random << 43;
                        input.at_mut(0, j)[i] = match i {
                            0..=3 => values[i],
                            4 => values[4 + j % 2],
                            5 => (random as i64 >> 1) as i128,
                            6 => values[j % 2],
                            _ => random as i128 >> 1,
                        };
                    }
                }
                let mut small_input = VecZnx::<Vec<u8>, i64>::from_data(vec![0; 8 * N * a_size], N, 1, a_size);
                for j in 0..a_size {
                    for i in 0..N {
                        small_input.at_mut(0, j)[i] = if (4..=5).contains(&i) {
                            input.at(0, j)[i] as i64
                        } else {
                            input.at(0, j)[i] as i64 >> 1
                        };
                    }
                }
                for res_size in 1..=8 {
                    let mut output = VecZnx::<Vec<u8>, i64>::from_data(vec![0; 8 * N * res_size], N, 1, res_size);
                    let mut small_output = VecZnx::<Vec<u8>, i64>::from_data(vec![0; 8 * N * res_size], N, 1, res_size);
                    let bracket = (a_size * a_base2k + res_size * res_base2k + 128) as i64;
                    let gap = (a_size * a_base2k) as i64 - (res_size * res_base2k) as i64;
                    for offset in [
                        -bracket,
                        -(a_base2k as i64) - 1,
                        -1,
                        0,
                        1,
                        a_base2k as i64,
                        bracket,
                        gap - a_base2k as i64 - 1,
                        gap - a_base2k as i64,
                        gap - 1,
                    ] {
                        ntt4x30_vec_znx_big_normalize::<_, _, NTT4x30Ref>(
                            &mut output,
                            res_base2k,
                            res_size * res_base2k,
                            offset,
                            0,
                            &input,
                            a_base2k,
                            0,
                            &mut carry,
                        );
                        vec_znx_normalize::<FFT64Ref>(
                            &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut small_output),
                            res_base2k,
                            res_size * res_base2k,
                            offset,
                            0,
                            &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&small_input),
                            a_base2k,
                            0,
                            &mut [0i64; 3 * N],
                        );
                        for i in 4..=5 {
                            for j in 0..res_size {
                                assert_eq!(output.at(0, j)[i], small_output.at(0, j)[i]);
                            }
                        }
                        for i in 0..N {
                            let limbs: Vec<_> = (0..a_size).map(|j| input.at(0, j)[i]).collect();
                            let want = normalize_integer_oracle(&limbs, a_base2k, res_base2k, res_size, offset);
                            let got: Vec<_> = (0..res_size).map(|j| output.at(0, j)[i]).collect();
                            assert_eq!(
                                got, want,
                                "ka={a_base2k} kr={res_base2k} a={limbs:?} size={res_size} offset={offset}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_normalize_exact_canonical_precision() {
    use crate::reference::{
        ntt4x30::ntt4x30_vec_znx_big_normalize,
        vec_znx::{vec_znx_normalize, vec_znx_normalize_assign},
    };
    use poulpy_hal::layouts::{VecZnx, VecZnxBig, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut};

    type Case<'a> = (&'a [i64], usize, usize, usize, i64, &'a [i64]);
    let cases: &[Case<'_>] = &[
        (&[3, 7, 5, 0], 4, 4, 6, 0, &[4, -8]),
        (&[-3, -7, -5, 0], 4, 4, 6, 0, &[-3, -8]),
        (&[3, 7, 7], 5, 4, 6, 0, &[2, -8]),
        (&[3, 7, 7], 5, 4, 4, 0, &[2, 0]),
        (&[3, 7, 7], 5, 4, 0, 0, &[0, 0]),
        (&[1400], 4, 4, 14, -20, &[0, 0, 0, 4]),
        (&[1400], 4, 4, 14, i64::MIN, &[0, 0, 0, 0]),
        (&[1400], 4, 4, 14, i64::MAX, &[0, 0, 0, 0]),
        (&[i64::MAX], 64, 64, 62, 0, &[i64::MIN]),
    ];
    for &(values, a_base2k, res_base2k, res_k, offset, expected) in cases {
        let mut small = VecZnx::<Vec<u8>, i64>::from_data(vec![0; 8 * values.len()], 1, 1, values.len());
        let mut big = VecZnxBig::<Vec<u8>, i128, NTT4x30Ref>::from_data(vec![0; 16 * values.len()], 1, 1, values.len());
        for (j, &value) in values.iter().enumerate() {
            small.at_mut(0, j)[0] = value;
            big.at_mut(0, j)[0] = value as i128;
        }
        let mut output = VecZnx::<Vec<u8>, i64>::from_data(vec![93; 8 * expected.len()], 1, 1, expected.len());
        ntt4x30_vec_znx_big_normalize::<_, _, NTT4x30Ref>(
            &mut output,
            res_base2k,
            res_k,
            offset,
            0,
            &big,
            a_base2k,
            0,
            &mut [0; 3],
        );
        assert_eq!((0..expected.len()).map(|j| output.at(0, j)[0]).collect::<Vec<_>>(), expected);
        if a_base2k <= 62 && res_base2k <= 62 {
            vec_znx_normalize::<FFT64Ref>(
                &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut output),
                res_base2k,
                res_k,
                offset,
                0,
                &<VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&small),
                a_base2k,
                0,
                &mut [0; 3],
            );
            assert_eq!((0..expected.len()).map(|j| output.at(0, j)[0]).collect::<Vec<_>>(), expected);
        } else if a_base2k == 64 && res_base2k == 64 {
            vec_znx_normalize_assign::<FFT64Ref>(
                res_base2k,
                res_k,
                &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut small),
                0,
                &mut [0],
            );
            assert_eq!(small.at(0, 0), expected);
        }
    }
}

#[test]
fn test_vec_znx_big_normalize_assign_and_ranges() {
    use crate::reference::ntt4x30::{
        ntt4x30_vec_znx_big_normalize, ntt4x30_vec_znx_big_normalize_assign, ntt4x30_vec_znx_big_normalize_range_raw,
        vec_znx_big::{AddOp, SubOp},
    };
    use poulpy_hal::layouts::{VecZnx, VecZnxBig, ZnxView, ZnxViewMut};
    let mut regression_input = VecZnxBig::<Vec<u8>, i128, NTT4x30Ref>::from_data(vec![0; 16], 1, 1, 1);
    regression_input.at_mut(0, 0)[0] = 3;
    let mut regression_output = VecZnx::<Vec<u8>, i64>::from_data(vec![0; 24], 1, 1, 3);
    ntt4x30_vec_znx_big_normalize_assign::<SubOp, _, _, NTT4x30Ref>(
        &mut regression_output,
        2,
        -2,
        0,
        &regression_input,
        1,
        0,
        &mut [0; 3],
    );
    assert_eq!((0..3).map(|j| regression_output.at(0, j)[0]).collect::<Vec<_>>(), [2, 2, 0]);
    const N: usize = 17;
    for a_base2k in [1, 2, 17, 50, 62] {
        for res_base2k in [1, 2, 19, 51, 62] {
            for offset in [i64::MIN, -400, -63, -1, 0, 1, 63, 400, i64::MAX] {
                let mut input = VecZnxBig::<Vec<u8>, i128, NTT4x30Ref>::from_data(vec![0; 16 * N * 3], N, 1, 3);
                for j in 0..3 {
                    for i in 0..N {
                        input.at_mut(0, j)[i] = (NORMALIZE_IDFT_BOUND / (i as i128 + 1)) * if (i + j) % 2 == 0 { -1 } else { 1 };
                    }
                }
                let alloc = || VecZnx::<Vec<u8>, i64>::from_data(vec![0; 8 * N * 3], N, 1, 3);
                let mut want = alloc();
                let mut carry = [0i128; 3 * N];
                ntt4x30_vec_znx_big_normalize::<_, _, NTT4x30Ref>(
                    &mut want,
                    res_base2k,
                    3 * res_base2k,
                    offset,
                    0,
                    &input,
                    a_base2k,
                    0,
                    &mut carry,
                );
                let mut split = alloc();
                let ptr = split.data_mut().as_mut_ptr().cast::<i64>();
                for (start, len) in [(0, 3), (3, 7), (10, 7)] {
                    let mut private = vec![0i128; 3 * len];
                    unsafe {
                        ntt4x30_vec_znx_big_normalize_range_raw::<_, NTT4x30Ref>(
                            ptr,
                            N,
                            1,
                            3,
                            res_base2k,
                            3 * res_base2k,
                            offset,
                            0,
                            &input,
                            a_base2k,
                            0,
                            start,
                            len,
                            &mut private,
                        );
                    }
                }
                assert_eq!(split, want);
                for sub in [false, true] {
                    let mut got = alloc();
                    for j in 0..3 {
                        got.at_mut(0, j).fill(37);
                    }

                    if sub {
                        ntt4x30_vec_znx_big_normalize_assign::<SubOp, _, _, NTT4x30Ref>(
                            &mut got, res_base2k, offset, 0, &input, a_base2k, 0, &mut carry,
                        );
                    } else {
                        ntt4x30_vec_znx_big_normalize_assign::<AddOp, _, _, NTT4x30Ref>(
                            &mut got, res_base2k, offset, 0, &input, a_base2k, 0, &mut carry,
                        );
                    }
                    for j in 0..3 {
                        for i in 0..N {
                            let expected = if sub { 37 - want.at(0, j)[i] } else { 37 + want.at(0, j)[i] };
                            assert_eq!(
                                got.at(0, j)[i],
                                expected,
                                "ka={a_base2k} kr={res_base2k} offset={offset} sub={sub}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_i128_normalization_kernel_integer() {
    use crate::reference::ntt4x30::I128NormalizeOps;
    use dashu_int::IBig;
    for base2k in 1..=63 {
        let half = 1i128 << (base2k - 1);
        let a = [
            -(1i128 << 126),
            1i128 << 126,
            -NORMALIZE_IDFT_BOUND,
            NORMALIZE_IDFT_BOUND,
            -half - 1,
            -half,
            half - 1,
            half,
            -1,
            0,
            1,
        ];
        for lsh in 0..base2k {
            for offset in 0..a.len() {
                let mut carry: Vec<_> = (0..a.len())
                    .map(|i| {
                        let total = IBig::from(a[(i + offset) % a.len()]) << lsh;
                        i128::try_from((total + IBig::from(half)) >> base2k).unwrap()
                    })
                    .collect();
                let mut want_res = Vec::new();
                let mut want_carry = Vec::new();
                for (&input, &previous) in a.iter().zip(&carry) {
                    let total = (IBig::from(input) << lsh) + IBig::from(previous);
                    let next = (&total + IBig::from(half)) >> base2k;
                    want_res.push(i64::try_from(total - (&next << base2k)).unwrap());
                    want_carry.push(i128::try_from(next).unwrap());
                }
                let mut res = vec![0i64; a.len()];
                <NTT4x30Ref as I128NormalizeOps>::nfc_middle_step(base2k, lsh, &mut res, &a, &mut carry);
                assert_eq!(res, want_res, "base2k={base2k} lsh={lsh}");
                assert_eq!(carry, want_carry, "base2k={base2k} lsh={lsh}");
            }
        }
    }
}

#[test]
fn test_i128_normalize_fused_reference() {
    use crate::reference::{
        ntt4x30::I128NormalizeOps,
        znx::{get_carry_i128, get_digit_i128},
    };
    for base2k in 1..=63 {
        for take in 1..=base2k {
            let scale = base2k - take;
            let mut src = [-NORMALIZE_IDFT_BOUND, NORMALIZE_IDFT_BOUND, -1, 0, 1];
            let mut carry = [-1, 0, -1, 0, 0];
            let quarter = if base2k > 1 { 1i64 << (base2k - 2) } else { 0 };
            let mut res = [-quarter, quarter.saturating_sub(1), -1, 0, 1];
            if base2k == 1 {
                res.fill(0);
            }
            let mut want_src = src;
            let mut want_carry = carry;
            let mut want_res = res;
            for (r, s) in want_res.iter_mut().zip(&mut want_src) {
                let digit = get_digit_i128(take, *s);
                *s = get_carry_i128(take, *s, digit);
                *r = r.wrapping_add((digit as i64).wrapping_shl(scale as u32));
            }
            <NTT4x30Ref as I128NormalizeOps>::nfc_middle_step_assign(base2k, 0, &mut want_res, &mut want_carry);
            <NTT4x30Ref as I128NormalizeOps>::znx_extract_digit_addmul_normalize_i128::<false>(
                take, scale, base2k, &mut res, &mut src, &mut carry,
            );
            assert_eq!((res, src, carry), (want_res, want_src, want_carry));
        }
    }
}

#[test]
fn test_vec_znx_big_normalize_wide_radices() {
    use crate::reference::{
        ntt4x30::{
            ntt4x30_vec_znx_big_normalize, ntt4x30_vec_znx_big_normalize_assign,
            vec_znx_big::{AddOp, SubOp},
        },
        vec_znx::normalize_integer_oracle,
    };
    use poulpy_hal::layouts::{VecZnx, VecZnxBig, ZnxView, ZnxViewMut};
    for a_base2k in [1, 63, 64, 65, 126, 127] {
        for res_base2k in [1, 17, 63, 64] {
            for size in 1..=3 {
                let mut input = VecZnxBig::<Vec<u8>, i128, NTT4x30Ref>::from_data(vec![0; 32 * size], 2, 1, size);
                for j in 0..size {
                    input
                        .at_mut(0, j)
                        .copy_from_slice(&[-NORMALIZE_IDFT_BOUND, NORMALIZE_IDFT_BOUND]);
                }
                let mut output = VecZnx::<Vec<u8>, i64>::from_data(vec![0; 16 * size], 2, 1, size);
                for offset in [-400, -64, -(a_base2k as i64), -1, 0, 1, 64, 400] {
                    ntt4x30_vec_znx_big_normalize::<_, _, NTT4x30Ref>(
                        &mut output,
                        res_base2k,
                        size * res_base2k,
                        offset,
                        0,
                        &input,
                        a_base2k,
                        0,
                        &mut [0; 6],
                    );
                    for sub in [false, true] {
                        let mut assigned = VecZnx::<Vec<u8>, i64>::from_data(vec![0; 16 * size], 2, 1, size);
                        for j in 0..size {
                            assigned.at_mut(0, j).copy_from_slice(&[-(1i64 << 62), 1i64 << 62]);
                        }
                        if sub {
                            ntt4x30_vec_znx_big_normalize_assign::<SubOp, _, _, NTT4x30Ref>(
                                &mut assigned,
                                res_base2k,
                                offset,
                                0,
                                &input,
                                a_base2k,
                                0,
                                &mut [0; 6],
                            );
                        } else {
                            ntt4x30_vec_znx_big_normalize_assign::<AddOp, _, _, NTT4x30Ref>(
                                &mut assigned,
                                res_base2k,
                                offset,
                                0,
                                &input,
                                a_base2k,
                                0,
                                &mut [0; 6],
                            );
                        }
                        for i in 0..2 {
                            let initial = [-(1i64 << 62), 1i64 << 62][i] as i128;
                            let total: Vec<_> = (0..size)
                                .map(|j| {
                                    initial
                                        + if sub {
                                            -(output.at(0, j)[i] as i128)
                                        } else {
                                            output.at(0, j)[i] as i128
                                        }
                                })
                                .collect();
                            let stored: Vec<_> = (0..size).map(|j| assigned.at(0, j)[i] as i128).collect();
                            assert_eq!(
                                normalize_integer_oracle(&stored, res_base2k, res_base2k, size, 0),
                                normalize_integer_oracle(&total, res_base2k, res_base2k, size, 0)
                            );
                        }
                    }
                    for i in 0..2 {
                        let limbs: Vec<_> = (0..size).map(|j| input.at(0, j)[i]).collect();
                        let want = normalize_integer_oracle(&limbs, a_base2k, res_base2k, size, offset);
                        let got: Vec<_> = (0..size).map(|j| output.at(0, j)[i]).collect();
                        assert_eq!(got, want, "ka={a_base2k} kr={res_base2k} size={size} offset={offset}");
                    }
                }
            }
        }
    }
}
mod canonical_precision_tests {
    use crate::{
        FFT64Ref, NTT4x30Ref,
        reference::{
            ntt4x30::{ntt4x30_vec_znx_big_normalize, ntt4x30_vec_znx_big_normalize_range_raw},
            vec_znx::{
                vec_znx_normalize, vec_znx_normalize_assign, vec_znx_normalize_assign_range_raw, vec_znx_normalize_range_raw,
            },
        },
    };
    use dashu_int::IBig;
    use poulpy_hal::layouts::{VecZnx, VecZnxBig, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut};

    const N: usize = 9;
    const IDFT_BOUND: i128 = (1_073_479_681i128 * 1_071_513_601 * 1_070_727_169 * 1_068_236_801 - 1) / 2;

    // One rounding decision on the exact integer value, independent of limb scheduling.
    fn oracle(a: &[i128], ka: usize, kr: usize, k: usize, size: usize, offset: i64) -> Vec<i64> {
        let mut result = vec![0; size];
        let shift = k as i128 + offset as i128 - (a.len() * ka) as i128;
        if k == 0 || offset as i128 >= (a.len() * ka) as i128 || shift < -((a.len() * ka + 130) as i128) {
            return result;
        }
        let mut value = a.iter().fold(IBig::ZERO, |sum, &limb| (sum << ka) + IBig::from(limb));
        if shift >= 0 {
            value <<= shift as usize;
        } else {
            let drop = (-shift) as usize;
            value = (value + (IBig::ONE << (drop - 1))) >> drop;
        }
        value <<= size * kr - k;
        for digit in result.iter_mut().rev() {
            let carry = (&value + (IBig::ONE << (kr - 1))) >> kr;
            *digit = i64::try_from(&value - (&carry << kr)).unwrap();
            value = carry;
        }
        result
    }

    fn small(size: usize) -> VecZnx<Vec<u8>, i64> {
        VecZnx::from_data(vec![0xa5; 8 * N * 2 * size], N, 2, size)
    }

    #[allow(clippy::too_many_arguments)]
    fn check(a: &[Vec<i128>], ka: usize, kr: usize, k: usize, size: usize, offset: i64, ranges: bool) {
        let mut input = small(a.len());
        let mut wide = VecZnxBig::<Vec<u8>, i128, NTT4x30Ref>::from_data(vec![0; 16 * N * 2 * a.len()], N, 2, a.len());
        for (j, values) in a.iter().enumerate() {
            for (i, &value) in values.iter().enumerate() {
                input.at_mut(1, j)[i] = value as i64;
                wide.at_mut(1, j)[i] = value;
            }
        }
        let narrow = ka <= 62 && kr <= 62 && a.iter().flatten().all(|&x| (-(1i128 << 62)..=1i128 << 62).contains(&x));
        let expected: Vec<_> = (0..N)
            .map(|i| oracle(&a.iter().map(|v| v[i]).collect::<Vec<_>>(), ka, kr, k, size, offset))
            .collect();
        let verify = |name: &str, output: &VecZnx<Vec<u8>, i64>| {
            for (i, want) in expected.iter().enumerate() {
                let got: Vec<_> = (0..size).map(|j| output.at(1, j)[i]).collect();
                assert_eq!(
                    got, *want,
                    "{name}: ka={ka} kr={kr} k={k} size={size} offset={offset} i={i} a={a:?}"
                );
            }
            for j in 0..size {
                assert!(output.at(0, j).iter().all(|&x| x == i64::from_ne_bytes([0xa5; 8])));
            }
        };
        let mut output = small(size);
        if narrow {
            let src = <VecZnx<Vec<u8>, i64> as VecZnxToBackendRef<FFT64Ref>>::to_backend_ref(&input);
            if ranges {
                let ptr = output.data_mut().as_mut_ptr().cast::<i64>();
                for (start, len) in [(0, 2), (2, 3), (5, 4)] {
                    unsafe {
                        vec_znx_normalize_range_raw::<FFT64Ref>(
                            ptr,
                            N,
                            2,
                            size,
                            kr,
                            k,
                            offset,
                            1,
                            &src,
                            ka,
                            1,
                            start,
                            len,
                            &mut vec![73; 3 * len],
                        );
                    }
                }
            } else {
                vec_znx_normalize::<FFT64Ref>(
                    &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut output),
                    kr,
                    k,
                    offset,
                    1,
                    &src,
                    ka,
                    1,
                    &mut [73; 3 * N],
                );
            }
            verify("narrow", &output);
        }
        output = small(size);
        if ranges {
            let ptr = output.data_mut().as_mut_ptr().cast::<i64>();
            for (start, len) in [(0, 2), (2, 3), (5, 4)] {
                unsafe {
                    ntt4x30_vec_znx_big_normalize_range_raw::<_, NTT4x30Ref>(
                        ptr,
                        N,
                        2,
                        size,
                        kr,
                        k,
                        offset,
                        1,
                        &wide,
                        ka,
                        1,
                        start,
                        len,
                        &mut vec![73; 3 * len],
                    );
                }
            }
        } else {
            ntt4x30_vec_znx_big_normalize::<_, _, NTT4x30Ref>(&mut output, kr, k, offset, 1, &wide, ka, 1, &mut [73; 3 * N]);
        }
        verify("wide", &output);
        if narrow && ka == kr && size == a.len() && offset == 0 {
            if ranges {
                let ptr = input.data_mut().as_mut_ptr().cast::<i64>();
                for (start, len) in [(0, 2), (2, 3), (5, 4)] {
                    unsafe {
                        vec_znx_normalize_assign_range_raw::<FFT64Ref>(ptr, N, 2, size, kr, k, 1, start, len, &mut vec![73; len]);
                    }
                }
            } else {
                vec_znx_normalize_assign::<FFT64Ref>(
                    kr,
                    k,
                    &mut <VecZnx<Vec<u8>, i64> as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut input),
                    1,
                    &mut [73; N],
                );
            }
            verify("assign", &input);
        }
    }

    #[test]
    fn test_canonical_precision_round_once_regressions() {
        for (a, ka, kr, k, size) in [
            (vec![1, -8], 4, 4, 3, 1),
            (vec![0, 1, 2], 2, 3, 2, 1),
            (vec![1, -8, -8], 4, 4, 4, 3),
            (vec![1, -8, -8], 4, 4, 4, 1),
            (vec![1, -(1i128 << 49)], 50, 50, 49, 1),
        ] {
            for sign in [-1, 1] {
                check(
                    &a.iter().map(|&x| vec![sign * x; N]).collect::<Vec<_>>(),
                    ka,
                    kr,
                    k,
                    size,
                    0,
                    false,
                );
            }
        }
    }

    #[test]
    fn test_canonical_precision_integer_oracle() {
        for ka in [1, 2, 17, 19, 21, 50, 51, 62] {
            for kr in [1, 2, 17, 19, 21, 50, 51, 62] {
                for a_size in [1, 3] {
                    let half = 1i128 << (ka - 1);
                    let digits = [0, 1, -1, 1i128 << 62, -(1i128 << 62), half, -half, half - 1, -half - 1];
                    let input: Vec<Vec<_>> = (0..a_size)
                        .map(|j| (0..N).map(|i| digits[(i + 2 * j) % N]).collect())
                        .collect();
                    for size in [1, 3] {
                        let mut precisions = vec![0, 1, kr - 1, kr, size * kr / 2, size * kr - 1, size * kr];
                        precisions.sort_unstable();
                        precisions.dedup();
                        for k in precisions {
                            for offset in [-(ka as i64), -1, 0, 1, ka as i64, i64::MIN, i64::MAX] {
                                check(&input, ka, kr, k, size, offset, k.is_multiple_of(2));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_canonical_precision_cross_boundaries() {
        for (ka, kr) in [(1, 2), (2, 1), (17, 50), (50, 17), (50, 51), (51, 50), (51, 62), (62, 51)] {
            let digits = [
                0,
                1,
                -1,
                1i128 << 62,
                -(1i128 << 62),
                (1i128 << 62) - 1,
                -(1i128 << 62) + 1,
                7,
                -7,
            ];
            let input: Vec<Vec<_>> = (0..8).map(|j| (0..N).map(|i| digits[(i + 2 * j) % N]).collect()).collect();
            // Every width of the last active limb, including the 50 -> 51 dispatch boundary at k=399/400.
            let mut precisions: Vec<_> = (7 * kr + 1..=8 * kr).chain([1, kr - 1, kr, 4 * kr]).collect();
            precisions.sort_unstable();
            precisions.dedup();
            for k in precisions {
                for offset in [-(ka as i64) - 1, -1, 0, 1, ka as i64 + 1] {
                    for size in [8, 9] {
                        check(&input, ka, kr, k, size, offset, k.is_multiple_of(2));
                    }
                    let drop = 8 * ka as i64 - k as i64 - offset;
                    if !(1..8 * ka as i64).contains(&drop) {
                        continue;
                    }
                    // Encode exact values immediately below, at and above both signed rounding ties.
                    let half = IBig::ONE << (drop as usize - 1);
                    let mut ties = vec![vec![0; N]; 8];
                    let mut values: Vec<_> = (0..N).map(|i| &half * (i / 3) as i32 - &half + (i % 3) as i32 - 1).collect();
                    for limb in ties[1..].iter_mut().rev() {
                        for (digit, value) in limb.iter_mut().zip(values.iter_mut()) {
                            let carry = &*value >> ka;
                            *digit = i128::try_from(&*value - (&carry << ka)).unwrap();
                            *value = carry;
                        }
                    }
                    for (digit, value) in ties[0].iter_mut().zip(values) {
                        *digit = i128::try_from(value).unwrap();
                    }
                    check(&ties, ka, kr, k, 9, offset, !k.is_multiple_of(2));
                }
            }
            let wide_digits = [IDFT_BOUND, -IDFT_BOUND, i64::MAX as i128, i64::MIN as i128, 0, 1, -1, 7, -7];
            let wide: Vec<Vec<_>> = (0..8)
                .map(|j| (0..N).map(|i| wide_digits[(i + 2 * j) % N]).collect())
                .collect();
            for k in [1, kr, 4 * kr, 8 * kr - 1, 8 * kr] {
                for offset in [-(ka as i64) - 1, 0, ka as i64 + 1] {
                    check(&wide, ka, kr, k, 9, offset, k.is_multiple_of(2));
                }
            }
        }
    }

    #[test]
    fn test_canonical_precision_wide_integer_oracle() {
        for ka in [1, 17, 50, 62, 63, 64, 119, 127] {
            for kr in [1, 19, 51, 62, 63, 64] {
                let digits = [
                    IDFT_BOUND,
                    -IDFT_BOUND,
                    i64::MAX as i128,
                    i64::MIN as i128,
                    0,
                    1,
                    -1,
                    (1i128 << 100) + 1,
                    -(1i128 << 100) - 1,
                ];
                let input: Vec<Vec<_>> = (0..3).map(|j| (0..N).map(|i| digits[(i + 2 * j) % N]).collect()).collect();
                for k in [0, 1, kr - 1, kr, 2 * kr + 1, 3 * kr] {
                    for offset in [-(ka as i64) - 1, 0, 1, ka as i64 + 1] {
                        check(&input, ka, kr, k, 3, offset, k.is_multiple_of(2));
                    }
                }
            }
        }
    }
}
