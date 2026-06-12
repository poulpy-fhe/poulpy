use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{test_convolution, test_convolution_by_const, test_convolution_pairwise},
};

use crate::{FFT64Ref, NTT120Ref};

#[cfg(feature = "enable-ckks")]
mod ckks_tests;
#[cfg(feature = "enable-core")]
mod delegating_backend;

#[test]
fn test_convolution_by_const_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution_by_const(&module, 17);
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
fn test_convolution_by_const_ntt120_ref() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(8);
    test_convolution_by_const(&module, 50);
}

#[test]
fn test_convolution_ntt120_ref() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(8);
    test_convolution(&module, 50);
}

#[test]
fn test_convolution_pairwise_ntt120_ref() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(8);
    test_convolution_pairwise(&module, 50);
}

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
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
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}
cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
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
    backend_test = crate::NTT120Ref,
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
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
    }
}
cross_backend_test_suite! {
    mod vmp,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT120Ref,
    params = TestParams { size: 1<<12, base2k: 12 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_vec_znx_seed_sampling_matches_source_wrappers => poulpy_hal::test_suite::vec_znx::test_vec_znx_seed_sampling_matches_source_wrappers,
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
    mod ntt120,
    backend = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 52 },
);

#[test]
fn test_vec_znx_rsh_assign_multi_limb_matches_rsh() {
    use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRshAssignBackend, VecZnxRshBackend, VecZnxRshTmpBytes};
    use poulpy_hal::layouts::{FillUniform, HostBytesBackend, ScratchOwned, VecZnx};
    use poulpy_hal::source::Source;
    use poulpy_hal::test_suite::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};

    let n = 8usize;
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(n as u64);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(n as u64);
    let mut scratch: ScratchOwned<NTT120Ref> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes());
    let base2k = 52usize;
    let mut source = Source::new([3u8; 32]);

    // shifts spanning >= 2 limbs previously corrupted the in-place variant
    for size in [2usize, 3, 4] {
        for k in [60usize, 90, 105, 116] {
            if k / base2k + 1 > size {
                continue;
            }
            let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(1, size);
            a.fill_uniform(base2k, &mut source);
            let a_be = upload_vec_znx::<NTT120Ref>(&a);
            let mut want_be = upload_vec_znx::<NTT120Ref>(&module_host.vec_znx_alloc(1, size));
            module.vec_znx_rsh_backend(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT120Ref>(&mut want_be),
                0,
                &vec_znx_backend_ref::<NTT120Ref>(&a_be),
                0,
                &mut scratch.borrow(),
            );
            let mut got_be = upload_vec_znx::<NTT120Ref>(&a);
            module.vec_znx_rsh_assign_backend(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT120Ref>(&mut got_be),
                0,
                &mut scratch.borrow(),
            );
            assert_eq!(
                download_vec_znx::<NTT120Ref>(&got_be),
                download_vec_znx::<NTT120Ref>(&want_be),
                "vec_znx_rsh_assign mismatch for size={size} k={k}"
            );
        }
    }
}

#[test]
fn test_ntt120_vmp_apply_truncated_res_matches_full_prefix() {
    use poulpy_hal::api::{
        ScratchOwnedAlloc, VecZnxDftAlloc, VecZnxDftApply, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
        VmpPrepareTmpBytes,
    };
    use poulpy_hal::layouts::{
        Backend, FillUniform, HostBytesBackend, MatZnx, MatZnxToBackendRef, ScratchOwned, VecZnx, VecZnxDft,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMat, VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxView,
    };
    use poulpy_hal::source::Source;
    use poulpy_hal::test_suite::{upload_mat_znx, upload_vec_znx, vec_znx_backend_ref};

    let n = 8usize;
    let base2k = 52usize;
    let (rows, cols_in, cols_out, mat_size) = (2usize, 1usize, 1usize, 4usize);

    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(n as u64);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(n as u64);
    let mut source = Source::new([7u8; 32]);

    let mut scratch: ScratchOwned<NTT120Ref> = ScratchOwned::alloc(
        module
            .vmp_apply_dft_to_dft_tmp_bytes(mat_size, rows, rows, cols_in, cols_out, mat_size)
            .max(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, mat_size)),
    );

    let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_in, rows);
    a.fill_uniform(base2k, &mut source);
    let a_be = upload_vec_znx::<NTT120Ref>(&a);
    let mut a_dft: VecZnxDft<Vec<u8>, NTT120Ref> = module.vec_znx_dft_alloc(cols_in, rows);
    module.vec_znx_dft_apply(
        1,
        0,
        &mut a_dft.to_backend_mut(),
        0,
        &vec_znx_backend_ref::<NTT120Ref>(&a_be),
        0,
    );

    let mut mat: MatZnx<Vec<u8>> = module_host.mat_znx_alloc(rows, cols_in, cols_out, mat_size);
    mat.fill_uniform(base2k, &mut source);
    let mat_be = upload_mat_znx::<NTT120Ref>(&mat);
    let mut pmat: VmpPMat<Vec<u8>, NTT120Ref> = module.vmp_pmat_alloc(rows, cols_in, cols_out, mat_size);
    module.vmp_prepare(
        &mut pmat.to_backend_mut(),
        &<MatZnx<<NTT120Ref as Backend>::OwnedBuf> as MatZnxToBackendRef<NTT120Ref>>::to_backend_ref(&mat_be),
        &mut scratch.arena(),
    );

    let mut res_full: VecZnxDft<Vec<u8>, NTT120Ref> = module.vec_znx_dft_alloc(cols_out, mat_size);
    module.vmp_apply_dft_to_dft(
        &mut res_full.to_backend_mut(),
        &a_dft.to_backend_ref(),
        &pmat.to_backend_ref(),
        0,
        &mut scratch.arena(),
    );
    let full_u64: Vec<u64> = bytemuck::cast_slice(res_full.raw()).to_vec();

    // odd truncated sizes hit the trailing-column path on a paired pmat column
    for res_size in [1usize, 3] {
        let mut res_trunc: VecZnxDft<Vec<u8>, NTT120Ref> = module.vec_znx_dft_alloc(cols_out, res_size);
        module.vmp_apply_dft_to_dft(
            &mut res_trunc.to_backend_mut(),
            &a_dft.to_backend_ref(),
            &pmat.to_backend_ref(),
            0,
            &mut scratch.arena(),
        );
        let trunc_u64: &[u64] = bytemuck::cast_slice(res_trunc.raw());
        assert_eq!(
            trunc_u64,
            &full_u64[..trunc_u64.len()],
            "truncated vmp output mismatch for res_size={res_size}"
        );
    }
}
