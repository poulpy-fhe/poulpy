use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_add, test_convolution_by_const, test_convolution_by_const_add,
        test_convolution_pairwise, test_convolution_sum,
    },
};

use crate::{FFT64Portable, NTT4x30Portable};

#[cfg(feature = "enable-blas")]
mod blas_overrides;
#[cfg(feature = "enable-blas")]
mod blas_tests;
#[cfg(feature = "enable-ckks")]
mod ckks_tests;
#[cfg(feature = "enable-core")]
mod delegating_backend;
mod derived_scratch;
#[cfg(feature = "enable-ckks")]
mod libm_tests;
#[cfg(feature = "enable-core")]
mod linear_transformation_batch;

/// A square's constant coefficient has mean `(2n - 1) * 2^(2K) / 12` on the conjugate-invariant ring: one radix past
/// the model's, it exceeds `Q/2` by itself.
#[test]
fn conjugate_invariant_square_mean_sets_the_ntt_radix() {
    use poulpy_hal::{api::*, layouts::*};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    type BE = crate::NTT4x30CIPortable;
    let n = 1 << 13;
    assert_eq!(Module::<NTT4x30Portable>::max_base2k(n, 1, 128, true), 55);
    let k = Module::<BE>::max_base2k(n, 1, 128, true);
    assert_eq!(k, 54);
    let module = Module::<BE>::new(n as u64);
    let mut scratch = ScratchOwned::<BE>::alloc(module.svp_apply_dft_tmp_bytes(1).max(module.vec_znx_idft_apply_tmp_bytes()));
    let mut rng = StdRng::seed_from_u64(0);
    for radix in [k, k + 1] {
        let half = 1i64 << (radix - 1);
        let mut a = module.vec_znx_alloc(n, 1, 1);
        let mut b = module.scalar_znx_alloc(n, 1);
        for j in 0..n {
            let x = rng.random_range(-half..half);
            a.at_mut(0, 0)[j] = x;
            b.at_mut(0, 0)[j] = x;
        }
        let coeffs = a.at(0, 0);
        let want = (coeffs[0] as i128).pow(2) + coeffs[1..].iter().map(|&x| 2 * (x as i128).pow(2)).sum::<i128>();
        let mut ppol = module.svp_ppol_alloc(n, 1, PrepareHint::Reuse);
        let mut dft = module.vec_znx_dft_alloc(n, 1, 1);
        let mut big = module.vec_znx_big_alloc(n, 1, 1);
        module.svp_prepare(
            &mut ppol.to_backend_mut(),
            0,
            &poulpy_hal::test_suite::scalar_znx_backend_ref::<BE>(&b),
            0,
        );
        module.svp_apply_dft(
            &mut dft.to_backend_mut(),
            0,
            &ppol.to_backend_ref(),
            0,
            &vec_znx_backend_ref::<BE>(&a),
            0,
            &mut scratch.arena(),
        );
        module.vec_znx_idft_apply(&mut big.to_backend_mut(), 0, &dft.to_backend_ref(), 0, &mut scratch.arena());
        assert_eq!(big.at(0, 0)[0] == want, radix == k, "radix {radix}");
    }
}

#[test]
fn bootstrapping_presets_take_the_backend_radix() {
    use poulpy_ckks::{
        layouts::EvalModRing,
        presets::bootstrapping::all,
        test_suite::presets::{BOOTSTRAP_FAILURE_BITS, preset_for_backend},
    };
    use poulpy_core::layouts::LWEInfos;

    let (standard, ci): (Vec<_>, Vec<_>) = all()
        .unwrap()
        .into_iter()
        .partition(|preset| preset.plan().eval_mod_ring() == EvalModRing::Standard);
    for preset in ci {
        // An output limb of EvalMod's squares on the CI ring holds one square: its mean keeps
        // the radix at the nominal 52 on NTT4x30. FFT64's model of the CI ring takes 14, at
        // which the preset is re-derived with digits of 7 limbs.
        assert_eq!(
            preset.max_backend_base2k::<NTT4x30Portable, crate::NTT4x30CIPortable>(BOOTSTRAP_FAILURE_BITS),
            52
        );
        let adapted = preset_for_backend::<NTT4x30Portable, crate::NTT4x30CIPortable>(&preset).unwrap();
        assert_eq!((adapted.base2k(), adapted.bootstrap_k()), (52, preset.bootstrap_k()));
        let fft = preset_for_backend::<FFT64Portable, crate::FFT64CIPortable>(&preset).unwrap();
        assert_eq!(
            (fft.base2k(), fft.key_dsize(), fft.bootstrap_k()),
            (14, 7, preset.bootstrap_k())
        );
    }
    for preset in standard {
        // At `N = 2^16` and `2^15`, the giant-step passes sum a few hundred products:
        // NTT4x30 takes one bit above the nominal radix, FFT64 drops to 17.
        assert_eq!(
            preset.max_backend_base2k::<NTT4x30Portable, crate::NTT4x30CIPortable>(BOOTSTRAP_FAILURE_BITS),
            53
        );
        assert_eq!(
            preset.max_backend_base2k::<FFT64Portable, crate::FFT64CIPortable>(BOOTSTRAP_FAILURE_BITS),
            17
        );
        let ntt = preset_for_backend::<NTT4x30Portable, crate::NTT4x30CIPortable>(&preset).unwrap();
        let fft = preset_for_backend::<FFT64Portable, crate::FFT64CIPortable>(&preset).unwrap();
        let fft_dsize = if preset.log_n() == 15 { 2 } else { 7 };
        assert_eq!((ntt.base2k(), ntt.dense_to_sparse_dsize()), (53, 1));
        assert_eq!(
            (fft.base2k(), fft.key_dsize(), fft.dense_to_sparse_dsize()),
            (17, fft_dsize, 1)
        );
        for adapted in [fft, ntt] {
            assert_eq!(
                (adapted.input_k(), adapted.output_k(), adapted.bootstrap_k()),
                (preset.input_k(), preset.output_k(), preset.bootstrap_k())
            );
            assert_eq!(adapted.plan().c2s_guard_bits(), preset.plan().c2s_guard_bits());
            assert!(adapted.keys_layout().automorphism_key.k().as_usize() <= adapted.max_dense_modulus());
            let small_key = &adapted.keys_layout().encapsulation.as_ref().unwrap().dense_to_sparse;
            assert!(small_key.k().as_usize() <= adapted.max_sparse_modulus());
        }
    }
}

#[test]
fn test_convolution_by_const_fft64_portable() {
    let module: Module<FFT64Portable> = Module::<FFT64Portable>::new(64);
    test_convolution_by_const(&module, 8, 17);
    test_convolution_by_const(&module, 64, 17);
    test_convolution_by_const_add(&module, 8, 17);
    test_convolution_by_const_add(&module, 64, 17);
}

#[test]
fn test_convolution_fft64_portable() {
    let module: Module<FFT64Portable> = Module::<FFT64Portable>::new(64);
    test_convolution(&module, 8, 17);
    test_convolution(&module, 64, 17);
}

#[test]
fn test_convolution_pairwise_fft64_portable() {
    let module: Module<FFT64Portable> = Module::<FFT64Portable>::new(64);
    test_convolution_pairwise(&module, 8, 17);
    test_convolution_pairwise(&module, 64, 17);
}

#[test]
fn test_convolution_add_fft64_portable() {
    let module: Module<FFT64Portable> = Module::<FFT64Portable>::new(64);
    test_convolution_add(&module, 8, 17);
    test_convolution_add(&module, 64, 17);
}

#[test]
fn test_convolution_sum_fft64_portable() {
    let module: Module<FFT64Portable> = Module::<FFT64Portable>::new(64);
    test_convolution_sum(&module, 8, 17);
    test_convolution_sum(&module, 64, 17);
}

#[test]
fn test_convolution_by_const_ntt4x30_portable() {
    let module: Module<NTT4x30Portable> = Module::<NTT4x30Portable>::new(64);
    test_convolution_by_const(&module, 8, 50);
    test_convolution_by_const(&module, 64, 50);
    test_convolution_by_const_add(&module, 8, 50);
    test_convolution_by_const_add(&module, 64, 50);
}

#[test]
fn test_convolution_ntt4x30_portable() {
    let module: Module<NTT4x30Portable> = Module::<NTT4x30Portable>::new(64);
    test_convolution(&module, 8, 50);
    test_convolution(&module, 64, 50);
}

#[test]
fn test_convolution_pairwise_ntt4x30_portable() {
    let module: Module<NTT4x30Portable> = Module::<NTT4x30Portable>::new(64);
    test_convolution_pairwise(&module, 8, 50);
    test_convolution_pairwise(&module, 64, 50);
}

#[test]
fn test_convolution_add_ntt4x30_portable() {
    let module: Module<NTT4x30Portable> = Module::<NTT4x30Portable>::new(64);
    test_convolution_add(&module, 8, 50);
    test_convolution_add(&module, 64, 50);
}

#[test]
fn test_convolution_sum_ntt4x30_portable() {
    let module: Module<NTT4x30Portable> = Module::<NTT4x30Portable>::new(64);
    test_convolution_sum(&module, 8, 50);
    test_convolution_sum(&module, 64, 50);
}

#[test]
fn test_transform_domain_packed_byte_sizes() {
    use poulpy_hal::layouts::{Backend, PrepareHint};
    let (n, cols, size) = (256, 3, 5);
    let packed_bytes = n * cols * size * 4 * size_of::<u32>();
    assert_eq!(
        <NTT4x30Portable as Backend>::bytes_of_vec_znx_dft(n, cols, size),
        packed_bytes
    );
    assert_eq!(
        <NTT4x30Portable as Backend>::bytes_of_cnv_pvec_left(n, cols, size, PrepareHint::Reuse),
        packed_bytes
    );
    assert_eq!(
        <NTT4x30Portable as Backend>::bytes_of_cnv_pvec_right(n, cols, size, PrepareHint::Reuse),
        packed_bytes
    );
    assert_eq!(
        <NTT4x30Portable as Backend>::bytes_of_svp_ppol(n, cols, PrepareHint::Reuse),
        n * cols * 4 * size_of::<u32>()
    );
    let (rows, cols_in, cols_out) = (3, 2, 4);
    assert_eq!(
        <NTT4x30Portable as Backend>::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size, PrepareHint::Reuse),
        n * rows * cols_in * cols_out * size * 4 * size_of::<u32>()
    );
}

/// The packed kernels reject a result column past the last one and an operand of another degree.
#[test]
fn test_packed_kernels_reject_out_of_range_operands() {
    use poulpy_hal::{
        api::{
            CnvPVecAlloc, Convolution, ScratchOwnedAlloc, VecZnxDftAdd, VecZnxDftAlloc, VecZnxDftAutomorphism,
            VecZnxDftAutomorphismPlan,
        },
        layouts::{
            CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, PrepareHint, ScratchOwned,
            VecZnxDftToBackendMut, VecZnxDftToBackendRef, vec_znx_backend_ref,
        },
    };
    let (n, cols, size) = (64, 2, 3);
    let module = Module::<NTT4x30Portable>::new(2 * n as u64);
    let rejects = |what: &str, op: &mut dyn FnMut()| {
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(op)).is_err();
        assert!(panicked, "{what} accepted an out-of-range operand");
    };

    let a = module.vec_znx_alloc(n, cols, size);
    let mut left = module.cnv_pvec_left_alloc(n, cols, size, PrepareHint::Reuse);
    let mut right = module.cnv_pvec_right_alloc(n, cols, size, PrepareHint::Reuse);
    let mut scratch: ScratchOwned<NTT4x30Portable> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_add_tmp_bytes(0, size, size, size)
            .max(module.cnv_prepare_left_tmp_bytes(size, size))
            .max(module.cnv_prepare_right_tmp_bytes(size, size)),
    );
    module.cnv_prepare_left(
        &mut left.to_backend_mut(),
        &vec_znx_backend_ref::<NTT4x30Portable>(&a),
        &mut scratch.arena(),
    );
    module.cnv_prepare_right(
        &mut right.to_backend_mut(),
        &vec_znx_backend_ref::<NTT4x30Portable>(&a),
        &mut scratch.arena(),
    );

    let mut res = module.vec_znx_dft_alloc(n, cols, size);
    let b = module.vec_znx_dft_alloc(n, cols, size);
    let wide = module.vec_znx_dft_alloc(2 * n, cols, size);
    rejects("cnv_apply_dft_add", &mut || {
        module.cnv_apply_dft_add(
            0,
            &mut res.to_backend_mut(),
            cols,
            &left.to_backend_ref(),
            0,
            &right.to_backend_ref(),
            0,
            1,
            &mut scratch.arena(),
        )
    });
    rejects("vec_znx_dft_add", &mut || {
        module.vec_znx_dft_add(
            &mut res.to_backend_mut(),
            0,
            &wide.to_backend_ref(),
            0,
            &b.to_backend_ref(),
            0,
        )
    });
    let plan = module.vec_znx_dft_automorphism_plan(n, 5);
    rejects("vec_znx_dft_automorphism_add_with_plan", &mut || {
        module.vec_znx_dft_automorphism_add_with_plan(
            &plan,
            &mut res.to_backend_mut(),
            cols,
            &b.to_backend_ref(),
            0,
            &mut scratch.arena(),
        )
    });
}

/// The fused interleaved-digit product against the core reference body, on the same backend.
#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_digits_strided_bit_identical() {
    poulpy_core::test_suite::parity::test_gglwe_product_digits_strided(&Module::<NTT4x30Portable>::new(64), 50);
}

/// Rank-one tensor on enough limbs for the convolution kernels to reduce their accumulators several times per output.
#[cfg(feature = "enable-core")]
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
    let comparison = Module::<poulpy_cpu_oracle::NTT4x30Oracle>::new(u64::from(layout.n.0));
    let tested = Module::<NTT4x30Portable>::new(u64::from(layout.n.0));
    test_glwe_tensor_parity_for_layout(&layout, &[0, 51], &comparison, &tested);
}

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

cross_backend_test_suite! {
    mod vec_znx_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_zero_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_zero_matches_wrapper,
        test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_mat_znx_add_assign => poulpy_hal::test_suite::mat_znx::test_mat_znx_add_assign,
        test_mat_znx_copy => poulpy_hal::test_suite::mat_znx::test_mat_znx_copy,
        test_mat_znx_normalize => poulpy_hal::test_suite::mat_znx::test_mat_znx_normalize,
        test_vec_znx_add_assign_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign_matches_wrapper,
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
        test_vec_znx_gadget_decompose => poulpy_hal::test_suite::vec_znx::test_vec_znx_gadget_decompose,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_switch_ring_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring_matches_wrapper,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        test_vec_znx_copy_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_matches_wrapper,
    }
}

// Above the gadget-decomposition kernel's chunk length, against the oracle.
cross_backend_test_suite! {
    mod vec_znx_gadget_decompose_wide_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<11, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_gadget_decompose => poulpy_hal::test_suite::vec_znx::test_vec_znx_gadget_decompose,
    }
}

cross_backend_test_suite! {
    mod vec_znx_gadget_decompose_wide_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<11, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_gadget_decompose => poulpy_hal::test_suite::vec_znx::test_vec_znx_gadget_decompose,
    }
}

cross_backend_test_suite! {
    mod vec_znx_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_zero_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_zero_matches_wrapper,
        test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_mat_znx_add_assign => poulpy_hal::test_suite::mat_znx::test_mat_znx_add_assign,
        test_mat_znx_copy => poulpy_hal::test_suite::mat_znx::test_mat_znx_copy,
        test_mat_znx_normalize => poulpy_hal::test_suite::mat_znx::test_mat_znx_normalize,
        test_vec_znx_add_assign_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign_matches_wrapper,
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
        test_vec_znx_gadget_decompose => poulpy_hal::test_suite::vec_znx::test_vec_znx_gadget_decompose,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_switch_ring_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring_matches_wrapper,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        test_vec_znx_copy_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_matches_wrapper,
    }
}
cross_backend_test_suite! {
    mod vec_znx_ci_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_ci_embed_trace => poulpy_hal::test_suite::vec_znx::test_vec_znx_ci_embed_trace,
        test_scalar_znx_ci_embed => poulpy_hal::test_suite::vec_znx::test_scalar_znx_ci_embed,
    }
}

cross_backend_test_suite! {
    mod vec_znx_ci_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_ci_embed_trace => poulpy_hal::test_suite::vec_znx::test_vec_znx_ci_embed_trace,
        test_scalar_znx_ci_embed => poulpy_hal::test_suite::vec_znx::test_scalar_znx_ci_embed,
    }
}
cross_backend_test_suite! {
    mod svp_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

cross_backend_test_suite! {
    mod svp_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}
cross_backend_test_suite! {
    mod vec_znx_big_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_big_add => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
        test_vec_znx_big_add_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small,
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
        test_vec_znx_big_from_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_from_small,
        test_vec_znx_big_inner_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_inner_sum,
        test_vec_znx_big_col_weighted_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_col_weighted_sum,
        test_vec_znx_scalar_product => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_scalar_product,
    }
}

cross_backend_test_suite! {
    mod vec_znx_big_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_big_add => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
        test_vec_znx_big_add_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small,
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
        test_vec_znx_big_from_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_from_small,
        test_vec_znx_big_inner_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_inner_sum,
        test_vec_znx_big_col_weighted_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_col_weighted_sum,
        test_vec_znx_scalar_product => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_scalar_product,
    }
}
cross_backend_test_suite! {
    mod vec_znx_dft_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_zero => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_zero,
    }
}

cross_backend_test_suite! {
    mod vec_znx_dft_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_zero => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_zero,
    }
}
cross_backend_test_suite! {
    mod vec_znx_dft_automorphism_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize,
    }
}

cross_backend_test_suite! {
    mod vec_znx_dft_automorphism_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize,
    }
}
cross_backend_test_suite! {
    mod vmp_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
        test_word_compat_prepare_hint_sizes => poulpy_hal::test_suite::word_compat::test_word_compat_prepare_hint_sizes,
    }
}

cross_backend_test_suite! {
    mod vmp_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
        test_word_compat_prepare_hint_sizes => poulpy_hal::test_suite::word_compat::test_word_compat_prepare_hint_sizes,
    }
}

backend_test_suite! {
    mod vec_znx_dft_operand_validation_ntt4x30,
    backend = crate::NTT4x30Portable,
    params = TestParams { size: 16, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_dft_rejects_invalid_operands => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_rejects_invalid_operands,
        test_vec_znx_dft_transforms_reject_invalid_operands => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_transforms_reject_invalid_operands,
    }
}

backend_test_suite! {
    mod derived_fft64,
    backend = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_derived,
        test_vmp_apply_dft_to_dft_add_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_to_dft_add_derived,
        test_vec_znx_lsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_derived,
        test_vec_znx_rsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_derived,
        test_vec_znx_lsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_add_derived,
        test_vec_znx_lsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_sub_derived,
        test_vec_znx_rsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_add_derived,
        test_vec_znx_rsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_sub_derived,
        test_vec_znx_lsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_assign_derived,
        test_vec_znx_rsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_assign_derived,
        test_vec_znx_mul_xp_minus_one_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_derived,
        test_vec_znx_mul_xp_minus_one_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_assign_derived,
        test_vec_znx_fill_uniform_source_all_derived => poulpy_hal::test_suite::derived::test_vec_znx_fill_uniform_source_all_derived,
        test_vec_znx_add_scalar_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_add_scalar_assign_derived,
        test_scalar_znx_automorphism_derived => poulpy_hal::test_suite::derived::test_scalar_znx_automorphism_derived,
        test_scalar_znx_ci_embed_derived => poulpy_hal::test_suite::derived::test_scalar_znx_ci_embed_derived,
        test_vec_znx_big_add_small_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_add_small_derived,
        test_vec_znx_big_sub_small_a_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_a_derived,
        test_vec_znx_big_sub_small_b_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_b_derived,
        test_vec_znx_idft_normalize_derived => poulpy_hal::test_suite::derived::test_vec_znx_idft_normalize_derived,
        test_vec_znx_dft_automorphism_add_with_plan_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_add_with_plan_derived,
        test_vec_znx_dft_automorphism_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_derived,
        test_svp_apply_dft_derived => poulpy_hal::test_suite::derived::test_svp_apply_dft_derived,
        test_cnv_prepare_self_derived => poulpy_hal::test_suite::derived::test_cnv_prepare_self_derived,
        test_cnv_apply_dft_add_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_add_derived,
        test_cnv_apply_dft_sum_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_sum_derived,
        test_cnv_pairwise_apply_dft_derived => poulpy_hal::test_suite::derived::test_cnv_pairwise_apply_dft_derived,
        test_cnv_by_const_apply_add_derived => poulpy_hal::test_suite::derived::test_cnv_by_const_apply_add_derived,
    }
}

backend_test_suite! {
    mod derived_ntt4x30,
    backend = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_derived,
        test_vmp_apply_dft_to_dft_add_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_to_dft_add_derived,
        test_vec_znx_lsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_derived,
        test_vec_znx_rsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_derived,
        test_vec_znx_lsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_add_derived,
        test_vec_znx_lsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_sub_derived,
        test_vec_znx_rsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_add_derived,
        test_vec_znx_rsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_sub_derived,
        test_vec_znx_lsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_assign_derived,
        test_vec_znx_rsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_assign_derived,
        test_vec_znx_mul_xp_minus_one_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_derived,
        test_vec_znx_mul_xp_minus_one_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_assign_derived,
        test_vec_znx_fill_uniform_source_all_derived => poulpy_hal::test_suite::derived::test_vec_znx_fill_uniform_source_all_derived,
        test_vec_znx_add_scalar_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_add_scalar_assign_derived,
        test_scalar_znx_automorphism_derived => poulpy_hal::test_suite::derived::test_scalar_znx_automorphism_derived,
        test_scalar_znx_ci_embed_derived => poulpy_hal::test_suite::derived::test_scalar_znx_ci_embed_derived,
        test_vec_znx_big_add_small_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_add_small_derived,
        test_vec_znx_big_sub_small_a_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_a_derived,
        test_vec_znx_big_sub_small_b_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_b_derived,
        test_vec_znx_idft_normalize_derived => poulpy_hal::test_suite::derived::test_vec_znx_idft_normalize_derived,
        test_vec_znx_dft_automorphism_add_with_plan_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_add_with_plan_derived,
        test_vec_znx_dft_automorphism_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_derived,
        test_svp_apply_dft_derived => poulpy_hal::test_suite::derived::test_svp_apply_dft_derived,
        test_cnv_prepare_self_derived => poulpy_hal::test_suite::derived::test_cnv_prepare_self_derived,
        test_cnv_apply_dft_add_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_add_derived,
        test_cnv_apply_dft_sum_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_sum_derived,
        test_cnv_pairwise_apply_dft_derived => poulpy_hal::test_suite::derived::test_cnv_pairwise_apply_dft_derived,
        test_cnv_by_const_apply_add_derived => poulpy_hal::test_suite::derived::test_cnv_by_const_apply_add_derived,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT4x30Portable,
    params = TestParams { size: 1<<12, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
    }
}

// Gated on `enable-core`, like `core_impl`, which implements the noise seam for both portable backends.
#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core,
    backend = crate::NTT4x30Portable,
    // the core suite computes at the module degree, no sweep
    params = TestParams { size: 1<<12, base2k: 12, n: 1<<12 },
    tests = {
        test_vec_znx_add_noise => poulpy_core::test_suite::sampling::test_vec_znx_add_noise,
        test_vec_znx_big_add_noise => poulpy_core::test_suite::sampling::test_vec_znx_big_add_noise,
    }
}

#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core_fft64,
    backend = crate::FFT64Portable,
    // the core suite computes at the module degree, no sweep
    params = TestParams { size: 1<<12, base2k: 17, n: 1<<12 },
    tests = {
        test_vec_znx_add_noise => poulpy_core::test_suite::sampling::test_vec_znx_add_noise,
        test_vec_znx_big_add_noise => poulpy_core::test_suite::sampling::test_vec_znx_big_add_noise,
    }
}

backend_test_suite! {
    mod window_fft64,
    backend = crate::FFT64Portable,
    params = TestParams { size: 1 << 8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
        test_vec_znx_dft_rejects_zero_step => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_rejects_zero_step,
        test_cnv_prepare_rejects_mismatched_shape => poulpy_hal::test_suite::convolution::test_cnv_prepare_rejects_mismatched_shape,
        test_vmp_apply_dft_to_dft_rejects_column_mismatch => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_rejects_column_mismatch,
        test_vec_znx_embed_add_sub => poulpy_hal::test_suite::embed::test_vec_znx_embed_add_sub,
        test_vec_znx_big_embed_add_sub => poulpy_hal::test_suite::embed::test_vec_znx_big_embed_add_sub,
        test_convolution_embed => poulpy_hal::test_suite::convolution::test_convolution_embed,
        test_convolution_galois => poulpy_hal::test_suite::convolution::test_convolution_galois,
        test_cnv_by_const_rejects_degree_mismatch => poulpy_hal::test_suite::convolution::test_cnv_by_const_rejects_degree_mismatch,
        test_transfer_padded_lengths => poulpy_hal::test_suite::transfer::test_transfer_padded_lengths,
    }
}

backend_test_suite! {
    mod window_ntt4x30,
    backend = crate::NTT4x30Portable,
    params = TestParams { size: 1 << 8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
        test_vec_znx_dft_rejects_zero_step => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_rejects_zero_step,
        test_cnv_prepare_rejects_mismatched_shape => poulpy_hal::test_suite::convolution::test_cnv_prepare_rejects_mismatched_shape,
        test_vmp_apply_dft_to_dft_rejects_column_mismatch => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_rejects_column_mismatch,
        test_vec_znx_embed_add_sub => poulpy_hal::test_suite::embed::test_vec_znx_embed_add_sub,
        test_vec_znx_big_embed_add_sub => poulpy_hal::test_suite::embed::test_vec_znx_big_embed_add_sub,
        test_convolution_embed => poulpy_hal::test_suite::convolution::test_convolution_embed,
        test_convolution_galois => poulpy_hal::test_suite::convolution::test_convolution_galois,
        test_cnv_by_const_rejects_degree_mismatch => poulpy_hal::test_suite::convolution::test_cnv_by_const_rejects_degree_mismatch,
        test_transfer_padded_lengths => poulpy_hal::test_suite::transfer::test_transfer_padded_lengths,
    }
}

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod fft64,
    backend = crate::FFT64Portable,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 17, n: 1<<8 },
);

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod ntt4x30,
    backend = crate::NTT4x30Portable,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
);

#[test]
fn test_vec_znx_rsh_assign_multi_limb_matches_rsh() {
    use poulpy_hal::api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceAll, VecZnxRsh, VecZnxRshAssign, VecZnxRshTmpBytes,
    };
    use poulpy_hal::layouts::ScratchOwned;
    use poulpy_hal::source::Source;
    use poulpy_hal::test_suite::{download_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};

    let n = 8usize;
    let module: Module<NTT4x30Portable> = Module::<NTT4x30Portable>::new(n as u64);
    let mut scratch: ScratchOwned<NTT4x30Portable> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes(4));
    let base2k = 52usize;
    let mut source = Source::new([3u8; 32]);

    // shifts spanning >= 2 limbs previously corrupted the in-place variant
    for size in [2usize, 3, 4] {
        for k in [60usize, 90, 105, 116] {
            if k / base2k + 1 > size {
                continue;
            }
            let mut a_be = module.vec_znx_alloc(n, 1, size);
            module.vec_znx_fill_uniform_source_all(base2k, size * base2k, &mut a_be, &mut source);
            let mut want_be = module.vec_znx_alloc(n, 1, size);
            module.vec_znx_rsh(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Portable>(&mut want_be),
                0,
                &vec_znx_backend_ref::<NTT4x30Portable>(&a_be),
                0,
                &mut scratch.borrow(),
            );
            let mut got_be = a_be.clone();
            module.vec_znx_rsh_assign(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Portable>(&mut got_be),
                0,
                &mut scratch.borrow(),
            );
            assert_eq!(
                download_vec_znx::<NTT4x30Portable>(&got_be),
                download_vec_znx::<NTT4x30Portable>(&want_be),
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
    requires_eq::<poulpy_hal::layouts::VecZnxDftOwned<crate::FFT64Portable>>();
    requires_eq::<poulpy_hal::layouts::VecZnxBigOwned<crate::FFT64Portable>>();
    requires_eq::<poulpy_hal::layouts::SvpPPolOwned<crate::FFT64Portable>>();
    requires_eq::<poulpy_hal::layouts::VmpPMatOwned<crate::FFT64Portable>>();
}

#[cfg(feature = "enable-core")]
#[cfg(feature = "enable-bin-fhe")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64, backend = crate::FFT64Portable);

#[test]
fn test_hal_serialization_fft64_portable() {
    poulpy_hal::test_suite::serialization::test_serialization(&Module::<FFT64Portable>::new(1024));
}

#[cfg(feature = "enable-core")]
#[test]
fn test_glwe_public_key_rank1_golden() {
    use poulpy_core::test_suite::noise::encryption::glwe_public_key_rank1_digests;
    use poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples;
    use poulpy_cpu_oracle::test_suite::{ControlledSamplingFFT64Oracle, ControlledSamplingNTT4x30Oracle};
    // Pins the discrete-Gaussian sampler and PNM3 component-noise wire format.
    // Digests cover the public key, encryption of a message, and encryption of zero.
    const FFT64: [u64; 3] = [9615310425194628740, 1969118651227205468, 6537548132563225856];
    const NTT4X30: [u64; 3] = [9915393708676454290, 9177372829677904665, 15298276994712787542];
    assert_eq!(
        (
            glwe_public_key_rank1_digests(&Module::<FFT64Portable>::new(256), 17),
            glwe_public_key_rank1_digests(&Module::<NTT4x30Portable>::new(256), 52),
        ),
        (FFT64, NTT4X30)
    );
    let fft_oracle = with_backend_samples(Module::<FFT64Portable>::new(256), |_| {
        glwe_public_key_rank1_digests(&Module::<ControlledSamplingFFT64Oracle>::new(256), 17)
    });
    let ntt_oracle = with_backend_samples(Module::<NTT4x30Portable>::new(256), |_| {
        glwe_public_key_rank1_digests(&Module::<ControlledSamplingNTT4x30Oracle>::new(256), 52)
    });
    assert_eq!((fft_oracle, ntt_oracle), (FFT64, NTT4X30));
}

#[cfg(feature = "enable-core")]
#[test]
fn test_core_serialization_fft64_portable() {
    poulpy_core::test_suite::serialization::test_serialization(&Module::<FFT64Portable>::new(64));
}

#[cfg(feature = "enable-bin-fhe")]
#[test]
fn test_blind_rotation_key_serialization_fft64_portable() {
    poulpy_bin_fhe::blind_rotation::test_suite::serialization::test_blind_rotation_key_serialization(
        &Module::<FFT64Portable>::new(256),
    );
}

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_fft64_portable() {
    poulpy_core::test_suite::parity::test_gglwe_product_dft_selected(&Module::<FFT64Portable>::new(64), 12);
}

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_ntt4x30_portable() {
    poulpy_core::test_suite::parity::test_gglwe_product_dft_selected(&Module::<NTT4x30Portable>::new(64), 12);
}

#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 12, n: 1<<8 },
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
        glwe_trace => poulpy_core::test_suite::parity::test_glwe_trace_parity,
        trace_packing => poulpy_core::test_suite::parity::test_trace_packing_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        linear_transformation_powers_of_two => poulpy_core::test_suite::parity::test_linear_transformation_powers_of_two_parity,
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

#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 12, n: 1<<8 },
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
        glwe_trace => poulpy_core::test_suite::parity::test_glwe_trace_parity,
        trace_packing => poulpy_core::test_suite::parity::test_trace_packing_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        linear_transformation_powers_of_two => poulpy_core::test_suite::parity::test_linear_transformation_powers_of_two_parity,
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

fn vec_znx_big_normalize_limb_bounds<BE>(module: &Module<BE>)
where
    BE: poulpy_hal::test_suite::TestBackend,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    Module<BE>: poulpy_hal::api::VecZnxBigAlloc<BE>
        + poulpy_hal::api::VecZnxBigFromSmall<BE>
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxFillUniformSourceAll<BE>
        + poulpy_hal::api::VecZnxAddAssign<BE>
        + poulpy_hal::api::VecZnxSubNegateAssign<BE>,
    poulpy_hal::layouts::ScratchOwned<BE>: poulpy_hal::api::ScratchOwnedAlloc<BE>,
{
    use poulpy_hal::{
        api::{
            ScratchOwnedAlloc, VecZnxAddAssign, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize,
            VecZnxBigNormalizeTmpBytes, VecZnxFillUniformSourceAll, VecZnxSubNegateAssign,
        },
        layouts::{ScratchOwned, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendMut, ZnxView},
        source::Source,
        test_suite::{vec_znx_backend_mut, vec_znx_backend_ref},
    };
    let mut source = Source::new([2u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
    for a_base2k in 1..=51usize {
        for res_base2k in 1..=51usize {
            for offset in [-(a_base2k as i64), -3, -1, 0, 1, 3, a_base2k as i64] {
                for a_size in 1..=3usize {
                    for res_size in 1..=3usize {
                        // 2x - y (x at radix 62, y at radix 1) is uniform on [-2^62, 2^62), the radix-63 digits the fill cannot draw.
                        let mut a = module.vec_znx_alloc(module.n(), 1, a_size);
                        let mut x = module.vec_znx_alloc(module.n(), 1, a_size);
                        module.vec_znx_fill_uniform_source_all(1, a_size, &mut a, &mut source);
                        module.vec_znx_fill_uniform_source_all(62, a_size * 62, &mut x, &mut source);
                        module.vec_znx_sub_negate_assign(
                            &mut vec_znx_backend_mut::<BE>(&mut a),
                            0,
                            &vec_znx_backend_ref::<BE>(&x),
                            0,
                        );
                        module.vec_znx_add_assign(&mut vec_znx_backend_mut::<BE>(&mut a), 0, &vec_znx_backend_ref::<BE>(&x), 0);
                        let mut big = module.vec_znx_big_alloc(module.n(), 1, a_size);
                        module.vec_znx_big_from_small(&mut big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a), 0);
                        let mut res = module.vec_znx_alloc(module.n(), 1, res_size);
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
fn test_vec_znx_big_normalize_limb_bounds_fft64_portable() {
    vec_znx_big_normalize_limb_bounds(&Module::<FFT64Portable>::new(8));
}

#[test]
fn test_vec_znx_big_normalize_limb_bounds_ntt4x30_portable() {
    vec_znx_big_normalize_limb_bounds(&Module::<NTT4x30Portable>::new(8));
}

const NORMALIZE_IDFT_BOUND: i128 = (1_073_479_681i128 * 1_071_513_601 * 1_070_727_169 * 1_068_236_801 - 1) / 2;

#[test]
fn test_vec_znx_big_normalize_input_bound_integer() {
    use crate::{
        kernels::{vec_znx::vec_znx_normalize_portable, vec_znx_big_normalize::vec_znx_big_normalize_portable},
        test_suite::znx_normalize::oracle,
    };
    use poulpy_hal::layouts::{
        VecZnx, VecZnxBig, VecZnxBigToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    };

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
    let mut random = 0x123456789abcdef0u128;
    for a_base2k in 1..=62 {
        for res_base2k in 1..=62 {
            for a_size in 0..=8 {
                let mut input = VecZnxBig::<AlignedBuf, i128, NTT4x30Portable>::from_data(
                    AlignedBuf::from(vec![0; 16 * N * a_size]),
                    N,
                    1,
                    a_size,
                );
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
                let mut small_input =
                    VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0; 8 * N * a_size]), N, 1, a_size);
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
                    let mut output =
                        VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0; 8 * N * res_size]), N, 1, res_size);
                    let mut small_output =
                        VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0; 8 * N * res_size]), N, 1, res_size);
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
                        vec_znx_big_normalize_portable::<NTT4x30Portable>(
                            &mut VecZnxToBackendMut::<NTT4x30Portable>::to_backend_mut(&mut output),
                            res_base2k,
                            res_size * res_base2k,
                            offset,
                            0,
                            &input.to_backend_ref(),
                            a_base2k,
                            0,
                        );
                        vec_znx_normalize_portable::<FFT64Portable>(
                            &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(
                                &mut small_output,
                            ),
                            res_base2k,
                            res_size * res_base2k,
                            offset,
                            0,
                            &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&small_input),
                            a_base2k,
                            0,
                        );
                        for i in 4..=5 {
                            for j in 0..res_size {
                                assert_eq!(output.at(0, j)[i], small_output.at(0, j)[i]);
                            }
                        }
                        for i in 0..N {
                            let limbs: Vec<_> = (0..a_size).map(|j| input.at(0, j)[i]).collect();
                            let want = oracle(&limbs, a_base2k, res_base2k, res_size * res_base2k, res_size, offset);
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
    use crate::kernels::{
        vec_znx::{vec_znx_normalize_assign_portable, vec_znx_normalize_portable},
        vec_znx_big_normalize::vec_znx_big_normalize_portable,
    };
    use poulpy_hal::layouts::{
        VecZnx, VecZnxBig, VecZnxBigToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    };

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
        let mut small = VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0; 8 * values.len()]), 1, 1, values.len());
        let mut big = VecZnxBig::<AlignedBuf, i128, NTT4x30Portable>::from_data(
            AlignedBuf::from(vec![0; 16 * values.len()]),
            1,
            1,
            values.len(),
        );
        for (j, &value) in values.iter().enumerate() {
            small.at_mut(0, j)[0] = value;
            big.at_mut(0, j)[0] = value as i128;
        }
        let mut output =
            VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![93; 8 * expected.len()]), 1, 1, expected.len());
        vec_znx_big_normalize_portable::<NTT4x30Portable>(
            &mut VecZnxToBackendMut::<NTT4x30Portable>::to_backend_mut(&mut output),
            res_base2k,
            res_k,
            offset,
            0,
            &big.to_backend_ref(),
            a_base2k,
            0,
        );
        assert_eq!((0..expected.len()).map(|j| output.at(0, j)[0]).collect::<Vec<_>>(), expected);
        if a_base2k <= 62 && res_base2k <= 62 {
            vec_znx_normalize_portable::<FFT64Portable>(
                &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut output),
                res_base2k,
                res_k,
                offset,
                0,
                &<VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&small),
                a_base2k,
                0,
            );
            assert_eq!((0..expected.len()).map(|j| output.at(0, j)[0]).collect::<Vec<_>>(), expected);
        } else if a_base2k == 64 && res_base2k == 64 {
            vec_znx_normalize_assign_portable::<FFT64Portable>(
                res_base2k,
                res_k,
                0,
                &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut small),
                0,
            );
            assert_eq!(small.at(0, 0), expected);
        }
    }
}

#[test]
fn test_vec_znx_big_normalize_wide_radices() {
    use crate::{kernels::vec_znx_big_normalize::vec_znx_big_normalize_portable, test_suite::znx_normalize::oracle};
    use poulpy_hal::layouts::{VecZnx, VecZnxBig, VecZnxBigToBackendRef, VecZnxToBackendMut, ZnxView, ZnxViewMut};
    for a_base2k in [1, 63, 64, 65, 126, 127] {
        for res_base2k in [1, 17, 63, 64] {
            for size in 1..=3 {
                let mut input =
                    VecZnxBig::<AlignedBuf, i128, NTT4x30Portable>::from_data(AlignedBuf::from(vec![0; 32 * size]), 2, 1, size);
                for j in 0..size {
                    input
                        .at_mut(0, j)
                        .copy_from_slice(&[-NORMALIZE_IDFT_BOUND, NORMALIZE_IDFT_BOUND]);
                }
                let mut output = VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0; 16 * size]), 2, 1, size);
                for offset in [-400, -64, -(a_base2k as i64), -1, 0, 1, 64, 400] {
                    vec_znx_big_normalize_portable::<NTT4x30Portable>(
                        &mut VecZnxToBackendMut::<NTT4x30Portable>::to_backend_mut(&mut output),
                        res_base2k,
                        size * res_base2k,
                        offset,
                        0,
                        &input.to_backend_ref(),
                        a_base2k,
                        0,
                    );
                    for i in 0..2 {
                        let limbs: Vec<_> = (0..size).map(|j| input.at(0, j)[i]).collect();
                        let want = oracle(&limbs, a_base2k, res_base2k, size * res_base2k, size, offset);
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
        FFT64Portable, NTT4x30Portable,
        kernels::{
            vec_znx::{
                vec_znx_normalize_assign_portable, vec_znx_normalize_assign_range_raw_portable, vec_znx_normalize_portable,
                vec_znx_normalize_range_raw_portable,
            },
            vec_znx_big_normalize::{vec_znx_big_normalize_portable, vec_znx_big_normalize_range_raw_portable},
        },
    };
    use dashu_int::IBig;
    use poulpy_hal::AlignedBuf;
    use poulpy_hal::layouts::{
        VecZnx, VecZnxBig, VecZnxBigToBackendRef, VecZnxShape, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    };

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
            value = crate::test_suite::znx_normalize::round_ties_even(value, drop);
        }
        value <<= size * kr - k;
        for digit in result.iter_mut().rev() {
            let carry = (&value + (IBig::ONE << (kr - 1))) >> kr;
            *digit = i64::try_from(&value - (&carry << kr)).unwrap();
            value = carry;
        }
        result
    }

    fn small(size: usize) -> VecZnx<AlignedBuf, i64> {
        VecZnx::from_data(AlignedBuf::from(vec![0xa5; 8 * N * 2 * size]), N, 2, size)
    }

    #[allow(clippy::too_many_arguments)]
    fn check(a: &[Vec<i128>], ka: usize, kr: usize, k: usize, size: usize, offset: i64, ranges: bool) {
        let mut input = small(a.len());
        let mut wide = VecZnxBig::<AlignedBuf, i128, NTT4x30Portable>::from_data(
            AlignedBuf::from(vec![0; 16 * N * 2 * a.len()]),
            N,
            2,
            a.len(),
        );
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
        let verify = |name: &str, output: &VecZnx<AlignedBuf, i64>| {
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
            let src = <VecZnx<AlignedBuf, i64> as VecZnxToBackendRef<FFT64Portable>>::to_backend_ref(&input);
            if ranges {
                let ptr = output.data_mut().as_mut_ptr().cast::<i64>();
                for (start, len) in [(0, 2), (2, 3), (5, 4)] {
                    unsafe {
                        vec_znx_normalize_range_raw_portable::<FFT64Portable>(
                            ptr,
                            VecZnxShape::new(N, 2, size),
                            kr,
                            k,
                            offset,
                            1,
                            &src,
                            ka,
                            1,
                            start,
                            len,
                        );
                    }
                }
            } else {
                vec_znx_normalize_portable::<FFT64Portable>(
                    &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut output),
                    kr,
                    k,
                    offset,
                    1,
                    &src,
                    ka,
                    1,
                );
            }
            verify("narrow", &output);
        }
        output = small(size);
        let wide = wide.to_backend_ref();
        if ranges {
            let ptr = output.data_mut().as_mut_ptr().cast::<i64>();
            for (start, len) in [(0, 2), (2, 3), (5, 4)] {
                unsafe {
                    vec_znx_big_normalize_range_raw_portable::<NTT4x30Portable>(
                        ptr,
                        VecZnxShape::new(N, 2, size),
                        kr,
                        k,
                        offset,
                        1,
                        &wide,
                        ka,
                        1,
                        start,
                        len,
                    );
                }
            }
        } else {
            vec_znx_big_normalize_portable::<NTT4x30Portable>(
                &mut VecZnxToBackendMut::<NTT4x30Portable>::to_backend_mut(&mut output),
                kr,
                k,
                offset,
                1,
                &wide,
                ka,
                1,
            );
        }
        verify("wide", &output);
        if narrow && ka == kr && size == a.len() && offset == 0 {
            if ranges {
                let ptr = input.data_mut().as_mut_ptr().cast::<i64>();
                for (start, len) in [(0, 2), (2, 3), (5, 4)] {
                    unsafe {
                        vec_znx_normalize_assign_range_raw_portable::<FFT64Portable>(
                            ptr,
                            VecZnxShape::new(N, 2, size),
                            kr,
                            k,
                            0,
                            1,
                            start,
                            len,
                        );
                    }
                }
            } else {
                vec_znx_normalize_assign_portable::<FFT64Portable>(
                    kr,
                    k,
                    0,
                    &mut <VecZnx<AlignedBuf, i64> as VecZnxToBackendMut<FFT64Portable>>::to_backend_mut(&mut input),
                    1,
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

#[cfg(feature = "enable-core")]
poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption_fft64portable,
    backend_ref = poulpy_cpu_oracle::test_suite::ControlledSamplingFFT64Oracle,
    backend_test = crate::FFT64Portable,
);

#[cfg(feature = "enable-core")]
poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption_ntt4x30portable,
    backend_ref = poulpy_cpu_oracle::test_suite::ControlledSamplingFFT64Oracle,
    backend_test = crate::NTT4x30Portable,
);

// Encryption on a module of twice the operand degree samples and computes as a
// module of the operand degree.
#[cfg(feature = "enable-core")]
poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption_sub_degree_fft64portable,
    backend_ref = poulpy_cpu_oracle::test_suite::ControlledSamplingFFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<7, n: 1<<7, base2k: 12 },
    test_size = 1<<8,
);

// A module computes every operation of a smaller degree as a module of that
// degree does.
#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_sub_degree_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<7, base2k: 12, n: 1<<7 },
    test_size = 1<<8,
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
        glwe_trace => poulpy_core::test_suite::parity::test_glwe_trace_parity,
        trace_packing => poulpy_core::test_suite::parity::test_trace_packing_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        linear_transformation_powers_of_two => poulpy_core::test_suite::parity::test_linear_transformation_powers_of_two_parity,
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

// A module computes every operation of a smaller degree as a module of that
// degree does.
#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_sub_degree_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<7, base2k: 12, n: 1<<7 },
    test_size = 1<<8,
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
        glwe_trace => poulpy_core::test_suite::parity::test_glwe_trace_parity,
        trace_packing => poulpy_core::test_suite::parity::test_trace_packing_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        linear_transformation_powers_of_two => poulpy_core::test_suite::parity::test_linear_transformation_powers_of_two_parity,
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

#[cfg(feature = "enable-ckks")]
mod ckks_overrides;

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_overrides;

// A lazy op into a compact destination must not drop the carries a
// non-canonical source holds past its `k`.
#[cfg(feature = "enable-ckks")]
mod ckks_noncanonical_compact_dst {
    use crate::FFT64Portable;
    use poulpy_ckks::api::{CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSNegOps, CKKSPow2Ops, CKKSSubOps};
    use poulpy_ckks::layouts::{CKKSCiphertextOwned, CKKSModuleAlloc};
    use poulpy_ckks::{GLWEPlaintextMeta, Scale, SetCKKSInfos};
    use poulpy_core::layouts::{GLWELayout, GLWEToBackendMut};
    use poulpy_core::{GLWECopy, GLWENormalize};
    use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow};
    use poulpy_hal::layouts::{Module, ScratchOwned, ZnxView, ZnxViewMut};

    fn ct(module: &Module<FFT64Portable>, k: u32) -> CKKSCiphertextOwned<FFT64Portable> {
        let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&GLWELayout {
            n: 64u32.into(),
            base2k: 16u32.into(),
            k: k.into(),
            rank: 1u32.into(),
        });
        ct.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(16),
            ..Default::default()
        });
        ct
    }

    #[test]
    fn lazy_ops_keep_carries_past_k() {
        let module = Module::<FFT64Portable>::new(64);
        let mut scratch = ScratchOwned::<FFT64Portable>::alloc(1 << 20);
        // k = 32 over 4 limbs: 3 units of limb 1 parked in limb 2.
        let mut src = ct(&module, 64);
        src.set_log_budget(16);
        src.data_mut().at_mut(1, 2)[0] = 3 << 16;
        GLWEToBackendMut::<FFT64Portable>::set_canonical(&mut src, false);

        let limbs = |dst: &mut CKKSCiphertextOwned<FFT64Portable>, scratch: &mut ScratchOwned<FFT64Portable>| {
            module.glwe_normalize_assign(dst, &mut scratch.borrow());
            [dst.data().at(1, 1)[0], dst.data().at(1, 1)[32]]
        };
        type Op = fn(
            &Module<FFT64Portable>,
            &mut CKKSCiphertextOwned<FFT64Portable>,
            &CKKSCiphertextOwned<FFT64Portable>,
            &mut ScratchOwned<FFT64Portable>,
        );
        let ops: [(&str, Op, [i64; 2]); 10] = [
            ("glwe_copy", |m, d, s, sc| m.glwe_copy(d, s, &mut sc.borrow()), [3, 0]),
            (
                "add_into",
                |m, d, s, sc| m.ckks_add_into(d, s, &ct(m, 32), &mut sc.borrow()).unwrap(),
                [3, 0],
            ),
            (
                "sub_into",
                |m, d, s, sc| m.ckks_sub_into(d, &ct(m, 32), s, &mut sc.borrow()).unwrap(),
                [-3, 0],
            ),
            (
                "add_assign",
                |m, d, s, sc| m.ckks_add_assign(d, s, &mut sc.borrow()).unwrap(),
                [3, 0],
            ),
            (
                "sub_assign",
                |m, d, s, sc| m.ckks_sub_assign(d, s, &mut sc.borrow()).unwrap(),
                [-3, 0],
            ),
            ("copy", |m, d, s, sc| m.ckks_copy(d, s, &mut sc.borrow()).unwrap(), [3, 0]),
            (
                "double",
                |m, d, s, sc| m.ckks_double_into(d, s, &mut sc.borrow()).unwrap(),
                [6, 0],
            ),
            ("neg", |m, d, s, sc| m.ckks_neg_into(d, s, &mut sc.borrow()).unwrap(), [-3, 0]),
            (
                "mul_i",
                |m, d, s, sc| m.ckks_mul_i_into(d, s, &mut sc.borrow()).unwrap(),
                [0, 3],
            ),
            (
                "div_i",
                |m, d, s, sc| m.ckks_div_i_into(d, s, &mut sc.borrow()).unwrap(),
                [0, -3],
            ),
        ];
        for (name, op, want) in ops {
            let mut dst = ct(&module, 32);
            op(&module, &mut dst, &src, &mut scratch);
            assert_eq!(limbs(&mut dst, &mut scratch), want, "{name}");
        }
    }
}

backend_test_suite! {
    mod vec_znx_dft_operand_validation_fft64,
    backend = crate::FFT64Portable,
    params = TestParams { size: 16, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_rejects_invalid_operands => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_rejects_invalid_operands,
        test_vec_znx_dft_transforms_reject_invalid_operands => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_transforms_reject_invalid_operands,
    }
}
