use poulpy_hal::{backend_test_suite, cross_backend_test_suite};
use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_accumulate, test_convolution_by_const, test_convolution_pairwise,
    },
};

use crate::NTT4x30Avx512;

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vec_znx_add_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_into_backend_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
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
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
    }
}

cross_backend_test_suite! {
    mod svp,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vec_znx_big_add_into => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_into,
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
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
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
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
    }
}

cross_backend_test_suite! {
    mod vmp,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<12, base2k: 50 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
        test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
    }
}

backend_test_suite! {
    mod lwe_matrix,
    backend = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        glwe_expand_lwe_matrix_decrypt => poulpy_core::test_suite::noise::test_glwe_expand_lwe_matrix_decrypt,
    }
}

// NTT CHANGE_MODE_N boundary tests.
// CHANGE_MODE_N = 1024: for n <= 1024 the AVX NTT runs fully by-block;
// for n > 1024 it first completes upper levels by-level then switches to
// by-block for the remaining levels. These suites ensure both modes are
// exercised and agree with the reference backend.

// n = 1024: last size that uses by-block only.
cross_backend_test_suite! {
    mod ntt_n1024,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<10, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 8192: large size exercising many by-level stages.
cross_backend_test_suite! {
    mod ntt_n8192,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<13, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 16384: largest size before the AVX NTT switches to by-level mode only.
cross_backend_test_suite! {
    mod ntt_n16384,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<14, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 32768: largest size (still by-level only) included in the test suite.
cross_backend_test_suite! {
    mod ntt_n32768,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<15, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

#[test]
fn test_convolution_direct() {
    let module = Module::<NTT4x30Avx512>::new(1 << 8);
    test_convolution(&module, 50);
    test_convolution_by_const(&module, 50);
    test_convolution_pairwise(&module, 50);
    test_convolution_accumulate(&module, 50);
}

#[test]
fn test_gglwe_product_digits_strided_bit_identical() {
    poulpy_core::test_suite::parity::test_gglwe_product_digits_strided(&Module::<NTT4x30Avx512>::new(64), 50);
}

#[test]
fn test_vmp_apply_dft_to_dft_digits_strided_bit_identical() {
    use poulpy_hal::{
        api::{
            ScratchOwnedAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftCopy, VmpApplyDftToDft, VmpApplyDftToDftAccumulate,
            VmpApplyDftToDftDigitsStrided, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
        },
        layouts::{
            FillUniform, MatZnxToBackendRef, ScratchOwned, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut,
            VecZnxDftToBackendRef, VecZnxToBackendRef, VmpPMatToBackendMut, VmpPMatToBackendRef,
        },
        source::Source,
    };

    let module = Module::<NTT4x30Avx512>::new(64);
    let mut source = Source::new([2u8; 32]);
    let cases: [(usize, usize, usize, usize); 6] = [
        (2, 1, 2, 4),
        (2, 2, 1, 5),
        (3, 1, 1, 7),
        (3, 2, 2, 2),
        (2, 1, 1, 1),
        (3, 2, 1, 8),
    ];

    let mut any_nonzero = false;
    for (dsize, cols_in, cols_out, a_size) in cases {
        let rows = a_size.div_ceil(dsize);
        let size_out = a_size;
        let mut scratch: ScratchOwned<NTT4x30Avx512> = ScratchOwned::alloc(
            module
                .vmp_apply_dft_to_dft_tmp_bytes(size_out, a_size, rows, cols_in, cols_out, size_out)
                .max(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size_out)),
        );

        let mut a = module.vec_znx_alloc(cols_in, a_size);
        a.fill_uniform(50, &mut source);
        let mut a_dft = module.vec_znx_dft_alloc(cols_in, a_size);
        for col in 0..cols_in {
            module.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft.to_backend_mut(),
                col,
                &VecZnxToBackendRef::<NTT4x30Avx512>::to_backend_ref(&a),
                col,
            );
        }

        let mut mat = module.mat_znx_alloc(rows, cols_in, cols_out, size_out);
        mat.fill_uniform(50, &mut source);
        let mut pmat = module.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
        module.vmp_prepare(
            &mut pmat.to_backend_mut(),
            &MatZnxToBackendRef::<NTT4x30Avx512>::to_backend_ref(&mat),
            &mut scratch.arena(),
        );

        let mut res_sequential = module.vec_znx_dft_alloc(cols_out, size_out);
        for di in 0..dsize {
            let digit_size = ((a_size + di) / dsize).min(rows);
            let mut digit = module.vec_znx_dft_alloc(cols_in, digit_size.max(1));
            let mut digit_backend = digit.to_backend_mut();
            let mut digit_view = digit_backend.with_size_mut(digit_size);
            for col in 0..cols_in {
                module.vec_znx_dft_copy(dsize, dsize - di - 1, &mut digit_view, col, &a_dft.to_backend_ref(), col);
            }
            let res_size = res_sequential.size() - ((dsize - di) as isize - 2).max(0) as usize;
            let mut res_backend = res_sequential.to_backend_mut();
            let mut res_view = res_backend.with_size_mut(res_size);
            if di == 0 {
                module.vmp_apply_dft_to_dft(
                    &mut res_view,
                    &digit_view.reborrow_backend_ref(),
                    &pmat.to_backend_ref(),
                    0,
                    &mut scratch.arena(),
                );
            } else {
                module.vmp_apply_dft_to_dft_accumulate(
                    &mut res_view,
                    &digit_view.reborrow_backend_ref(),
                    &pmat.to_backend_ref(),
                    di,
                    &mut scratch.arena(),
                );
            }
        }

        let mut res_strided = module.vec_znx_dft_alloc(cols_out, size_out);
        module.vmp_apply_dft_to_dft_digits_strided(
            &mut res_strided.to_backend_mut(),
            &a_dft.to_backend_ref(),
            dsize,
            &pmat.to_backend_ref(),
            &mut scratch.arena(),
        );

        let sequential = res_sequential.data.as_slice();
        let strided = res_strided.data.as_slice();
        any_nonzero |= sequential.iter().any(|&byte| byte != 0);
        assert_eq!(
            sequential, strided,
            "strided VMP differs for dsize={dsize}, cols_in={cols_in}, cols_out={cols_out}, a_size={a_size}"
        );
    }
    assert!(any_nonzero);
}

cross_backend_test_suite! {
    mod word_compat,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_word_compat_dft_bytes => poulpy_hal::test_suite::word_compat::test_word_compat_dft_bytes,
        test_word_compat_svp_prepare_bytes => poulpy_hal::test_suite::word_compat::test_word_compat_svp_prepare_bytes,
        test_word_compat_dft_cross_idft => poulpy_hal::test_suite::word_compat::test_word_compat_dft_cross_idft,
    }
}
