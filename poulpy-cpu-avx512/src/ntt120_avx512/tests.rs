use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{DataView, FillUniform, MatZnx, Module, ScratchOwned},
    source::Source,
    test_suite::convolution::{test_convolution, test_convolution_by_const, test_convolution_pairwise},
};
use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

use crate::NTT120Avx512;

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vec_znx_add_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_into,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
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
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
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
        test_vec_znx_big_normalize_fused => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize_fused,
        test_vec_znx_big_sub_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_assign,
        test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
        test_vec_znx_big_sub_small_a_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_assign,
        test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
        test_vec_znx_big_sub_small_b_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_assign,
    }
}

cross_backend_test_suite! {
    mod vec_znx_dft,
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vec_znx_dft_add_into => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_into,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
        test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
    }
}

cross_backend_test_suite! {
    mod vmp,
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT120Avx512,
    params = TestParams { size: 1<<12, base2k: 50 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
        test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
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
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<10, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 8192: large size exercising many by-level stages.
cross_backend_test_suite! {
    mod ntt_n8192,
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<13, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 16384: largest size before the AVX NTT switches to by-level mode only.
cross_backend_test_suite! {
    mod ntt_n16384,
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<14, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 32768: largest size (still by-level only) included in the test suite.
cross_backend_test_suite! {
    mod ntt_n32768,
    backend_ref =  poulpy_cpu_ref::NTT120Ref,
    backend_test = crate::NTT120Avx512,
    params = TestParams { size: 1<<15, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

#[test]
fn test_convolution_direct() {
    let module = Module::<NTT120Avx512>::new(1 << 8);
    test_convolution(&module, 50);
    test_convolution_by_const(&module, 50);
    test_convolution_pairwise(&module, 50);
}

#[test]
fn test_vmp_prepare_matches_ref_layout() {
    let n = 1usize << 8;
    let rows = 3;
    let cols_in = 2;
    let cols_out = 2;
    let size = 4;

    let module_ref = Module::<poulpy_cpu_ref::NTT120Ref>::new(n as u64);
    let module_avx = Module::<NTT120Avx512>::new(n as u64);

    let mut source = Source::new([0u8; 32]);
    let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(n, rows, cols_in, cols_out, size);
    mat.fill_uniform(50, &mut source);

    let mut pmat_ref = module_ref.vmp_pmat_alloc(rows, cols_in, cols_out, size);
    let mut pmat_avx = module_avx.vmp_pmat_alloc(rows, cols_in, cols_out, size);

    let mut scratch_ref: ScratchOwned<poulpy_cpu_ref::NTT120Ref> =
        ScratchOwned::alloc(module_ref.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));
    let mut scratch_avx: ScratchOwned<NTT120Avx512> =
        ScratchOwned::alloc(module_avx.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));

    module_ref.vmp_prepare(&mut pmat_ref, &mat, scratch_ref.borrow());
    module_avx.vmp_prepare(&mut pmat_avx, &mat, scratch_avx.borrow());

    let nrows = rows * cols_in;
    let ncols = cols_out * size;
    let n_block_pairs = n / 4;
    let plane_stride = n_block_pairs * ncols * nrows * 4;
    let offset_u64 = nrows * ncols * 8;
    let pmat_ref_u64: &[u64] = bytemuck::cast_slice(pmat_ref.data().as_ref());

    let mut expected = vec![0u64; 4 * plane_stride];
    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let dst_base_u64 = if col_i == ncols - 1 && !ncols.is_multiple_of(2) {
                col_i * nrows * 8 + row_i * 8
            } else {
                (col_i / 2) * (nrows * 16) + row_i * 16 + (col_i % 2) * 8
            };

            for bp in 0..n_block_pairs {
                let blk0 = 2 * bp;
                let blk1 = blk0 + 1;
                let chunk0 = &pmat_ref_u64[dst_base_u64 + blk0 * offset_u64..dst_base_u64 + blk0 * offset_u64 + 8];
                let chunk1 = &pmat_ref_u64[dst_base_u64 + blk1 * offset_u64..dst_base_u64 + blk1 * offset_u64 + 8];

                for p in 0..4usize {
                    let dst = p * plane_stride + bp * (ncols * nrows * 4) + col_i * (nrows * 4) + row_i * 4;
                    expected[dst..dst + 4].copy_from_slice(&[chunk0[p], chunk0[4 + p], chunk1[p], chunk1[4 + p]]);
                }
            }
        }
    }

    assert_eq!(pmat_avx.data().as_ref(), bytemuck::cast_slice::<u64, u8>(&expected));
}
