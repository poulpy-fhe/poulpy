use poulpy_hal::{
    DEFAULTALIGN, is_aligned,
    layouts::{Backend, Module},
    test_suite::convolution::{
        test_convolution, test_convolution_accumulate, test_convolution_by_const, test_convolution_pairwise,
    },
};

use crate::NTT3x42Ifma;

#[cfg(test)]
mod ntt3x42_ifma_tests {
    use crate::ntt3x42_ifma::{
        primes::Primes42, reference::arithmetic::b_ntt3x42_ifma_to_znx128_ref, vec_znx_dft::simd_b_ntt3x42_ifma_to_znx128,
    };
    use poulpy_hal::{
        backend_test_suite, cross_backend_test_suite,
        layouts::{Backend, SvpPPolOwned, VecZnxDftOwned, ZnxView, ZnxViewMut, ZnxZero},
    };

    cross_backend_test_suite! {
        mod vec_znx,
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
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
            test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
            test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
            test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        }
    }

    cross_backend_test_suite! {
        mod vec_znx_dft,
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
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
        mod vec_znx_dft_large,
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
        params = TestParams { size: 1<<12, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
        }
    }

    cross_backend_test_suite! {
        mod svp,
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
            test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
        }
    }

    cross_backend_test_suite! {
        mod vmp,
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
            test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
        }
    }

    cross_backend_test_suite! {
        mod vec_znx_big,
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
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
        backend = crate::NTT3x42Ifma,
        params = TestParams { size: 1<<12, base2k: 50 },
        tests = {
            test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
            test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
            test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
        }
    }

    // NTT size-range coverage.
    //
    // The planar IFMA NTT runs breadth-first level loops with fused head/tail
    // stages. These sizes cover the scalar-only edges, the fused tail, and
    // larger mixed-width levels, confirming bit-exact agreement with the
    // reference backend.

    // n = 1024: only block-local inner levels run.
    cross_backend_test_suite! {
        mod ntt_n1024,
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
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
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
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
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
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
        backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
        backend_test = crate::NTT3x42Ifma,
        params = TestParams { size: 1<<15, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }

    #[test]
    fn test_packed_layout_zero_and_display() {
        const N: usize = 64;
        const COLS: usize = 2;
        const SIZE: usize = 3;

        let mut dft = VecZnxDftOwned::<crate::NTT3x42Ifma>::alloc(N, COLS, SIZE);
        let block_bytes = <crate::NTT3x42Ifma as Backend>::bytes_of_vec_znx_dft(N, 1, 1);
        let byte_len = <crate::NTT3x42Ifma as Backend>::bytes_of_vec_znx_dft(N, COLS, SIZE);
        dft.data[..byte_len].fill(0xa5);

        dft.zero_at(1, 1);

        let offset = (COLS + 1) * block_bytes;
        assert!(dft.data[..offset].iter().all(|&byte| byte == 0xa5));
        assert!(dft.data[offset..offset + block_bytes].iter().all(|&byte| byte == 0));
        assert!(dft.data[offset + block_bytes..byte_len].iter().all(|&byte| byte == 0xa5));

        let display = format!("{dft}");
        assert!(display.contains("<backend-packed representation:"));

        dft.zero();
        assert!(dft.data[..byte_len].iter().all(|&byte| byte == 0));

        let mut svp = SvpPPolOwned::<crate::NTT3x42Ifma>::alloc(N, COLS);
        let display = format!("{svp}");
        assert!(display.contains("<backend-packed representation:"));
        assert!(
            std::panic::catch_unwind(|| {
                let _ = svp.at(1, 0);
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = svp.at_mut(1, 0);
            }))
            .is_err()
        );
    }

    #[test]
    fn test_b_to_znx128_ifma_asm_edges_vs_ref() {
        const Q: [u64; 3] = <Primes42 as poulpy_hal::layouts::PrimeSet>::Q;
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
            let n = values.len();
            for (i, &value) in values.iter().enumerate() {
                for k in 0..3 {
                    let residue = (value % q[k] as u128) as u64;
                    dst[k * n + i] = if (i + k).is_multiple_of(2) { residue } else { residue + q[k] };
                }
            }
        }

        fn assert_matches_ref(n: usize, b: &[u64]) {
            let mut got = vec![0i128; n];
            let mut expected = vec![0i128; n];
            unsafe { simd_b_ntt3x42_ifma_to_znx128(n, &mut got, b) };
            b_ntt3x42_ifma_to_znx128_ref(n, &mut expected, b);
            assert_eq!(got, expected);
        }

        let n = values.len();
        let mut b = vec![0u64; 3 * n];
        fill_b_format(&mut b, &values, &Q);
        assert_matches_ref(n, &b);

        let mut skewed = vec![0u64; 3 * n + 1];
        fill_b_format(&mut skewed[1..], &values, &Q);
        assert_matches_ref(n, &skewed[1..]);
    }
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

    let n = 64;
    let base2k = 50;
    let module = Module::<NTT3x42Ifma>::new(n as u64);
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
        let mut scratch: ScratchOwned<NTT3x42Ifma> = ScratchOwned::alloc(
            module
                .vmp_apply_dft_to_dft_tmp_bytes(size_out, a_size, rows, cols_in, cols_out, size_out)
                .max(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size_out)),
        );

        let mut a = module.vec_znx_alloc(cols_in, a_size);
        a.fill_uniform(base2k, &mut source);
        let mut a_dft = module.vec_znx_dft_alloc(cols_in, a_size);
        for col in 0..cols_in {
            module.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft.to_backend_mut(),
                col,
                &VecZnxToBackendRef::<NTT3x42Ifma>::to_backend_ref(&a),
                col,
            );
        }

        let mut mat = module.mat_znx_alloc(rows, cols_in, cols_out, size_out);
        mat.fill_uniform(base2k, &mut source);
        let mut pmat = module.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
        module.vmp_prepare(
            &mut pmat.to_backend_mut(),
            &MatZnxToBackendRef::<NTT3x42Ifma>::to_backend_ref(&mat),
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

#[test]
fn test_convolution_by_const_ntt3x42_ifma() {
    let module: Module<NTT3x42Ifma> = Module::<NTT3x42Ifma>::new(8);
    test_convolution_by_const(&module, 12);
}

#[test]
fn test_convolution_ntt3x42_ifma() {
    let module: Module<NTT3x42Ifma> = Module::<NTT3x42Ifma>::new(8);
    test_convolution(&module, 12);
}

#[test]
fn test_convolution_pairwise_ntt3x42_ifma() {
    let module: Module<NTT3x42Ifma> = Module::<NTT3x42Ifma>::new(8);
    test_convolution_pairwise(&module, 12);
}

#[test]
fn test_gglwe_product_digits_strided_bit_identical() {
    poulpy_core::test_suite::parity::test_gglwe_product_digits_strided(&Module::<NTT3x42Ifma>::new(64), 50);
}

#[test]
fn test_convolution_accumulate_ntt3x42_ifma() {
    let module: Module<NTT3x42Ifma> = Module::<NTT3x42Ifma>::new(8);
    test_convolution_accumulate(&module, 12);
}

#[test]
#[should_panic(expected = "NTT3x42Ifma requires n >= 8")]
fn test_ntt3x42_ifma_rejects_too_small_ring() {
    let _ = Module::<NTT3x42Ifma>::new(4);
}

#[test]
fn test_ntt3x42_ifma_zeroed_allocation_alignment_and_padding() {
    for len in [1usize, 63, 64, 65, 4_096, (1 << 20) + 1] {
        let bytes = <NTT3x42Ifma as Backend>::alloc_zeroed_bytes(len);
        assert_eq!(bytes.len(), len.next_multiple_of(DEFAULTALIGN));
        assert!(is_aligned(bytes.as_ptr()));
        assert!(bytes.iter().all(|&byte| byte == 0));
    }
}
