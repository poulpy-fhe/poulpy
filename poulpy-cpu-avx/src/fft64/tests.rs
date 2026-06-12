use poulpy_hal::{backend_test_suite, cross_backend_test_suite};
use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_accumulate, test_convolution_by_const, test_convolution_pairwise,
    },
};

use crate::FFT64Avx;

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    params = TestParams { size: 1<<8, base2k: 12 },
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
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
    }
}

cross_backend_test_suite! {
    mod svp,
    backend_ref =  poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    params = TestParams { size: 1<<8, base2k: 12 },
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
    backend_ref =  poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    params = TestParams { size: 1<<8, base2k: 12 },
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
    backend_ref =  poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::FFT64Avx,
    params = TestParams { size: 1<<12, base2k: 12 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
        test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
    }
}

backend_test_suite! {
    mod lwe_matrix,
    backend = crate::FFT64Avx,
    params = TestParams { size: 1<<8, base2k: 17 },
    tests = {
        glwe_expand_lwe_matrix_decrypt => poulpy_core::test_suite::test_glwe_expand_lwe_matrix_decrypt,
        lwe_matrix_mul_identity => poulpy_core::test_suite::test_lwe_matrix_mul_identity,
        lwe_matrix_mul_decrypts_to_plain_product => poulpy_core::test_suite::test_lwe_matrix_mul_decrypts_to_plain_product,
    }
}

#[test]
fn test_convolution_direct() {
    let module = Module::<FFT64Avx>::new(1 << 8);
    test_convolution(&module, 12);
    test_convolution_by_const(&module, 12);
    test_convolution_pairwise(&module, 12);
    test_convolution_accumulate(&module, 12);
}

#[test]
fn lwe_matrix_mul_bounded_u_matches_unbounded() {
    use poulpy_core::{
        LWEMatrixMul,
        layouts::{Base2K, CoeffMatrix, CoeffMatrixLayout, Degree, LWEMatrix, LWEMatrixLayout, ModuleCoreAlloc, TorusPrecision},
    };
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
        source::Source,
    };

    let module = Module::<FFT64Avx>::new(1 << 8);
    let rows_in = 200usize;
    let rows_out = 200usize;
    let lwe_n = 200usize;
    let base2k = Base2K(12);
    let size = 3usize;
    let k = TorusPrecision((base2k.0 as usize * size) as u32);

    let u_infos = CoeffMatrixLayout {
        n: Degree(rows_in as u32),
        rows_out,
        base2k,
        k,
    };
    let a_infos = LWEMatrixLayout {
        rows: rows_in,
        n: Degree(lwe_n as u32),
        base2k,
        k,
    };
    let res_infos = LWEMatrixLayout {
        rows: rows_out,
        n: Degree(lwe_n as u32),
        base2k,
        k,
    };

    let mut src = Source::new([0u8; 32]);
    let mask = (1i64 << base2k.0) - 1;
    fn fill(v: &mut VecZnx<Vec<u8>>, s: &mut Source, mask: i64) {
        for x in v.raw_mut() {
            *x = s.next_i64() & mask;
        }
    }

    let mut u16: CoeffMatrix<Vec<u8>, i16> = module.coeff_matrix_alloc_from_infos(&u_infos);
    let mut u64: CoeffMatrix<Vec<u8>, i64> = module.coeff_matrix_alloc_from_infos(&u_infos);
    let mut a: LWEMatrix<Vec<u8>> = module.lwe_matrix_alloc_from_infos(&a_infos);
    fill(u16.data_mut(), &mut src, mask);
    u64.data_mut().raw_mut().copy_from_slice(u16.data().raw());
    fill(a.body_mut(), &mut src, mask);
    fill(a.mask_mut(), &mut src, mask);

    let mut res16: LWEMatrix<Vec<u8>> = module.lwe_matrix_alloc_from_infos(&res_infos);
    let mut res64: LWEMatrix<Vec<u8>> = module.lwe_matrix_alloc_from_infos(&res_infos);
    let mut scratch = ScratchOwned::alloc(module.lwe_matrix_mul_tmp_bytes(&res_infos, &u_infos, &a_infos));
    module.lwe_matrix_mul(&mut res16, &u16, &a, &mut scratch.borrow());
    module.lwe_matrix_mul(&mut res64, &u64, &a, &mut scratch.borrow());

    assert_eq!(res16.body().raw(), res64.body().raw(), "K16 body != K64 body");
    assert_eq!(res16.mask().raw(), res64.mask().raw(), "K16 mask != K64 mask");
}

#[test]
fn lwe_matrix_mul_bodies_matches_per_column() {
    use poulpy_core::{
        LWEMatrixMul,
        layouts::{Base2K, CoeffMatrix, CoeffMatrixLayout, Degree, ModuleCoreAlloc, TorusPrecision},
    };
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
        source::Source,
    };

    let module = Module::<FFT64Avx>::new(1 << 8);
    let rows_in = 256usize;
    let rows_out = 256usize;
    let base2k = Base2K(12);
    let size = 2usize;
    let k_prec = TorusPrecision((base2k.0 as usize * size) as u32);
    let num_bodies = 5usize; // not a multiple of the SIMD width; exercises tails
    let bk = base2k.0 as usize;

    let u_infos = CoeffMatrixLayout {
        n: Degree(rows_in as u32),
        rows_out,
        base2k,
        k: k_prec,
    };

    let mut src = Source::new([0u8; 32]);
    let mask = (1i64 << base2k.0) - 1;

    // i16-bounded U so the K16 batched kernel is exercised.
    let mut u: CoeffMatrix<Vec<u8>, i16> = module.coeff_matrix_alloc_from_infos(&u_infos);
    for x in u.data_mut().raw_mut() {
        *x = src.next_i64() & mask;
    }

    let mut bodies: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
    for x in bodies.raw_mut() {
        *x = src.next_i64() & mask;
    }

    let mut res_all: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
    let mut scratch = ScratchOwned::alloc(module.lwe_matrix_mul_bodies_tmp_bytes(&u_infos, num_bodies, size, size));
    module.lwe_matrix_mul_bodies(&mut res_all, bk, &u, &bodies, bk, &mut scratch.borrow());

    for kk in 0..num_bodies {
        let mut body1: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, size);
        for limb in 0..size {
            body1.at_mut(0, limb).copy_from_slice(bodies.at(kk, limb));
        }
        let mut res1: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, size);
        let mut sc = ScratchOwned::alloc(module.lwe_matrix_mul_bodies_tmp_bytes(&u_infos, 1, size, size));
        module.lwe_matrix_mul_bodies(&mut res1, bk, &u, &body1, bk, &mut sc.borrow());
        for limb in 0..size {
            assert_eq!(
                &res_all.at(kk, limb)[..rows_out],
                &res1.at(0, limb)[..rows_out],
                "batched column {kk} limb {limb} != single-body result"
            );
        }
    }
}

#[test]
fn lwe_matrix_mul_bodies_prepared_matches_unprepared() {
    use poulpy_core::{
        CoeffMatrixPrepare, LWEMatrixMul,
        layouts::{Base2K, CoeffBound, CoeffMatrix, CoeffMatrixLayout, Degree, ModuleCoreAlloc, TorusPrecision},
    };
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
        source::Source,
    };

    fn check<BU: CoeffBound>(module: &Module<FFT64Avx>) {
        let rows_in = 256usize;
        let rows_out = 256usize;
        let base2k = Base2K(12);
        let size = 2usize;
        let k_prec = TorusPrecision((base2k.0 as usize * size) as u32);
        let num_bodies = 7usize;
        let bk = base2k.0 as usize;
        let u_infos = CoeffMatrixLayout {
            n: Degree(rows_in as u32),
            rows_out,
            base2k,
            k: k_prec,
        };

        let mut src = Source::new([0u8; 32]);
        let mask = (1i64 << base2k.0) - 1;

        let mut u: CoeffMatrix<Vec<u8>, BU> = module.coeff_matrix_alloc_from_infos(&u_infos);
        for x in u.data_mut().raw_mut() {
            *x = src.next_i64() & mask;
        }
        let mut bodies: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
        for x in bodies.raw_mut() {
            *x = src.next_i64() & mask;
        }

        let mut res_np: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
        let mut s1 = ScratchOwned::alloc(module.lwe_matrix_mul_bodies_tmp_bytes(&u_infos, num_bodies, size, size));
        module.lwe_matrix_mul_bodies(&mut res_np, bk, &u, &bodies, bk, &mut s1.borrow());

        let pu = module.coeff_matrix_prepare(&u);
        let mut res_p: VecZnx<Vec<u8>> = module.vec_znx_alloc(num_bodies, size);
        let mut s2 = ScratchOwned::alloc(module.lwe_matrix_mul_bodies_prepared_tmp_bytes(&pu, num_bodies, size, size));
        module.lwe_matrix_mul_bodies_prepared(&mut res_p, bk, &pu, &bodies, bk, &mut s2.borrow());

        for c in 0..num_bodies {
            for limb in 0..size {
                assert_eq!(
                    &res_p.at(c, limb)[..rows_out],
                    &res_np.at(c, limb)[..rows_out],
                    "prepared != unprepared, col {c} limb {limb}"
                );
            }
        }
    }

    let module = Module::<FFT64Avx>::new(1 << 8);
    check::<i16>(&module);
    check::<i32>(&module);
    check::<i64>(&module);
}
