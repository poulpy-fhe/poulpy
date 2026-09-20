/// HAL `VecZnx` methods excluding normalization.
#[macro_export]
macro_rules! hal_impl_vec_znx_without_normalize {
    () => {
        $crate::hal_impl_vec_znx_without_normalize!(standard);
    };
    ($automorphism:ident) => {
        fn vec_znx_zero(module: &Module<Self>, res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>, res_col: usize) {
            <Self as HalVecZnxDefault>::vec_znx_zero_default(module, res, res_col)
        }

        fn vec_znx_normalize_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault>::vec_znx_normalize_tmp_bytes_default(module)
        }

        fn vec_znx_add(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_add_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_add_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_add_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_sub(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_sub_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_sub_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_sub_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_sub_negate_assign(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_sub_negate_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_negate(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_negate_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_negate_assign(module: &Module<Self>, a: &mut VecZnxBackendMut<'_, Self>, a_col: usize) {
            <Self as HalVecZnxDefault>::vec_znx_negate_assign_default(module, a, a_col)
        }

        fn vec_znx_lsh_assign(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault>::vec_znx_lsh_assign_default(module, base2k, k, a, a_col, &mut scratch);
        }

        fn vec_znx_rotate(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            assert_eq!(
                <Self as ::poulpy_hal::layouts::Backend>::cyclotomic_order(module),
                2 * module.n() as i64,
                "monomial multiplication is not defined for the invariant ring"
            );
            <Self as HalVecZnxDefault>::vec_znx_rotate_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_rotate_assign_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault>::vec_znx_rotate_assign_tmp_bytes_default(module)
        }

        fn vec_znx_rotate_assign(
            module: &Module<Self>,
            k: i64,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            assert_eq!(
                <Self as ::poulpy_hal::layouts::Backend>::cyclotomic_order(module),
                2 * module.n() as i64,
                "monomial multiplication is not defined for the invariant ring"
            );
            <Self as HalVecZnxDefault>::vec_znx_rotate_assign_default(module, k, a, a_col, &mut scratch);
        }

        $crate::hal_impl_vec_znx_automorphism!($automorphism);

        fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(module: &Module<Self>, _size: usize) -> usize {
            <Self as HalVecZnxDefault>::vec_znx_mul_xp_minus_one_assign_tmp_bytes_default(module)
        }

        fn vec_znx_mul_xp_minus_one_assign(
            module: &Module<Self>,
            k: i64,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            assert_eq!(
                <Self as ::poulpy_hal::layouts::Backend>::cyclotomic_order(module),
                2 * module.n() as i64,
                "monomial multiplication is not defined for the invariant ring"
            );
            <Self as HalVecZnxDefault>::vec_znx_mul_xp_minus_one_assign_default(module, k, res, res_col, &mut scratch);
        }

        fn vec_znx_switch_ring(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_switch_ring_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_copy(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_copy_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_fill_uniform(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault>::vec_znx_fill_uniform_default(module, base2k, k, res, res_col, seed)
        }
    };
}

/// HAL `VecZnx` normalization methods.
#[macro_export]
macro_rules! hal_impl_vec_znx_normalize {
    () => {
        fn vec_znx_normalize(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_base2k: usize,
            res_k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_base2k: usize,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault>::vec_znx_normalize_default(
                module,
                res,
                res_base2k,
                res_k,
                res_offset,
                res_col,
                a,
                a_base2k,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_normalize_assign(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            a_offset: i64,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault>::vec_znx_normalize_assign_default(module, base2k, k, a_offset, a, a_col, &mut scratch);
        }
    };
}

#[macro_export]
macro_rules! hal_impl_vec_znx {
    () => {
        $crate::hal_impl_vec_znx_without_normalize!(standard);
        $crate::hal_impl_vec_znx_normalize!();
    };
    (fft64) => {
        $crate::hal_impl_vec_znx_without_normalize!(fft64);
        $crate::hal_impl_vec_znx_normalize!();
    };
}

#[macro_export]
macro_rules! hal_impl_vec_znx_automorphism {
    (standard) => {
        $crate::hal_impl_vec_znx_automorphism!(fft64);
    };
    (fft64) => {
        fn vec_znx_automorphism(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            if <Self as ::poulpy_hal::layouts::Backend>::cyclotomic_order(module) == 4 * module.n() as i64 {
                $crate::reference::vec_znx::vec_znx_ci_automorphism::<Self>(k, res, res_col, a, a_col)
            } else {
                <Self as HalVecZnxDefault>::vec_znx_automorphism_default(module, k, res, res_col, a, a_col)
            }
        }

        fn vec_znx_automorphism_assign_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault>::vec_znx_automorphism_assign_tmp_bytes_default(module)
        }

        fn vec_znx_automorphism_assign(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            if <Self as ::poulpy_hal::layouts::Backend>::cyclotomic_order(module) == 4 * module.n() as i64 {
                let (tmp, _) = scratch.take_region(res.n() * ::std::mem::size_of::<i64>());
                let tmp = ::bytemuck::cast_slice_mut::<u8, i64>(::poulpy_hal::api::HostBufMut::into_bytes(tmp));
                $crate::reference::vec_znx::vec_znx_ci_automorphism_assign::<Self>(k, res, res_col, tmp)
            } else {
                <Self as HalVecZnxDefault>::vec_znx_automorphism_assign_default(module, k, res, res_col, &mut scratch);
            }
        }
    };
}
