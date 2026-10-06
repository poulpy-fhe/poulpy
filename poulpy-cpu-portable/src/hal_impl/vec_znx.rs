/// HAL `VecZnx` methods excluding normalization.
#[macro_export]
macro_rules! hal_impl_vec_znx_without_normalize {
    () => {
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

        fn vec_znx_automorphism(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault>::vec_znx_automorphism_default(module, k, res, res_col, a, a_col)
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
            <Self as HalVecZnxDefault>::vec_znx_automorphism_assign_default(module, k, res, res_col, &mut scratch);
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

/// HAL `VecZnx` maps between the conjugate-invariant and standard rings.
#[macro_export]
macro_rules! hal_impl_vec_znx_ci {
    () => {
        fn vec_znx_ci_embed(
            _module: &::poulpy_hal::layouts::Module<Self>,
            res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            $crate::kernels::vec_znx::vec_znx_ci_embed_portable::<Self>(res, res_col, a, a_col)
        }

        fn vec_znx_ci_trace(
            _module: &::poulpy_hal::layouts::Module<Self>,
            res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            $crate::kernels::vec_znx::vec_znx_ci_trace_portable::<Self>(res, res_col, a, a_col)
        }
    };
}

/// HAL `VecZnx` monomial methods (`X^k`).
#[macro_export]
macro_rules! hal_impl_vec_znx_monomial {
    () => {
        fn vec_znx_rotate(
            module: &::poulpy_hal::layouts::Module<Self>,
            k: i64,
            res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $crate::hal_defaults::HalVecZnxDefault>::vec_znx_rotate_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_rotate_assign_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>) -> usize {
            <Self as $crate::hal_defaults::HalVecZnxDefault>::vec_znx_rotate_assign_tmp_bytes_default(module)
        }

        fn vec_znx_rotate_assign(
            module: &::poulpy_hal::layouts::Module<Self>,
            k: i64,
            a: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            a_col: usize,
            scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $crate::hal_defaults::HalVecZnxDefault>::vec_znx_rotate_assign_default(module, k, a, a_col, &mut scratch);
        }

        fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, _size: usize) -> usize {
            <Self as $crate::hal_defaults::HalVecZnxDefault>::vec_znx_mul_xp_minus_one_assign_tmp_bytes_default(module)
        }

        fn vec_znx_mul_xp_minus_one_assign(
            module: &::poulpy_hal::layouts::Module<Self>,
            k: i64,
            res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $crate::hal_defaults::HalVecZnxDefault>::vec_znx_mul_xp_minus_one_assign_default(
                module,
                k,
                res,
                res_col,
                &mut scratch,
            );
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
        $crate::hal_impl_vec_znx_without_normalize!();
        $crate::hal_impl_vec_znx_normalize!();
    };
}
