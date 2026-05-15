#[macro_export]
macro_rules! hal_impl_vec_znx {
    () => {
        fn scalar_znx_fill_ternary_hw_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::ScalarZnxBackendMut<'_, Self>,
            res_col: usize,
            hw: usize,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::scalar_znx_fill_ternary_hw_backend_default(module, res, res_col, hw, seed)
        }

        fn scalar_znx_fill_ternary_prob_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::ScalarZnxBackendMut<'_, Self>,
            res_col: usize,
            prob: f64,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::scalar_znx_fill_ternary_prob_backend_default(module, res, res_col, prob, seed)
        }

        fn scalar_znx_fill_binary_hw_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::ScalarZnxBackendMut<'_, Self>,
            res_col: usize,
            hw: usize,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::scalar_znx_fill_binary_hw_backend_default(module, res, res_col, hw, seed)
        }

        fn scalar_znx_fill_binary_prob_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::ScalarZnxBackendMut<'_, Self>,
            res_col: usize,
            prob: f64,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::scalar_znx_fill_binary_prob_backend_default(module, res, res_col, prob, seed)
        }

        fn scalar_znx_fill_binary_block_backend(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::ScalarZnxBackendMut<'_, Self>,
            res_col: usize,
            block_size: usize,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::scalar_znx_fill_binary_block_backend_default(module, res, res_col, block_size, seed)
        }

        fn vec_znx_zero_backend<'r>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_zero_backend_default(module, res, res_col)
        }

        fn vec_znx_sub_inner_product_assign_backend<'r, 'a, 'b>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            res_limb: usize,
            res_offset: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
            a_limb: usize,
            a_offset: usize,
            b: &poulpy_hal::layouts::ScalarZnxBackendRef<'b, Self>,
            b_col: usize,
            b_offset: usize,
            len: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_sub_inner_product_assign_backend_default(
                module, res, res_col, res_limb, res_offset, a, a_col, a_limb, a_offset, b, b_col, b_offset, len,
            )
        }

        fn vec_znx_normalize_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_normalize_tmp_bytes_backend_default(module)
        }

        fn vec_znx_normalize_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_base2k: usize,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_normalize_backend_default(
                module,
                res,
                res_base2k,
                res_offset,
                res_col,
                a,
                a_base2k,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_normalize_assign_backend<'s, 'r>(
            module: &Module<Self>,
            base2k: usize,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_normalize_assign_backend_default(module, base2k, a, a_col, &mut scratch);
        }

        fn vec_znx_normalize_coeff_assign_backend<'s, 'r>(
            module: &Module<Self>,
            base2k: usize,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            a_col: usize,
            a_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_normalize_coeff_assign_backend_default(
                module,
                base2k,
                a,
                a_col,
                a_coeff,
                &mut scratch,
            );
        }

        fn vec_znx_normalize_coeff_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_base2k: usize,
            a_col: usize,
            a_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_normalize_coeff_backend_default(
                module,
                res,
                res_base2k,
                res_offset,
                res_col,
                a,
                a_base2k,
                a_col,
                a_coeff,
                &mut scratch,
            );
        }

        fn vec_znx_add_into_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            b_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_add_into_backend_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_add_assign_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_add_assign_backend_default(module, res, res_col, a, a_col)
        }

        #[allow(clippy::too_many_arguments)]
        fn vec_znx_add_const_into_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
            cnst: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            cnst_col: usize,
            cnst_coeff: usize,
            res_limb: usize,
            res_coeff: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_add_const_into_backend_default(
                module, res, res_col, a, a_col, cnst, cnst_col, cnst_coeff, res_limb, res_coeff,
            )
        }

        fn vec_znx_add_const_assign_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            cnst: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            cnst_col: usize,
            cnst_coeff: usize,
            res_limb: usize,
            res_coeff: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_add_const_assign_backend_default(
                module, res, res_col, cnst, cnst_col, cnst_coeff, res_limb, res_coeff,
            )
        }

        fn vec_znx_extract_coeff_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
            a_coeff: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_extract_coeff_backend_default(module, res, res_col, a, a_col, a_coeff)
        }

        fn vec_znx_add_scalar_into_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'a, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            b_col: usize,
            b_limb: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_add_scalar_into_backend_default(
                module, res, res_col, a, a_col, b, b_col, b_limb,
            )
        }

        fn vec_znx_add_scalar_assign_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            res_limb: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_add_scalar_assign_backend_default(module, res, res_col, res_limb, a, a_col)
        }

        fn vec_znx_sub_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            b_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_sub_backend_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_sub_assign_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_sub_assign_backend_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_sub_negate_assign_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_sub_negate_assign_backend_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_sub_scalar_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'a, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            b_col: usize,
            b_limb: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_sub_scalar_backend_default(module, res, res_col, a, a_col, b, b_col, b_limb)
        }

        fn vec_znx_sub_scalar_assign_backend<'r, 'a>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            res_limb: usize,
            a: &poulpy_hal::layouts::ScalarZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_sub_scalar_assign_backend_default(module, res, res_col, res_limb, a, a_col)
        }

        fn vec_znx_negate_backend(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_negate_backend_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_negate_assign_backend(module: &Module<Self>, a: &mut VecZnxBackendMut<'_, Self>, a_col: usize) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_negate_assign_backend_default(module, a, a_col)
        }

        fn vec_znx_rsh_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_tmp_bytes_backend_default(module)
        }

        fn vec_znx_rsh_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_rsh_coeff_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            a_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_coeff_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                a_coeff,
                &mut scratch,
            );
        }

        fn vec_znx_rsh_add_into_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_add_into_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_rsh_add_coeff_into_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            a_coeff: usize,
            res_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_add_coeff_into_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                a_coeff,
                res_coeff,
                &mut scratch,
            );
        }

        fn vec_znx_rsh_sub_coeff_into_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            a_coeff: usize,
            res_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_sub_coeff_into_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                a_coeff,
                res_coeff,
                &mut scratch,
            );
        }

        fn vec_znx_lsh_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_lsh_tmp_bytes_backend_default(module)
        }

        fn vec_znx_lsh_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_lsh_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_lsh_coeff_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            a_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_lsh_coeff_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                a_coeff,
                &mut scratch,
            );
        }

        fn vec_znx_lsh_add_into_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_lsh_add_into_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_lsh_add_coeff_into_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            a_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_lsh_add_coeff_into_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                a_coeff,
                &mut scratch,
            );
        }

        fn vec_znx_lsh_sub_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_lsh_sub_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_rsh_sub_backend<'s, 'r, 'a>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            res: &mut VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'a, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_sub_backend_default(
                module,
                base2k,
                k,
                res,
                res_col,
                a,
                a_col,
                &mut scratch,
            );
        }

        fn vec_znx_rsh_assign_backend<'s, 'r>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rsh_assign_backend_default(module, base2k, k, a, a_col, &mut scratch);
        }

        fn vec_znx_lsh_assign_backend<'s, 'r>(
            module: &Module<Self>,
            base2k: usize,
            k: usize,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_lsh_assign_backend_default(module, base2k, k, a, a_col, &mut scratch);
        }

        fn vec_znx_rotate_backend<'r, 'a>(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_rotate_backend_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_rotate_assign_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_rotate_assign_tmp_bytes_backend_default(module)
        }

        fn vec_znx_rotate_assign_backend<'s, 'r>(
            module: &Module<Self>,
            k: i64,
            a: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_rotate_assign_backend_default(module, k, a, a_col, &mut scratch);
        }

        fn vec_znx_automorphism_backend<'r, 'a>(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'a, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_automorphism_backend_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_automorphism_assign_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_automorphism_assign_tmp_bytes_backend_default(module)
        }

        fn vec_znx_automorphism_assign_backend<'s, 'r>(
            module: &Module<Self>,
            k: i64,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'r, Self>,
            res_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_automorphism_assign_backend_default(module, k, res, res_col, &mut scratch);
        }

        fn vec_znx_mul_xp_minus_one_backend(
            module: &Module<Self>,
            k: i64,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_mul_xp_minus_one_backend_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_mul_xp_minus_one_assign_tmp_bytes_backend_default(module)
        }

        fn vec_znx_mul_xp_minus_one_assign_backend<'s>(
            module: &Module<Self>,
            k: i64,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_mul_xp_minus_one_assign_backend_default(
                module,
                k,
                res,
                res_col,
                &mut scratch,
            );
        }

        fn vec_znx_split_ring_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_split_ring_tmp_bytes_backend_default(module)
        }

        fn vec_znx_split_ring_backend<'s>(
            module: &Module<Self>,
            res: &mut [VecZnxBackendMut<'_, Self>],
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_split_ring_backend_default(module, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_merge_rings_tmp_bytes_backend(module: &Module<Self>) -> usize {
            <Self as HalVecZnxDefault<Self>>::vec_znx_merge_rings_tmp_bytes_backend_default(module)
        }

        fn vec_znx_merge_rings_backend<'s>(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &[VecZnxBackendRef<'_, Self>],
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as HalVecZnxDefault<Self>>::vec_znx_merge_rings_backend_default(module, res, res_col, a, a_col, &mut scratch);
        }

        fn vec_znx_switch_ring_backend(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_switch_ring_backend_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_copy_backend(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_copy_backend_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_copy_range_backend(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            res_limb: usize,
            res_offset: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
            a_limb: usize,
            a_offset: usize,
            len: usize,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_copy_range_backend_default(
                module, res, res_col, res_limb, res_offset, a, a_col, a_limb, a_offset, len,
            )
        }

        fn vec_znx_fill_uniform_backend(
            module: &Module<Self>,
            base2k: usize,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_fill_uniform_backend_default(module, base2k, res, res_col, seed)
        }

        fn vec_znx_fill_normal_backend(
            module: &Module<Self>,
            res_base2k: usize,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            noise_infos: NoiseInfos,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_fill_normal_backend_default(
                module,
                res_base2k,
                res,
                res_col,
                noise_infos,
                seed,
            )
        }

        fn vec_znx_add_normal_backend(
            module: &Module<Self>,
            res_base2k: usize,
            res: &mut VecZnxBackendMut<'_, Self>,
            res_col: usize,
            noise_infos: NoiseInfos,
            seed: [u8; 32],
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_add_normal_backend_default(
                module,
                res_base2k,
                res,
                res_col,
                noise_infos,
                seed,
            )
        }
    };
}
