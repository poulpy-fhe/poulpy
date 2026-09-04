/// HAL `VecZnxBig` methods other than full-width normalization.
#[macro_export]
macro_rules! hal_impl_vec_znx_big_without_normalize {
    ($defaults:ident) => {
        fn vec_znx_big_from_small_backend(
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_from_small_default(&mut res, res_col, a, a_col)
        }

        fn vec_znx_big_add_normal_backend(
            module: &Module<Self>,
            res_base2k: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            noise_infos: NoiseInfos,
            seed: [u8; 32],
        ) {
            <Self as $defaults<Self>>::vec_znx_big_add_normal_seed_default(
                module,
                res_base2k,
                &mut res,
                res_col,
                noise_infos,
                seed,
            )
        }

        fn vec_znx_big_add_into(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_add_into_default(module, &mut res, res_col, &a, a_col, &b, b_col)
        }

        fn vec_znx_big_add_assign(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_add_assign_default(module, &mut res, res_col, &a, a_col)
        }

        fn vec_znx_big_add_small_into_backend(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_add_small_into_default(module, &mut res, res_col, &a, a_col, b, b_col)
        }

        fn vec_znx_big_add_small_assign(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_add_small_assign_default(module, &mut res, res_col, a, a_col)
        }

        fn vec_znx_big_sub(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_sub_default(module, &mut res, res_col, &a, a_col, &b, b_col)
        }

        fn vec_znx_big_sub_assign(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_sub_assign_default(module, &mut res, res_col, &a, a_col)
        }

        fn vec_znx_big_sub_negate_assign(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_sub_negate_assign_default(module, &mut res, res_col, &a, a_col)
        }

        fn vec_znx_big_sub_small_a_backend(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_sub_small_a_default(module, &mut res, res_col, a, a_col, &b, b_col)
        }

        fn vec_znx_big_sub_small_assign(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_sub_small_assign_default(module, &mut res, res_col, a, a_col)
        }

        fn vec_znx_big_sub_small_b_backend(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_sub_small_b_default(module, &mut res, res_col, &a, a_col, b, b_col)
        }

        fn vec_znx_big_sub_small_negate_assign(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_sub_small_negate_assign_default(module, &mut res, res_col, a, a_col)
        }

        fn vec_znx_big_inner_sum_backend(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            res_coeff: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_inner_sum_default(module, &mut res, res_col, res_coeff, &a, a_col)
        }

        fn vec_znx_scalar_product(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            a_col: usize,
            b: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_scalar_product_default(module, &mut res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_col_weighted_sum(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            weights: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, Self>,
            weights_col: usize,
            cols: usize,
            coeffs: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_col_weighted_sum_default(
                module,
                &mut res,
                res_col,
                a,
                weights,
                weights_col,
                cols,
                coeffs,
            )
        }

        fn vec_znx_big_negate(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_negate_default(module, &mut res, res_col, &a, a_col)
        }

        fn vec_znx_big_negate_assign(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_negate_assign_default(module, &mut res, res_col)
        }

        fn vec_znx_big_normalize_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as $defaults<Self>>::vec_znx_big_normalize_tmp_bytes_default(module)
        }

        fn vec_znx_big_automorphism(
            module: &Module<Self>,
            k: i64,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as $defaults<Self>>::vec_znx_big_automorphism_default(module, k, &mut res, res_col, &a, a_col)
        }

        fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as $defaults<Self>>::vec_znx_big_automorphism_assign_tmp_bytes_default(module)
        }

        fn vec_znx_big_automorphism_assign(
            module: &Module<Self>,
            k: i64,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vec_znx_big_automorphism_assign_default(module, k, &mut res, res_col, &mut scratch);
        }
    };
}

/// Full-width HAL `VecZnxBig` normalization.
#[macro_export]
macro_rules! hal_impl_vec_znx_big_normalize {
    ($defaults:ident) => {
        fn vec_znx_big_normalize(
            module: &Module<Self>,
            mut res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self, impl poulpy_hal::layouts::ArithmeticState>,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
            a_base2k: usize,
            a_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vec_znx_big_normalize_default(
                module,
                &mut res,
                res_base2k,
                res_offset,
                res_col,
                &a,
                a_base2k,
                a_col,
                &mut scratch,
            );
        }
    };
}

#[macro_export]
macro_rules! hal_impl_vec_znx_big {
    ($defaults:ident) => {
        $crate::hal_impl_vec_znx_big_without_normalize!($defaults);
        $crate::hal_impl_vec_znx_big_normalize!($defaults);
    };
}
