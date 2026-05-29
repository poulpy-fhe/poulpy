#[macro_export]
macro_rules! hal_impl_vec_znx_matmul {
    ($defaults:ident) => {
        fn vec_znx_matmul_tmp_bytes(
            module: &Module<Self>,
            rows_in: usize,
            rows_out: usize,
            cols: usize,
            res_size: usize,
            u_size: usize,
            a_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::vec_znx_matmul_tmp_bytes_default(module, rows_in, rows_out, cols, res_size, u_size, a_size)
        }

        fn vec_znx_matmul(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            res_base2k: usize,
            u: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            u_base2k: usize,
            u_bound_bits: u32,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
            cols: usize,
            a_base2k: usize,
            rows_in: usize,
            rows_out: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::vec_znx_matmul_default(
                module,
                res,
                res_col,
                res_base2k,
                u,
                u_base2k,
                u_bound_bits,
                a,
                a_col,
                cols,
                a_base2k,
                rows_in,
                rows_out,
                &mut scratch,
            );
        }

        fn coeff_gemm_panel_wp(u_bound_bits: u32) -> (u32, usize) {
            $crate::hal_defaults::vec_znx_matmul::coeff_gemm_panel_wp_default::<Self>(u_bound_bits)
        }

        fn coeff_gemm_prepare(
            module: &Module<Self>,
            panel: &mut poulpy_hal::layouts::CoeffGemmPanelBackendMut<'_, Self>,
            u: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
        ) {
            $crate::hal_defaults::vec_znx_matmul::coeff_gemm_prepare_default::<Self>(module, panel, u)
        }

        #[allow(clippy::too_many_arguments)]
        fn vec_znx_matmul_prepared(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
            res_col: usize,
            res_base2k: usize,
            panel: &poulpy_hal::layouts::CoeffGemmPanelBackendRef<'_, Self>,
            u_base2k: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
            cols: usize,
            a_base2k: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            $crate::hal_defaults::vec_znx_matmul::vec_znx_matmul_prepared_default::<Self>(
                module,
                res,
                res_col,
                res_base2k,
                panel,
                u_base2k,
                a,
                a_col,
                cols,
                a_base2k,
                &mut scratch,
            );
        }
    };
}
