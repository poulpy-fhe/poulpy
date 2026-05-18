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
                a,
                a_col,
                cols,
                a_base2k,
                rows_in,
                rows_out,
                &mut scratch,
            );
        }
    };
}
