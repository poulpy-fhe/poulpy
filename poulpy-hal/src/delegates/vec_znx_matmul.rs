use crate::{
    api::{VecZnxMatMul, VecZnxMatMulTmpBytes},
    layouts::{Backend, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef},
    oep::HalVecZnxMatMulImpl,
};

macro_rules! impl_vec_znx_matmul_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend + HalVecZnxMatMulImpl<B>,
        {
            $($body)+
        }
    };
}

impl_vec_znx_matmul_delegate!(
    VecZnxMatMulTmpBytes,
    fn vec_znx_matmul_tmp_bytes(
        &self,
        rows_in: usize,
        rows_out: usize,
        cols: usize,
        res_size: usize,
        u_size: usize,
        a_size: usize,
    ) -> usize {
        B::vec_znx_matmul_tmp_bytes(self, rows_in, rows_out, cols, res_size, u_size, a_size)
    }
);

impl_vec_znx_matmul_delegate!(
    VecZnxMatMul<B>,
    fn vec_znx_matmul(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_base2k: usize,
        u: &VecZnxBackendRef<'_, B>,
        u_base2k: usize,
        u_bound_bits: u32,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        cols: usize,
        a_base2k: usize,
        rows_in: usize,
        rows_out: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vec_znx_matmul(
            self,
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
            scratch,
        )
    }
);
