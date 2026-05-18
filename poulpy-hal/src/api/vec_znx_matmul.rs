use crate::layouts::{Backend, ScratchArena, VecZnxBackendMut, VecZnxBackendRef};

#[allow(clippy::too_many_arguments)]
pub trait VecZnxMatMulTmpBytes {
    fn vec_znx_matmul_tmp_bytes(
        &self,
        rows_in: usize,
        rows_out: usize,
        cols: usize,
        res_size: usize,
        u_size: usize,
        a_size: usize,
    ) -> usize;
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxMatMul<B: Backend> {
    /// Computes a range of packed coefficient-matrix product columns.
    ///
    /// `u` stores `U[out, in]` as `u[out][in]`, while `a[a_col][in]` stores
    /// the first input vector. For each `j < cols`, the result is written as
    /// `res[res_col + j][out] = sum_in U[out, in] * a[a_col + j][in]`.
    fn vec_znx_matmul(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_base2k: usize,
        u: &VecZnxBackendRef<'_, B>,
        u_base2k: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        cols: usize,
        a_base2k: usize,
        rows_in: usize,
        rows_out: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}
