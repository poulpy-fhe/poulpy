use crate::layouts::{
    Backend, CoeffGemmPanelBackendMut, CoeffGemmPanelBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
};

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
        // Compile-time entry bound of `U` in bits (8/16/32/64); selects the kernel width.
        u_bound_bits: u32,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        cols: usize,
        a_base2k: usize,
        rows_in: usize,
        rows_out: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Prepare a `U` matrix into a reusable [`crate::layouts::CoeffGemmPanel`] so the
/// kernel-specific decomposition is computed once and amortized across many
/// [`VecZnxMatMulPrepared`] calls.
pub trait CoeffGemmPrepare<B: Backend> {
    /// `(w, up)` — the SIMD piece width and `U`-piece count this backend uses
    /// for an entry bound of `u_bound_bits`. Use it to size the panel via
    /// [`crate::layouts::CoeffGemmPanel::alloc`].
    fn coeff_gemm_panel_wp(&self, u_bound_bits: u32) -> (u32, usize);

    /// Fills `panel` with the prepared decomposition of `u` (shape already set
    /// by alloc with the `(w, up)` from [`Self::coeff_gemm_panel_wp`]).
    fn coeff_gemm_prepare(&self, panel: &mut CoeffGemmPanelBackendMut<'_, B>, u: &VecZnxBackendRef<'_, B>);
}

/// Applies a prepared [`crate::layouts::CoeffGemmPanel`] to `cols` RHS columns
/// of `a`: `res[res_col + j][out] = sum_in U[out,in] * a[a_col + j][in]`.
#[allow(clippy::too_many_arguments)]
pub trait VecZnxMatMulPrepared<B: Backend> {
    fn vec_znx_matmul_prepared(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_base2k: usize,
        panel: &CoeffGemmPanelBackendRef<'_, B>,
        u_base2k: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        cols: usize,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}
