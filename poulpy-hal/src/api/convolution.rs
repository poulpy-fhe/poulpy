use crate::layouts::{
    Backend, CnvDftAccTerm, CnvDftStore, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLOwned, CnvPVecRBackendMut,
    CnvPVecRBackendRef, CnvPVecROwned, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
};

/// Allocates prepared convolution operands ([`CnvPVecL`](crate::layouts::CnvPVecL), [`CnvPVecR`](crate::layouts::CnvPVecR)).
pub trait CnvPVecAlloc<BE: Backend> {
    fn cnv_pvec_left_alloc(&self, cols: usize, size: usize) -> CnvPVecLOwned<BE>;
    fn cnv_pvec_right_alloc(&self, cols: usize, size: usize) -> CnvPVecROwned<BE>;
}

/// Returns the byte sizes for prepared convolution operands.
pub trait CnvPVecBytesOf {
    fn bytes_of_cnv_pvec_left(&self, cols: usize, size: usize) -> usize;
    fn bytes_of_cnv_pvec_right(&self, cols: usize, size: usize) -> usize;
}

/// Bivariate convolution over `Z[X, Y] mod (X^N + 1)` where `Y = 2^{-K}`.
///
/// Provides methods to prepare left/right operands and apply the convolution.
/// See method-level documentation for the mathematical formulation.
pub trait Convolution<BE: Backend> {
    /// Returns scratch bytes required for [`cnv_prepare_left`](Convolution::cnv_prepare_left).
    fn cnv_prepare_left_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the left
    /// operand of a bivariate convolution.
    fn cnv_prepare_left(
        &self,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_right`](Convolution::cnv_prepare_right).
    fn cnv_prepare_right_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    /// Prepares a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) as the right
    /// operand of a bivariate convolution.
    fn cnv_prepare_right(
        &self,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_apply_dft`](Convolution::cnv_apply_dft).
    fn cnv_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Returns scratch bytes required for [`cnv_by_const_apply`](Convolution::cnv_by_const_apply).
    fn cnv_by_const_apply_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Evaluates a bivariate convolution over Z\[X, Y\] (x) Z\[Y\] mod (X^N + 1) where Y = 2^-K over the
    /// selected columns and stores the result on the selected column, scaled by 2^{cnv_offset * Base2K}
    ///
    /// Behavior is identical to [Convolution::cnv_apply_dft] with `b` treated as a constant polynomial
    /// in the X variable, for example:
    ///```text
    ///       1    X   X^2  X^3
    /// a = 1 [a00, a10, a20, a30] = (a00 + a01 * 2^-K) + (a10 + a11 * 2^-K) * X ...
    ///     Y [a01, a11, a21, a31]
    ///
    /// b = 1 [b0] = (b00 + b01 * 2^-K)
    ///     Y [b0]
    /// ```
    /// This method is intended to be used for multiplications by constants that are greater than the base2k.
    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    /// Evaluates a bivariate convolution over Z\[X, Y\] (x) Z\[X, Y\] mod (X^N + 1) where Y = 2^-K over the
    /// selected columns and stores the result on the selected column, scaled by 2^{cnv_offset * Base2K}
    ///
    /// # Example
    ///```text
    ///       1    X   X^2  X^3
    /// a = 1 [a00, a10, a20, a30] = (a00 + a01 * 2^-K) + (a10 + a11 * 2^-K) * X ...
    ///     Y [a01, a11, a21, a31]
    ///
    /// b = 1 [b00, b10, b20, b30] = (b00 + b01 * 2^-K) + (b10 + b11 * 2^-K) * X ...
    ///     Y [b01, b11, b21, b31]
    ///
    /// If cnv_offset = 0:
    ///
    ///            1    X   X^2  X^3
    /// res = 1  [r00, r10, r20, r30] = (r00 + r01 * 2^-K + r02 * 2^-2K + r03 * 2^-3K) + ... * X + ...
    ///       Y  [r01, r11, r21, r31]
    ///       Y^2[r02, r12, r22, r32]
    ///       Y^3[r03, r13, r23, r33]
    ///
    /// If cnv_offset = 1:
    ///
    ///            1    X   X^2  X^3
    /// res = 1  [r01, r11, r21, r31]  = (r01 + r02 * 2^-K + r03 * 2^-2K) + ... * X + ...
    ///       Y  [r02, r12, r22, r32]
    ///       Y^2[r03, r13, r23, r33]
    ///       Y^3[  0,   0,   0 ,  0]
    /// ```
    /// If res.size() < a.size() + b.size() + k, result is truncated accordingly in the Y dimension.
    fn cnv_apply_dft(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Lazy-canonicalization convolution used by `glwe_mul_plain`; bit-identical to
    /// the eager `cnv_prepare_left/right` + `cnv_apply_dft`.
    fn cnv_prepare_left_lazy_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    fn cnv_prepare_left_lazy(
        &self,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );
    fn cnv_prepare_right_lazy_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
    fn cnv_prepare_right_lazy(
        &self,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );
    fn cnv_apply_dft_lazy_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_lazy(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Accumulating variant of [`cnv_apply_dft`](Convolution::cnv_apply_dft):
    /// `res[res_col] += a[a_col] (x) b[b_col]`, bit-identical to `cnv_apply_dft`
    /// followed by a DFT-domain add. Limbs `>= min(res.size(), a.size() + b.size())`
    /// are left untouched. Scratch requirement is
    /// [`cnv_apply_dft_tmp_bytes`](Convolution::cnv_apply_dft_tmp_bytes).
    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_accumulate(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_accumulate_dft`](Convolution::cnv_accumulate_dft)
    /// and [`cnv_accumulate_dft_columns`](Convolution::cnv_accumulate_dft_columns).
    ///
    /// `a_size` and `b_size` are upper bounds over the sizes of the term operands.
    /// The budget also covers the per-term `cnv_apply_dft{,_accumulate}` fallback
    /// those methods may take, so one number sizes the whole family.
    fn cnv_accumulate_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Evaluates a sum of bivariate convolutions: `res[res_col] = Σ_t a_t ⊛ b_t`,
    /// scaled by `2^{cnv_offset * Base2K}`, overwriting `res[res_col]`.
    ///
    /// Each term behaves like one [`Convolution::cnv_apply_dft`] call over the
    /// selected columns and the per-term results are summed; with an empty
    /// `terms` slice the output column is zeroed. Backends may fuse the
    /// accumulation (one lazy reduction per output limb, destination written
    /// once), so the result is congruent to — but not necessarily bit-identical
    /// with — a sequence of [`Convolution::cnv_apply_dft_accumulate`] calls.
    fn cnv_accumulate_dft<'a>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        terms: &[CnvDftAccTerm<'a, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: 'a;

    /// Multi-column variant of [`cnv_accumulate_dft`](Convolution::cnv_accumulate_dft):
    /// for every output column `c < cols`,
    /// `res[res_col + c] (=|+=) Σ_t a_t[a_col + c] ⊛ b_t[b_col]`.
    ///
    /// The right operand of each term is broadcast across the output columns
    /// (one GLWE mask/body sweep against one diagonal), so a whole BSGS giant
    /// step is one call. `store` selects whether the destination columns are
    /// overwritten or accumulated into; `Overwrite` also zeroes the limbs past
    /// the convolution bound, `Accumulate` leaves them untouched. Scratch
    /// requirement is [`cnv_accumulate_dft_tmp_bytes`](Convolution::cnv_accumulate_dft_tmp_bytes).
    #[allow(clippy::too_many_arguments)]
    fn cnv_accumulate_dft_columns<'a>(
        &self,
        cnv_offset: usize,
        store: CnvDftStore,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        cols: usize,
        terms: &[CnvDftAccTerm<'a, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: 'a;

    /// Batched variant of [`cnv_accumulate_dft_columns`](Convolution::cnv_accumulate_dft_columns):
    /// evaluates `results.len()` independent multi-column accumulations, one per
    /// term set, so a backend can share a prepared left operand appearing in
    /// several sets across their launches.
    ///
    /// `results[g][res_col + c] (=|+=) Σ_t a_t[a_col + c] ⊛ b_t[b_col]` over
    /// `term_sets[g]`, for every `g` and every `c < cols`. `results.len()` must
    /// equal `term_sets.len()`; the sets are independent and may differ in
    /// length, order, and left operands. `store` applies to each result
    /// independently, exactly as in
    /// [`cnv_accumulate_dft_columns`](Convolution::cnv_accumulate_dft_columns):
    /// an empty set zeroes its result under `Overwrite` and is a no-op under
    /// `Accumulate`. An empty batch is a no-op, a one-result batch is one
    /// ordinary call. Results must be pairwise non-overlapping; their sizes may
    /// differ. Scratch requirement is
    /// [`cnv_accumulate_dft_tmp_bytes`](Convolution::cnv_accumulate_dft_tmp_bytes)
    /// taken over the batch maxima.
    #[allow(clippy::too_many_arguments)]
    fn cnv_accumulate_dft_columns_batch<'a>(
        &self,
        cnv_offset: usize,
        store: CnvDftStore,
        results: &mut [VecZnxDftBackendMut<'_, BE>],
        res_col: usize,
        cols: usize,
        term_sets: &[&[CnvDftAccTerm<'a, BE>]],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: 'a;

    /// Returns scratch bytes required for [`cnv_pairwise_apply_dft`](Convolution::cnv_pairwise_apply_dft).
    fn cnv_pairwise_apply_dft_tmp_bytes(&self, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    /// Evaluates the bivariate pair-wise convolution res = (a\[i\] + a\[j\]) * (b\[i\] + b\[j\]).
    /// If i == j then calls [Convolution::cnv_apply_dft], i.e. res = a\[i\] * b\[i\].
    /// See [Convolution::cnv_apply_dft] for information about the bivariate convolution.
    fn cnv_pairwise_apply_dft(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`cnv_prepare_self`](Convolution::cnv_prepare_self).
    fn cnv_prepare_self_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;

    /// Prepares both left and right convolution operands from the same input polynomial,
    /// sharing the FFT/NTT computation. This is an optimization for self-convolution
    /// (squaring) where both operands are the same polynomial.
    fn cnv_prepare_self(
        &self,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
