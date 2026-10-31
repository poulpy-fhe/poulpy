use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, Data, ScratchArena, VecZnxDftBackendMut};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
};

/// Sum over a slice of ciphertexts.
pub trait CKKSAddManyOps<BE: Backend> {
    /// Scratch of [`Self::ckks_add_many`] from terms of layout `a` into `res`: the additions, the normalizations and,
    /// for one term, the copy.
    fn ckks_add_many_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds;

    /// Computes `res = a[0] + a[1] + … + a[n-1]`, adding the terms to `res`
    /// from left to right, and normalizing `res` in place, by poulpy-core's
    /// `glwe_normalize_assign`, after every `normalize_after` additions that
    /// another addition follows.
    ///
    /// The sum is lazy, so `res` leaves unnormalized unless it holds one
    /// term. `normalize_after` bounds the canonical digits a coefficient of
    /// `res` accumulates to `normalize_after + 1` times the most a term
    /// holds, one for a canonical term, which poulpy-core's `max_lazy_terms`
    /// sizes for independent terms; `usize::MAX` never normalizes.
    ///
    /// # Metadata
    ///
    /// Follows from repeated application of `ckks_add_into`:
    ///
    /// ```text
    /// log_delta_out  = min over all terms of log_delta
    /// log_budget_out = min over all terms of log_budget
    /// ```
    ///
    /// No capacity is consumed.  The destination-capacity offset from the
    /// final write applies:
    ///
    /// ```text
    /// offset         = max(0, result_k − res.k())
    /// log_budget_out = (min log_budget) − offset
    /// ```
    ///
    /// # Errors
    ///
    /// Before any write: `InvalidOperand` if `normalize_after` is 0, `a` is empty or a term differs from `res` in
    /// degree, radix or rank, then `CKKSError::Internal` with less scratch than [`Self::ckks_add_many_tmp_bytes`]
    /// reports for `res` and `a[0]`, then the errors of
    /// [`CKKSAddOps::ckks_add_into`](crate::api::CKKSAddOps::ckks_add_into) on the first two terms or of
    /// [`CKKSCopyOps::ckks_copy`](crate::api::CKKSCopyOps::ckks_copy) on a single one.
    fn ckks_add_many<R, A>(
        &self,
        normalize_after: usize,
        res: &mut R,
        a: &[&A],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds;
}

/// Fused multiply-accumulate: `dst += a * b`.
///
/// Each variant accepts any CKKS-compatible backend layout and uses caller
/// scratch for the intermediate product instead of requiring a caller-visible
/// ciphertext allocation.
pub trait CKKSMulAddOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_mul_add_ct_into`] with result `res` and
    /// multiplication operands `a`, `b` (the internal product is carved at the
    /// widest of the three).
    fn ckks_mul_add_ct_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_mul_add_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Computes `dst += a * b` using tensor-product keyswitching via `tsk`.
    ///
    /// # Metadata
    ///
    /// First the product `a * b` is computed (see [`CKKSMulOps`](crate::api::CKKSMulOps)
    /// for its metadata rule), then that product is added into `dst`.
    /// The final metadata is the result of the addition:
    ///
    /// ```text
    /// // product metadata (capacity consumed by the mul):
    /// prod_delta  = min(a.log_delta, b.log_delta)
    /// prod_budget = min(a.log_budget, b.log_budget) − max(a.log_delta, b.log_delta)
    ///
    /// // addition with dst:
    /// log_delta_out  = min(dst.log_delta, prod_delta)
    /// log_budget_out = min(dst.log_budget, prod_budget)
    ///                  − max(0, min(dst.k(), prod_k) − dst.k())
    /// ```
    fn ckks_mul_add_ct_into<Dst, A, B, H>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Computes `dst += a * pt` where `pt` is a full plaintext polynomial.
    ///
    /// # Metadata
    ///
    /// ```text
    /// // product metadata:
    /// prod_delta  = a.log_delta
    /// prod_budget = a.log_budget − pt.log_delta
    ///
    /// // addition with dst:
    /// log_delta_out  = min(dst.log_delta, prod_delta)
    /// log_budget_out = min(dst.log_budget, prod_budget)
    ///                  − max(0, min(dst.k(), prod_k) − dst.k())
    /// ```
    fn ckks_mul_add_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;

    /// Computes `dst += a * pt[pt_coeff]`.
    ///
    /// Metadata follows the same rule as [`Self::ckks_mul_add_pt_vec_into`].
    fn ckks_mul_add_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;
}

/// Fused affine evaluation: `dst = a * scale + offset`.
///
/// Combines a plaintext multiplication followed by a plaintext addition.
/// Both `scale` and `offset` must be CKKS plaintexts compatible with the
/// ciphertext's remaining capacity.
pub trait CKKSAffineOps<BE: Backend> {
    fn ckks_affine_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, affine_const: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Computes `dst = a * affine_const[scale_coeff] + affine_const[offset_coeff]`.
    ///
    /// Reads two quantized scalar constants from `affine_const`: one for the
    /// multiplicative scale (at `scale_coeff`) and one for the additive offset
    /// (at `offset_coeff`).  The offset is always applied to ZNX coefficient 0
    /// of `dst` (the real-slot constant term).  Use
    /// [`CKKSAddOps::ckks_add_pt_const_into`](crate::api::CKKSAddOps::ckks_add_pt_const_into)
    /// directly if you need to target a different destination coefficient.
    ///
    /// # Metadata
    ///
    /// The operation is `mul` then `add`.  The `add` step does not consume
    /// capacity, so the net cost is from the `mul` step only:
    ///
    /// ```text
    /// natural_eff_k  = a.k() − affine_const.log_delta
    /// offset         = max(0, natural_eff_k − dst.k())
    ///
    /// log_delta_out  = a.log_delta
    /// log_budget_out = a.log_budget − affine_const.log_delta − offset
    /// ```
    ///
    /// **Net capacity consumed**: `affine_const.log_delta + offset` bits.
    fn ckks_affine_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        affine_const: &P,
        offset_coeff: usize,
        scale_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;

    /// Computes `dst = dst * affine_const[scale_coeff] + affine_const[offset_coeff]` in-place.
    ///
    /// For an in-place operation `dst` acts as both source and destination.
    /// No destination-capacity offset applies:
    ///
    /// ```text
    /// log_delta_out  = dst.log_delta
    /// log_budget_out = dst.log_budget − affine_const.log_delta
    /// ```
    fn ckks_affine_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        affine_const: &P,
        offset_coeff: usize,
        scale_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;

    fn ckks_affine_pt_vec_tmp_bytes<R, A, S>(&self, res: &R, a: &A, scale: &S) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        S: CKKSInfos;

    /// Computes `dst = a * scale + offset` where `scale` and `offset` are full
    /// plaintext polynomials in the ZNX domain.
    ///
    /// # Metadata
    ///
    /// The `add offset` step does not consume capacity; the cost comes from the
    /// `mul scale` step:
    ///
    /// ```text
    /// natural_eff_k  = a.k() − scale.log_delta
    /// offset_bits    = max(0, natural_eff_k − dst.k())
    ///
    /// log_delta_out  = a.log_delta
    /// log_budget_out = a.log_budget − scale.log_delta − offset_bits
    /// ```
    ///
    /// **Net capacity consumed**: `scale.log_delta + offset_bits` bits.
    ///
    /// **Precondition**: `(a.log_budget − scale.log_delta) + offset.log_delta >= offset.k()`
    /// (the offset plaintext must fit in the ciphertext headroom after the multiply).
    fn ckks_affine_pt_vec_into<Dst, A, S, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        scale: &S,
        offset: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;

    /// Computes `dst = dst * scale + offset` in-place.
    ///
    /// For an in-place operation no destination-capacity offset applies:
    ///
    /// ```text
    /// log_delta_out  = dst.log_delta
    /// log_budget_out = dst.log_budget − scale.log_delta
    /// ```
    fn ckks_affine_pt_vec_assign<Dst, S, P>(
        &self,
        dst: &mut Dst,
        scale: &S,
        offset: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;
}

/// Fused multiply-subtract: `dst -= a * b`.
pub trait CKKSMulSubOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_mul_sub_ct_into`] with result `res` and
    /// multiplication operands `a`, `b` (the internal product is carved at the
    /// widest of the three).
    fn ckks_mul_sub_ct_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_mul_sub_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Computes `dst -= a * b` using tensor-product keyswitching via `tsk`.
    ///
    /// Metadata follows the same rule as
    /// [`CKKSMulAddOps::ckks_mul_add_ct_into`] with `+=` replaced by `-=`.
    fn ckks_mul_sub_ct_into<Dst, A, B, H>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Computes `dst -= a * pt` where `pt` is a full plaintext polynomial.
    ///
    /// Metadata follows the same rule as
    /// [`CKKSMulAddOps::ckks_mul_add_pt_vec_into`].
    fn ckks_mul_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;

    /// Computes `dst -= a * pt[pt_coeff]`.
    ///
    /// Metadata follows the same rule as
    /// [`CKKSMulAddOps::ckks_mul_add_pt_const_into`].
    fn ckks_mul_sub_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;
}

/// Inner product (dot product) of ciphertext and plaintext slices.
///
/// Computes the weighted sum `dst = Σ a[i] * b[i]` over all pairs.
pub trait CKKSDotProductOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_dot_product_ct`] over `n` input pairs;
    /// `a`/`b` describe the widest input pair (the internal products are carved
    /// at `max(a.k, b.k)`). They also cover [`Self::ckks_dot_product_prepared`]
    /// over `n` pairs prepared from such ciphertexts: pass the prepared
    /// operands, whose infos report them.
    fn ckks_dot_product_ct_tmp_bytes<R, A, B, T>(&self, n: usize, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_dot_product_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Computes `dst = Σ a[i] * b[i]` over ciphertext pairs, with the bounds of
    /// [`CKKSMulOps::ckks_mul_into`](crate::api::CKKSMulOps::ckks_mul_into).
    ///
    /// Two pairs or more, every `a[i]` of one layout and scale, every `b[i]` of
    /// one, and `dst.base2k() <= 62`: the products are summed and relinearized
    /// once, with the tensor key at `max(a[0].k(), b[0].k())`. Where
    /// `ckks_mul_into` takes its one-pass product, the sum is core's
    /// `glwe_mul_relinearize_sum`, which adds every pair in the transform domain
    /// in slice order: `base2k` must hold `a.len()` times the products of one
    /// pair (`Module::max_base2k`). This path returns its errors before writing
    /// `dst` and leaves it canonical. Other inputs are `ckks_mul_into` then
    /// `ckks_mul_add_ct_into` per pair, so a later pair can fail after `dst` was
    /// written.
    ///
    /// # Metadata
    ///
    /// Both paths stamp the `ckks_mul_into` rule, the smallest `log_sparsity`
    /// and the joined slot kind of all operands. With uniform-precision inputs:
    ///
    /// ```text
    /// natural_eff_k  = a[i].log_budget   (= a[i].k() − a[i].log_delta)
    /// offset         = max(0, natural_eff_k − dst.k())
    ///
    /// log_delta_out  = a[i].log_delta
    /// log_budget_out = a[i].log_budget − a[i].log_delta − offset
    /// ```
    fn ckks_dot_product_ct<Dst, A, B, H>(
        &self,
        dst: &mut Dst,
        a: &[&A],
        b: &[&B],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// [`Self::ckks_dot_product_ct`] of the ciphertexts `a[i]` and `b[i]` were
    /// prepared from by
    /// [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare_alloc`](crate::api::CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare_alloc)
    /// and
    /// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc`](crate::api::CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc),
    /// bit for bit, under either hint and for any number of pairs, with no
    /// operand transformed in the call. Its scratch is
    /// [`Self::ckks_dot_product_ct_tmp_bytes`] of the prepared operands.
    ///
    /// Returns, before writing `dst`, `PreparedOperandLayoutMismatch` for an
    /// operand of another degree, `base2k` or rank than `dst`, and
    /// `InvalidOperand` for a side whose operands do not share one layout and
    /// scale, or past `base2k` 62: prepared pairs have no eager path.
    fn ckks_dot_product_prepared<Dst, H>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSLeftOperandPrepared<BE>],
        b: &[&CKKSRightOperandPrepared<BE>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>;

    /// Scratch bytes of [`Self::ckks_dot_product_prepared_pt_vec`] for terms prepared like `a` and plaintexts prepared like
    /// `b`, whatever their number: pass the prepared operands, whose infos report their sources, or those sources.
    fn ckks_dot_product_prepared_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos;

    /// Computes `dst = Σ a[t] · σ_{galois[t]}(b[t])` over ciphertexts prepared by
    /// [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare_alloc`](crate::api::CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare_alloc)
    /// and plaintexts prepared by
    /// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc`](crate::api::CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc),
    /// under either hint, each plaintext read through its odd Galois element: `1` for the plaintext itself,
    /// [`CKKSModuleInfos::ckks_galois_element`](crate::CKKSModuleInfos::ckks_galois_element) of `r` for its slots
    /// rotated by `r`. It is core's `glwe_mul_prepared_plain_sum`: no operand is transformed in the call, so prepared
    /// terms serve every sum that shares them, and the products are added in the transform domain in slice order and
    /// normalized once. One term through `1` is
    /// [`CKKSMulOps::ckks_mul_pt_vec_into`](crate::api::CKKSMulOps::ckks_mul_pt_vec_into) of the ciphertext and the
    /// plaintext it was prepared from, bit for bit.
    ///
    /// # Metadata
    ///
    /// The rule of `ckks_mul_pt_vec_into`, which the terms, sharing one layout and scale, give alike, with the joined
    /// slot kind and the smallest `log_sparsity` of the pairs:
    ///
    /// ```text
    /// natural_eff_k  = a[t].k() − b[t].log_delta
    /// offset         = max(0, natural_eff_k − dst.k())
    ///
    /// log_delta_out  = a[t].log_delta
    /// log_budget_out = a[t].log_budget − b[t].log_delta − offset
    /// ```
    ///
    /// # Errors
    ///
    /// Before writing `dst`: `InvalidOperand` for `a` and `b` empty or of two lengths, or for `galois` of another
    /// length; as [`Self::ckks_dot_product_prepared`], `PreparedOperandLayoutMismatch` for a term of another degree,
    /// `base2k` or rank than `dst`; `InvalidOperand` for an even Galois element, a plaintext not of rank 0 at `dst`'s `base2k`, or a side whose operands do not share one
    /// layout and scale; `PlaintextDegreeMismatch` for a plaintext whose degree does not embed into `dst`'s; and the
    /// errors of the rule.
    fn ckks_dot_product_prepared_pt_vec<Dst>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSLeftOperandPrepared<BE>],
        b: &[&CKKSRightOperandPrepared<BE>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Scratch bytes of [`Self::ckks_dot_product_prepared_pt_vec_add`] into `res` for terms prepared like `a` and plaintexts
    /// prepared like `b`, whatever their number: pass the prepared operands, whose infos report their sources, or those
    /// sources.
    fn ckks_dot_product_prepared_pt_vec_add_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos;

    /// Computes `res += Σ a[t] · σ_{galois[t]}(b[t])`: the sum of [`Self::ckks_dot_product_prepared_pt_vec`], added to `res`
    /// before its one normalization (core's `glwe_mul_prepared_plain_sum_add`), so the total is rounded once, with no
    /// temporary.
    ///
    /// `res` holds the sum's frame: the scale of the terms and the width the rule of
    /// [`CKKSMulOps::ckks_mul_pt_vec_into`](crate::api::CKKSMulOps::ckks_mul_pt_vec_into) leaves into it, so the sum is
    /// written at `res`'s width and scale and `res`'s value enters exactly. A masked insertion into a ciphertext of
    /// diagonals reads one prepared mask through the Galois element of each shift, then re-prepares the updated diagonals
    /// with
    /// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare`](crate::api::CKKSRightOperandPreparedFactory::ckks_right_operand_prepare).
    ///
    /// # Metadata
    ///
    /// `res`'s width, scale and budget are kept, its slots joined with the pairs' and its sparsity the smallest of
    /// `res`'s and the pairs'; `res` is canonical, its noise tag cleared.
    ///
    /// # Errors
    ///
    /// Before writing `res`: those of [`Self::ckks_dot_product_prepared_pt_vec`], `InvalidOperand` when the sum's rule leaves
    /// another scale or budget than `res`'s, and `Internal` for a `scratch` below
    /// [`Self::ckks_dot_product_prepared_pt_vec_add_tmp_bytes`].
    fn ckks_dot_product_prepared_pt_vec_add<R>(
        &self,
        res: &mut R,
        a: &[&CKKSLeftOperandPrepared<BE>],
        b: &[&CKKSRightOperandPrepared<BE>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Scratch bytes of [`Self::ckks_dot_product_prepared_pt_vec_dft`] for terms prepared like `a` and plaintexts prepared
    /// like `b`, whatever their number.
    fn ckks_dot_product_prepared_pt_vec_dft_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos;

    /// [`Self::ckks_dot_product_prepared_pt_vec`] into `acc`, before its normalization: `acc` receives, in the transform
    /// domain, the sum poulpy-core's `glwe_mul_prepared_plain_sum_dft` forms at the offset of the rule of
    /// `ckks_mul_pt_vec_into` for a destination of `res`'s layout and metadata. `acc` holds the degree of `res`, its
    /// `rank + 1` columns and `a[0].size() + b[0].size()` limbs. Sums added by
    /// [`Self::ckks_dot_product_prepared_pt_vec_dft_add`] for the same `res`, over terms of one layout and scale and
    /// plaintexts of another, are finished by [`Self::ckks_dot_product_prepared_pt_vec_from_dft`]: that is
    /// `ckks_dot_product_prepared_pt_vec` of the terms of every call, bit for bit, metadata included when every pair has the
    /// slot kind and sparsity of the pair the finish reads. `acc` carries no metadata, and nothing checks the calls of one
    /// sum against each other: that is the caller's.
    ///
    /// # Errors
    ///
    /// Before writing `acc`: those of [`Self::ckks_dot_product_prepared_pt_vec`] with `res` as the destination, and
    /// `InvalidOperand` for `acc` of another degree, column count or limb count.
    fn ckks_dot_product_prepared_pt_vec_dft<R>(
        &self,
        acc: &mut VecZnxDftBackendMut<'_, BE>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<BE>],
        b: &[&CKKSRightOperandPrepared<BE>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: CKKSCtBounds;

    /// Scratch bytes of [`Self::ckks_dot_product_prepared_pt_vec_dft_add`] for terms prepared like `a` and plaintexts
    /// prepared like `b`, whatever their number.
    fn ckks_dot_product_prepared_pt_vec_dft_add_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos;

    /// `acc += Σ a[t] · σ_{galois[t]}(b[t])`, the sum of [`Self::ckks_dot_product_prepared_pt_vec_dft`] added to `acc`.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_dot_product_prepared_pt_vec_dft`].
    fn ckks_dot_product_prepared_pt_vec_dft_add<R>(
        &self,
        acc: &mut VecZnxDftBackendMut<'_, BE>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<BE>],
        b: &[&CKKSRightOperandPrepared<BE>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: CKKSCtBounds;

    /// Scratch bytes of [`Self::ckks_dot_product_prepared_pt_vec_from_dft`] into `res` from terms prepared like `a` and
    /// plaintexts prepared like `b`.
    fn ckks_dot_product_prepared_pt_vec_from_dft_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos;

    /// Finishes a sum of [`Self::ckks_dot_product_prepared_pt_vec_dft`] into `res`: stamps `res` with the metadata
    /// `ckks_dot_product_prepared_pt_vec` gives the pair `(a, b)`, a term and a plaintext of the sum, then normalizes each
    /// column of `acc` once into `res`, at `res.k()`, canonical. `acc` is unspecified afterwards.
    ///
    /// # Errors
    ///
    /// Before writing `res`: `PreparedOperandLayoutMismatch` for `a` of another degree, `base2k` or rank than `res`;
    /// `InvalidOperand` for `b` not a plaintext at `res`'s `base2k` or for `acc` of another degree, column count or limb
    /// count than [`Self::ckks_dot_product_prepared_pt_vec_dft`] takes; `PlaintextDegreeMismatch` for `b` of a degree that
    /// does not embed into `res`'s; and the errors of the rule.
    fn ckks_dot_product_prepared_pt_vec_from_dft<Dst>(
        &self,
        res: &mut Dst,
        acc: &mut VecZnxDftBackendMut<'_, BE>,
        a: &CKKSLeftOperandPrepared<BE>,
        b: &CKKSRightOperandPrepared<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Computes `dst = Σ a[i] * b[i]` over ciphertext–plaintext-polynomial pairs.
    ///
    /// # Metadata
    ///
    /// ```text
    /// natural_eff_k  = a[i].k() − b[i].log_delta
    /// offset         = max(0, natural_eff_k − dst.k())
    ///
    /// log_delta_out  = a[i].log_delta
    /// log_budget_out = a[i].log_budget − b[i].log_delta − offset
    /// ```
    fn ckks_dot_product_pt_vec<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord, BE::Ring>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos;

    /// Computes `dst = Σ a[i] * b[i][pt_coeffs[i]]` over ciphertext–scalar-constant pairs.
    ///
    /// Each `b[i]` contributes a single ZNX coefficient `pt_coeffs[i]`.
    /// Metadata follows the same rule as [`Self::ckks_dot_product_pt_vec`].
    fn ckks_dot_product_pt_const<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord, BE::Ring>],
        b: &[&E],
        pt_coeffs: &[usize],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos;
}
