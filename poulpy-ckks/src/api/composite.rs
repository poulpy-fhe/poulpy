use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSPreparedLeft, CKKSPreparedRight},
};

/// Sum over a slice of ciphertexts.
pub trait CKKSAddManyOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_add_many`] into `res` from inputs laid out like
    /// `input`: the larger of the addition query at `res` and the copy query from
    /// `input` to `res`, the copy serving a single input.
    fn ckks_add_many_tmp_bytes<R, A>(&self, res: &R, input: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds;

    /// Computes `dst = inputs[0] + inputs[1] + … + inputs[n-1]`, adding the
    /// inputs to `dst` from left to right.
    ///
    /// # Metadata
    ///
    /// Follows from repeated application of `ckks_add_into`:
    ///
    /// ```text
    /// log_delta_out  = min over all inputs of log_delta
    /// log_budget_out = min over all inputs of log_budget
    /// ```
    ///
    /// No capacity is consumed.  The destination-capacity offset from the
    /// final write applies:
    ///
    /// ```text
    /// offset         = max(0, result_k − dst.k())
    /// log_budget_out = (min log_budget) − offset
    /// ```
    ///
    /// Errors if `inputs` is empty.
    fn ckks_add_many<Dst, Src>(&self, dst: &mut Dst, inputs: &[&Src], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
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
    /// prepared from by [`CKKSMulOps::ckks_prepare_left`](crate::api::CKKSMulOps::ckks_prepare_left)
    /// and [`CKKSMulOps::ckks_prepare_right`](crate::api::CKKSMulOps::ckks_prepare_right),
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
        a: &[&CKKSPreparedLeft<BE>],
        b: &[&CKKSPreparedRight<BE>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>;

    /// Scratch bytes of [`Self::ckks_dot_product_prepared_pt`] for terms prepared like `a` and plaintexts prepared like
    /// `b`, whatever their number: pass the prepared operands, whose infos report their sources, or those sources.
    fn ckks_dot_product_prepared_pt_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos;

    /// Computes `dst = Σ a[t] · σ_{galois[t]}(b[t])` over ciphertexts prepared by
    /// [`CKKSMulOps::ckks_prepare_left`](crate::api::CKKSMulOps::ckks_prepare_left) and plaintexts prepared by
    /// [`CKKSMulOps::ckks_prepare_right`](crate::api::CKKSMulOps::ckks_prepare_right), under either hint, each plaintext
    /// read through its odd Galois element: `1` for the plaintext itself,
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
    /// Before writing `dst`: an error for `a` and `b` empty or of two lengths; `InvalidOperand` for `galois` of another
    /// length; as [`Self::ckks_dot_product_prepared`], `PreparedOperandLayoutMismatch` for a term of another degree,
    /// `base2k` or rank than `dst`; `InvalidOperand` for an even Galois element, a plaintext not of rank 0 at `dst`'s `base2k`, or a side whose operands do not share one
    /// layout and scale; `PlaintextDegreeMismatch` for a plaintext whose degree does not embed into `dst`'s; and the
    /// errors of the rule.
    fn ckks_dot_product_prepared_pt<Dst>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSPreparedLeft<BE>],
        b: &[&CKKSPreparedRight<BE>],
        galois: &[i64],
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
