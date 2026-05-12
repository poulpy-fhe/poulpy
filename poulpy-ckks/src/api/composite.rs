use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos,
    layouts::{CKKSCiphertext, UnnormalizedCKKSCiphertext},
};

/// Tree-reduction sum over a slice of ciphertexts.
pub trait CKKSAddManyOps<BE: Backend> {
    fn ckks_add_many_tmp_bytes(&self) -> usize;

    /// Computes `dst = inputs[0] + inputs[1] + … + inputs[n-1]` using a
    /// balanced binary tree of pairwise additions.
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
    /// offset         = max(0, result_effective_k − dst.max_k())
    /// log_budget_out = (min log_budget) − offset
    /// ```
    ///
    /// Errors if `inputs` is empty.
    fn ckks_add_many<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>;
}

/// Fused multiply-accumulate: `dst += a * b`.
///
/// Each variant computes the product of two operands and adds it to `dst`
/// without a separate allocation for the intermediate product.
pub trait CKKSMulAddOps<BE: Backend> {
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
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
    ///                  − max(0, min(dst.effective_k(), prod_effective_k) − dst.max_k())
    /// ```
    fn ckks_mul_add_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>;

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
    ///                  − max(0, min(dst.effective_k(), prod_effective_k) − dst.max_k())
    /// ```
    fn ckks_mul_add_pt_vec_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += a * pt[pt_coeff]`.
    ///
    /// Metadata follows the same rule as [`Self::ckks_mul_add_pt_vec_into`].
    fn ckks_mul_add_pt_const_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += a * pt[pt_coeff]` without normalizing `dst`.
    ///
    /// The accumulator `dst` carries un-propagated carries in its limb digits.
    /// Use this to fuse several multiply-add steps before a single
    /// [`UnnormalizedCKKSCiphertext::normalize`] call.  See
    /// [`crate::api::CKKSAddOpsUnnormalized`] for the digit-growth analysis
    /// and safety bound.
    fn ckks_mul_add_pt_const_into_unnormalized<Dst: Data, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += a * pt` without normalizing `dst`.
    ///
    /// Metadata follows the same rule as
    /// [`Self::ckks_mul_add_pt_const_into_unnormalized`].
    fn ckks_mul_add_pt_vec_into_unnormalized<Dst: Data, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
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
    /// natural_eff_k  = a.effective_k() − affine_const.log_delta
    /// offset         = max(0, natural_eff_k − dst.max_k())
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
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

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
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

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
    /// natural_eff_k  = a.effective_k() − scale.log_delta
    /// offset_bits    = max(0, natural_eff_k − dst.max_k())
    ///
    /// log_delta_out  = a.log_delta
    /// log_budget_out = a.log_budget − scale.log_delta − offset_bits
    /// ```
    ///
    /// **Net capacity consumed**: `scale.log_delta + offset_bits` bits.
    ///
    /// **Precondition**: `(a.log_budget − scale.log_delta) + offset.log_delta >= offset.effective_k()`
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
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

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
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
}

/// Fused multiply-subtract: `dst -= a * b`.
pub trait CKKSMulSubOps<BE: Backend> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
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
    fn ckks_mul_sub_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst -= a * pt` where `pt` is a full plaintext polynomial.
    ///
    /// Metadata follows the same rule as
    /// [`CKKSMulAddOps::ckks_mul_add_pt_vec_into`].
    fn ckks_mul_sub_pt_vec_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst -= a * pt[pt_coeff]`.
    ///
    /// Metadata follows the same rule as
    /// [`CKKSMulAddOps::ckks_mul_add_pt_const_into`].
    fn ckks_mul_sub_pt_const_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
}

/// Inner product (dot product) of ciphertext and plaintext slices.
///
/// Computes the weighted sum `dst = Σ a[i] * b[i]` over all pairs.
pub trait CKKSDotProductOps<BE: Backend> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
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

    /// Computes `dst = Σ a[i] * b[i]` over ciphertext–ciphertext pairs.
    ///
    /// # Metadata
    ///
    /// Equivalent to a sequence of `ckks_mul_add_ct_into` calls.  With
    /// uniform-precision inputs:
    ///
    /// ```text
    /// natural_eff_k  = a[i].log_budget   (= a[i].effective_k() − a[i].log_delta)
    /// offset         = max(0, natural_eff_k − dst.max_k())
    ///
    /// log_delta_out  = a[i].log_delta
    /// log_budget_out = a[i].log_budget − a[i].log_delta − offset
    /// ```
    fn ckks_dot_product_ct<Dst: Data, D: Data, E: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        CKKSCiphertext<E>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst = Σ a[i] * b[i]` over ciphertext–plaintext-polynomial pairs.
    ///
    /// # Metadata
    ///
    /// ```text
    /// natural_eff_k  = a[i].effective_k() − b[i].log_delta
    /// offset         = max(0, natural_eff_k − dst.max_k())
    ///
    /// log_delta_out  = a[i].log_delta
    /// log_budget_out = a[i].log_budget − b[i].log_delta − offset
    /// ```
    fn ckks_dot_product_pt_vec<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = Σ a[i] * b[i][pt_coeffs[i]]` over ciphertext–scalar-constant pairs.
    ///
    /// Each `b[i]` contributes a single ZNX coefficient `pt_coeffs[i]`.
    /// Metadata follows the same rule as [`Self::ckks_dot_product_pt_vec`].
    fn ckks_dot_product_pt_const<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        pt_coeffs: &[usize],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds;
}
