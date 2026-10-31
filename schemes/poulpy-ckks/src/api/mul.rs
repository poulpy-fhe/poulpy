use crate::CKKSResult as Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendRef, GetTensorKey};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf},
    layouts::{Backend, PrepareHint, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
};

/// Ciphertext–ciphertext and ciphertext–plaintext multiplication.
///
/// Multiplication is the **primary consumer** of homomorphic capacity.  Every
/// multiplication reduces `log_budget` by an amount proportional to the
/// precision of the operands, plus an additional reduction if the destination
/// buffer cannot hold the full natural result.
///
/// # Metadata
///
/// ## Ciphertext–ciphertext multiplication (`ckks_mul_*`, `ckks_square_*`)
///
/// Let:
/// ```text
/// natural_budget = min(a.log_budget, b.log_budget) − max(a.log_delta, b.log_delta)
/// log_delta_out  = min(a.log_delta, b.log_delta)
/// natural_eff_k  = natural_budget + log_delta_out
/// offset         = max(0, natural_eff_k − dst.k())
///
/// log_budget_out = natural_budget − offset
/// ```
///
/// **Capacity consumed by the multiplication itself**: `max(a.log_delta, b.log_delta)` bits.
/// **Additional reduction from small `dst`**: `offset` bits.
///
/// The result is produced at the destination's **requested `dst.k()`** (value-preserving
/// rounding when narrower than the natural width), not the buffer's allocation — the
/// same contract as the `pt_vec` variant below.
///
/// For the common case of equal-precision operands (`a.log_delta == b.log_delta == Δ`):
///
/// ```text
/// natural_eff_k  = a.log_budget   (= b.log_budget when budgets are also equal)
/// log_delta_out  = Δ
/// offset         = max(0, a.log_budget − dst.k())
/// log_budget_out = a.log_budget − Δ − offset
/// ```
///
/// Errors with `MultiplicationPrecisionUnderflow` if `natural_budget < 0`
/// (i.e. `min(log_budget) < max(log_delta)`).
///
/// ## Ciphertext–plaintext-vector multiplication (`ckks_mul_pt_vec_*`)
///
/// ```text
/// natural_budget = a.log_budget − pt.log_delta
/// log_delta_out  = a.log_delta
/// natural_eff_k  = natural_budget + a.log_delta
///                = a.k() − pt.log_delta
/// offset         = max(0, natural_eff_k − dst.k())
///
/// log_budget_out = natural_budget − offset
/// ```
///
/// **Capacity consumed**: `pt.log_delta` bits (precision of the plaintext
/// multiplier), plus `offset`.
///
/// The result is produced at the destination's **requested `dst.k()`** (with
/// value-preserving rounding of the low bits when it is narrower than the natural
/// width), not at the buffer's limb-aligned allocation. Allocate `dst` at the
/// exact `k` you want the product at — this is how a leveled consumer (e.g. the PaCo
/// blind rotation) evaluates the whole downstream circuit at a lower, cheaper width.
///
/// **Plaintext operand**: `pt` is multiplied in as a full-width **integer
/// polynomial** (bottom-up encoding) — every stored limb participates
/// (the convolution masks it at its declared `pt.encoded_k()` — plaintext
/// operands are integer polynomials, not Torus elements, and are bounded by
/// `IntPolyInfos` to state that width).
/// Allocate `pt` at exactly the precision you want folded in;
/// a reduced `pt.k()` / `log_budget` does not narrow it.
///
/// ## Ciphertext–plaintext-constant multiplication (`ckks_mul_pt_const_*`)
///
/// Identical metadata rule to the `pt_vec` variant above, using
/// `pt.log_delta` as the plaintext precision.
///
/// # Rescaling after multiplication
///
/// After a ciphertext–ciphertext multiplication the result has a lower
/// `log_budget` but the same `log_delta`; the destination buffer keeps its
/// allocated width (allocate the destination at exactly the `k` you want the
/// product at). To trade further budget for precision under `log_delta` — the
/// closest analogue of an RNS "rescale" — use
/// [`CKKSPow2Ops::ckks_div_pow2_assign`](crate::api::CKKSPow2Ops::ckks_div_pow2_assign).
pub trait CKKSMulOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_mul_into`] / [`Self::ckks_mul_assign`] /
    /// [`Self::ckks_mul_prepared_assign`] with result `res` and operands `a`, `b`.
    ///
    /// The operands must be passed: the internal tensor intermediate is carved
    /// at `max(a.k(), b.k())`; `res` supplies its physical result shape and
    /// capacity, but its pre-call precision does not widen the intermediate.
    /// For `_assign` pass `dst` as both `res` and `a`; for
    /// `_prepared_assign` pass the [`CKKSRightOperandPrepared`], whose infos report
    /// the operand it was prepared from.
    fn ckks_mul_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos;

    /// Scratch bytes for [`Self::ckks_square_into`] / [`Self::ckks_square_assign`]
    /// with result `res` and operand `a` (pass `dst` twice for `_assign`).
    fn ckks_square_tmp_bytes<R, A, T>(&self, res: &R, a: &A, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Computes `dst = a * b` using tensor-product keyswitching via `tsk`.
    ///
    /// See the trait-level documentation for the exact metadata rule including
    /// the capacity offset.
    fn ckks_mul_into<Dst, A, B, H>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Computes `dst *= a` in-place using tensor-product keyswitching via `tsk`.
    fn ckks_mul_assign<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Computes `dst *= prepared` in-place against a caller-prepared right operand,
    /// relinearizing via `tsk`.
    ///
    /// Equivalent to [`Self::ckks_mul_assign`] with `a` pre-prepared by
    /// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc`]; same metadata rule and scratch bound
    /// ([`Self::ckks_mul_tmp_bytes`]).
    fn ckks_mul_prepared_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSRightOperandPrepared<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>;

    /// Computes `dst = a * a` (squaring) using tensor-product keyswitching.
    ///
    /// Equivalent to `ckks_mul_into(dst, a, a, tsk)` with the same metadata rule.
    fn ckks_square_into<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Computes `dst = dst * dst` (squaring in-place) using tensor-product keyswitching.
    fn ckks_square_assign<Dst, H>(&self, dst: &mut Dst, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>;

    /// Computes `dst = a * pt` where `pt` is a full plaintext polynomial.
    ///
    /// See the trait-level documentation for the exact metadata rule including
    /// the capacity offset.
    fn ckks_mul_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;

    /// Computes `dst *= pt` in-place.
    fn ckks_mul_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;

    /// Computes `dst = a * pt[pt_coeff]`, multiplying by a single
    /// quantized constant from coefficient `pt_coeff` of `pt`.
    ///
    /// See the trait-level documentation for the exact metadata rule including
    /// the capacity offset.
    fn ckks_mul_pt_const_into<Dst, A, P>(
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

    /// Computes `dst *= pt[pt_coeff]` in-place.
    fn ckks_mul_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds;
}

/// Allocates and prepares [`CKKSLeftOperandPrepared`] operands.
pub trait CKKSLeftOperandPreparedFactory<BE: Backend>
where
    Self: CnvPVecAlloc<BE> + CnvPVecBytesOf,
{
    /// A left operand for a ciphertext of `infos`, on its `ceil(k / base2k)` limbs and `rank + 1` columns, in the
    /// representation `hint` selects, as
    /// [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc`] describes it, which
    /// [`Self::ckks_left_operand_prepare`] fills. Its contents are unspecified until then, and its infos report `infos`
    /// and no metadata.
    fn ckks_left_operand_prepared_alloc_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> CKKSLeftOperandPrepared<BE>
    where
        A: GLWEInfos,
    {
        let size = infos.size();
        CKKSLeftOperandPrepared {
            prep: self.cnv_pvec_left_alloc(infos.n().as_usize(), infos.rank().as_usize() + 1, size, hint),
            size,
            layout: infos.glwe_layout(),
            meta: None,
        }
    }

    /// Bytes of a left operand [`Self::ckks_left_operand_prepared_alloc_from_infos`] allocates for `infos` under `hint`.
    fn ckks_left_operand_prepared_bytes_of_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        A: GLWEInfos,
    {
        self.cnv_pvec_left_bytes_of(infos.n().as_usize(), infos.rank().as_usize() + 1, infos.size(), hint)
    }

    /// Scratch bytes of [`Self::ckks_left_operand_prepare`] and [`Self::ckks_left_operand_prepare_alloc`] for an
    /// operand shaped like `a`: a GLWE shaped like `a`, to normalize it into when its canonical flag is clear, and the
    /// larger of the normalization's and the forward transform's scratch.
    fn ckks_left_operand_prepare_tmp_bytes<A>(&self, a: &A) -> usize
    where
        A: CKKSCtBounds;

    /// Prepares `a` into `res`, a left operand allocated for `a`'s layout, with no allocation:
    /// [`Self::ckks_left_operand_prepare_alloc`] of `a` under `res`'s hint, bit for bit. `res` then reports `a`'s layout
    /// and metadata; preparing again into it refills it. Takes [`Self::ckks_left_operand_prepare_tmp_bytes`] of scratch.
    ///
    /// # Errors
    ///
    /// `InvalidOperand` for an `a` of another degree, radix, width `k` or rank than `res` was allocated for, before
    /// any write to `res`.
    fn ckks_left_operand_prepare<A>(
        &self,
        res: &mut CKKSLeftOperandPrepared<BE>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Prepares `a` as a left operand of
    /// [`CKKSDotProductOps::ckks_dot_product_prepared`](crate::api::CKKSDotProductOps::ckks_dot_product_prepared),
    /// as [`CKKSRightOperandPreparedFactory::ckks_right_operand_prepare_alloc`] prepares a right one.
    fn ckks_left_operand_prepare_alloc<A>(
        &self,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<CKKSLeftOperandPrepared<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds;
}

/// Allocates and prepares [`CKKSRightOperandPrepared`] operands.
pub trait CKKSRightOperandPreparedFactory<BE: Backend>
where
    Self: CnvPVecAlloc<BE> + CnvPVecBytesOf,
{
    /// As [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepared_alloc_from_infos`], a right operand, which
    /// [`Self::ckks_right_operand_prepare`] fills.
    fn ckks_right_operand_prepared_alloc_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> CKKSRightOperandPrepared<BE>
    where
        A: GLWEInfos,
    {
        let size = infos.size();
        CKKSRightOperandPrepared {
            prep: self.cnv_pvec_right_alloc(infos.n().as_usize(), infos.rank().as_usize() + 1, size, hint),
            size,
            layout: infos.glwe_layout(),
            meta: None,
        }
    }

    /// Bytes of a right operand [`Self::ckks_right_operand_prepared_alloc_from_infos`] allocates for `infos` under
    /// `hint`.
    fn ckks_right_operand_prepared_bytes_of_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        A: GLWEInfos,
    {
        self.cnv_pvec_right_bytes_of(infos.n().as_usize(), infos.rank().as_usize() + 1, infos.size(), hint)
    }

    /// As [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare_tmp_bytes`], for
    /// [`Self::ckks_right_operand_prepare`] and [`Self::ckks_right_operand_prepare_alloc`].
    fn ckks_right_operand_prepare_tmp_bytes<A>(&self, a: &A) -> usize
    where
        A: CKKSCtBounds;

    /// As [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare`], into a right operand:
    /// [`Self::ckks_right_operand_prepare_alloc`] of `a`, bit for bit. Takes
    /// [`Self::ckks_right_operand_prepare_tmp_bytes`] of scratch.
    ///
    /// # Errors
    ///
    /// As [`CKKSLeftOperandPreparedFactory::ckks_left_operand_prepare`].
    fn ckks_right_operand_prepare<A>(
        &self,
        res: &mut CKKSRightOperandPrepared<BE>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Prepares `a` as a reusable right operand for [`CKKSMulOps::ckks_mul_prepared_assign`]
    /// and [`CKKSDotProductOps::ckks_dot_product_prepared`](crate::api::CKKSDotProductOps::ckks_dot_product_prepared).
    ///
    /// Hoists the forward transform of `a` so the same operand can multiply many
    /// destinations without re-preparing it (e.g. one `X^{gsp}` across a BSGS
    /// giant-step level): `a`, normalized first if its canonical flag is clear,
    /// on its `ceil(k / base2k)` limbs. `hint` tells the backend how the operand
    /// will be used, [`PrepareHint::OneShot`] for one product and
    /// [`PrepareHint::Reuse`] for many; it may pick another representation for
    /// each, but the products give the same bits under either. The returned
    /// [`CKKSRightOperandPrepared`] is backend-resident (heap-owned), so it draws no
    /// scratch once produced, and its infos report `a`'s layout and metadata
    /// (no noise estimate: only fresh ciphertexts carry one). Producing it takes
    /// [`Self::ckks_right_operand_prepare_tmp_bytes`] of scratch, which [`CKKSMulOps::ckks_mul_tmp_bytes`] covers. It
    /// is [`Self::ckks_right_operand_prepare`] of `a` into [`Self::ckks_right_operand_prepared_alloc_from_infos`] of
    /// `a` under `hint`.
    fn ckks_right_operand_prepare_alloc<A>(
        &self,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<CKKSRightOperandPrepared<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds;
}
