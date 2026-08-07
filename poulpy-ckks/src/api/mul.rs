use crate::CKKSResult as Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos, layouts::CKKSPreparedRight};

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
    /// The operands must be passed: the internal tensor intermediate is carved at
    /// the widest of `res`/`a`/`b`, so a destination narrower than its operands
    /// (a supported call) needs more scratch than `res` alone describes. For
    /// `_assign` pass `dst` as both `res` and `a`; for `_prepared_assign` pass
    /// the operand the [`CKKSPreparedRight`] was prepared from.
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
    fn ckks_mul_into<Dst, A, B, T>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst *= a` in-place using tensor-product keyswitching via `tsk`.
    fn ckks_mul_assign<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Prepares `a` as a reusable right operand for [`Self::ckks_mul_prepared_assign`].
    ///
    /// Hoists the forward transform of `a` so the same operand can multiply many
    /// destinations without re-preparing it (e.g. one `X^{gsp}` across a BSGS
    /// giant-step level). The returned [`CKKSPreparedRight`] is backend-resident
    /// (heap-owned), so it draws no scratch once produced. The scratch needed to
    /// produce it is bounded by [`Self::ckks_mul_tmp_bytes`].
    fn ckks_prepare_right<A>(&self, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedRight<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst *= prepared` in-place against a caller-prepared right operand,
    /// relinearizing via `tsk`.
    ///
    /// Equivalent to [`Self::ckks_mul_assign`] with `a` pre-prepared by
    /// [`Self::ckks_prepare_right`]; same metadata rule and scratch bound
    /// ([`Self::ckks_mul_tmp_bytes`]).
    fn ckks_mul_prepared_assign<Dst, T>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst = a * a` (squaring) using tensor-product keyswitching.
    ///
    /// Equivalent to `ckks_mul_into(dst, a, a, tsk)` with the same metadata rule.
    fn ckks_square_into<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst = dst * dst` (squaring in-place) using tensor-product keyswitching.
    fn ckks_square_assign<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

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
