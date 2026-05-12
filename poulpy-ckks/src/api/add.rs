use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos, layouts::CKKSCiphertext, layouts::UnnormalizedCKKSCiphertext};

/// Normalized ciphertext and plaintext addition.
///
/// All operations in this trait produce a fully normalized [`CKKSCiphertext`]
/// whose limb digits fit within `base2k` bits, safe for any subsequent
/// DFT-domain operation (keyswitching, convolution, automorphisms).
///
/// # Metadata
///
/// ## Ciphertext–ciphertext addition (`ckks_add_into` / `ckks_add_assign`)
///
/// Both operands are automatically shifted to the same torus level before
/// addition, so different `log_budget` values are accepted without manual
/// alignment.
///
/// For `_into` variants the destination capacity can reduce the result:
///
/// ```text
/// offset         = max(0, min(a.effective_k(), b.effective_k()) − dst.max_k())
///
/// log_delta_out  = min(a.log_delta,  b.log_delta)
/// log_budget_out = min(a.log_budget, b.log_budget) − offset
/// ```
///
/// For `_assign` variants `dst` is already the buffer being operated on so
/// `offset = 0` and the formula reduces to:
///
/// ```text
/// log_delta_out  = min(dst.log_delta, a.log_delta)
/// log_budget_out = min(dst.log_budget, a.log_budget)
/// ```
///
/// Addition does not consume homomorphic capacity beyond the offset.
///
/// ## Ciphertext–plaintext-vector addition (`ckks_add_pt_vec_*`)
///
/// The full plaintext polynomial is added coefficient-wise in the ZNX domain.
///
/// ```text
/// offset         = max(0, a.effective_k() − dst.max_k())
///
/// log_delta_out  = a.log_delta
/// log_budget_out = a.log_budget − offset
/// ```
///
/// **Precondition**: `a.log_budget + pt.log_delta >= pt.effective_k()`.
/// Returns `PlaintextAlignmentImpossible` otherwise.
///
/// ## Ciphertext–plaintext-constant addition (`ckks_add_pt_const_*`)
///
/// Adds a single quantized constant (one ZNX coefficient of a plaintext) to
/// one coefficient slot of the ciphertext.  Metadata follows the same rule as
/// the `pt_vec` variants above.
pub trait CKKSAddOps<BE: Backend> {
    fn ckks_add_tmp_bytes(&self) -> usize;

    /// Computes `dst = a + b`.
    ///
    /// Operands with differing `log_budget` are aligned automatically.
    /// See the trait-level documentation for the full metadata rule.
    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += a` in-place.
    ///
    /// `dst` and `a` are aligned automatically if their `log_budget` differs.
    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += 1` in-place.
    ///
    /// The exact integer constant is added to coefficient slot `0`.
    /// Metadata is preserved.
    fn ckks_add_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_add_pt_vec_tmp_bytes(&self) -> usize;

    /// Computes `dst = a + pt` where `pt` is a full plaintext polynomial.
    ///
    /// `pt` is added coefficient-wise after being aligned to the ciphertext's
    /// torus level.  Metadata is inherited from `a` with the capacity offset
    /// applied (see trait-level doc).
    fn ckks_add_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += pt` in-place, where `pt` is a full plaintext polynomial.
    fn ckks_add_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize;

    /// Computes `dst = a + pt[pt_coeff]`, adding one quantized constant to
    /// a single coefficient slot of the ciphertext.
    ///
    /// - `dst_coeff`: target ZNX coefficient of `dst`.  Use `0` for the
    ///   real-slot constant term and `n/2` for the imaginary-slot constant
    ///   term (standard CKKS real/imaginary packing split).
    /// - `pt_coeff`: source coefficient index in `pt`.
    ///
    /// Metadata is inherited from `a` with the capacity offset applied.
    fn ckks_add_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += pt[pt_coeff]` in-place.
    ///
    /// See [`Self::ckks_add_pt_const_into`] for the semantics of `dst_coeff`
    /// and `pt_coeff`.
    fn ckks_add_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
}

/// Unnormalized add variants for explicit fusion loops.
///
/// Each method writes into an [`UnnormalizedCKKSCiphertext`], whose limb
/// digits may hold un-propagated carries (wider than `base2k` bits).  An
/// unnormalized ciphertext cannot be passed as an operand to any primitive
/// that works in the DFT domain (keyswitching, convolution, automorphisms).
/// Call [`UnnormalizedCKKSCiphertext::normalize`] to propagate carries and
/// recover a [`CKKSCiphertext`] safe to pass to those operations.
///
/// # Metadata
///
/// Metadata rules are identical to the corresponding normalized variants in
/// [`CKKSAddOps`]; the output `UnnormalizedCKKSCiphertext` carries the same
/// `log_delta` and `log_budget` as its normalized counterpart would.
///
/// # Digit growth under repeated addition
///
/// Limb digits are signed integers in `[−2^(base2k−1), 2^(base2k−1))`.
/// Each un-normalized addition accumulates one more term per digit without
/// propagating carries.
///
/// **Worst case** — adversarial inputs all having the same sign: digit
/// magnitude grows linearly as `n · 2^(base2k−1)`.  This overflows a 64-bit
/// signed word when `n · 2^(base2k−1) ≥ 2^63`, i.e. `n ≥ 2^(64 − base2k)`.
///
/// **Average / typical case** — CKKS torus values are signed and centered at
/// zero, so each per-digit summand is approximately a uniform variable on
/// `[−2^(base2k−1), 2^(base2k−1))`.  The sum of `n` such variables follows
/// an [Irwin–Hall](https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution)
/// distribution whose standard deviation grows as
/// `sqrt(n) · 2^(base2k−1) / sqrt(3)` — far below the worst-case linear
/// bound for realistic `n`.
///
/// Higher-level operations that accept an explicit accumulation count
/// (e.g. [`CKKSAddManyOps`](crate::api::CKKSAddManyOps),
/// [`CKKSDotProductOps`](crate::api::CKKSDotProductOps)) enforce the
/// conservative safety bound `n ≤ 2^(63 − base2k)` — half the worst-case
/// overflow threshold — to guarantee no digit can overflow `i64` even in the
/// worst case.  When the caller knows inputs are sign-balanced, the bound
/// can effectively be relaxed by `sqrt(n)` in expectation, but that is not
/// checked here.
///
/// # When to use
///
/// These variants avoid the normalize pass after every individual addition.
/// A typical use case is accumulating a sum of many terms, normalizing once
/// at the end:
///
/// ```text
/// for term in &terms {
///     module.ckks_add_assign_unnormalized(&mut acc, term, scratch)?;
/// }
/// let normalized = acc.normalize(&module, &mut scratch)?;
/// ```
pub trait CKKSAddOpsUnnormalized<BE: Backend> {
    fn ckks_add_into_unnormalized<Dst, A, B>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_add_assign_unnormalized<Dst, A>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos;

    fn ckks_add_pt_vec_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_add_pt_vec_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_add_pt_const_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_add_pt_const_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
}
