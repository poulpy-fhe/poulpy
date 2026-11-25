use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Ciphertext and plaintext addition.
///
/// The operand `b` (`a` in the assign forms) is a ciphertext or a plaintext:
/// the operations dispatch on its rank, a metadata read. Rank 0 takes the
/// plaintext route, any other rank the ciphertext route; a CKKS ciphertext has
/// rank at least 1, which its allocation and carving enforce.
///
/// The sum is not normalized: the destination's canonical flag is cleared and
/// the next operation that reads it through a DFT normalizes it first. An
/// operand aligned by a shift is normalized by that shift.
///
/// # Metadata
///
/// ## Ciphertext operand
///
/// Both operands are automatically shifted to the same torus level before
/// addition, so different `log_budget` values are accepted without manual
/// alignment.
///
/// For `_into` variants the destination capacity can reduce the result:
///
/// ```text
/// offset           = max(0, min(a.k(), b.k()) − dst.k())
///
/// log_delta_out    = min(a.log_delta,  b.log_delta)
/// log_budget_out   = min(a.log_budget, b.log_budget) − offset
/// log_sparsity_out = min(a.log_sparsity, b.log_sparsity)
/// slots_out        = a.slots.join(b.slots)
/// ```
///
/// For `_assign` variants `dst` is already the buffer being operated on so
/// `offset = 0` and the formula reduces to:
///
/// ```text
/// log_delta_out    = min(dst.log_delta, a.log_delta)
/// log_budget_out   = min(dst.log_budget, a.log_budget)
/// log_sparsity_out = min(dst.log_sparsity, a.log_sparsity)
/// slots_out        = dst.slots.join(a.slots)
/// ```
///
/// Addition does not consume homomorphic capacity beyond the offset.
///
/// ## Plaintext operand
///
/// The plaintext `b`, of any power-of-two degree `n` dividing the ciphertext's
/// `N`, is broadcast through `X -> X^(N/n)` and added coefficient-wise in the
/// ZNX domain, a plaintext of degree 1 adding an integer to coefficient `0`.
/// It is aligned to the ciphertext's budget, without the ciphertext moving.
///
/// ```text
/// offset           = max(0, a.k() − dst.k())          (_assign: 0, dst for a)
///
/// log_delta_out    = a.log_delta
/// log_budget_out   = a.log_budget − offset
/// log_sparsity_out = min(a.log_sparsity, log2(N / n))
/// slots_out        = a.slots.join(b.slots)
/// ```
///
/// **Precondition**: `log_budget_out + b.log_delta >= b.k()`.
/// Returns `PlaintextAlignmentImpossible` otherwise.
///
/// A constant is a plaintext of its natural degree: a real constant of degree
/// 1, a complex one `[re, im]` of degree 2 (`i = X^(N/2)`), or `c·π_S` of
/// degree `2R` on a replicated layout.
///
/// ## Slot layouts
///
/// Both operands must share one slot layout, which the sum carries.
pub trait CKKSAddOps<BE: Backend, Rg: Ring = Standard> {
    /// Scratch bytes for [`Self::ckks_add_into`] and [`Self::ckks_add_assign`]:
    /// the maximum over the ciphertext and plaintext routes, as it takes only sizes.
    fn ckks_add_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = a + b`, `b` a ciphertext or a plaintext.
    ///
    /// Ciphertext operands with differing `log_budget` are aligned automatically.
    /// See the trait-level documentation for the full metadata rule.
    ///
    /// # Errors
    ///
    /// Before any write to `dst`, in this order. A ciphertext `b`:
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for an
    /// operand above the module's degree, for slot layouts that differ and, on a
    /// replicated layout, for scales that differ, then
    /// [`InsufficientHomomorphicCapacity`](crate::CKKSCompositionError::InsufficientHomomorphicCapacity)
    /// for an `offset` above the operands' budget. A plaintext `b`: `InvalidOperand`
    /// for slot layouts that differ, for a plaintext of a finer scale than `a` on a
    /// replicated layout and for `dst` or `a` above the module's degree,
    /// `InsufficientHomomorphicCapacity` for an `offset` above `a`'s budget, then
    /// [`PlaintextBase2KMismatch`](crate::CKKSCompositionError::PlaintextBase2KMismatch),
    /// [`PlaintextDegreeMismatch`](crate::CKKSCompositionError::PlaintextDegreeMismatch)
    /// for a degree that does not divide the ciphertext's and
    /// [`PlaintextAlignmentImpossible`](crate::CKKSCompositionError::PlaintextAlignmentImpossible).
    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds;

    /// Computes `dst += a` in-place, `a` a ciphertext or a plaintext.
    ///
    /// A ciphertext `a` and `dst` are aligned automatically if their `log_budget` differs.
    ///
    /// # Errors
    ///
    /// Before any write to `dst`, in this order. A ciphertext `a`:
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for an
    /// operand above the module's degree, for slot layouts that differ and, on a
    /// replicated layout, for scales that differ. A plaintext `a`: `InvalidOperand`
    /// for `dst` above the module's degree, for slot layouts that differ and for a
    /// plaintext of a finer scale than `dst` on a replicated layout, then
    /// [`PlaintextBase2KMismatch`](crate::CKKSCompositionError::PlaintextBase2KMismatch),
    /// [`PlaintextDegreeMismatch`](crate::CKKSCompositionError::PlaintextDegreeMismatch)
    /// for a degree that does not divide `dst`'s and
    /// [`PlaintextAlignmentImpossible`](crate::CKKSCompositionError::PlaintextAlignmentImpossible).
    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds;

    /// Scratch bytes for [`Self::ckks_add_one_assign`].
    fn ckks_add_one_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst += 1` in-place.
    ///
    /// The exact integer constant is added to coefficient slot `0`.
    /// On a replicated slot layout, the masked one `π_S` is added instead, so the guard slots stay zero.
    /// CKKS layout and scale metadata is preserved; component noise metadata is cleared.
    /// Size scratch with [`Self::ckks_add_one_tmp_bytes`].
    fn ckks_add_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos;
}
