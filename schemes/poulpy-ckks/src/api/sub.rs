use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Ciphertext and plaintext subtraction.
///
/// Subtraction is the additive inverse of addition. Operand dispatch,
/// normalization and metadata rules are identical to those of
/// [`CKKSAddOps`](crate::api::CKKSAddOps): `b` (`a` in the assign forms) is a
/// ciphertext or a plaintext, rank 0 taking the plaintext route; `dst` and the
/// `a` of the `_into` forms are ciphertexts.
///
/// # Metadata
///
/// ## Ciphertext operand
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
/// For `_assign` variants `offset = 0`, `dst` standing for `a` and `a` for `b`.
///
/// ## Plaintext operand
///
/// The plaintext `b` of degree `n` is broadcast into the ciphertext of degree
/// `N` and aligned to its budget.
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
/// A constant is a plaintext of its natural degree, broadcast as for addition.
///
/// ## Slot layouts
///
/// Both operands must share one slot layout, which the difference carries.
pub trait CKKSSubOps<BE: Backend, Rg: Ring = Standard> {
    /// Scratch bytes for [`Self::ckks_sub_into`] and [`Self::ckks_sub_assign`]:
    /// the maximum over the ciphertext and plaintext routes, as it takes only sizes.
    fn ckks_sub_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = a - b`, `b` a ciphertext or a plaintext.
    ///
    /// Ciphertext operands with differing `log_budget` are aligned automatically.
    ///
    /// # Errors
    ///
    /// Those of [`CKKSAddOps::ckks_add_into`](crate::api::CKKSAddOps::ckks_add_into), in its order, under
    /// `ckks_sub_into`, before any write to `dst`.
    fn ckks_sub_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds;

    /// Computes `dst -= a` in-place, `a` a ciphertext or a plaintext.
    ///
    /// # Errors
    ///
    /// Those of [`CKKSAddOps::ckks_add_assign`](crate::api::CKKSAddOps::ckks_add_assign), in its order, under
    /// `ckks_sub_assign`, before any write to `dst`.
    fn ckks_sub_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds;

    /// Scratch bytes for [`Self::ckks_sub_one_assign`].
    fn ckks_sub_one_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst -= 1` in-place.
    ///
    /// The exact integer constant is subtracted from coefficient slot `0`.
    /// On a replicated slot layout, the masked one `π_S` is subtracted instead, so the guard slots stay zero.
    /// CKKS layout and scale metadata is preserved; component noise metadata is cleared.
    /// Size scratch with [`Self::ckks_sub_one_tmp_bytes`].
    fn ckks_sub_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos;
}
