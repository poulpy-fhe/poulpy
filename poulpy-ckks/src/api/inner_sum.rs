use crate::CKKSResult as Result;
use poulpy_core::layouts::{Degree, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Inner sums and replications: the sum of the rotations of a ciphertext by multiples of a power-of-two batch, and
/// the same sum in the opposite direction.
///
/// `ckks_inner_sum` computes `res = sum_{j < 2^log_count} Rot(a, j 2^log_batch)`: on slots holding `2^log_count`
/// batches of `2^log_batch` values side by side, each batch receives the sum of all of them, position by position.
/// With `normalize` set it returns their mean instead, the sum divided by `2^log_count`: a slot every rotation by
/// `2^log_batch` leaves unchanged then comes back unchanged, where the sum multiplies it by `2^log_count`, so the mean
/// over all the slots projects onto those slots. The division is `ckks_div_pow2_assign`, a relabel that moves no limb:
/// it adds `log_count` to `log_delta` and spends as many bits of budget. `ckks_replicate` computes
/// `res = sum_{j < 2^log_count} Rot(a, -j 2^log_batch)`: a batch whose neighbours hold zeros is copied into the
/// `2^log_count - 1` batches that follow it, the inverse direction of the inner sum.
///
/// Both are poulpy-core's summed trace over the levels `log_batch + 1 .. log_batch + 1 + log_count`, forward for the
/// inner sum and reversed for the replicate, on `res` after a `ckks_copy` of `a`: each level adds to `res` its
/// rotation by `2^(level - 1)` slots, or by the opposite, the automorphism leaving the sum canonical. The `_assign`
/// forms are the same on `res` in place. The keys are those of [`Self::ckks_inner_sum_galois_elements`] and
/// [`Self::ckks_replicate_galois_elements`].
///
/// # Metadata
///
/// ```text
/// offset         = max(0, a.k − res.k)                 (0 in place)
/// k_out          = a.k − offset
/// log_delta_out  = a.log_delta + (log_count if normalize)
/// log_budget_out = a.log_budget − offset − (log_count if normalize)
/// log_sparsity   = a.log_sparsity
/// slots          = a.slots
/// canonical      = true, as ckks_copy leaves it when log_count = 0
/// noise          = None
/// ```
///
/// # Errors
///
/// In this order, before any write: `InvalidOperand` when `2^log_count` batches of `2^log_batch` slots exceed the
/// slots of `res`'s degree, or when `normalize` divides `Integer` slots; `MissingAutomorphismKey` for the first key `keys` lacks at
/// the width the sum switches at, `k_out`; `InsufficientHomomorphicCapacity` when the copy into `res` needs more than
/// `a.log_budget`, or the division more than the budget the copy leaves; `CKKSError::Internal` with less scratch than
/// [`Self::ckks_inner_sum_tmp_bytes`] reports for the keys the sum looks up.
pub trait CKKSInnerSumOps<BE: Backend> {
    /// Galois elements of the keys [`Self::ckks_inner_sum`] looks up on a degree-`n` ciphertext: poulpy-core's
    /// `glwe_trace_galois_elements` over its levels. Panics with "ckks_inner_sum_galois_elements: the window exceeds
    /// the slot count" when `2^log_count` batches of `2^log_batch` slots exceed the slots of degree `n`.
    fn ckks_inner_sum_galois_elements(&self, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64>;

    /// Galois elements of the keys [`Self::ckks_replicate`] looks up on a degree-`n` ciphertext: poulpy-core's
    /// `glwe_trace_galois_elements` over its levels in reverse. Panics with "ckks_replicate_galois_elements: the window
    /// exceeds the slot count" when `2^log_count` batches of `2^log_batch` slots exceed the slots of degree `n`.
    fn ckks_replicate_galois_elements(&self, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64>;

    /// Scratch of [`Self::ckks_inner_sum`], [`Self::ckks_inner_sum_assign`], [`Self::ckks_replicate`] and
    /// [`Self::ckks_replicate_assign`] from `a` into `res` with rotation keys of layout `key`, the `_assign` forms
    /// passing `res` as `a`.
    fn ckks_inner_sum_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos;

    /// `res = sum_{j < 2^log_count} Rot(a, j 2^log_batch)`, or the mean of those rotations when `normalize`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the sum's operands, batch, count and normalization, keys and scratch"
    )]
    fn ckks_inner_sum<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// [`Self::ckks_inner_sum`] on `res` in place.
    fn ckks_inner_sum_assign<R, H>(
        &self,
        res: &mut R,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// `res = sum_{j < 2^log_count} Rot(a, -j 2^log_batch)`.
    fn ckks_replicate<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// [`Self::ckks_replicate`] on `res` in place.
    fn ckks_replicate_assign<R, H>(
        &self,
        res: &mut R,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;
}
