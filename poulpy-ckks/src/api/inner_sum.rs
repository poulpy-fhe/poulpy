use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Inner sums and replications: the sum of the rotations of a ciphertext by multiples of a power-of-two batch, and
/// the same sum in the opposite direction.
///
/// `ckks_inner_sum_into` computes `dst = sum_{j < 2^log_count} Rot(src, j 2^log_batch)`: on slots holding `2^log_count`
/// batches of `2^log_batch` values side by side, each batch receives the sum of all of them, position by position.
/// With `normalize` set it returns their mean instead, the sum divided by `2^log_count`: a slot every rotation by
/// `2^log_batch` leaves unchanged then comes back unchanged, where the sum multiplies it by `2^log_count`, so the mean
/// over all the slots projects onto those slots. The division is `ckks_div_pow2_assign`, a relabel that moves no limb:
/// it adds `log_count` to `log_delta` and spends as many bits of budget. `ckks_replicate_into` computes
/// `dst = sum_{j < 2^log_count} Rot(src, -j 2^log_batch)`: a batch whose neighbours hold zeros is copied into the
/// `2^log_count - 1` batches that follow it, the inverse direction of the inner sum.
///
/// Both are poulpy-core's summed trace over the levels `log_batch + 1 .. log_batch + 1 + log_count`, forward for the
/// inner sum and reversed for the replicate, on `dst` after a `ckks_copy` of `src`: each level adds to `dst` its
/// rotation by `2^(level - 1)` slots, or by the opposite, the automorphism leaving the sum canonical. The `_assign`
/// forms are the same on `dst` in place.
///
/// # Metadata
///
/// ```text
/// k            = min(dst.k, src.k)                       (dst.k in place)
/// log_delta    = src.log_delta, + log_count when normalize
/// log_budget   = as ckks_copy leaves it, − log_count when normalize
/// log_sparsity = src.log_sparsity
/// slots        = src.slots
/// canonical    = true, as ckks_copy leaves it when log_count = 0
/// noise        = None
/// ```
///
/// # Errors
///
/// Before any write: `InvalidOperand` when `2^log_count` batches of `2^log_batch` slots exceed the slot count, or
/// when `normalize` divides `Integer` slots; `InsufficientHomomorphicCapacity` from the copy when `dst.k` is below
/// `src.log_delta`, or when the division needs more than the remaining budget; `MissingAutomorphismKey` for the key of
/// a rotation `keys` lacks at the width the sum has, `min(dst.k, src.k)`. With less scratch than its query reports for
/// the keys it looks up, an inner sum or a replicate panics before any write.
pub trait CKKSInnerSumOps<BE: Backend> {
    /// Scratch of [`Self::ckks_inner_sum_into`] from `a` into `res` with rotation keys of layout `key`.
    fn ckks_inner_sum_into_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch of [`Self::ckks_inner_sum_assign`] on `res` with rotation keys of layout `key`.
    fn ckks_inner_sum_assign_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: CKKSCtBounds,
        K: GGLWEInfos;

    /// `dst = sum_{j < 2^log_count} Rot(src, j 2^log_batch)`, or the mean of those rotations when `normalize`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the sum's operands, batch, count and normalization, keys and scratch"
    )]
    fn ckks_inner_sum_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// [`Self::ckks_inner_sum_into`] on `dst` in place.
    fn ckks_inner_sum_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// Scratch of [`Self::ckks_replicate_into`] from `a` into `res` with rotation keys of layout `key`.
    fn ckks_replicate_into_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch of [`Self::ckks_replicate_assign`] on `res` with rotation keys of layout `key`.
    fn ckks_replicate_assign_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: CKKSCtBounds,
        K: GGLWEInfos;

    /// `dst = sum_{j < 2^log_count} Rot(src, -j 2^log_batch)`.
    fn ckks_replicate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// [`Self::ckks_replicate_into`] on `dst` in place.
    fn ckks_replicate_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;
}
