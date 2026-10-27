use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Homomorphic cyclic slot rotation.
///
/// Rotates the standard ring's `N/2` complex slots or the invariant ring's
/// `N` real slots. Use [`super::CKKSModuleInfos::ckks_galois_element`] to obtain
/// the evaluation-key identifier. Shifts wrap modulo the slot count, so a
/// negative shift rotates backwards.
///
/// Non-identity rotation requires a set of automorphism evaluation keys (`keys`).  The
/// key collection `H` must contain the key for the Galois element
/// corresponding to shift `k`.
///
/// Rotation does not consume homomorphic capacity.
///
/// # Metadata
///
/// For `_into` variants:
///
/// ```text
/// offset         = max(0, src.k() − dst.k())
///
/// log_delta_out  = src.log_delta
/// log_budget_out = src.log_budget − offset
/// ```
///
/// For `_assign` variants `offset = 0` and CKKS layout and scale metadata is unchanged.
/// Component noise metadata is cleared.
///
/// # Slot layouts
///
/// A [`SlotLayout::Replicated`](crate::SlotLayout::Replicated) ciphertext
/// rotates only by multiples of its period `R`, which move the batch by
/// `k / R` values, and its key switch is rounded with shaping. A narrower
/// destination requires key coverage of `dst.k() + λ`, where `λ` is the
/// source layout's shaping depth at `src.log_delta()`. Otherwise the key
/// covers `src.k()`; a full layout requires `min(src.k(), dst.k())`.
pub trait CKKSRotateOps<BE: Backend> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Computes `dst = rotate(src, k)`: shifts all slots by `k` positions.
    ///
    /// `k` may be negative (shifts in the opposite direction).  The `keys`
    /// collection must contain the automorphism key for shift amount `k`.
    ///
    /// # Errors
    ///
    /// A shift of zero slots returns the errors of [`CKKSCopyOps::ckks_copy`](crate::api::CKKSCopyOps::ckks_copy).
    /// Otherwise `MissingAutomorphismKey` when `keys` has no key for the shift covering
    /// the consumed input width, including the shaping depth for a replicated layout, and `InsufficientHomomorphicCapacity` when `dst` is narrower
    /// than `src` by more than its budget. `dst` is untouched on `Err`.
    fn ckks_rotate_into<Dst, Src, H>(
        &self,
        k: i64,
        dst: &mut Dst,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = rotate(dst, k)` in-place, preserving CKKS layout and scale metadata.
    /// Component noise metadata is cleared.
    ///
    /// # Errors
    ///
    /// `MissingAutomorphismKey` when `keys` has no key for the shift covering `dst.k()`. `dst` is untouched on `Err`.
    fn ckks_rotate_assign<Dst, H>(&self, k: i64, dst: &mut Dst, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
