use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Homomorphic cyclic slot rotation.
///
/// Rotates the standard ring's `N/2` complex slots or the invariant ring's
/// `N` real slots. Use [`super::CKKSModuleInfos::ckks_galois_element`] to obtain
/// the evaluation-key identifier. Invariant rotations wrap modulo `N`;
/// negative shifts rotate in the opposite direction.
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
/// For `_assign` variants `offset = 0` and metadata is unchanged.
pub trait CKKSRotateOps<BE: Backend> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Computes `dst = rotate(src, k)`: shifts all slots by `k` positions.
    ///
    /// `k` may be negative (shifts in the opposite direction).  The `keys`
    /// collection must contain the automorphism key for shift amount `k`.
    fn ckks_rotate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = rotate(dst, k)` in-place.  Metadata is unchanged.
    fn ckks_rotate_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        k: i64,
        keys: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
