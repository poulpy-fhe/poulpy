use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Homomorphic cyclic slot rotation.
///
/// Applies the automorphism `X ↦ X^(5^k mod 2n)` to the RLWE ciphertext,
/// which corresponds to a cyclic shift of the CKKS complex slot vector by
/// `k` positions: slot `j` moves to slot `(j + k) mod (n/2)`.
///
/// Rotation requires a set of automorphism evaluation keys (`keys`).  The
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
/// offset         = max(0, src.effective_k() − dst.max_k())
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

    /// Computes `dst = rotate(src, k)`: shifts all complex slots by `k` positions.
    ///
    /// `k` may be negative (shifts in the opposite direction).  The `keys`
    /// collection must contain the automorphism key for shift amount `k`.
    fn ckks_rotate_into<'s, Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        BE: 's;

    /// Computes `dst = rotate(dst, k)` in-place.  Metadata is unchanged.
    fn ckks_rotate_assign<'s, Dst, H, K>(
        &self,
        dst: &mut Dst,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        BE: 's;
}
