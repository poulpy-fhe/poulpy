use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, prepared::GLWEAutomorphismKeyPreparedToBackendRef};
use poulpy_core::layouts::{GGLWEPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Homomorphic complex conjugation.
///
/// Applies the automorphism `X ↦ X^(2n−1)` to the Module-LWE ciphertext, which
/// maps every complex slot value `z_j` to its conjugate `z̄_j`.
///
/// Conjugation requires one automorphism evaluation key (the key for the
/// Galois element `2n − 1`).
///
/// Conjugation does not consume homomorphic capacity.
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
pub trait CKKSConjugateOps<BE: Backend> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Computes `dst = conj(src)`: takes the complex conjugate of every slot.
    fn ckks_conjugate_into<'s, Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        BE: 's;

    /// Computes `dst = conj(dst)` in-place.  Metadata is unchanged.
    fn ckks_conjugate_assign<'s, Dst, K>(&self, dst: &mut Dst, key: &K, scratch: &mut ScratchArena<'s, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        BE: 's;
}
