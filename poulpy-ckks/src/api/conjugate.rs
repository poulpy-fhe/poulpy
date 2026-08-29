use crate::CKKSResult as Result;
use poulpy_core::layouts::GGLWEInfos;
use poulpy_core::layouts::GetAutomorphismKey;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
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
/// offset         = max(0, src.k() − dst.k())
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

    /// Computes `dst = conj(src)`, the automorphism of Galois element `-1`.
    fn ckks_conjugate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        self.ckks_conjugate_rotate_into(dst, src, 0, keys, scratch)
    }

    /// Computes `dst = conj(dst)` in-place. Metadata is unchanged.
    fn ckks_conjugate_assign<Dst, H>(&self, dst: &mut Dst, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// Computes `dst = conj(rot_k(src))`: conjugation composed with a rotation
    /// of `k` slots, as one key switch under the Galois element
    /// `-galois_element(k)`. PaCo's psi tail is the caller; `k = 0` is plain
    /// conjugation, spelled [`Self::ckks_conjugate_into`].
    fn ckks_conjugate_rotate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;
}
