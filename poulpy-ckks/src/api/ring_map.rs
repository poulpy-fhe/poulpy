use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Data, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::CKKSCiphertext};

/// Maps between ciphertexts of the conjugate-invariant module of degree `N` and
/// standard ciphertexts of degree `2N`.
///
/// Not implemented on standard backends. Unfolded ciphertexts decrypt under the
/// unfolded secret ([`GLWESecretCIUnfold`](poulpy_core::layouts::GLWESecretCIUnfold));
/// switching between it and a standard secret is an ordinary key switch of the
/// standard module.
///
/// # Metadata
///
/// ```text
/// unfold: log_delta_out = src.log_delta
/// fold:   log_delta_out = src.log_delta + 1   (dst holds 2·Re(m))
/// ```
///
/// Both keep `src.k()` and the sparsity, and set the slot kind to `SlotsKind::Real`.
/// The unfolded digits are exact but not renormalized: key switching normalizes them.
pub trait CKKSCIRingMapOps<BE: Backend> {
    /// Unfolds `src` into `dst`, at the radix of `src`.
    fn ckks_ci_unfold<D, Src>(&self, dst: &mut CKKSCiphertext<D, BE::ZnxWord, Standard>, src: &Src) -> Result<()>
    where
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_ci_fold_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    /// Folds `src`, which decrypts under the unfolded secret, into `dst`.
    fn ckks_ci_fold<Dst, D>(
        &self,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, BE::ZnxWord, Standard>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>;
}
