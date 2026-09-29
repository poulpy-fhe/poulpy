use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};

/// Unfolding of a GLWE of the conjugate-invariant module of degree `N` into the
/// standard ring of degree `2N`.
///
/// The result decrypts under the unfolded conjugate-invariant secret
/// ([`GLWESecretCIUnfold`](crate::layouts::GLWESecretCIUnfold)).
pub trait GLWECIUnfold<BE: Backend> {
    /// Writes `a_0 + Σ a_i (X^i + X^-i)` of every column of `a` into `res`, at
    /// the radix of `a`. The digits are exact but not renormalized, so `res` is
    /// flagged non-canonical.
    fn glwe_ci_unfold<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Folding of a standard GLWE of degree `2N` onto the conjugate-invariant module
/// of degree `N`.
///
/// `a` must decrypt under the unfolded conjugate-invariant secret; the result
/// decrypts under the conjugate-invariant secret to `m(X) + m(X^-1)`.
pub trait GLWECIFold<BE: Backend> {
    fn glwe_ci_fold_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    /// Writes the compressed `a(X) + a(X^-1)` of every column of `a` into `res`,
    /// normalized to the layout of `res`.
    fn glwe_ci_fold<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}
