use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Homomorphic multiplexer (CMux) operation on GLWE ciphertexts.
///
/// Given two GLWE ciphertexts `t` (true branch) and `f` (false branch) and a
/// GGSW ciphertext `s` encrypting a selector bit `b`, computes:
///
/// ```text
/// res = (t - f) · s + f
/// ```
///
/// so that `res` encrypts `t` when `b = 1` and `f` when `b = 0`.  This is the
/// fundamental gate used throughout BDD circuit evaluation.
pub trait Cmux<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns the minimum scratch-space size in bytes required by [`cmux`][Self::cmux].
    fn cmux_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn cmux<'k, R, T, F>(
        &self,
        res: &mut R,
        t: &T,
        f: &F,
        s: &GGSWPreparedBackendRef<'k, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        T: GLWEToBackendRef<BE>,
        F: GLWEToBackendRef<BE>,
        BE: 'k;
    #[allow(clippy::too_many_arguments)]
    fn cmux_assign_neg<'k, R, A>(
        &self,
        res: &mut R,
        a: &A,
        s: &GGSWPreparedBackendRef<'k, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        BE: 'k;
    #[allow(clippy::too_many_arguments)]
    fn cmux_assign<'k, R, A>(&self, res: &mut R, a: &A, s: &GGSWPreparedBackendRef<'k, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        BE: 'k;
}
