use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Homomorphic conditional swap of two GLWE ciphertexts.
///
/// Given a GGSW ciphertext `s` encrypting a bit `b ∈ {0, 1}`, swaps the
/// contents of `res_a` and `res_b` if `b = 1`, and leaves them unchanged if
/// `b = 0`.  The operation is equivalent to:
///
/// ```text
/// (new_res_a, new_res_b) = if b == 1 { (res_b, res_a) } else { (res_a, res_b) }
/// ```
///
/// but is performed entirely in the ciphertext domain.  Used by
/// `GLWEBlindRetrieval` to implement oblivious array access.
pub trait Cswap<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns the minimum scratch-space size in bytes required by [`cswap`][Self::cswap].
    fn cswap_tmp_bytes<R, A, S>(&self, res_a_infos: &R, res_b_infos: &A, s_infos: &S) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        S: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn cswap<'k, A, B>(
        &self,
        res_a: &mut A,
        res_b: &mut B,
        s: &GGSWPreparedBackendRef<'k, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        BE: 'k;
}
