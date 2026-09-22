use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Homomorphic rotation of a GLWE ciphertext by an encrypted exponent.
///
/// Given a GLWE ciphertext `a` and a set of GGSW ciphertexts encoding the bits
/// of an integer `k`, computes:
///
/// ```text
/// res = a * X^{sign * ((k >> bit_rsh) % 2^bit_mask) << bit_lsh}
/// ```
///
/// where `sign` controls whether the rotation is positive or negative.
/// The operation is performed using `bit_mask` successive CMux gates, one per
/// bit of the shift amount.
pub trait GLWEBlindRotation<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns the minimum scratch-space size in bytes required by
    /// [`glwe_blind_rotation`][Self::glwe_blind_rotation].
    /// res <- a * X^{sign * ((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn glwe_blind_rotation_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;
    /// Scratch required for the in-place rotation.
    fn glwe_blind_rotation_assign_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_assign<R, K>(
        &self,
        res: &mut R,
        value: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static;
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static;
}
