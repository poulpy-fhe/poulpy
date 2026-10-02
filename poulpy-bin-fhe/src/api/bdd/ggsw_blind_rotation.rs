use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Extends [`GLWEBlindRotation`] to lift blind rotation to GGSW matrices and
/// to construct a GGSW from a scalar test-vector via blind rotation.
///
/// A GGSW matrix is a `(dnum × (rank+1))` array of GLWE ciphertexts.  The two
/// methods in this trait apply [`GLWEBlindRotation`] row-by-row:
///
/// - `ggsw_blind_rotation`: rotates each GLWE row of an existing GGSW by the
///   encrypted exponent derived from `fhe_uint`.
/// - `scalar_to_ggsw_blind_rotation`: constructs a fresh GGSW by first placing
///   the scalar test-vector into each row of a temporary GLWE and then rotating.
pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns the minimum scratch-space size in bytes required by
    /// [`ggsw_blind_rotation`][Self::ggsw_blind_rotation].
    fn ggsw_to_ggsw_blind_rotation_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;
    /// Scratch required for rotating each GGSW row in place.
    fn ggsw_blind_rotation_assign_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    /// res <- res * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    /// res <- a * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_blind_rotation_assign<R, K>(
        &self,
        res: &mut R,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64>;
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation<R, A, K>(
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
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64>;
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        test_vector: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: ScalarZnxToBackendRef<BE>,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64>;
}
