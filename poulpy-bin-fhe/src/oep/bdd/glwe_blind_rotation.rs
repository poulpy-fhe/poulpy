use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`GLWEBlindRotation`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait GLWEBlindRotationImpl: Backend {
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;
    /// Scratch required for the in-place rotation.
    fn glwe_blind_rotation_assign_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_assign<R, K>(
        module: &Module<Self>,
        res: &mut R,
        value: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64>;
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self>,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64>;
}
