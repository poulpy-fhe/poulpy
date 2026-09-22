use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`GGSWBlindRotation`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait GGSWBlindRotationImpl<T: UnsignedInteger>: Backend + crate::oep::GLWEBlindRotationImpl {
    #[allow(clippy::too_many_arguments)]
    fn ggsw_to_ggsw_blind_rotation_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos,
    {
        crate::oep::derived::bdd::ggsw_to_ggsw_blind_rotation_tmp_bytes_derived::<Self, _, _, _>(
            module, res_infos, a_infos, k_infos,
        )
    }
    fn ggsw_blind_rotation_assign_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        crate::oep::derived::bdd::ggsw_blind_rotation_assign_tmp_bytes_derived::<Self, _, _>(module, res_infos, k_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation_assign<R, K>(
        module: &Module<Self>,
        res: &mut R,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64> + 'static,
    {
        crate::oep::derived::bdd::ggsw_blind_rotation_assign_derived::<Self, _, _>(
            module, res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation<R, A, K>(
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
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWAtViewRef<Self> + GGSWInfos,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64> + 'static,
    {
        crate::oep::derived::bdd::ggsw_blind_rotation_derived::<Self, _, _, _>(
            module, res, a, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        test_vector: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: ScalarZnxToBackendRef<Self>,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64> + 'static;
}
