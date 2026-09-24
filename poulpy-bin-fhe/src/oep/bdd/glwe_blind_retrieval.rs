use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`GLWEBlindRetrieval`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait GLWEBlindRetrievalImpl: Backend + crate::oep::CswapImpl {
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        crate::oep::derived::bdd::glwe_blind_retrieval_tmp_bytes_derived::<Self, _, _>(module, res_infos, k_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_statefull<R, K>(
        module: &Module<Self>,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        K: GetGGSWBit<Self>,
    {
        crate::oep::derived::bdd::glwe_blind_retrieval_statefull_derived::<Self, _, _>(
            module, res, bits, bit_rsh, bit_mask, scratch,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_statefull_rev<R, K>(
        module: &Module<Self>,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        K: GetGGSWBit<Self>,
    {
        crate::oep::derived::bdd::glwe_blind_retrieval_statefull_rev_derived::<Self, _, _>(
            module, res, bits, bit_rsh, bit_mask, scratch,
        )
    }
}
