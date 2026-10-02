use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`Cswap`](crate::api::Cswap).
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait CswapImpl: Backend {
    #[allow(clippy::too_many_arguments)]
    fn cswap_tmp_bytes<R, A, S>(module: &Module<Self>, res_a_infos: &R, res_b_infos: &A, s_infos: &S) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        S: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn cswap<A, B>(
        module: &Module<Self>,
        res_a: &mut A,
        res_b: &mut B,
        s: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        A: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        B: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos;
}
