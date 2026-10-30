use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`GLWECswap`](crate::api::GLWECswap).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWECswapImpl: Backend {
    /// Scratch bound of [`Self::glwe_cswap_assign`]; may report less than the reference, never less than the body takes.
    #[allow(clippy::too_many_arguments)]
    fn glwe_cswap_assign_tmp_bytes<R, A, S>(module: &Module<Self>, res_a_infos: &R, res_b_infos: &A, s_infos: &S) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        S: GGSWInfos;
    /// Reproduces [`glwe_cswap_assign_reference`](crate::reference::bdd::glwe_cswap_assign_reference) bit for bit, within
    /// [`Self::glwe_cswap_assign_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_cswap_assign<A, B>(
        module: &Module<Self>,
        res_a: &mut A,
        res_b: &mut B,
        s: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        A: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        B: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos;
}
