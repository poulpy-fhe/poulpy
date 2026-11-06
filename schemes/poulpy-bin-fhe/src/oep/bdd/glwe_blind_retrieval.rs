use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`GLWEBlindRetrieval`](crate::api::GLWEBlindRetrieval).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEBlindRetrievalImpl: Backend + crate::oep::GLWECswapImpl {
    /// Default body: `glwe_blind_retrieval_tmp_bytes_derived`, the selected `glwe_cswap_assign_tmp_bytes` with `res_infos` for both
    /// operands and `k_infos`.
    ///
    /// Scratch bound of [`Self::glwe_blind_retrieval_stateful_assign`] and [`Self::glwe_blind_retrieval_stateful_rev_assign`]; may
    /// report less than the reference, never less than the body takes.
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        crate::oep::derived::bdd::glwe_blind_retrieval_tmp_bytes_derived::<Self, _, _>(module, res_infos, k_infos)
    }
    /// Default body: `glwe_blind_retrieval_stateful_assign_derived`, the selected `glwe_cswap_assign` over the butterfly network from the
    /// most to the least significant selector bit, then `set_noise(None)` on every element, within
    /// [`Self::glwe_blind_retrieval_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_stateful_assign<R, K>(
        module: &Module<Self>,
        bit_rsh: usize,
        bit_mask: usize,
        res: &mut Vec<R>,
        bits: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        K: GetGGSWBit<Self>,
    {
        crate::oep::derived::bdd::glwe_blind_retrieval_stateful_assign_derived::<Self, _, _>(
            module, bit_rsh, bit_mask, res, bits, scratch,
        )
    }
    /// Default body: `glwe_blind_retrieval_stateful_rev_assign_derived`, the selected `glwe_cswap_assign` over the butterfly network in the
    /// reverse order of [`Self::glwe_blind_retrieval_stateful_assign`], then `set_noise(None)` on every element, within
    /// [`Self::glwe_blind_retrieval_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_retrieval_stateful_rev_assign<R, K>(
        module: &Module<Self>,
        bit_rsh: usize,
        bit_mask: usize,
        res: &mut Vec<R>,
        bits: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        K: GetGGSWBit<Self>,
    {
        crate::oep::derived::bdd::glwe_blind_retrieval_stateful_rev_assign_derived::<Self, _, _>(
            module, bit_rsh, bit_mask, res, bits, scratch,
        )
    }
}
