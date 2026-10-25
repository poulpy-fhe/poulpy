use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`ExecuteBDDCircuit`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait ExecuteBDDCircuitImpl: Backend {
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_tmp_bytes<R, G>(module: &Module<Self>, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_tmp_bytes_for<R, G, C>(module: &Module<Self>, res_infos: &R, circuit: &C, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo,
    {
        crate::oep::derived::bdd::execute_bdd_circuit_tmp_bytes_for_derived::<Self, _, _, _>(
            module, res_infos, circuit, ggsw_infos,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit<C, G, O>(
        module: &Module<Self>,
        out: &mut [O],
        inputs: &G,
        circuit: &C,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        G: GetGGSWBit<Self> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<Self> + GLWEInfos + Send,
    {
        crate::oep::derived::bdd::execute_bdd_circuit_derived::<Self, _, _, _>(module, out, inputs, circuit, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_multi_thread<C, G, O>(
        module: &Module<Self>,
        threads: usize,
        out: &mut [O],
        inputs: &G,
        circuit: &C,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        G: GetGGSWBit<Self> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<Self> + GLWEInfos + Send;
}
