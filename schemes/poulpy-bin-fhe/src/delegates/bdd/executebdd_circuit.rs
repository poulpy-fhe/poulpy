use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BE: Backend> ExecuteBDDCircuit<BE> for Module<BE>
where
    BE: crate::oep::ExecuteBDDCircuitImpl,
{
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_tmp_bytes<R, G>(&self, state_size: usize, res_infos: &R, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
    {
        BE::execute_bdd_circuit_tmp_bytes::<R, G>(self, state_size, res_infos, ggsw_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_tmp_bytes_for<R, G, C>(&self, res_infos: &R, ggsw_infos: &G, circuit: &C) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo,
    {
        BE::execute_bdd_circuit_tmp_bytes_for::<R, G, C>(self, res_infos, ggsw_infos, circuit)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit<C, G, O>(&self, out: &mut [O], inputs: &G, circuit: &C, scratch: &mut ScratchArena<'_, BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE> + GLWEInfos + Send,
    {
        BE::execute_bdd_circuit::<C, G, O>(self, out, inputs, circuit, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_multi_thread<C, G, O>(
        &self,
        threads: usize,
        out: &mut [O],
        inputs: &G,
        circuit: &C,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE> + GLWEInfos + Send,
    {
        BE::execute_bdd_circuit_multi_thread::<C, G, O>(self, threads, out, inputs, circuit, scratch)
    }
}
