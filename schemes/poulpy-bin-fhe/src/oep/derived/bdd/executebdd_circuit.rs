use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_bdd_circuit_tmp_bytes_for_derived<BE: Backend, R, G, C>(
    module: &Module<BE>,
    res_infos: &R,
    ggsw_infos: &G,
    circuit: &C,
) -> usize
where
    R: GLWEInfos,
    G: GGSWInfos,
    C: GetBitCircuitInfo,
    Module<BE>: ExecuteBDDCircuit<BE>,
{
    module.execute_bdd_circuit_tmp_bytes(circuit.max_state_size(), res_infos, ggsw_infos)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_bdd_circuit_derived<BE: Backend, C, G, O>(
    module: &Module<BE>,
    out: &mut [O],
    inputs: &G,
    circuit: &C,
    scratch: &mut ScratchArena<'_, BE>,
) where
    G: GetGGSWBit<BE> + BitSize,
    C: GetBitCircuitInfo,
    O: GLWEToBackendMut<BE> + GLWEInfos + Send,
    Module<BE>: ExecuteBDDCircuit<BE>,
{
    module.execute_bdd_circuit_multi_thread(1, out, inputs, circuit, scratch);
}
