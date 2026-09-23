use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`ExecuteBDDCircuit::execute_bdd_circuit_tmp_bytes`].
pub fn execute_bdd_circuit_tmp_bytes_reference<BE: Backend<ZnxWord = i64>, R, G>(
    module: &Module<BE>,
    res_infos: &R,
    state_size: usize,
    ggsw_infos: &G,
) -> usize
where
    R: GLWEInfos,
    G: GGSWInfos,
    Module<BE>: GLWEBytesOf<BE>
        + Cmux<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + VecZnxAddScalarAssign<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + Sync,
{
    2 * state_size * module.glwe_bytes_of_from_infos(res_infos)
        + module
            .cmux_tmp_bytes(res_infos, res_infos, ggsw_infos)
            .max(module.glwe_copy_tmp_bytes(res_infos, res_infos))
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`ExecuteBDDCircuit::execute_bdd_circuit_multi_thread`].
pub fn execute_bdd_circuit_multi_thread_reference<BE: Backend<ZnxWord = i64>, C, G, O>(
    module: &Module<BE>,
    threads: usize,
    out: &mut [O],
    inputs: &G,
    circuit: &C,
    scratch: &mut ScratchArena<'_, BE>,
) where
    G: GetGGSWBit<BE> + BitSize,
    C: GetBitCircuitInfo,
    O: GLWEToBackendMut<BE> + GLWEInfos + Send,
    Module<BE>: GLWEBytesOf<BE>
        + Cmux<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + VecZnxAddScalarAssign<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + Sync,
{
    crate::bdd_arithmetic::eval::execute_bdd_circuit_reference(module, threads, out, inputs, circuit, scratch);
}
