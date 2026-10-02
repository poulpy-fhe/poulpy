use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_bdd_circuit_2w_to_1w_derived<BE, C, H, T>(
    module: &Module<BE>,
    out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
    circuit: &C,
    a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
    b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
    key: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    T: UnsignedInteger,
    C: GetBitCircuitInfo,
    H: GetAutomorphismKey<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: ExecuteBDDCircuit2WTo1W<BE>,
{
    module.execute_bdd_circuit_2w_to_1w_multi_thread(1, out, circuit, a, b, key, scratch);
}

/// The serial wrapper uses the selected one-worker execution budget.
pub(crate) fn execute_bdd_circuit_2w_to_1w_tmp_bytes_derived<BE, C, T, R, G, H>(
    module: &Module<BE>,
    circuit: &C,
    res_infos: &R,
    ggsw_infos: &G,
    key: &H,
) -> usize
where
    BE: Backend,
    C: GetBitCircuitInfo,
    T: UnsignedInteger,
    R: GLWEInfos,
    G: GGSWInfos,
    H: GetAutomorphismKey<BE>,
    Module<BE>: ExecuteBDDCircuit2WTo1W<BE>,
{
    module.execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes::<C, T, R, G, H>(1, circuit, res_infos, ggsw_infos, key)
}
