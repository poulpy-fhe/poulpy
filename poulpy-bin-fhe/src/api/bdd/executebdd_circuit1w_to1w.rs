use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend-level executor for single-input BDD circuits (`Z → Z`).
///
/// Evaluates a BDD circuit that reads one encrypted integer and produces
/// one encrypted integer.  After evaluating the per-bit BDD levels, the
/// output bits are repacked into a single [`FheUint`] polynomial via
/// [`GLWEPacking`](poulpy_core::GLWEPacking).
pub trait ExecuteBDDCircuit1WTo1W<BE: Backend> {
    /// Workspace for this evaluator and output packing.
    fn execute_bdd_circuit_1w_to_1w_tmp_bytes<C, T, R, G, H>(&self, circuit: &C, res_infos: &R, ggsw_infos: &G, key: &H) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<BE>;
    /// Workspace for this evaluator and output packing.
    fn execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
        &self,
        threads: usize,
        circuit: &C,
        res_infos: &R,
        ggsw_infos: &G,
        key: &H,
    ) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<BE>;
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_1w_to_1w<C, H, T>(
        &self,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>;
    #[allow(clippy::too_many_arguments)]
    /// Operations Z x Z -> Z
    fn execute_bdd_circuit_1w_to_1w_multi_thread<C, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>;
}
