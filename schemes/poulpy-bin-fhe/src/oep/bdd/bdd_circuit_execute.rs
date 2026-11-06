use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`BDDCircuitExecute`](crate::api::BDDCircuitExecute).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BDDCircuitExecuteImpl: Backend {
    /// Scratch bound of one worker of [`Self::bdd_circuit_execute_multi_thread`] for a circuit with at most `state_size`
    /// live intermediate values, rounded by `worker_scratch_bytes`; may report less than the reference, never less than
    /// the body takes.
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_tmp_bytes<R, G>(module: &Module<Self>, state_size: usize, res_infos: &R, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos;
    /// Default body: `bdd_circuit_execute_circuit_tmp_bytes_derived`, the selected [`Self::bdd_circuit_execute_tmp_bytes`] at
    /// `circuit.max_state_size()`.
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_circuit_tmp_bytes<R, G, C>(module: &Module<Self>, res_infos: &R, ggsw_infos: &G, circuit: &C) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo,
    {
        crate::oep::derived::bdd::bdd_circuit_execute_circuit_tmp_bytes_derived::<Self, _, _, _>(
            module, res_infos, ggsw_infos, circuit,
        )
    }
    /// Default body: `bdd_circuit_execute_derived`, the selected [`Self::bdd_circuit_execute_multi_thread`] with `threads
    /// = 1`.
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute<C, G, O>(
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
        crate::oep::derived::bdd::bdd_circuit_execute_derived::<Self, _, _, _>(module, out, inputs, circuit, scratch)
    }
    /// Reproduces
    /// [`bdd_circuit_execute_multi_thread_reference`](crate::reference::bdd::bdd_circuit_execute_multi_thread_reference)
    /// bit for bit, within [`Self::bdd_circuit_execute_tmp_bytes`]; free in kernels, scheduling and scratch use within
    /// one worker bound per active worker, output bit-identical for every `threads` and executor.
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_multi_thread<C, G, O>(
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
