use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend-level BDD circuit evaluator.
///
/// Evaluates a multi-output BDD circuit on a set of encrypted input bits,
/// producing one GLWE ciphertext per output bit.  The circuit is represented as
/// a sequence of [`Node`] entries arranged in BDD levels; each level is evaluated
/// using [`Cmux`] gates.
pub trait ExecuteBDDCircuit<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns the minimum scratch-space size in bytes required by a single
    /// thread of BDD circuit evaluation.
    ///
    /// `state_size` is the maximum number of live intermediate GLWE values
    /// (i.e. `max_inter_state` from [`BitCircuit`]).
    /// Single-threaded BDD circuit evaluation.
    ///
    /// Evaluates `circuit` on `inputs`, writing one GLWE ciphertext per output
    /// bit into `out[0..circuit.output_size()]`.  Elements beyond
    /// `output_size` are zeroed.
    ///
    /// Delegates to [`execute_bdd_circuit_multi_thread`][Self::execute_bdd_circuit_multi_thread]
    /// with `threads = 1`.
    fn execute_bdd_circuit_tmp_bytes<R, G>(&self, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_tmp_bytes_for<R, G, C>(&self, res_infos: &R, circuit: &C, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo;
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit<C, G, O>(&self, out: &mut [O], inputs: &G, circuit: &C, scratch: &mut ScratchArena<'_, BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE> + GLWEInfos + Send;
    #[allow(clippy::too_many_arguments)]
    /// Multi-threaded BDD circuit evaluation.
    ///
    /// Partitions the output bits across at most `threads` workers using the
    /// selected backend's task executor. Each worker receives a dedicated slice of the
    /// scratch arena of size
    /// [`execute_bdd_circuit_tmp_bytes`][Self::execute_bdd_circuit_tmp_bytes].
    ///
    /// # Panics
    ///
    /// Panics if the arena cannot provide one scratch slice per active worker.
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
        O: GLWEToBackendMut<BE> + GLWEInfos + Send;
}
