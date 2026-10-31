use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`BDDCircuitExecute2WTo1W`](crate::api::BDDCircuitExecute2WTo1W), on the standard
/// ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BDDCircuitExecute2WTo1WImpl: Backend<Ring = Standard> {
    /// Default body: `bdd_circuit_execute_2w_to_1w_derived`, the selected
    /// [`Self::bdd_circuit_execute_2w_to_1w_multi_thread`] with `threads = 1`, within
    /// [`Self::bdd_circuit_execute_2w_to_1w_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_2w_to_1w<C, H, T>(
        module: &Module<Self>,
        out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
        a: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        b: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        circuit: &C,
        key: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<Self>,
        Self: Backend<ZnxWord = i64>,
    {
        crate::oep::derived::bdd::bdd_circuit_execute_2w_to_1w_derived::<Self, _, _, T>(module, out, a, b, circuit, key, scratch)
    }
    /// Default body: `bdd_circuit_execute_2w_to_1w_tmp_bytes_derived`, the selected
    /// [`Self::bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes`] with `threads = 1`.
    ///
    /// Scratch bound of [`Self::bdd_circuit_execute_2w_to_1w`]; may report less than the reference, never less than the
    /// body takes.
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_2w_to_1w_tmp_bytes<C, T, R, G, H>(
        module: &Module<Self>,
        res_infos: &R,
        ggsw_infos: &G,
        circuit: &C,
        key: &H,
    ) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::bdd::bdd_circuit_execute_2w_to_1w_tmp_bytes_derived::<Self, C, T, R, G, H>(
            module, res_infos, ggsw_infos, circuit, key,
        )
    }
    /// Scratch bound of [`Self::bdd_circuit_execute_2w_to_1w_multi_thread`] at `threads`; may report less than the
    /// reference, never less than the body takes. The bound covers one rounded worker bound per active worker.
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
        module: &Module<Self>,
        threads: usize,
        res_infos: &R,
        ggsw_infos: &G,
        circuit: &C,
        key: &H,
    ) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<Self>;
    /// Reproduces
    /// [`bdd_circuit_execute_2w_to_1w_multi_thread_reference`](crate::reference::bdd::bdd_circuit_execute_2w_to_1w_multi_thread_reference)
    /// bit for bit, within [`Self::bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes`]; free in kernels, scheduling and
    /// scratch use, output bit-identical for every `threads` and executor.
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_2w_to_1w_multi_thread<C, H, T>(
        module: &Module<Self>,
        threads: usize,
        out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
        a: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        b: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        circuit: &C,
        key: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<Self>,
        Self: Backend<ZnxWord = i64>;
}
