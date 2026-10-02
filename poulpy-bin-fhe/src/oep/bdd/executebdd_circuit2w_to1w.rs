use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`ExecuteBDDCircuit2WTo1W`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait ExecuteBDDCircuit2WTo1WImpl: Backend {
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w<C, H, T>(
        module: &Module<Self>,
        out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        b: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        key: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<Self>,
        Self: Backend<ZnxWord = i64>,
    {
        crate::oep::derived::bdd::execute_bdd_circuit_2w_to_1w_derived::<Self, _, _, T>(module, out, circuit, a, b, key, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w_tmp_bytes<C, T, R, G, H>(
        module: &Module<Self>,
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
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::bdd::execute_bdd_circuit_2w_to_1w_tmp_bytes_derived::<Self, C, T, R, G, H>(
            module, circuit, res_infos, ggsw_infos, key,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
        module: &Module<Self>,
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
        H: GetAutomorphismKey<Self>;
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w_multi_thread<C, H, T>(
        module: &Module<Self>,
        threads: usize,
        out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        b: &FheUintPrepared<Self::OwnedBuf, T, Self>,
        key: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<Self>,
        Self: Backend<ZnxWord = i64>;
}
