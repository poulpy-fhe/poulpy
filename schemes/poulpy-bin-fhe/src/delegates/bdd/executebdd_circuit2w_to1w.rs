use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BE: Backend> ExecuteBDDCircuit2WTo1W<BE> for Module<BE>
where
    BE: crate::oep::ExecuteBDDCircuit2WTo1WImpl,
{
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w<C, H, T>(
        &self,
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
    {
        BE::execute_bdd_circuit_2w_to_1w::<C, H, T>(self, out, circuit, a, b, key, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w_tmp_bytes<C, T, R, G, H>(&self, circuit: &C, res_infos: &R, ggsw_infos: &G, key: &H) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::execute_bdd_circuit_2w_to_1w_tmp_bytes::<C, T, R, G, H>(self, circuit, res_infos, ggsw_infos, key)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
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
        H: GetAutomorphismKey<BE>,
    {
        BE::execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes::<C, T, R, G, H>(
            self, threads, circuit, res_infos, ggsw_infos, key,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_bdd_circuit_2w_to_1w_multi_thread<C, H, T>(
        &self,
        threads: usize,
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
    {
        BE::execute_bdd_circuit_2w_to_1w_multi_thread::<C, H, T>(self, threads, out, circuit, a, b, key, scratch)
    }
}
