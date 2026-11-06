use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BE: Backend> BDDCircuitExecute1WTo1W<BE> for Module<BE>
where
    BE: crate::oep::BDDCircuitExecute1WTo1WImpl,
{
    fn bdd_circuit_execute_1w_to_1w_tmp_bytes<C, T, R, G, H>(&self, res_infos: &R, ggsw_infos: &G, circuit: &C, key: &H) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::bdd_circuit_execute_1w_to_1w_tmp_bytes::<C, T, R, G, H>(self, res_infos, ggsw_infos, circuit, key)
    }
    fn bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
        &self,
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
        H: GetAutomorphismKey<BE>,
    {
        BE::bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes::<C, T, R, G, H>(
            self, threads, res_infos, ggsw_infos, circuit, key,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_1w_to_1w<C, H, T>(
        &self,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        circuit: &C,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        BE::bdd_circuit_execute_1w_to_1w::<C, H, T>(self, out, a, circuit, key, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_1w_to_1w_multi_thread<C, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        circuit: &C,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        BE::bdd_circuit_execute_1w_to_1w_multi_thread::<C, H, T>(self, threads, out, a, circuit, key, scratch)
    }
}
