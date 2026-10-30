use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};
use std::marker::PhantomData;

struct FheUintHelper<'a, T: UnsignedInteger, BE: Backend<ZnxWord = i64>> {
    data: Vec<&'a dyn GetGGSWBit<BE>>,
    _phantom: PhantomData<T>,
}

impl<'a, T: UnsignedInteger, BE: Backend<ZnxWord = i64>> GetGGSWBit<BE> for FheUintHelper<'a, T, BE> {
    fn get_bit(&self, bit: usize) -> &GGSWPrepared<BE::OwnedBuf, BE> {
        let lo: usize = bit % T::BITS as usize;
        let hi: usize = bit / T::BITS as usize;
        self.data[hi].get_bit(lo)
    }
}

impl<'a, T: UnsignedInteger, BE: Backend<ZnxWord = i64>> BitSize for FheUintHelper<'a, T, BE> {
    fn bit_size(&self) -> usize {
        T::BITS as usize * self.data.len()
    }
}

#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDCircuitExecute2WTo1W::bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes`].
pub fn bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes_reference<BE: Backend<ZnxWord = i64>, C, T, R, G, H>(
    module: &Module<BE>,
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
    Module<BE>: Sized + GLWEBytesOf<BE> + ModuleLogN + BDDCircuitExecute<BE> + GLWEPack<BE> + GLWECopy<BE>,
{
    let glwe_slot_bytes = T::BITS as usize * module.glwe_bytes_of_from_infos(res_infos);
    let bdd_per_thread = poulpy_hal::execution::worker_scratch_bytes::<BE>(module.bdd_circuit_execute_tmp_bytes(
        circuit.max_state_size(),
        res_infos,
        ggsw_infos,
    ));
    let atk = key.get_automorphism_key(-1, res_infos.k()).unwrap_or_else(|e| panic!("{e}"));
    let pack_bytes = module.glwe_pack_tmp_bytes(res_infos, res_infos, &atk);
    let workers = poulpy_hal::execution::worker_count::<BE::TaskExecutor>(threads, circuit.output_size());
    glwe_slot_bytes
        + workers
            .checked_mul(bdd_per_thread)
            .expect("BDD scratch size overflow")
            .max(pack_bytes)
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDCircuitExecute2WTo1W::bdd_circuit_execute_2w_to_1w_multi_thread`].
pub fn bdd_circuit_execute_2w_to_1w_multi_thread_reference<BE, C, H, T>(
    module: &Module<BE>,
    threads: usize,
    out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
    a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
    b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
    circuit: &C,
    key: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    T: UnsignedInteger,
    C: GetBitCircuitInfo,
    H: GetAutomorphismKey<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: Sized + GLWEBytesOf<BE> + ModuleLogN + BDDCircuitExecute<BE> + GLWEPack<BE> + GLWECopy<BE>,
{
    // Collects inputs into a single array
    let inputs: Vec<&dyn GetGGSWBit<BE>> = [a as &dyn GetGGSWBit<BE>, b as &dyn GetGGSWBit<BE>].to_vec();
    let helper: FheUintHelper<'_, T, BE> = FheUintHelper {
        data: inputs,
        _phantom: PhantomData,
    };

    assert!(
        scratch.available()
            >= bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes_reference::<BE, C, T, _, _, _>(
                module,
                threads,
                out,
                a.get_bit(0),
                circuit,
                key
            ),
        "bdd_circuit_execute_2w_to_1w: insufficient scratch"
    );
    let (mut out_bits, mut scratch_1) = scratch.borrow().take_glwe_slice_scratch(T::BITS as usize, out);

    // Evaluates out[i] = circuit[i](a, b)
    module.bdd_circuit_execute_multi_thread(threads, &mut out_bits, &helper, circuit, &mut scratch_1);

    // Repacks the bits
    out.pack(module, out_bits, key, &mut scratch_1);
}

/// Reference bodies of the required [`BDDCircuitExecute2WTo1WImpl`](crate::oep::BDDCircuitExecute2WTo1WImpl) methods,
/// which [`impl_bdd_circuit_execute_2w_to_1w_reference_full!`](crate::impl_bdd_circuit_execute_2w_to_1w_reference_full)
/// forwards to.
pub trait BDDCircuitExecute2WTo1WReference<BE: Backend> {
    fn bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes_reference<C, T, R, G, H>(
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
        H: GetAutomorphismKey<BE>;

    #[allow(clippy::too_many_arguments)]
    fn bdd_circuit_execute_2w_to_1w_multi_thread_reference<C, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        circuit: &C,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>;
}

impl<BE: Backend<ZnxWord = i64>> BDDCircuitExecute2WTo1WReference<BE> for Module<BE>
where
    Module<BE>: GLWEBytesOf<BE> + ModuleLogN + BDDCircuitExecute<BE> + GLWEPack<BE> + GLWECopy<BE>,
{
    fn bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes_reference<C, T, R, G, H>(
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
        bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes_reference::<BE, C, T, R, G, H>(
            self, threads, res_infos, ggsw_infos, circuit, key,
        )
    }

    fn bdd_circuit_execute_2w_to_1w_multi_thread_reference<C, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        circuit: &C,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
    {
        bdd_circuit_execute_2w_to_1w_multi_thread_reference(self, threads, out, a, b, circuit, key, scratch)
    }
}
