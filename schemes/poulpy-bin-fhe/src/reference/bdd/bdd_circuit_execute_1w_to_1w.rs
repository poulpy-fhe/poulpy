use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};

/// Workspace for the retained output bits and the larger of evaluation or packing.
pub fn bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes_reference<BE, C, T, R, G, H>(
    module: &Module<BE>,
    threads: usize,
    res_infos: &R,
    ggsw_infos: &G,
    circuit: &C,
    key: &H,
) -> usize
where
    BE: Backend<ZnxWord = i64>,
    C: GetBitCircuitInfo,
    T: UnsignedInteger,
    R: GLWEInfos,
    G: GGSWInfos,
    H: GetAutomorphismKey<BE>,
    Module<BE>: BDDCircuitExecute<BE> + GLWEPack<BE>,
{
    let outputs = T::BITS as usize * module.glwe_bytes_of_from_infos(res_infos);
    let per_worker = poulpy_hal::execution::worker_scratch_bytes::<BE>(module.bdd_circuit_execute_tmp_bytes(
        circuit.max_state_size(),
        res_infos,
        ggsw_infos,
    ));
    let workers = poulpy_hal::execution::worker_count::<BE::TaskExecutor>(threads, circuit.output_size());
    let atk = key.get_automorphism_key(-1, res_infos.k()).expect("missing packing key");
    outputs
        + workers
            .checked_mul(per_worker)
            .expect("BDD scratch size overflow")
            .max(module.glwe_pack_tmp_bytes(res_infos, res_infos, &atk))
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDCircuitExecute1WTo1W::bdd_circuit_execute_1w_to_1w_multi_thread`].
pub fn bdd_circuit_execute_1w_to_1w_multi_thread_reference<BE, C, H, T>(
    module: &Module<BE>,
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
    Module<BE>: GLWEBytesOf<BE>,
    Module<BE>: Sized + ModuleLogN + BDDCircuitExecute<BE> + GLWEPack<BE> + GLWECopy<BE>,
{
    assert!(
        scratch.available()
            >= bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes_reference::<BE, C, T, _, _, _>(
                module,
                threads,
                out,
                a.get_bit(0),
                circuit,
                key
            ),
        "bdd_circuit_execute_1w_to_1w: insufficient scratch"
    );
    let (mut out_bits, mut scratch_1) = scratch.borrow().take_glwe_slice_scratch(T::BITS as usize, out);

    // Evaluates out[i] = circuit[i](a, b)
    module.bdd_circuit_execute_multi_thread(threads, &mut out_bits, a, circuit, &mut scratch_1);

    // Repacks the bits
    out.pack(module, out_bits, key, &mut scratch_1);
}

/// Reference bodies of the required [`BDDCircuitExecute1WTo1WImpl`](crate::oep::BDDCircuitExecute1WTo1WImpl) methods,
/// which [`impl_bdd_circuit_execute_1w_to_1w_reference_full!`](crate::impl_bdd_circuit_execute_1w_to_1w_reference_full)
/// forwards to.
pub trait BDDCircuitExecute1WTo1WReference<BE: Backend> {
    fn bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes_reference<C, T, R, G, H>(
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

    fn bdd_circuit_execute_1w_to_1w_multi_thread_reference<C, H, T>(
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
        H: GetAutomorphismKey<BE>;
}

impl<BE: Backend<ZnxWord = i64>> BDDCircuitExecute1WTo1WReference<BE> for Module<BE>
where
    Module<BE>: GLWEBytesOf<BE> + ModuleLogN + BDDCircuitExecute<BE> + GLWEPack<BE> + GLWECopy<BE>,
{
    fn bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes_reference<C, T, R, G, H>(
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
        bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes_reference::<BE, C, T, R, G, H>(
            self, threads, res_infos, ggsw_infos, circuit, key,
        )
    }

    fn bdd_circuit_execute_1w_to_1w_multi_thread_reference<C, H, T>(
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
    {
        bdd_circuit_execute_1w_to_1w_multi_thread_reference(self, threads, out, a, circuit, key, scratch)
    }
}
