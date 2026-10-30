use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};

/// Workspace for the retained output bits and the larger of evaluation or packing.
pub fn execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes_reference<BE, C, T, R, G, H>(
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
    Module<BE>: ExecuteBDDCircuit<BE> + GLWEPack<BE>,
{
    let outputs = T::BITS as usize * module.glwe_bytes_of_from_infos(res_infos);
    let per_worker = poulpy_hal::execution::worker_scratch_bytes::<BE>(module.execute_bdd_circuit_tmp_bytes(
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
/// Independently callable canonical implementation of [`ExecuteBDDCircuit1WTo1W::execute_bdd_circuit_1w_to_1w_multi_thread`].
pub fn execute_bdd_circuit_1w_to_1w_multi_thread_reference<BE, C, H, T>(
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
    Module<BE>: Sized + ModuleLogN + ExecuteBDDCircuit<BE> + GLWEPack<BE> + GLWECopy<BE>,
{
    assert!(
        scratch.available()
            >= execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes_reference::<BE, C, T, _, _, _>(
                module,
                threads,
                out,
                a.get_bit(0),
                circuit,
                key
            ),
        "execute_bdd_circuit_1w_to_1w: insufficient scratch"
    );
    let (mut out_bits, mut scratch_1) = scratch.borrow().take_glwe_slice_scratch(T::BITS as usize, out);

    // Evaluates out[i] = circuit[i](a, b)
    module.execute_bdd_circuit_multi_thread(threads, &mut out_bits, a, circuit, &mut scratch_1);

    // Repacks the bits
    out.pack(module, out_bits, key, &mut scratch_1);
}
