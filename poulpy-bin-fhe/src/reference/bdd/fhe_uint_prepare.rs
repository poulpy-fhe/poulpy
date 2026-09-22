use crate::{
    bdd_arithmetic::*,
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
    circuit_bootstrapping::*,
};
use poulpy_core::{layouts::*, *};
use poulpy_hal::layouts::*;
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`FheUintPrepare::fhe_uint_prepare_tmp_bytes`].
pub fn fhe_uint_prepare_tmp_bytes_reference<BRA: BlindRotationAlgo, BE, R, A, B>(
    module: &Module<BE>,
    block_size: usize,
    extension_factor: usize,
    res_infos: &R,
    bits_infos: &A,
    bdd_infos: &B,
) -> usize
where
    R: GGSWInfos,
    A: GLWEInfos,
    B: BDDKeyInfos,
    Module<BE>: LWEFromGLWE<BE> + GLWEKeyswitch<BE> + CircuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    BE: Backend<ZnxWord = i64> + 'static,
{
    let mut lwe_infos = bits_infos.lwe_layout();
    lwe_infos.n = bdd_infos.cbt_infos().brk_infos().n_lwe();
    let extraction = if let Some(key) = bdd_infos.ks_glwe_infos() {
        let bridge = GLWELayout {
            n: key.n(),
            base2k: key.base2k(),
            k: key.k(),
            rank: key.rank_out(),
        };
        module
            .glwe_keyswitch_tmp_bytes(&bridge, bits_infos, &key)
            .max(module.lwe_from_glwe_tmp_bytes(&lwe_infos, &bridge, &bdd_infos.ks_lwe_infos()))
    } else {
        module.lwe_from_glwe_tmp_bytes(&lwe_infos, bits_infos, &bdd_infos.ks_lwe_infos())
    };
    let operation = extraction
        .max(module.circuit_bootstrapping_execute_tmp_bytes(block_size, extension_factor, res_infos, &bdd_infos.cbt_infos()))
        .max(module.ggsw_prepare_tmp_bytes(res_infos));
    poulpy_hal::execution::worker_scratch_bytes::<BE>(
        module.ggsw_bytes_of_from_infos(res_infos) + module.lwe_bytes_of_from_infos(&lwe_infos) + operation,
    )
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`FheUintPrepare::fhe_uint_prepare_custom_multi_thread`].
pub fn fhe_uint_prepare_custom_multi_thread_reference<BRA: BlindRotationAlgo, BE, K, T: UnsignedInteger>(
    module: &Module<BE>,
    threads: usize,
    res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
    bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
    bit_start: usize,
    bit_count: usize,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    Module<BE>: LWEFromGLWE<BE> + GLWEKeyswitch<BE> + CircuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    BE: Backend<ZnxWord = i64> + 'static,
{
    let bit_end = bit_start.checked_add(bit_count).expect("FheUint bit range overflow");
    let (cbt, ks_glwe, ks_lwe) = key.get_cbt_key();
    let mut lwe_infos = bits.lwe_layout();
    lwe_infos.n = cbt.brk_infos().n_lwe();

    assert!(bit_end <= T::BITS as usize);

    let workers = poulpy_hal::execution::worker_count::<BE::TaskExecutor>(threads, bit_count);
    let scratch_thread_size =
        fhe_uint_prepare_tmp_bytes_reference::<BRA, BE, _, _, _>(module, cbt.block_size(), 1, res, bits, key);
    let needed = workers
        .checked_mul(scratch_thread_size)
        .expect("FheUint parallel scratch size overflows usize");

    assert!(
        scratch.available() >= needed,
        "scratch.available():{} < parallel FheUint scratch bytes:{needed}",
        scratch.available()
    );

    let ggsw_infos: &GGSWLayout = &res.ggsw_layout();

    for i in 0..bit_start {
        module.ggsw_zero(&mut res.bits[i]);
    }

    for i in bit_end..T::BITS as usize {
        module.ggsw_zero(&mut res.bits[i]);
    }
    if bit_count == 0 {
        return;
    }

    let (worker_scratch, _) = scratch.borrow().split(workers, scratch_thread_size);
    poulpy_hal::execution::for_each_with_scratch::<BE::TaskExecutor, BE, _, _>(
        &mut res.bits[bit_start..bit_end],
        bit_start,
        worker_scratch,
        &|bit, res_bit, scratch| {
            let (mut tmp_ggsw, scratch_bit) = scratch.borrow().take_ggsw_scratch(ggsw_infos);
            let (mut tmp_lwe, mut scratch_bit) = scratch_bit.take_lwe_scratch(&lwe_infos);
            let ks_glwe_ref = ks_glwe.map(GGLWEPreparedToBackendRef::to_backend_ref);
            bits.get_bit_lwe(
                module,
                bit,
                &mut tmp_lwe,
                ks_glwe_ref.as_ref(),
                &GGLWEPreparedToBackendRef::to_backend_ref(ks_lwe),
                &mut scratch_bit,
            );
            cbt.execute_to_constant(module, &mut tmp_ggsw.to_backend_mut(), &tmp_lwe, 1, 1, &mut scratch_bit);
            module.ggsw_prepare(res_bit, &tmp_ggsw, &mut scratch_bit);
        },
    );
}
