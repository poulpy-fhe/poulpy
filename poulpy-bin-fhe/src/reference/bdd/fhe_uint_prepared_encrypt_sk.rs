use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*, source::Source};
/// Workspace reused by bit encryption and preparation; temporary ciphertexts are owned allocations.
pub fn fhe_uint_prepared_encrypt_sk_tmp_bytes_reference<BE: Backend, A: GGSWInfos>(module: &Module<BE>, infos: &A) -> usize
where
    Module<BE>: GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
{
    module
        .ggsw_encrypt_sk_tmp_bytes(infos)
        .max(module.ggsw_prepare_tmp_bytes(infos))
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`FheUintPreparedEncryptSk::fhe_uint_prepared_encrypt_sk`].
pub fn fhe_uint_prepared_encrypt_sk_reference<T: UnsignedInteger + ToBits, BE: Backend<ZnxWord = i64> + 'static, S, E>(
    module: &Module<BE>,
    res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
    value: T,
    sk: &S,
    enc_infos: &E,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    E: EncryptionInfos,
    Module<BE>: Sized
        + ModuleN
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    assert!(module.n().is_multiple_of(T::BITS as usize));
    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(sk.n(), module.n() as u32);

    let mut tmp_ggsw: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(res);
    let mut pt = vec![0u8; module.n() * size_of::<i64>()];
    let mut scratch_1 = scratch.borrow();

    for i in 0..T::BITS as usize {
        pt[..size_of::<i64>()].copy_from_slice(&(value.bit(i) as i64).to_ne_bytes());
        let pt_backend = ScalarZnx::from_data(BE::from_host_bytes(&pt), module.n(), 1);
        let mut scratch_bit = scratch_1.borrow();
        module.ggsw_encrypt_sk(
            &mut tmp_ggsw,
            &pt_backend,
            sk,
            enc_infos,
            source_xe,
            source_xa,
            &mut scratch_bit,
        );
        module.ggsw_prepare(&mut res.bits[i], &tmp_ggsw, &mut scratch_bit);
    }
}
