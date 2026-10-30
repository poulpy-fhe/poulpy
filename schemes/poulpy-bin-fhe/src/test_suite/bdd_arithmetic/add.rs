use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWENoise,
    api::GLWESecretPreparedFactory,
    layouts::{GGSWLayout, GLWELayout, GLWESecretPrepared},
};
use poulpy_hal::layouts::HostDataRef;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchOwned},
    source::Source,
};
use rand::Rng;

use crate::test_suite::bdd_arithmetic::decrypt_checked;
use crate::test_suite::bdd_arithmetic::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext};
use crate::test_suite::parity::{assert_short_scratch_panics, snapshot_glwe};
use crate::{
    bdd_arithmetic::{
        Add, BDDKeyEncryptSk, BDDKeyPrepared, BDDKeyPreparedFactory, ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare,
        FheUintPrepareDebug, FheUintPrepared, FheUintPreparedEncryptSk, FheUintPreparedFactory,
    },
    blind_rotation::BlindRotationAlgo,
};
use poulpy_core::test_suite::with_scratch;

pub fn test_bdd_add<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWENoise<BE>
        + FheUintPreparedFactory<u32, BE>
        + FheUintPreparedEncryptSk<u32, BE>
        + FheUintPrepareDebug<BRA, u32, BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + GGSWNoise<BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<BE>
        + GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk_glwe_prep: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let bdd_key_prepared: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut res: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let mut a_enc_prep: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::<BE::OwnedBuf, u32, BE>::alloc_from_infos(module, &ggsw_infos);
    let mut b_enc_prep: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::<BE::OwnedBuf, u32, BE>::alloc_from_infos(module, &ggsw_infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        FheUintPreparedEncryptSk::<u32, BE>::fhe_uint_prepared_encrypt_sk_tmp_bytes(module, &ggsw_infos)
            .max(res.decrypt_tmp_bytes::<_, BE>(module)),
    );

    let a: u32 = source.next_u32();
    let b: u32 = source.next_u32();

    source.fill_bytes(scratch.data.as_mut());
    a_enc_prep.encrypt_sk(module, a, sk_glwe_prep, &mut source_xe, &mut source_xa, &mut scratch.borrow());
    source.fill_bytes(scratch.data.as_mut());
    b_enc_prep.encrypt_sk(module, b, sk_glwe_prep, &mut source_xe, &mut source_xa, &mut scratch.borrow());

    let add_bytes: usize = res.add_tmp_bytes(module, &glwe_infos, &ggsw_infos, bdd_key_prepared);
    let before = snapshot_glwe::<BE, _>(&res);
    assert_short_scratch_panics::<BE, _>(add_bytes, "execute_bdd_circuit_2w_to_1w: insufficient scratch", |s| {
        res.add(module, &a_enc_prep, &b_enc_prep, bdd_key_prepared, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&res));
    with_scratch::<BE, _>(add_bytes, |s| res.add(module, &a_enc_prep, &b_enc_prep, bdd_key_prepared, s));
    assert_eq!(decrypt_checked(module, &res, sk_glwe_prep), a.wrapping_add(b));

    let mt_threads: usize = 4;
    let add_mt_bytes: usize = res.add_multi_thread_tmp_bytes(module, mt_threads, &glwe_infos, &ggsw_infos, bdd_key_prepared);
    let before = snapshot_glwe::<BE, _>(&res);
    assert_short_scratch_panics::<BE, _>(add_mt_bytes, "execute_bdd_circuit_2w_to_1w: insufficient scratch", |s| {
        res.add_multi_thread(module, mt_threads, &a_enc_prep, &b_enc_prep, bdd_key_prepared, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&res));
    with_scratch::<BE, _>(add_mt_bytes, |s| {
        res.add_multi_thread(module, mt_threads, &a_enc_prep, &b_enc_prep, bdd_key_prepared, s)
    });
    assert_eq!(decrypt_checked(module, &res, sk_glwe_prep), a.wrapping_add(b));
}
