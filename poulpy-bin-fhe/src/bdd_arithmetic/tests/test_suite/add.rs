use poulpy_core::{
    EncryptionLayout, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWENoise,
    layouts::{GGSWLayout, GLWELayout, GLWESecretPrepared, GLWESecretPreparedFactory},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena, ScratchOwned},
    source::Source,
};
use rand::Rng;

use crate::{
    bdd_arithmetic::{
        Add, BDDKeyEncryptSk, BDDKeyPrepared, BDDKeyPreparedFactory, ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare,
        FheUintPrepareDebug, FheUintPrepared, FheUintPreparedEncryptSk, FheUintPreparedFactory,
        tests::test_suite::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_bdd_add<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend>(test_context: &TestContext<BRA, BE>)
where
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
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
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

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut res: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
    let mut a_enc_prep: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::<BE::OwnedBuf, u32, BE>::alloc_from_infos(module, &ggsw_infos);
    let mut b_enc_prep: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::<BE::OwnedBuf, u32, BE>::alloc_from_infos(module, &ggsw_infos);

    let a: u32 = source.next_u32();
    let b: u32 = source.next_u32();

    let ggsw_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();
    source.fill_bytes(scratch.data.as_mut());
    a_enc_prep.encrypt_sk(
        module,
        a,
        sk_glwe_prep,
        &ggsw_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    source.fill_bytes(scratch.data.as_mut());
    b_enc_prep.encrypt_sk(
        module,
        b,
        sk_glwe_prep,
        &ggsw_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let add_bytes: usize = res.add_tmp_bytes(module, &glwe_infos, &ggsw_infos, bdd_key_prepared);
    let mut scratch_add: ScratchOwned<BE> = ScratchOwned::alloc(add_bytes);
    res.add(module, &a_enc_prep, &b_enc_prep, bdd_key_prepared, &mut scratch_add.borrow());
    assert_eq!(res.decrypt(module, sk_glwe_prep, &mut scratch.borrow()), a.wrapping_add(b));

    let mt_threads: usize = 4;
    let add_mt_bytes: usize = res.add_multi_thread_tmp_bytes(module, mt_threads, &glwe_infos, &ggsw_infos, bdd_key_prepared);
    let mut scratch_mt: ScratchOwned<BE> = ScratchOwned::alloc(add_mt_bytes);
    res.add_multi_thread(
        mt_threads,
        module,
        &a_enc_prep,
        &b_enc_prep,
        bdd_key_prepared,
        &mut scratch_mt.borrow(),
    );
    assert_eq!(res.decrypt(module, sk_glwe_prep, &mut scratch.borrow()), a.wrapping_add(b));
}
