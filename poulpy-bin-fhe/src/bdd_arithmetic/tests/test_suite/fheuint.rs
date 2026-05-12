use poulpy_core::{
    EncryptionLayout, GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWERotate, GLWESub, GLWETrace,
    layouts::{GLWELayout, GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena, ScratchOwned},
    source::Source,
};
use rand::Rng;

use crate::{
    bdd_arithmetic::{
        BDDKeyPrepared, FheUint, ToBits,
        tests::test_suite::{TEST_GLWE_INFOS, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_fhe_uint_sext<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);

    for j in 0..3 {
        let a: u32 = 0x8483_8281;
        a_enc.encrypt_sk(
            module,
            a,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        a_enc.sext(module, j, keys, &mut scratch.borrow());

        assert_eq!(
            sext(a, ((1 + j as u32) << 3) - 1),
            a_enc.decrypt(module, sk, &mut scratch.borrow())
        );
    }

    for j in 0..3 {
        let a: u32 = 0x4443_4241;
        a_enc.encrypt_sk(
            module,
            a,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        a_enc.sext(module, j, keys, &mut scratch.borrow());

        assert_eq!(
            sext(a, ((1 + j as u32) << 3) - 1),
            a_enc.decrypt(module, sk, &mut scratch.borrow())
        );
    }
}

pub(crate) fn sext(x: u32, bits: u32) -> u32 {
    let lo: u32 = x << (u32::BITS - bits) >> (u32::BITS - bits);
    let hi: u32 = ((x >> bits) & 1) * (0xFFFF_FFFF << bits);
    hi | lo
}

pub fn test_fhe_uint_splice_u8<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
    let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    b_enc.encrypt_sk(
        module,
        b,
        sk,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    a_enc.encrypt_sk(
        module,
        a,
        sk,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    for dst in 0..4 {
        for src in 0..4 {
            c_enc.splice_u8(module, dst, src, &a_enc, &b_enc, keys, &mut scratch.borrow());

            let rj: u32 = (dst << 3) as u32;
            let ri: u32 = (src << 3) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);

            let c_want: u32 = ((a_r & 0xFFFF_FF00) | (b_r & 0x0000_00FF)).rotate_left(rj);

            assert_eq!(c_want, c_enc.decrypt(module, sk, &mut scratch.borrow()));
        }
    }
}

pub fn test_fhe_uint_splice_u16<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
    let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    b_enc.encrypt_sk(
        module,
        b,
        sk,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    a_enc.encrypt_sk(
        module,
        a,
        sk,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    for dst in 0..2 {
        for src in 0..2 {
            c_enc.splice_u16(module, dst, src, &a_enc, &b_enc, keys, &mut scratch.borrow());
            let rj: u32 = (dst << 4) as u32;
            let ri: u32 = (src << 4) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);
            let c_want: u32 = ((a_r & 0xFFFF_0000) | (b_r & 0x0000_FFFF)).rotate_left(rj);
            assert_eq!(c_want, c_enc.decrypt(module, sk, &mut scratch.borrow()));
        }
    }
}

pub fn test_fhe_uint_get_bit_glwe<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = source_xa.next_u32();

    let mut scratch_enc: ScratchOwned<BE> = ScratchOwned::alloc(a_enc.encrypt_sk_tmp_bytes(module));
    a_enc.encrypt_sk(
        module,
        a,
        sk,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch_enc.borrow(),
    );

    let mut scratch_dec: ScratchOwned<BE> = ScratchOwned::alloc(c_enc.decrypt_tmp_bytes(module));

    for i in 0..32 {
        a_enc.get_bit_glwe(module, i, &mut c_enc, keys, &mut scratch.borrow());
        assert_eq!(a.bit(i) as u32, c_enc.decrypt(module, sk, &mut scratch_dec.borrow()));
    }
}
