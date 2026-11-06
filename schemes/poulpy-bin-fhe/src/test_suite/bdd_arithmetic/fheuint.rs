use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWEPack, GLWEPlaintextBytesOf, GLWERotate, GLWESub,
    GLWETrace, LWEDecrypt, LWEFillMask, LWEFromGLWE,
    api::GLWESecretPreparedFactory,
    layouts::GLWEPlaintextAlloc,
    layouts::{
        GGLWEPreparedToBackendRef, GGSWLayout, GLWELayout, GLWESecretPrepared, LWE, LWEAlloc, LWEInfos, LWEPlaintextAlloc,
        ModuleCoreAlloc, TorusPrecision,
    },
    test_suite::{noise::assert_glwe_flag_honest, with_scratch},
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::HostDataRef;
use poulpy_hal::{
    api::ModuleNew,
    layouts::{Backend, HostBackend, HostDataMut, Module},
    oep::HalModuleImpl,
    source::Source,
};
use rand::Rng;

use crate::test_suite::bdd_arithmetic::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext};
use crate::test_suite::parity::{assert_panics_with, assert_short_scratch_panics, snapshot_glwe};
use crate::{
    bdd_arithmetic::{
        BDDKeyPrepared, FheUint, FheUintPrepared, FheUintPreparedEncryptSk, FheUintPreparedFactory, GLWECmux, ToBits,
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
};

type Uint<BE> = FheUint<<BE as Backend>::OwnedBuf, u32, i64>;

/// Decrypts `a` on exactly its query once its canonical flag is checked honest.
pub(crate) fn decrypt_checked<BE, M, S>(module: &M, a: &Uint<BE>, sk: &S) -> u32
where
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
    M: GLWEPlaintextBytesOf<BE>
        + GLWEPlaintextAlloc
        + poulpy_core::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleLogN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GLWEDecrypt<BE>,
    S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
{
    assert_glwe_flag_honest::<BE, _>(a);
    with_scratch::<BE, _>(a.decrypt_tmp_bytes(module), |s| a.decrypt(module, sk, s))
}

fn encrypt<BE, M, S>(module: &M, a: &mut Uint<BE>, value: u32, sk: &S, source_xe: &mut Source, source_xa: &mut Source)
where
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64>,
    M: GLWEPlaintextBytesOf<BE>
        + GLWEPlaintextAlloc
        + poulpy_core::GLWEBytesOf<BE>
        + poulpy_hal::api::ModuleLogN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GLWEEncryptSk<BE>,
    S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
{
    with_scratch::<BE, _>(a.encrypt_sk_tmp_bytes(module), |s| {
        a.encrypt_sk(module, value, sk, source_xe, source_xa, s)
    });
}

pub fn test_fhe_uint_sext<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEDecrypt<BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);
    let bytes = Uint::<BE>::sext_tmp_bytes(module, &a_enc, keys);

    for a in [0x8483_8281u32, 0x4443_4241] {
        for j in 0..3 {
            encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);
            with_scratch::<BE, _>(bytes, |s| a_enc.sext(module, j, keys, s));
            assert_eq!(sext(a, ((1 + j as u32) << 3) - 1), decrypt_checked(module, &a_enc, sk));
        }
    }

    // Extending from the top byte leaves the value unchanged, yet is an evaluation.
    let a: u32 = 0x8483_8281;
    encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);
    assert!(a_enc.noise().is_some());
    let before = snapshot_glwe::<BE, _>(&a_enc);
    assert_short_scratch_panics::<BE, _>(bytes, "sext: insufficient scratch", |s| a_enc.sext(module, 3, keys, s));
    assert_eq!(before, snapshot_glwe::<BE, _>(&a_enc));
    with_scratch::<BE, _>(bytes, |s| a_enc.sext(module, 3, keys, s));
    assert!(a_enc.noise().is_none());
    assert_eq!(a, decrypt_checked(module, &a_enc, sk));
}

pub(crate) fn sext(x: u32, bits: u32) -> u32 {
    let lo: u32 = x << (u32::BITS - bits) >> (u32::BITS - bits);
    let hi: u32 = ((x >> bits) & 1) * (0xFFFF_FFFF << bits);
    hi | lo
}

pub fn test_fhe_uint_splice_u8<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEDecrypt<BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);
    let mut b_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    encrypt(module, &mut b_enc, b, sk, &mut source_xe, &mut source_xa);
    encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);

    let bytes = Uint::<BE>::splice_u8_tmp_bytes(module, &c_enc, &a_enc, &b_enc, keys);
    let before = snapshot_glwe::<BE, _>(&c_enc);
    assert_short_scratch_panics::<BE, _>(bytes, "splice_u8: insufficient scratch", |s| {
        c_enc.splice_u8(module, 0, 0, &a_enc, &b_enc, keys, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&c_enc));

    for dst in 0..4 {
        for src in 0..4 {
            with_scratch::<BE, _>(bytes, |s| c_enc.splice_u8(module, dst, src, &a_enc, &b_enc, keys, s));

            let rj: u32 = (dst << 3) as u32;
            let ri: u32 = (src << 3) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);

            let c_want: u32 = ((a_r & 0xFFFF_FF00) | (b_r & 0x0000_00FF)).rotate_left(rj);

            assert_eq!(c_want, decrypt_checked(module, &c_enc, sk));
        }
    }
}

pub fn test_fhe_uint_splice_u16<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEDecrypt<BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);
    let mut b_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    encrypt(module, &mut b_enc, b, sk, &mut source_xe, &mut source_xa);
    encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);

    let bytes = Uint::<BE>::splice_u16_tmp_bytes(module, &c_enc, &a_enc, &b_enc, keys);
    let before = snapshot_glwe::<BE, _>(&c_enc);
    assert_short_scratch_panics::<BE, _>(bytes, "splice_u16: insufficient scratch", |s| {
        c_enc.splice_u16(module, 0, 0, &a_enc, &b_enc, keys, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&c_enc));

    for dst in 0..2 {
        for src in 0..2 {
            with_scratch::<BE, _>(bytes, |s| c_enc.splice_u16(module, dst, src, &a_enc, &b_enc, keys, s));
            let rj: u32 = (dst << 4) as u32;
            let ri: u32 = (src << 4) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);
            let c_want: u32 = ((a_r & 0xFFFF_0000) | (b_r & 0x0000_FFFF)).rotate_left(rj);
            assert_eq!(c_want, decrypt_checked(module, &c_enc, sk));
        }
    }
}

pub fn test_fhe_uint_zero_byte<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEDecrypt<BE>,
{
    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xm: Source = Source::new([4u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &TEST_GLWE_INFOS);
    let a: u32 = source_xm.next_u32();
    let bytes = Uint::<BE>::zero_byte_tmp_bytes(module, &a_enc, keys);

    encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);
    let before = snapshot_glwe::<BE, _>(&a_enc);
    assert_short_scratch_panics::<BE, _>(bytes, "zero_byte: insufficient scratch", |s| {
        a_enc.zero_byte(module, 0, keys, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&a_enc));

    for byte in 0..4 {
        encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);
        with_scratch::<BE, _>(bytes, |s| a_enc.zero_byte(module, byte, keys, s));
        assert_eq!(a & !(0xFF << (8 * byte)), decrypt_checked(module, &a_enc, sk));
    }
}

pub fn test_fhe_uint_get_bit_glwe<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWEDecrypt<BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xm: Source = Source::new([4u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = source_xm.next_u32();
    encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);

    let bytes = Uint::<BE>::get_bit_glwe_tmp_bytes(module, &c_enc, keys);
    let before = snapshot_glwe::<BE, _>(&c_enc);
    assert_short_scratch_panics::<BE, _>(bytes, "get_bit_glwe: insufficient scratch", |s| {
        a_enc.get_bit_glwe(module, 0, &mut c_enc, keys, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&c_enc));

    for i in 0..32 {
        with_scratch::<BE, _>(bytes, |s| a_enc.get_bit_glwe(module, i, &mut c_enc, keys, s));
        assert_eq!(a.bit(i) as u32, decrypt_checked(module, &c_enc, sk));
    }
}

pub fn test_fhe_uint_get_byte<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWERotate<BE> + GLWETrace<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWEDecrypt<BE>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xm: Source = Source::new([4u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = source_xm.next_u32();
    encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);

    let bytes = Uint::<BE>::get_byte_tmp_bytes(module, &c_enc, keys);
    let before = snapshot_glwe::<BE, _>(&c_enc);
    assert_short_scratch_panics::<BE, _>(bytes, "get_byte: insufficient scratch", |s| {
        a_enc.get_byte(module, 0, &mut c_enc, keys, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&c_enc));

    for byte in 0..4 {
        with_scratch::<BE, _>(bytes, |s| a_enc.get_byte(module, byte, &mut c_enc, keys, s));
        assert_eq!((a >> (8 * byte)) & 0xFF, decrypt_checked(module, &c_enc, sk));
    }
}

/// Each extracted bit decrypts under the LWE secret of the key to the bit of the message, over a dirty destination.
pub fn test_fhe_uint_get_bit_lwe<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + LWEFromGLWE<BE>
        + LWEDecrypt<BE>
        + LWEFillMask<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + LWEAlloc
        + LWEPlaintextAlloc,
{
    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xm: Source = Source::new([4u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &TEST_GLWE_INFOS);
    let a: u32 = source_xm.next_u32();
    encrypt(module, &mut a_enc, a, sk, &mut source_xe, &mut source_xa);

    let mut lwe_infos = a_enc.lwe_layout();
    lwe_infos.n = keys.cbt.blind_rotation_key().n_lwe();
    let ks_glwe = keys.ks_glwe.as_ref().map(|k| k.to_backend_ref());
    let ks_lwe = keys.ks_lwe.to_backend_ref();
    let bytes = Uint::<BE>::get_bit_lwe_tmp_bytes(module, &lwe_infos, &a_enc, ks_glwe.as_ref(), &ks_lwe);

    let mut lwe: LWE<BE::OwnedBuf, BE::ZnxWord> = module.lwe_alloc_from_infos(&lwe_infos);
    module.lwe_fill_mask_from_source(lwe_infos.base2k.as_usize(), &mut lwe, &mut source_xa);
    let before = lwe.clone();
    assert_short_scratch_panics::<BE, _>(bytes, "get_bit_lwe: insufficient scratch", |s| {
        a_enc.get_bit_lwe(module, 0, &mut lwe, ks_glwe.as_ref(), &ks_lwe, s)
    });
    assert!(before == lwe, "a refused extraction changed its destination");

    let mut pt = module.lwe_plaintext_alloc_from_infos(&lwe_infos);
    for bit in [0usize, 1, 17, 31] {
        module.lwe_fill_mask_from_source(lwe_infos.base2k.as_usize(), &mut lwe, &mut source_xa);
        with_scratch::<BE, _>(bytes, |s| {
            a_enc.get_bit_lwe(module, bit, &mut lwe, ks_glwe.as_ref(), &ks_lwe, s)
        });
        with_scratch::<BE, _>(module.lwe_decrypt_tmp_bytes(&lwe_infos), |s| {
            module.lwe_decrypt(&lwe, &mut pt, &test_context.sk_lwe, s)
        });
        assert_eq!(a.bit(bit) as i64, pt.decode_i64(TorusPrecision(2)), "bit {bit}");
    }
}

pub fn test_fhe_uint_from_fhe_uint_prepared<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWECmux<BE>
        + GLWEPack<BE>
        + GLWECopy<BE>
        + FheUintPreparedEncryptSk<u32, BE>
        + FheUintPreparedFactory<u32, BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let ggsw_infos: GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xm: Source = Source::new([4u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let value: u32 = source_xm.next_u32();
    let mut prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> = FheUintPrepared::alloc_from_infos(module, &ggsw_infos);
    with_scratch::<BE, _>(
        FheUintPreparedEncryptSk::<u32, BE>::fhe_uint_prepared_encrypt_sk_tmp_bytes(module, &ggsw_infos),
        |s| prepared.encrypt_sk(module, value, sk, &mut source_xe, &mut source_xa, s),
    );

    let mut c_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &TEST_GLWE_INFOS);
    let bytes = Uint::<BE>::from_fhe_uint_prepared_tmp_bytes(module, &c_enc, &prepared, keys);
    let before = snapshot_glwe::<BE, _>(&c_enc);
    assert_short_scratch_panics::<BE, _>(bytes, "from_fhe_uint_prepared: insufficient scratch", |s| {
        c_enc.from_fhe_uint_prepared(module, &prepared, keys, s)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&c_enc));

    with_scratch::<BE, _>(bytes, |s| c_enc.from_fhe_uint_prepared(module, &prepared, keys, s));
    assert_eq!(value, decrypt_checked(module, &c_enc, sk));
}

/// A ciphertext at another degree than the module's, and a module degree no multiple of the integer width.
fn assert_rejects_degree<BE>(module: &Module<BE>, run: impl Fn(&Module<BE>, &mut Uint<BE>))
where
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend + HalModuleImpl,
    Module<BE>: ModuleNew<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let half: Module<BE> = Module::<BE>::new((module.n() / 2) as u64);
    let mut foreign: Uint<BE> = Uint::<BE>::alloc(&half, TEST_GLWE_INFOS.base2k, TEST_GLWE_INFOS.k, TEST_GLWE_INFOS.rank);
    let before = snapshot_glwe::<BE, _>(&foreign);
    assert_panics_with("the ciphertext degree differs from the module degree", || {
        run(module, &mut foreign)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&foreign));

    if BE::MIN_DEGREE < u32::BITS as usize {
        let narrow: Module<BE> = Module::<BE>::new(BE::MIN_DEGREE as u64);
        let mut small: Uint<BE> = Uint::<BE>::alloc(&narrow, TEST_GLWE_INFOS.base2k, TEST_GLWE_INFOS.k, TEST_GLWE_INFOS.rank);
        let before = snapshot_glwe::<BE, _>(&small);
        assert_panics_with("the module degree is not a multiple of the integer width", || {
            run(&narrow, &mut small)
        });
        assert_eq!(before, snapshot_glwe::<BE, _>(&small));
    }
}

/// A secret at another degree than the module's.
fn assert_rejects_secret_degree<BE>(module: &Module<BE>, run: impl Fn(&GLWESecretPrepared<BE::OwnedBuf, BE>, &mut Uint<BE>))
where
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend + HalModuleImpl,
    Module<BE>: ModuleNew<BE> + GLWESecretPreparedFactory<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let half: Module<BE> = Module::<BE>::new((module.n() / 2) as u64);
    let sk_half: GLWESecretPrepared<BE::OwnedBuf, BE> = half.glwe_secret_prepared_alloc(TEST_GLWE_INFOS.rank);
    let mut a_enc: Uint<BE> = Uint::<BE>::alloc_from_infos(module, &TEST_GLWE_INFOS);
    let before = snapshot_glwe::<BE, _>(&a_enc);
    assert_panics_with("the secret degree differs from the module degree", || {
        run(&sk_half, &mut a_enc)
    });
    assert_eq!(before, snapshot_glwe::<BE, _>(&a_enc));
}

pub fn test_fhe_uint_encrypt_sk_rejects_degree<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend + HalModuleImpl,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let encrypt_refused = |module: &Module<BE>, sk: &GLWESecretPrepared<BE::OwnedBuf, BE>, a: &mut Uint<BE>| {
        let mut source_xa: Source = Source::new([2u8; 32]);
        let mut source_xe: Source = Source::new([3u8; 32]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_scratch::<BE, _>(0, |s| a.encrypt_sk(module, 1, sk, &mut source_xe, &mut source_xa, s))
        }));
        assert!(
            source_xa.new_seed() == Source::new([2u8; 32]).new_seed()
                && source_xe.new_seed() == Source::new([3u8; 32]).new_seed(),
            "a refused encryption drew from its sources"
        );
        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
    };
    assert_rejects_degree(module, |module, a| encrypt_refused(module, sk, a));
    assert_rejects_secret_degree(module, |sk, a| encrypt_refused(module, sk, a));
}

pub fn test_fhe_uint_decrypt_rejects_degree<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend + HalModuleImpl,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>:
        ModuleNew<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    assert_rejects_degree(module, |module, a| {
        with_scratch::<BE, _>(0, |s| a.decrypt(module, sk, s));
    });
    assert_rejects_secret_degree(module, |sk, a| {
        with_scratch::<BE, _>(0, |s| a.decrypt(module, sk, s));
    });
}

pub fn test_fhe_uint_noise_stats_rejects_degree<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend + HalModuleImpl,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEDecrypt<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    assert_rejects_degree(module, |module, a| {
        with_scratch::<BE, _>(0, |s| a.noise_stats(module, 1, sk, s));
    });
    assert_rejects_secret_degree(module, |sk, a| {
        with_scratch::<BE, _>(0, |s| a.noise_stats(module, 1, sk, s));
    });
}

/// A prepared integer, then a secret, at another degree than the module's, each refused before any write or draw.
pub fn test_fhe_uint_prepared_encrypt_sk_rejects_degree<
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend + HalModuleImpl,
>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>:
        ModuleNew<BE> + FheUintPreparedEncryptSk<u32, BE> + FheUintPreparedFactory<u32, BE> + GLWESecretPreparedFactory<BE>,
{
    use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;
    use poulpy_hal::layouts::DataView;

    let module: &Module<BE> = &test_context.module;
    let half: Module<BE> = Module::<BE>::new((module.n() / 2) as u64);
    let bytes = |p: &FheUintPrepared<AlignedBuf, u32, BE>| -> Vec<Vec<u8>> {
        p.bits
            .iter()
            .map(|bit| {
                let view = bit.to_backend_ref();
                let mut out = vec![0u8; BE::len_bytes_ref(view.data().data())];
                BE::copy_view_to_host(view.data().data(), &mut out);
                out
            })
            .collect()
    };
    let half_infos = GGSWLayout {
        n: half.n().into(),
        ..TEST_GGSW_INFOS
    };
    let sk_half: GLWESecretPrepared<AlignedBuf, BE> = half.glwe_secret_prepared_alloc(TEST_GGSW_INFOS.rank);
    let cases = [
        (
            FheUintPrepared::alloc_from_infos(&half, &half_infos),
            &test_context.sk_glwe,
            "the ciphertext degree differs from the module degree",
        ),
        (
            FheUintPrepared::alloc_from_infos(module, &TEST_GGSW_INFOS),
            &sk_half,
            "the secret degree differs from the module degree",
        ),
    ];
    for (mut res, sk, message) in cases {
        let before = bytes(&res);
        let mut source_xe = Source::new([3u8; 32]);
        let mut source_xa = Source::new([2u8; 32]);
        assert_panics_with(message, || {
            with_scratch::<BE, _>(0, |s| res.encrypt_sk(module, 1u32, sk, &mut source_xe, &mut source_xa, s))
        });
        assert!(bytes(&res) == before, "a refused encryption changed its destination");
        assert!(
            source_xe.new_seed() == Source::new([3u8; 32]).new_seed()
                && source_xa.new_seed() == Source::new([2u8; 32]).new_seed(),
            "a refused encryption drew from its sources"
        );
    }
}
