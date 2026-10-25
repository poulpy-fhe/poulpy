use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEKeyswitch, GLWEPacking, GLWERotate, GLWESub, GLWETrace, LWEFromGLWE,
    layouts::{GGLWEPreparedToBackendRef, GGSWLayout, GLWELayout, GLWESecretPrepared, LWE, LWEInfos, ModuleCoreAlloc},
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{HostDataRef, WriterTo};
use poulpy_hal::{
    layouts::{Backend, HostBackend, HostDataMut, Module},
    source::Source,
};
use rand::Rng;

use crate::test_suite::bdd_arithmetic::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext};
use crate::test_suite::parity::with_scratch;
use crate::{
    bdd_arithmetic::{BDDKeyPrepared, Cmux, FheUint, FheUintPrepared, FheUintPreparedEncryptSk, FheUintPreparedFactory, ToBits},
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
};

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

    let mut a_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);

    for j in 0..3 {
        let a: u32 = 0x8483_8281;
        with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
            a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
        });

        with_scratch::<BE, _>(a_enc.sext_tmp_bytes(module, keys), |s| a_enc.sext(module, j, keys, s));

        assert_eq!(
            sext(a, ((1 + j as u32) << 3) - 1),
            with_scratch::<BE, _>(a_enc.decrypt_tmp_bytes(module), |s| a_enc.decrypt(module, sk, s))
        );
    }

    for j in 0..3 {
        let a: u32 = 0x4443_4241;
        with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
            a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
        });

        with_scratch::<BE, _>(a_enc.sext_tmp_bytes(module, keys), |s| a_enc.sext(module, j, keys, s));

        assert_eq!(
            sext(a, ((1 + j as u32) << 3) - 1),
            with_scratch::<BE, _>(a_enc.decrypt_tmp_bytes(module), |s| a_enc.decrypt(module, sk, s))
        );
    }

    // Extending from the top byte leaves the value unchanged, yet is an evaluation.
    let a: u32 = 0x8483_8281;
    with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
        a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
    });
    assert!(a_enc.noise().is_some());
    with_scratch::<BE, _>(a_enc.sext_tmp_bytes(module, keys), |s| a_enc.sext(module, 3, keys, s));
    assert!(a_enc.noise().is_none());
    assert_eq!(
        a,
        with_scratch::<BE, _>(a_enc.decrypt_tmp_bytes(module), |s| a_enc.decrypt(module, sk, s))
    );
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

    let mut a_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let mut b_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    with_scratch::<BE, _>(b_enc.encrypt_sk_tmp_bytes(module), |s| {
        b_enc.encrypt_sk(module, b, sk, &mut source_xe, &mut source_xa, s)
    });
    with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
        a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
    });

    for dst in 0..4 {
        for src in 0..4 {
            with_scratch::<BE, _>(c_enc.splice_u8_tmp_bytes(module, &a_enc, &b_enc, keys), |s| {
                c_enc.splice_u8(module, dst, src, &a_enc, &b_enc, keys, s)
            });

            let rj: u32 = (dst << 3) as u32;
            let ri: u32 = (src << 3) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);

            let c_want: u32 = ((a_r & 0xFFFF_FF00) | (b_r & 0x0000_00FF)).rotate_left(rj);

            assert_eq!(
                c_want,
                with_scratch::<BE, _>(c_enc.decrypt_tmp_bytes(module), |s| c_enc.decrypt(module, sk, s))
            );
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

    let mut a_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let mut b_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = 0xFFFFFFFF;
    let b: u32 = 0xAABBCCDD;

    with_scratch::<BE, _>(b_enc.encrypt_sk_tmp_bytes(module), |s| {
        b_enc.encrypt_sk(module, b, sk, &mut source_xe, &mut source_xa, s)
    });
    with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
        a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
    });

    for dst in 0..2 {
        for src in 0..2 {
            with_scratch::<BE, _>(c_enc.splice_u16_tmp_bytes(module, &a_enc, &b_enc, keys), |s| {
                c_enc.splice_u16(module, dst, src, &a_enc, &b_enc, keys, s)
            });
            let rj: u32 = (dst << 4) as u32;
            let ri: u32 = (src << 4) as u32;
            let a_r: u32 = a.rotate_right(rj);
            let b_r: u32 = b.rotate_right(ri);
            let c_want: u32 = ((a_r & 0xFFFF_0000) | (b_r & 0x0000_FFFF)).rotate_left(rj);
            assert_eq!(
                c_want,
                with_scratch::<BE, _>(c_enc.decrypt_tmp_bytes(module), |s| c_enc.decrypt(module, sk, s))
            );
        }
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

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = source_xa.next_u32();

    with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
        a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
    });

    for i in 0..32 {
        with_scratch::<BE, _>(a_enc.get_bit_glwe_tmp_bytes(module, &c_enc, keys), |s| {
            a_enc.get_bit_glwe(module, i, &mut c_enc, keys, s)
        });
        assert_eq!(
            a.bit(i) as u32,
            with_scratch::<BE, _>(c_enc.decrypt_tmp_bytes(module), |s| c_enc.decrypt(module, sk, s))
        );
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

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let mut c_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);

    let a: u32 = source_xa.next_u32();

    with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
        a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
    });

    for byte in 0..4 {
        with_scratch::<BE, _>(a_enc.get_byte_tmp_bytes(module, &c_enc, keys), |s| {
            a_enc.get_byte(module, byte, &mut c_enc, keys, s)
        });
        assert_eq!(
            (a >> (8 * byte)) & 0xFF,
            with_scratch::<BE, _>(c_enc.decrypt_tmp_bytes(module), |s| c_enc.decrypt(module, sk, s))
        );
    }
}

/// The extraction on exactly its query matches the one on a larger arena, guards intact.
pub fn test_fhe_uint_get_bit_lwe<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: GLWEEncryptSk<BE> + GLWEKeyswitch<BE> + LWEFromGLWE<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut a_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &glwe_infos);
    let a: u32 = source_xa.next_u32();
    with_scratch::<BE, _>(a_enc.encrypt_sk_tmp_bytes(module), |s| {
        a_enc.encrypt_sk(module, a, sk, &mut source_xe, &mut source_xa, s)
    });

    let mut lwe_infos = a_enc.lwe_layout();
    lwe_infos.n = keys.cbt.blind_rotation_key().n_lwe();
    let ks_glwe = keys.ks_glwe.as_ref().map(|k| k.to_backend_ref());
    let ks_lwe = keys.ks_lwe.to_backend_ref();
    let bytes = a_enc.get_bit_lwe_tmp_bytes(module, &lwe_infos, ks_glwe.as_ref(), &ks_lwe);

    for bit in [0usize, 1, 17, 31] {
        let extract = |scratch_bytes: usize| -> Vec<u8> {
            let mut lwe: LWE<BE::OwnedBuf, BE::ZnxWord> = module.lwe_alloc_from_infos(&lwe_infos);
            with_scratch::<BE, _>(scratch_bytes, |s| {
                a_enc.get_bit_lwe(module, bit, &mut lwe, ks_glwe.as_ref(), &ks_lwe, s)
            });
            let mut out = Vec::new();
            lwe.write_to(&mut out).unwrap();
            out
        };
        assert_eq!(extract(bytes), extract(bytes + 4096), "bit {bit}");
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
        + Cmux<BE>
        + GLWEPacking<BE>
        + GLWECopy<BE>
        + FheUintPreparedEncryptSk<u32, BE>
        + FheUintPreparedFactory<u32, BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let ggsw_infos: GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let keys: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let value: u32 = source_xa.next_u32();
    let mut prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> = FheUintPrepared::alloc_from_infos(module, &ggsw_infos);
    with_scratch::<BE, _>(
        FheUintPreparedEncryptSk::<u32, BE>::fhe_uint_prepared_encrypt_sk_tmp_bytes(module, &ggsw_infos),
        |s| prepared.encrypt_sk(module, value, sk, &mut source_xe, &mut source_xa, s),
    );

    let mut c_enc: FheUint<BE::OwnedBuf, u32, BE::ZnxWord> =
        FheUint::<BE::OwnedBuf, u32, i64>::alloc_from_infos(module, &TEST_GLWE_INFOS);
    with_scratch::<BE, _>(c_enc.from_fhe_uint_prepared_tmp_bytes(module, &prepared, keys), |s| {
        c_enc.from_fhe_uint_prepared(module, &prepared, keys, s)
    });
    assert_eq!(
        value,
        with_scratch::<BE, _>(c_enc.decrypt_tmp_bytes(module), |s| c_enc.decrypt(module, sk, s))
    );
}
