use itertools::Itertools;
use poulpy_core::{
    EncryptionLayout, GGSWEncryptSk, GLWEDecrypt, GLWEEncryptSk,
    layouts::{GGSW, GGSWPrepared, GGSWPreparedFactory, GLWELayout, GLWEPlaintext, GLWESecretPrepared, ModuleCoreAlloc},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, Module, ScalarZnx, ScratchArena, ScratchOwned, ZnxViewMut},
    source::Source,
};
use rand::Rng;

use crate::{
    bdd_arithmetic::{
        Cmux, Cswap, FheUint, FheUintPrepared, GLWEBlindRetrieval, GLWEBlindRetriever,
        tests::test_suite::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_cmux_direct<BRA, BE>(test_context: &TestContext<BRA, BE>)
where
    BRA: BlindRotationAlgo,
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + Cmux<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
    BE: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();
    let ggsw_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();

    let t: i64 = 11;
    let f: i64 = 37;
    let k_pt: usize = 8;

    for bit in [0_i64, 1_i64] {
        let mut pt_t: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        let mut pt_f: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        pt_t.encode_coeff_i64(t, k_pt.into(), 0);
        pt_f.encode_coeff_i64(f, k_pt.into(), 0);

        let mut ct_t = module.glwe_alloc_from_infos(&glwe_infos);
        let mut ct_f = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_encrypt_sk(
            &mut ct_t,
            &pt_t,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        module.glwe_encrypt_sk(
            &mut ct_f,
            &pt_f,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut s: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);
        let mut s_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw_infos);
        let mut pt_sel: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
        pt_sel.raw_mut()[0] = bit;
        module.ggsw_encrypt_sk(
            &mut s,
            &pt_sel,
            sk,
            &ggsw_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        module.ggsw_prepare(&mut s_prepared, &s, &mut scratch.borrow());

        let mut ct_res = module.glwe_alloc_from_infos(&glwe_infos);
        module.cmux(&mut ct_res, &ct_t, &ct_f, &s_prepared, &mut scratch.borrow());

        let mut pt_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.glwe_decrypt(&ct_res, &mut pt_have, sk, &mut scratch.borrow());

        let want = if bit == 0 { f } else { t };
        assert_eq!(want, pt_have.decode_coeff_i64(k_pt.into(), 0));
    }
}

pub fn test_cswap_direct<BRA, BE>(test_context: &TestContext<BRA, BE>)
where
    BRA: BlindRotationAlgo,
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + Cswap<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
    BE: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();
    let ggsw_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();

    let a: i64 = 19;
    let b: i64 = 53;
    let k_pt: usize = 8;

    for bit in [0_i64, 1_i64] {
        let mut pt_a: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        let mut pt_b: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        pt_a.encode_coeff_i64(a, k_pt.into(), 0);
        pt_b.encode_coeff_i64(b, k_pt.into(), 0);

        let mut ct_a = module.glwe_alloc_from_infos(&glwe_infos);
        let mut ct_b = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_encrypt_sk(
            &mut ct_a,
            &pt_a,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        module.glwe_encrypt_sk(
            &mut ct_b,
            &pt_b,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut s: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);
        let mut s_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw_infos);
        let mut pt_sel: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
        pt_sel.raw_mut()[0] = bit;
        module.ggsw_encrypt_sk(
            &mut s,
            &pt_sel,
            sk,
            &ggsw_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        module.ggsw_prepare(&mut s_prepared, &s, &mut scratch.borrow());

        module.cswap(&mut ct_a, &mut ct_b, &s_prepared, &mut scratch.borrow());

        let mut pt_a_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        let mut pt_b_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.glwe_decrypt(&ct_a, &mut pt_a_have, sk, &mut scratch.borrow());
        module.glwe_decrypt(&ct_b, &mut pt_b_have, sk, &mut scratch.borrow());

        let (a_want, b_want) = if bit == 0 { (a, b) } else { (b, a) };
        assert_eq!(a_want, pt_a_have.decode_coeff_i64(k_pt.into(), 0));
        assert_eq!(b_want, pt_b_have.decode_coeff_i64(k_pt.into(), 0));
    }
}

pub fn test_fhe_uint_swap<BRA, BE>(test_context: &TestContext<BRA, BE>)
where
    BRA: BlindRotationAlgo,
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + Cswap<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
    BE: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();
    let ggsw_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();

    let mut s: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);
    let mut s_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw_infos);

    let a: u32 = source_xa.next_u32();
    let b: u32 = source_xa.next_u32();

    for bit in [0, 1] {
        let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
        let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);

        a_enc.encrypt_sk(
            module,
            a,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        b_enc.encrypt_sk(
            module,
            b,
            sk,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut pt: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
        pt.raw_mut()[0] = bit;
        module.ggsw_encrypt_sk(
            &mut s,
            &pt,
            sk,
            &ggsw_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        module.ggsw_prepare(&mut s_prepared, &s, &mut scratch.borrow());

        module.cswap(&mut a_enc, &mut b_enc, &s_prepared, &mut scratch.borrow());

        let (a_want, b_want) = if bit == 0 { (a, b) } else { (b, a) };

        assert_eq!(a_want, a_enc.decrypt(module, sk, &mut scratch.borrow()));
        assert_eq!(b_want, b_enc.decrypt(module, sk, &mut scratch.borrow()));
    }
}

pub fn test_glwe_blind_retrieval_statefull<BRA, BE>(test_context: &TestContext<BRA, BE>)
where
    BRA: BlindRotationAlgo,
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWEBlindRetrieval<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
    BE: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();
    let ggsw_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();

    let data: Vec<u32> = (0..32).map(|i| i as u32).collect_vec();

    let mut data_enc: Vec<FheUint<Vec<u8>, u32>> = (0..data.len())
        .map(|i| {
            let mut ct: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
            ct.encrypt_sk(
                module,
                data[i],
                sk,
                &glwe_enc_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            ct
        })
        .collect_vec();

    for idx in 0..data.len() as u32 {
        let mut idx_enc = FheUintPrepared::alloc_from_infos(module, &ggsw_infos);
        idx_enc.encrypt_sk(
            module,
            idx,
            sk,
            &ggsw_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        module.glwe_blind_retrieval_statefull(&mut data_enc, &idx_enc, 0, 5, &mut scratch.borrow());

        assert_eq!(data[idx as usize], data_enc[0].decrypt(module, sk, &mut scratch.borrow()));

        module.glwe_blind_retrieval_statefull_rev(&mut data_enc, &idx_enc, 0, 5, &mut scratch.borrow());

        for i in 0..data.len() {
            assert_eq!(data[i], data_enc[i].decrypt(module, sk, &mut scratch.borrow()))
        }
    }
}

pub fn test_glwe_blind_retriever<BRA, BE>(test_context: &TestContext<BRA, BE>)
where
    BRA: BlindRotationAlgo,
    Module<BE>: GLWEEncryptSk<BE> + GLWEDecrypt<BE> + GLWEBlindRetrieval<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
    BE: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: poulpy_core::layouts::GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();
    let ggsw_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();

    let data: Vec<u32> = (0..25).map(|i| i as u32).collect_vec();

    let data_enc: Vec<FheUint<Vec<u8>, u32>> = (0..data.len())
        .map(|i| {
            let mut ct: FheUint<Vec<u8>, u32> = FheUint::<Vec<u8>, u32>::alloc_from_infos(module, &glwe_infos);
            ct.encrypt_sk(
                module,
                data[i],
                sk,
                &glwe_enc_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            ct
        })
        .collect_vec();

    let mut retriever: GLWEBlindRetriever = GLWEBlindRetriever::alloc(module, &glwe_infos, data.len());
    for idx in 0..data.len() as u32 {
        let offset = 2;
        let mut idx_enc: FheUintPrepared<BE::OwnedBuf, u32, BE> = FheUintPrepared::alloc_from_infos(module, &ggsw_infos);
        idx_enc.encrypt_sk(
            module,
            idx << offset,
            sk,
            &ggsw_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut res: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(module, &glwe_infos);
        retriever.retrieve(module, &mut res, &data_enc, &idx_enc, offset, &mut scratch.borrow());

        assert_eq!(data[idx as usize], res.decrypt(module, sk, &mut scratch.borrow()));
    }
}
