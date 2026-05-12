use std::collections::HashMap;

use poulpy_core::{
    EncryptionLayout, GGSWEncryptSk, GLWEDecrypt, GLWEEncryptSk,
    layouts::{
        Base2K, Dnum, Dsize, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout, GLWEPlaintext, GLWESecretPrepared,
        GLWESecretPreparedFactory, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, Module, ScratchArena, ScratchOwned},
    source::Source,
};
use rand::Rng;

use crate::{
    bdd_arithmetic::{
        FheUintPrepared, GLWEBlindSelection,
        tests::test_suite::{TEST_FHEUINT_BASE2K, TEST_RANK, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_glwe_blind_selection<BRA, BE>(test_context: &TestContext<BRA, BE>)
where
    BRA: BlindRotationAlgo,
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GGSWEncryptSk<BE>
        + GLWEBlindSelection<u32, BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
    BE: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    let module: &Module<BE> = &test_context.module;
    let sk_glwe_prep: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;

    let base2k: Base2K = TEST_FHEUINT_BASE2K.into();
    let rank: Rank = TEST_RANK.into();
    let k_glwe: TorusPrecision = TorusPrecision(26);
    let k_ggsw: TorusPrecision = TorusPrecision(39);
    let dnum: Dnum = Dnum(3);

    let glwe_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k,
        k: k_glwe,
        rank,
    };
    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: module.n().into(),
        base2k,
        k: k_ggsw,
        rank,
        dnum,
        dsize: Dsize(1),
    };

    let mut source: Source = Source::new([6u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut res: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);

    let k: u32 = source.next_u32();

    let ggsw_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();
    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();

    let mut k_enc_prep: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::<BE::OwnedBuf, u32, BE>::alloc_from_infos(module, &ggsw_infos);
    k_enc_prep.encrypt_sk(
        module,
        k,
        sk_glwe_prep,
        &ggsw_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let digit = 5;
    let mask: u32 = (1 << digit) - 1;

    // Starting bit
    let mut bit_start: usize = 0;

    let mut data = vec![0i64; 1 << digit];
    data.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    for _ in 0..32_usize.div_ceil(digit) {
        let mut pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

        let mut cts_map: HashMap<usize, &mut GLWE<Vec<u8>>> = HashMap::new();
        let mut cts: Vec<GLWE<Vec<u8>>> = Vec::new();

        for value in data.iter().take(1 << digit) {
            pt.encode_coeff_i64(*value, TorusPrecision(base2k.as_u32()), 0);
            let mut ct = module.glwe_alloc_from_infos(&glwe_infos);
            module.glwe_encrypt_sk(
                &mut ct,
                &pt,
                sk_glwe_prep,
                &glwe_enc_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            cts.push(ct);
        }

        for (i, ct) in cts.iter_mut().enumerate() {
            if i.is_multiple_of(3) {
                cts_map.insert(i, ct);
            }
        }

        // How many bits to take
        let bit_size: usize = (32 - bit_start).min(digit);

        module.glwe_blind_selection(&mut res, cts_map, &k_enc_prep, bit_start, bit_size, &mut scratch.borrow());

        module.glwe_decrypt(&res, &mut pt, sk_glwe_prep, &mut scratch.borrow());

        let idx = ((k >> bit_start) & mask) as usize;
        if !idx.is_multiple_of(3) {
            assert_eq!(0, pt.decode_coeff_i64(TorusPrecision(base2k.as_u32()), 0));
        } else {
            assert_eq!(data[idx], pt.decode_coeff_i64(TorusPrecision(base2k.as_u32()), 0));
        }

        bit_start += digit;

        if bit_start >= 32 {
            break;
        }
    }
}
