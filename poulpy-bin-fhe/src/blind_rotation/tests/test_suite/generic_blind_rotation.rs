use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, ScratchOwned, ZnxView},
    source::Source,
};

use crate::blind_rotation::{
    BlindRotationAlgo, BlindRotationExecute, BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyLayout,
    BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, LookUpTableLayout, LookupTable, LookupTableFactory, mod_switch_2n,
};

use poulpy_core::{
    EncryptionLayout, GLWEDecrypt, LWEEncryptSk,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, LWE, LWEInfos, LWELayout, LWEPlaintext,
        LWESecret, ModuleCoreAlloc, prepared::GLWESecretPrepared,
    },
};

pub fn test_blind_rotation<BRA: BlindRotationAlgo, M, BE: Backend<OwnedBuf = Vec<u8>>>(
    module: &M,
    n_lwe: usize,
    block_size: usize,
    extension_factor: usize,
) where
    M: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + BlindRotationExecute<BRA, BE>
        + LookupTableFactory
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut + AsMut<[u8]> + AsRef<[u8]> + Sync,
    for<'a> BE::OwnedBuf: HostDataRef,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    let n_glwe: usize = module.n();
    let base2k: usize = 19;
    let k_lwe: usize = 24;
    let k_brk: usize = 3 * base2k;
    let rows_brk: usize = 2; // Ensures first limb is noise-free.
    let k_lut: usize = base2k;
    let k_res: usize = 2 * base2k;
    let rank: usize = 1;

    let log_message_modulus: usize = 4;

    let message_modulus: usize = 1 << log_message_modulus;

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let brk_infos = EncryptionLayout::new_from_default_sigma(BlindRotationKeyLayout {
        n_glwe: n_glwe.into(),
        n_lwe: n_lwe.into(),
        base2k: base2k.into(),
        k: k_brk.into(),
        dnum: rows_brk.into(),
        rank: rank.into(),
    })
    .unwrap();

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_res.into(),
        rank: rank.into(),
    })
    .unwrap();

    let lwe_infos = EncryptionLayout::new_from_default_sigma(LWELayout {
        n: n_lwe.into(),
        k: k_lwe.into(),
        base2k: base2k.into(),
    })
    .unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::<BE>::alloc(BlindRotationKey::encrypt_sk_tmp_bytes(module, &brk_infos));

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_glwe_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk_glwe_dft, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut scratch_br: ScratchOwned<BE> = ScratchOwned::<BE>::alloc(BlindRotationKeyPrepared::execute_tmp_bytes(
        module,
        block_size,
        extension_factor,
        &glwe_infos,
        &brk_infos,
    ));

    let mut brk: BlindRotationKey<Vec<u8>, BRA> = BlindRotationKey::<Vec<u8>, BRA>::alloc(module, &brk_infos);

    module.blind_rotation_key_encrypt_sk(
        &mut brk,
        &sk_glwe_dft,
        &sk_lwe,
        &brk_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut lwe: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc_from_infos(&lwe_infos);

    let x: i64 = 15 % (message_modulus as i64);

    pt_lwe.encode_i64(x, (log_message_modulus + 1).into());

    module.lwe_encrypt_sk(
        &mut lwe,
        &pt_lwe,
        &sk_lwe,
        &lwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let f = |x: i64| -> i64 { 2 * x + 1 };

    let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
    f_vec.iter_mut().enumerate().for_each(|(i, x)| *x = f(i as i64));

    let lut_infos = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: k_lut.into(),
        base2k: base2k.into(),
    };

    let mut lut: LookupTable = LookupTable::alloc(module, &lut_infos);
    lut.set(module, &f_vec, log_message_modulus + 1);

    let mut res: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);

    let mut brk_prepared: BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> = BlindRotationKeyPrepared::alloc(module, &brk);
    brk_prepared.prepare(module, &brk, &mut scratch_br.borrow());

    brk_prepared.execute(module, &mut res, &lwe, &lut, &mut scratch_br.borrow());

    let mut pt_have: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

    module.glwe_decrypt(&res, &mut pt_have, &sk_glwe_dft, &mut scratch.borrow());

    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space

    mod_switch_2n::<HostBytesBackend, _>(2 * lut.domain_size(), &mut lwe_2n, &lwe, lut.rotation_direction());

    let pt_want: i64 =
        (lwe_2n[0] + lwe_2n[1..].iter().zip(sk_lwe.raw()).map(|(x, y)| x * y).sum::<i64>()) & (2 * lut.domain_size() - 1) as i64;

    lut.rotate(module, pt_want);

    // First limb should be exactly equal (test are parameterized such that the noise does not reach
    // the first limb)
    assert_eq!(pt_have.data.at(0, 0), lut.data[0].data().at(0, 0));

    // Verify that it effectively compute f(x)
    let mut have: i64 = pt_have.decode_coeff_i64((log_message_modulus + 1).into(), 0);

    // Get positive representative and assert equality
    have = (have + message_modulus as i64) % (message_modulus as i64);

    assert_eq!(have, f(x) % (message_modulus as i64));
}
