use std::collections::HashMap;

use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DeviceBuf, Module, Scratch, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWEPacker, GLWEPackerOps, GLWERotate, GLWESub,
    ScratchTakeCore,
    glwe_packer::{glwe_packer_add, glwe_packer_flush, glwe_packer_galois_elements, glwe_packer_tmp_bytes},
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
};

pub fn test_glwe_packer<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEPackerOps<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWEDecrypt<BE>
        + GLWERotate<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_ct: usize = 4 * out_base2k + 1;
    let pt_k: usize = 2 * out_base2k + 1;
    let rank: usize = 3;
    let dsize: usize = 1;
    let k_ksk: usize = k_ct + key_base2k * dsize;

    let dnum: usize = k_ct.div_ceil(key_base2k * dsize);

    let glwe_out_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n.into(),
        base2k: out_base2k.into(),
        k: k_ct.into(),
        rank: rank.into(),
    })
    .unwrap();

    let key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: key_base2k.into(),
        k: k_ksk.into(),
        rank: rank.into(),
        dsize: dsize.into(),
        dnum: dnum.into(),
    })
    .unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        (module).glwe_encrypt_sk_tmp_bytes(&glwe_out_infos)
            | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&key_infos)
            | glwe_packer_tmp_bytes(module, &glwe_out_infos, &key_infos),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_out_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_dft, &sk);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });

    pt.encode_vec_i64(&data, pt_k.into());

    let gal_els: Vec<i64> = glwe_packer_galois_elements(module);

    let mut auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE>> = HashMap::new();
    let mut tmp: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&key_infos);
    gal_els.iter().for_each(|gal_el| {
        module.glwe_automorphism_key_encrypt_sk(
            &mut tmp,
            *gal_el,
            &sk,
            &key_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );
        let mut atk_prepared: GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&tmp);
        module.glwe_automorphism_key_prepare(&mut atk_prepared, &tmp, scratch.borrow());
        auto_keys.insert(*gal_el, atk_prepared);
    });

    let log_batch: usize = 0;

    let mut packer: GLWEPacker = GLWEPacker::alloc(&glwe_out_infos, log_batch);

    let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);

    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        &sk_dft,
        &glwe_out_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let log_n: usize = module.log_n();

    (0..n >> log_batch).for_each(|i| {
        module.glwe_encrypt_sk(
            &mut ct,
            &pt,
            &sk_dft,
            &glwe_out_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );

        module.glwe_rotate_assign(-(1 << log_batch), &mut pt, scratch.borrow()); // X^-batch * pt

        if reverse_bits_msb(i, log_n as u32).is_multiple_of(5) {
            glwe_packer_add(module, &mut packer, Some(&ct), &auto_keys, scratch.borrow());
        } else {
            glwe_packer_add(module, &mut packer, None::<&GLWE<Vec<u8>>>, &auto_keys, scratch.borrow())
        }
    });

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
    glwe_packer_flush(module, &mut packer, &mut res, scratch.borrow());

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        if i.is_multiple_of(5) {
            *x = reverse_bits_msb(i, log_n as u32) as i64;
        }
    });

    pt_want.encode_vec_i64(&data, pt_k.into());

    module.glwe_decrypt(&res, &mut pt, &sk_dft, scratch.borrow());

    module.glwe_sub_assign(&mut pt, &pt_want);

    let noise_have: f64 = pt.stats().std().log2();

    assert!(noise_have < -((k_ct - out_base2k) as f64), "noise: {noise_have}");
}

#[inline(always)]
fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
