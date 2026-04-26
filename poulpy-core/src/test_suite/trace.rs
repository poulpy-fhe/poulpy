use std::collections::HashMap;

use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform, VecZnxNormalizeAssign, VecZnxSubAssign},
    layouts::{DeviceBuf, Module, Scratch, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    glwe_trace::GLWETrace,
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory, LWEInfos,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::var_noise_gglwe_product,
};

pub fn test_glwe_trace_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWETrace<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWESecretPreparedFactory<BE>
        + VecZnxSubAssign
        + VecZnxNormalizeAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k;
    let key_base2k: usize = base2k - 1;
    let k: usize = 4 * base2k + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();
        let k_autokey: usize = k + key_base2k;

        let dsize: usize = 1;
        let dnum: usize = k.div_ceil(key_base2k * dsize);

        let glwe_out_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();

        let key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: n.into(),
            base2k: key_base2k.into(),
            k: k_autokey.into(),
            rank: rank.into(),
            dsize: dsize.into(),
            dnum: dnum.into(),
        })
        .unwrap();

        let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module).glwe_encrypt_sk_tmp_bytes(&glwe_out_infos)
                | (module).glwe_decrypt_tmp_bytes(&glwe_out_infos)
                | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&key_infos)
                | module.glwe_trace_tmp_bytes(&glwe_out_infos, &glwe_out_infos, &key_infos),
        );

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_out_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk);

        let mut data_want: Vec<i64> = vec![0i64; n];

        data_want.iter_mut().for_each(|x| *x = source_xa.next_i64() & 0xFF);

        module.vec_znx_fill_uniform(out_base2k, &mut pt_have.data, 0, &mut source_xa);

        module.glwe_encrypt_sk(
            &mut glwe_out,
            &pt_have,
            &sk_dft,
            &glwe_out_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );

        let mut auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE>> = HashMap::new();
        let gal_els: Vec<i64> = module.glwe_trace_galois_elements();
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

        module.glwe_trace_assign(&mut glwe_out, 0, &auto_keys, scratch.borrow());

        (0..pt_want.size()).for_each(|i| pt_want.data.at_mut(0, i)[0] = pt_have.data.at(0, i)[0]);

        module.glwe_decrypt(&glwe_out, &mut pt_have, &sk_dft, scratch.borrow());

        module.vec_znx_sub_assign(&mut pt_want.data, 0, &pt_have.data, 0);
        module.vec_znx_normalize_assign(pt_want.base2k().as_usize(), &mut pt_want.data, 0, scratch.borrow());

        let noise_have: f64 = pt_want.stats().std().log2();

        let mut noise_want: f64 = var_noise_gglwe_product(
            n as f64,
            key_base2k * dsize,
            0.5,
            0.5,
            1.0 / 12.0,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            0.0,
            rank as f64,
            k,
            k_autokey,
        );
        noise_want += DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE * (-2.0 * (k) as f64).exp2();
        noise_want += n as f64 * 1.0 / 12.0 * 0.5 * rank as f64 * (-2.0 * (k) as f64).exp2();
        noise_want = noise_want.sqrt().log2();

        assert!(
            (noise_have - noise_want).abs() < 1.0,
            "{noise_have} > {noise_want} {}",
            noise_have - noise_want
        );
    }
}
