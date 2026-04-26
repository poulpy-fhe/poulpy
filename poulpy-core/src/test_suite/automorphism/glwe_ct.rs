use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismAssign, VecZnxFillUniform},
    layouts::{DeviceBuf, Module, Scratch, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWENoise, GLWENormalize,
    ScratchTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    var_noise_gglwe_product_v2,
};

pub fn test_glwe_automorphism<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssign<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = base2k - 2;
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + key_base2k * dsize;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(key_base2k * dsize);

            let ct_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ct_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
            })
            .unwrap();

            let mut autokey: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&autokey_infos);
            let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_in_infos);
            let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_out_infos);
            let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_in_infos);
            let mut pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(in_base2k, &mut pt_in.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                    | (module).glwe_decrypt_tmp_bytes(&ct_out)
                    | (module).glwe_encrypt_sk_tmp_bytes(&ct_in)
                    | module.glwe_automorphism_tmp_bytes(&ct_out, &ct_in, &autokey),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct_out);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut autokey,
                p,
                &sk,
                &autokey_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            module.glwe_encrypt_sk(
                &mut ct_in,
                &pt_in,
                &sk_prepared,
                &ct_in_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey_infos);
            module.glwe_automorphism_key_prepare(&mut autokey_prepared, &autokey, scratch.borrow());

            module.glwe_automorphism(&mut ct_out, &ct_in, &autokey_prepared, scratch.borrow());

            let max_noise: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum,
                max_dsize,
                key_base2k,
                0.5,
                0.5,
                0f64,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
                rank as f64,
            )
            .sqrt()
            .log2();

            module.glwe_normalize(&mut pt_out, &pt_in, scratch.borrow());
            module.vec_znx_automorphism_assign(p, &mut pt_out.data, 0, scratch.borrow());

            assert!(
                module
                    .glwe_noise(&ct_out, &pt_out, &sk_prepared, scratch.borrow())
                    .std()
                    .log2()
                    <= max_noise + 1.0
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_automorphism_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(key_base2k * dsize);

            let ct_out_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            })
            .unwrap();

            let autokey_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                rank: rank.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
            })
            .unwrap();

            let mut autokey: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&autokey_infos);
            let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_out_infos);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(out_base2k, &mut pt_want.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&autokey)
                    | (module).glwe_decrypt_tmp_bytes(&ct)
                    | (module).glwe_encrypt_sk_tmp_bytes(&ct)
                    | module.glwe_automorphism_tmp_bytes(&ct, &ct, &autokey),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.glwe_automorphism_key_encrypt_sk(
                &mut autokey,
                p,
                &sk,
                &autokey_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            module.glwe_encrypt_sk(
                &mut ct,
                &pt_want,
                &sk_prepared,
                &ct_out_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&autokey);
            module.glwe_automorphism_key_prepare(&mut autokey_prepared, &autokey, scratch.borrow());

            module.glwe_automorphism_assign(&mut ct, &autokey_prepared, scratch.borrow());

            let max_noise: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum,
                dsize,
                key_base2k,
                0.5,
                0.5,
                0f64,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
                rank as f64,
            )
            .sqrt()
            .log2();

            module.vec_znx_automorphism_assign(p, &mut pt_want.data, 0, scratch.borrow());

            assert!(module.glwe_noise(&ct, &pt_want, &sk_prepared, scratch.borrow()).std().log2() <= max_noise + 1.0)
        }
    }
}
