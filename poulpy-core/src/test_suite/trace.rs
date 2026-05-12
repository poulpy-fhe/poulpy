use std::collections::HashMap;

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeAssignBackend, VecZnxSubAssignBackend},
    layouts::{Module, ScratchOwned, VecZnxToBackendMut, VecZnxToBackendRef, ZnxViewMut},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWETrace, ScratchArenaTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::var_noise_gglwe_product,
    test_suite::{download_glwe_plaintext, upload_glwe, upload_glwe_automorphism_key, upload_glwe_plaintext, upload_glwe_secret},
};

pub fn test_glwe_trace_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETrace<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxSubAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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

        let glwe_out_template: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let pt_template: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>>;

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module).glwe_encrypt_sk_tmp_bytes(&glwe_out_infos)
                | (module).glwe_decrypt_tmp_bytes(&glwe_out_infos)
                | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&key_infos)
                | module.glwe_trace_tmp_bytes(&glwe_out_infos, &glwe_out_infos, &key_infos),
        );

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_out_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let sk_backend = upload_glwe_secret(module, &sk);

        let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut sk_dft, &sk_backend);

        let mut data_want: Vec<i64> = vec![0i64; n];

        data_want.iter_mut().for_each(|x| *x = source_xa.next_i64() & 0xFF);

        pt_want = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        for j in 0..pt_want.data.size() {
            pt_want.data.at_mut(0, j).fill(0);
        }
        pt_want.data.at_mut(0, 0)[0] = data_want[0];
        let pt_input = upload_glwe_plaintext(module, &pt_want);

        let mut glwe_out = upload_glwe(module, &glwe_out_template);
        module.glwe_encrypt_sk(
            &mut glwe_out,
            &pt_input,
            &sk_dft,
            &glwe_out_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
        let gal_els: Vec<i64> = module.glwe_trace_galois_elements();
        let tmp_template: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&key_infos);
        gal_els.iter().for_each(|gal_el| {
            let mut tmp = upload_glwe_automorphism_key(module, &tmp_template);
            module.glwe_automorphism_key_encrypt_sk(
                &mut tmp,
                *gal_el,
                &sk_backend,
                &key_infos,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );
            let mut atk_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&tmp);
            module.glwe_automorphism_key_prepare(&mut atk_prepared, &tmp, &mut scratch.borrow());
            auto_keys.insert(*gal_el, atk_prepared);
        });

        module.glwe_trace_assign(&mut glwe_out, 0, &auto_keys, key_infos.size(), &mut scratch.borrow());
        let mut pt_have_backend = upload_glwe_plaintext(module, &pt_template);
        module.glwe_decrypt(&glwe_out, &mut pt_have_backend, &sk_dft, &mut scratch.borrow());
        let pt_have: GLWEPlaintext<Vec<u8>> = download_glwe_plaintext(module, &pt_have_backend);

        {
            let mut pt_want_data =
                <poulpy_hal::layouts::VecZnx<Vec<u8>> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut pt_want.data);
            let pt_have_data = <poulpy_hal::layouts::VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(&pt_have.data);
            module.vec_znx_sub_assign_backend(&mut pt_want_data, 0, &pt_have_data, 0);
        }
        let mut pt_noise = upload_glwe_plaintext(module, &pt_want);
        module.vec_znx_normalize_assign_backend(
            pt_noise.base2k().as_usize(),
            &mut vec_znx_backend_mut::<BE>(&mut pt_noise.data),
            0,
            &mut scratch.borrow(),
        );
        pt_want = download_glwe_plaintext(module, &pt_noise);

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
