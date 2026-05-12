use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEEncryptPk, GLWEEncryptSk, GLWENoise, GLWEPublicKeyGenerate, GLWESub,
    ScratchArenaTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWEPreparedFactory, GLWEPublicKey, GLWEPublicKeyPreparedFactory,
        GLWESecret, GLWESecretPreparedFactory, ModuleCoreAlloc, ModuleCoreCompressedAlloc,
        compressed::{GLWECompressed, GLWEDecompress},
        prepared::{GLWEPublicKeyPrepared, GLWESecretPrepared},
    },
};

pub fn test_glwe_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>:
        GLWEEncryptSk<BE> + GLWENoise<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniformSourceBackend<BE> + GLWESub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k_ct: usize = base2k * 4 + 1;
    let k_pt: usize = base2k * 2 + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        })
        .unwrap();

        let pt_infos: GLWEPlaintextLayout = GLWEPlaintextLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_pt.into(),
        };

        let mut ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&pt_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
                .max(module.glwe_noise_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt_want.data), 0, &mut source_xa);

        module.glwe_encrypt_sk(
            &mut ct,
            &pt_want,
            &sk_prepared,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let noise_have: f64 = module
            .glwe_noise(&ct, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = DEFAULT_SIGMA_XE.log2() - (k_ct as f64) + 0.5;

        assert!(
            noise_have <= noise_want,
            "noise_have: {noise_have} > noise_want: {noise_want}"
        );
    }
}

pub fn test_glwe_compressed_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWECompressedEncryptSk<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GLWESub<BE>
        + GLWEDecompress<Backend = BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k_ct: usize = base2k * 4 + 1;
    let k_pt: usize = base2k * 2 + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        })
        .unwrap();

        let pt_infos: GLWEPlaintextLayout = GLWEPlaintextLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_pt.into(),
        };

        let mut ct_compressed: GLWECompressed<Vec<u8>> = module.glwe_compressed_alloc_from_infos(&glwe_infos);

        let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&pt_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_compressed_encrypt_sk_tmp_bytes(&glwe_infos)
                .max(module.glwe_noise_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt_want.data), 0, &mut source_xa);

        let seed_xa: [u8; 32] = [1u8; 32];

        module.glwe_compressed_encrypt_sk(
            &mut ct_compressed,
            &pt_want,
            &sk_prepared,
            seed_xa,
            &glwe_infos,
            &mut source_xe,
            &mut scratch.borrow(),
        );

        let mut ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
        module.decompress_glwe(&mut ct, &ct_compressed);

        let noise_have: f64 = module
            .glwe_noise(&ct, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = DEFAULT_SIGMA_XE.log2() - (k_ct as f64) + 0.5;
        assert!(
            noise_have <= noise_want,
            "noise_have: {noise_have} > noise_want: {noise_want}"
        );
    }
}

pub fn test_glwe_encrypt_zero_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>:
        GLWEEncryptSk<BE> + GLWENoise<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniformSourceBackend<BE> + GLWESub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k_ct: usize = base2k * 4 + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        })
        .unwrap();

        let pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_noise_tmp_bytes(&glwe_infos)
                .max((module).glwe_encrypt_sk_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let mut ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);

        module.glwe_encrypt_zero_sk(
            &mut ct,
            &sk_prepared,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let noise_have: f64 = module.glwe_noise(&ct, &pt, &sk_prepared, &mut scratch.borrow()).std().log2();
        let noise_want: f64 = DEFAULT_SIGMA_XE.log2() - (k_ct as f64) + 0.5;
        assert!(
            noise_have <= noise_want,
            "noise_have: {noise_have} > noise_want: {noise_want}"
        );
    }
}

pub fn test_glwe_encrypt_pk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GLWESub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let k_ct: usize = base2k * 4 + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        })
        .unwrap();

        let mut ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut source_xu: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_noise_tmp_bytes(&glwe_infos)
                .max(module.glwe_encrypt_pk_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let mut pk: GLWEPublicKey<Vec<u8>> = module.glwe_public_key_alloc_from_infos(&glwe_infos);
        module.glwe_public_key_generate(&mut pk, &sk_prepared, &glwe_infos, &mut source_xe, &mut source_xa);

        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt_want.data), 0, &mut source_xa);

        let mut pk_prepared: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_public_key_prepared_alloc_from_infos(&glwe_infos);
        module.glwe_prepare(&mut pk_prepared.key, &pk.key);
        pk_prepared.dist = pk.dist;

        module.glwe_encrypt_pk(
            &mut ct,
            &pt_want,
            &pk_prepared,
            &glwe_infos,
            &mut source_xu,
            &mut source_xe,
            &mut scratch.borrow(),
        );

        let noise_have: f64 = module
            .glwe_noise(&ct, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 =
            ((((rank as f64) + 1.0) * n as f64 * 0.5 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE).sqrt()).log2() - (k_ct as f64);
        let noise_want_tol: f64 = noise_want + 1.05_f64.log2();
        assert!(
            noise_have <= noise_want_tol,
            "noise_have: {noise_have} > noise_want_tol: {noise_want_tol} (noise_want: {noise_want})"
        );
    }
}
