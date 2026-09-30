use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{
        ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpPPolAlloc, SvpPrepare,
        VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftAlloc, VecZnxDftBytesOf, VecZnxDftZero, VecZnxFillUniformSource, VecZnxIdftApplyTmpA, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Module, PrepareHint, ScratchOwned, SvpPPolToBackendMut, SvpPPolToBackendRef, ToOwnedDeep, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, WriterTo, ZnxView,
    },
    source::Source,
    test_suite::{TestParams, scalar_znx_backend_mut, scalar_znx_backend_ref, vec_znx_backend_mut, vec_znx_backend_ref},
};

use crate::layouts::GLWESecretSampling;
use crate::test_suite::noise::glwe_noise_checked;
use crate::{
    EncryptionInfos, EncryptionLayout, GLWECompressedEncryptSk, GLWEEncryptPk, GLWEEncryptSk, GLWENoise, GLWENormalize,
    GLWEPublicKeyGenerate, GLWESub, GetDistribution, GetDistributionMut, ScalarZnxFillDistribution, VecZnxBigAddNormal,
    dist::Distribution,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWEPrepared, GLWEPreparedFactory, GLWEPublicKey,
        GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, Rank,
        compressed::{GLWECompressed, GLWEDecompress},
        prepared::{GLWEPublicKeyPrepared, GLWESecretPrepared},
    },
};

fn assert_canonical(ct: &GLWE<AlignedBuf, i64>) {
    let padding = ct.max_k().as_usize() - ct.k().as_usize();
    if padding == 0 {
        return;
    }

    let low_mask = (1i64 << padding) - 1;
    for col in 0..ct.data().cols() {
        assert!(ct.data().at(col, ct.max_size() - 1).iter().all(|value| value & low_mask == 0));
    }
}

pub fn test_glwe_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE> + GLWENoise<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniformSource<BE> + GLWESub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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

        let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&pt_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
                .max(module.glwe_noise_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.vec_znx_fill_uniform_source(
            base2k,
            pt_want.k().as_usize(),
            &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
            0,
            &mut source_xa,
        );

        module.glwe_encrypt_sk(
            &mut ct,
            &pt_want,
            &sk_prepared,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        assert_canonical(&ct);

        let noise_have: f64 = glwe_noise_checked(module, &ct, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = DEFAULT_SIGMA_XE.log2() - (k_ct as f64) + 0.5;

        assert!(
            noise_have <= noise_want,
            "noise_have: {noise_have} > noise_want: {noise_want}"
        );
    }
}

pub fn test_glwe_compressed_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWECompressedEncryptSk<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>
        + GLWESub<BE>
        + GLWEDecompress<Backend = BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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

        let mut ct_compressed: GLWECompressed<BE::OwnedBuf, BE::ZnxWord> = module.glwe_compressed_alloc_from_infos(&glwe_infos);

        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&pt_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            (module)
                .glwe_compressed_encrypt_sk_tmp_bytes(&glwe_infos)
                .max(module.glwe_noise_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.vec_znx_fill_uniform_source(
            base2k,
            pt_want.k().as_usize(),
            &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
            0,
            &mut source_xa,
        );

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

        let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        module.decompress_glwe(&mut ct, &ct_compressed);
        assert_canonical(&ct);

        let noise_have: f64 = glwe_noise_checked(module, &ct, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = DEFAULT_SIGMA_XE.log2() - (k_ct as f64) + 0.5;
        assert!(
            noise_have <= noise_want,
            "noise_have: {noise_have} > noise_want: {noise_want}"
        );

        let pt_zero: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        let mut source_xe_want: Source = Source::new([2u8; 32]);
        module.glwe_compressed_encrypt_sk(
            &mut ct_compressed,
            &pt_zero,
            &sk_prepared,
            seed_xa,
            &glwe_infos,
            &mut source_xe_want,
            &mut scratch.borrow(),
        );
        let mut ct_zero: GLWECompressed<BE::OwnedBuf, BE::ZnxWord> = module.glwe_compressed_alloc_from_infos(&glwe_infos);
        let mut source_xe_have: Source = Source::new([2u8; 32]);
        module.glwe_compressed_encrypt_zero_sk(
            &mut ct_zero,
            &sk_prepared,
            seed_xa,
            &glwe_infos,
            &mut source_xe_have,
            &mut scratch.borrow(),
        );
        let (mut bytes_want, mut bytes_have) = (Vec::new(), Vec::new());
        ct_compressed.write_to(&mut bytes_want).unwrap();
        ct_zero.write_to(&mut bytes_have).unwrap();
        assert_eq!(
            bytes_have, bytes_want,
            "zero encryption differs from encrypting a zero plaintext"
        );
        assert_eq!(source_xe_have.new_seed(), source_xe_want.new_seed());
    }
}

pub fn test_glwe_encrypt_zero_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE> + GLWENoise<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniformSource<BE> + GLWESub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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

        let pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_noise_tmp_bytes(&glwe_infos)
                .max((module).glwe_encrypt_sk_tmp_bytes(&glwe_infos)),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);

        module.glwe_encrypt_zero_sk(
            &mut ct,
            &sk_prepared,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        assert_canonical(&ct);

        let noise_have: f64 = glwe_noise_checked(module, &ct, &pt, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = DEFAULT_SIGMA_XE.log2() - (k_ct as f64) + 0.5;
        assert!(
            noise_have <= noise_want,
            "noise_have: {noise_have} > noise_want: {noise_want}"
        );
    }
}

/// FNV-1a digests of the serialized rank-1 public key and its two encryptions under fixed seeds.
pub fn glwe_public_key_rank1_digests<BE: crate::test_suite::noise::TestBackend>(module: &Module<BE>, base2k: usize) -> [u64; 3]
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    fn fnv1a(write: impl FnOnce(&mut Vec<u8>) -> std::io::Result<()>) -> u64 {
        let mut bytes: Vec<u8> = Vec::new();
        write(&mut bytes).unwrap();
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    let infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: (3 * base2k + 1).into(),
        rank: 1_usize.into(),
    })
    .unwrap();

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([3u8; 32]);
    let mut source_xu: Source = Source::new([4u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_pk_tmp_bytes(&infos, &infos)
            .max(module.glwe_public_key_generate_tmp_bytes(&infos))
            .max(module.glwe_public_key_prepare_tmp_bytes(&infos)),
    );

    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(1_usize.into());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut pk: GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_public_key_alloc_from_infos(&infos);
    module.glwe_public_key_generate(
        &mut pk,
        &sk_prepared,
        &infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&infos);
    module.vec_znx_fill_uniform_source(
        base2k,
        pt.k().as_usize(),
        &mut vec_znx_backend_mut::<BE>(&mut pt.data),
        0,
        &mut source_xa,
    );

    let mut pk_prepared: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&infos);
    module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());

    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    module.glwe_encrypt_pk(
        &mut ct,
        &pt,
        &pk_prepared,
        &infos,
        &mut source_xu,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let mut ct_zero: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    module.glwe_encrypt_zero_pk(
        &mut ct_zero,
        &pk_prepared,
        &infos,
        &mut source_xu,
        &mut source_xe,
        &mut scratch.borrow(),
    );

    // Hashes the rank-1 bytes from before the matrix layout, the distribution then entry 0, so the recorded digests still hold.
    [
        fnv1a(|b| {
            pk.dist().write_to(b)?;
            pk.at(0).write_to(b)
        }),
        fnv1a(|b| ct.write_to(b)),
        fnv1a(|b| ct_zero.write_to(b)),
    ]
}

pub fn test_glwe_encrypt_pk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>
        + GLWESub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k_ct: usize = base2k * 4 + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();
        let layout = |k: usize| GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };
        let glwe_infos = EncryptionLayout::new_from_default_sigma(layout(k_ct)).unwrap();

        // A public key more precise than the output needs scratch past the output's width.
        for k_pk in [k_ct, k_ct + base2k] {
            let pk_infos = EncryptionLayout::new_from_default_sigma(layout(k_pk)).unwrap();

            let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
            let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);
            let mut source_xu: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .glwe_noise_tmp_bytes(&glwe_infos)
                    .max(module.glwe_public_key_generate_tmp_bytes(&pk_infos))
                    .max(module.glwe_public_key_prepare_tmp_bytes(&pk_infos)),
            );
            let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_pk_tmp_bytes(&glwe_infos, &pk_infos));

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&glwe_infos);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            let mut pk: GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_public_key_alloc_from_infos(&pk_infos);
            module.glwe_public_key_generate(
                &mut pk,
                &sk_prepared,
                &pk_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            module.vec_znx_fill_uniform_source(
                base2k,
                pt_want.k().as_usize(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                0,
                &mut source_xa,
            );

            let mut pk_prepared: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_public_key_prepared_alloc_from_infos(&pk_infos);
            module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());

            module.glwe_encrypt_pk(
                &mut ct,
                &pt_want,
                &pk_prepared,
                &glwe_infos,
                &mut source_xu,
                &mut source_xe,
                &mut enc_scratch.borrow(),
            );
            assert_canonical(&ct);

            let noise_have: f64 = glwe_noise_checked(module, &ct, &pt_want, &sk_prepared, &mut scratch.borrow())
                .std()
                .log2();
            // Sum_l u_l e_l has rank terms, as Sum_j e_j s_j does.
            let noise_want: f64 =
                ((2.0 * rank as f64 * n as f64 * 0.5 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE).sqrt()).log2() - (k_ct as f64);
            let noise_want_tol: f64 = noise_want + 1.05_f64.log2();
            assert!(
                noise_have <= noise_want_tol,
                "noise_have: {noise_have} > noise_want_tol: {noise_want_tol} (noise_want: {noise_want})"
            );
        }
    }
}

/// Encrypting under a public key less precise than the output panics.
pub fn test_glwe_encrypt_pk_imprecise_key<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let layout = |k: usize| GLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1_usize.into(),
    };
    let glwe_infos = EncryptionLayout::new_from_default_sigma(layout(base2k * 4 + 1)).unwrap();
    let pk_layout = layout(base2k * 3 + 1);
    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
    let pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    let pk: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_pk_tmp_bytes(&glwe_infos, &pk_layout));
    module.glwe_encrypt_pk(
        &mut ct,
        &pt,
        &pk,
        &glwe_infos,
        &mut Source::new([0u8; 32]),
        &mut Source::new([0u8; 32]),
        &mut scratch.borrow(),
    );
}

/// `glwe_public_key_generate` and `glwe_encrypt_pk` equal their per-entry formulas replayed from the same sources.
pub fn test_glwe_encrypt_pk_replay<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE>
        + GLWEEncryptSk<BE>
        + GLWENormalize<BE>
        + GLWEPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>
        + ScalarZnxAlloc<BE>
        + ScalarZnxFillDistribution<BE>
        + SvpPPolAlloc<BE>
        + SvpPrepare<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxDftZero<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAddNormal<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = base2k * 3 + 1;
    let n: usize = module.n();

    for rank in 1_usize..3 {
        let infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_encrypt_pk_tmp_bytes(&infos, &infos)
                .max(module.glwe_public_key_generate_tmp_bytes(&infos))
                .max(module.glwe_public_key_prepare_tmp_bytes(&infos))
                .max(module.glwe_prepare_tmp_bytes(&infos))
                .max(module.vec_znx_big_normalize_tmp_bytes()),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([1u8; 32]));
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        let mut pk: GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_public_key_alloc_from_infos(&infos);
        module.glwe_public_key_generate(
            &mut pk,
            &sk_prepared,
            &infos,
            &mut Source::new([2u8; 32]),
            &mut Source::new([3u8; 32]),
            &mut scratch.borrow(),
        );

        let (mut xe, mut xa) = (Source::new([2u8; 32]), Source::new([3u8; 32]));
        let keys_want: Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> = (0..rank)
            .map(|_| {
                let mut key: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
                module.glwe_encrypt_zero_sk(&mut key, &sk_prepared, &infos, &mut xe, &mut xa, &mut scratch.borrow());
                module.glwe_normalize_assign(&mut key, &mut scratch.borrow());
                key
            })
            .collect();
        for (l, key) in keys_want.iter().enumerate() {
            assert_eq!(pk.at(l).to_owned_deep(), key.to_owned_deep(), "rank={rank} entry={l}");
        }

        let mut pk_prepared: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&infos);
        module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());

        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&infos);
        module.vec_znx_fill_uniform_source(
            base2k,
            pt.k().as_usize(),
            &mut vec_znx_backend_mut::<BE>(&mut pt.data),
            0,
            &mut Source::new([4u8; 32]),
        );

        let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
        module.glwe_encrypt_pk(
            &mut ct,
            &pt,
            &pk_prepared,
            &infos,
            &mut Source::new([5u8; 32]),
            &mut Source::new([6u8; 32]),
            &mut scratch.borrow(),
        );

        let mut source_xu: Source = Source::new([5u8; 32]);
        let mut source_xe: Source = Source::new([6u8; 32]);
        let entries: Vec<GLWEPrepared<BE::OwnedBuf, BE>> = keys_want
            .iter()
            .map(|entry| {
                let mut prepared: GLWEPrepared<BE::OwnedBuf, BE> = module.glwe_prepared_alloc_from_infos(entry);
                module.glwe_prepare(&mut prepared, entry, &mut scratch.borrow());
                prepared
            })
            .collect();
        let mut u = module.scalar_znx_alloc(n, rank);
        let mut u_prepared = module.svp_ppol_alloc(n, rank, PrepareHint::Reuse);
        for l in 0..rank {
            module.scalar_znx_fill_distribution(&mut scalar_znx_backend_mut::<BE>(&mut u), l, *pk.dist(), &mut source_xu);
            module.svp_prepare(&mut u_prepared.to_backend_mut(), l, &scalar_znx_backend_ref::<BE>(&u), l);
        }

        let size: usize = pk.size();
        let mut want: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
        let mut acc = module.vec_znx_dft_alloc(n, 1, size);
        let mut prod = module.vec_znx_dft_alloc(n, 1, size);
        let mut big = module.vec_znx_big_alloc(n, 1, size);
        for col in 0..rank + 1 {
            module.vec_znx_dft_zero(&mut acc.to_backend_mut(), 0);
            for (l, key) in entries.iter().enumerate() {
                module.svp_apply_dft_to_dft(
                    &mut prod.to_backend_mut(),
                    0,
                    &u_prepared.to_backend_ref(),
                    l,
                    &key.data.to_backend_ref(),
                    col,
                );
                module.vec_znx_dft_add_assign(&mut acc.to_backend_mut(), 0, &prod.to_backend_ref(), 0);
            }
            module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut acc.to_backend_mut(), 0);
            module.vec_znx_big_add_normal(base2k, &mut big.to_backend_mut(), 0, infos.noise_infos(), &mut source_xe);
            if col == 0 {
                module.vec_znx_big_add_small_assign(&mut big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&pt.data), 0);
            }
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut want.data),
                base2k,
                k,
                0,
                col,
                &big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.borrow(),
            );
        }
        assert_eq!(ct, want, "rank={rank}");

        // The ephemerals, their DFT, the product and its scratch lead the arena; left there they would decrypt `ct`.
        let bytes: usize = module.glwe_encrypt_pk_tmp_bytes(&infos, &infos);
        let mut zeroed: ScratchOwned<BE> = ScratchOwned {
            data: BE::from_host_bytes(&vec![0u8; bytes]),
            _phantom: std::marker::PhantomData,
        };
        module.glwe_encrypt_pk(
            &mut ct,
            &pt,
            &pk_prepared,
            &infos,
            &mut Source::new([5u8; 32]),
            &mut Source::new([6u8; 32]),
            &mut zeroed.borrow(),
        );
        let u_dft_start: usize = BE::scratch_aligned(BE::bytes_of_scalar_znx(n, rank));
        let product_start: usize = BE::scratch_aligned(u_dft_start + module.bytes_of_vec_znx_dft(n, rank, 1));
        let tail_start: usize = BE::scratch_aligned(product_start + module.bytes_of_vec_znx_dft(n, rank + 1, size));
        let vmp: usize = module.vmp_apply_dft_to_dft_tmp_bytes(size, 1, 1, rank, rank + 1, size);
        let wiped: usize = tail_start + BE::bytes_of_vec_znx(n, 1, vmp.div_ceil(BE::bytes_of_vec_znx(n, 1, 1)));
        let arena: Vec<u8> = BE::to_host_bytes(&zeroed.data);
        assert!(
            arena[..wiped].iter().all(|&b| b == 0),
            "rank={rank}: ephemerals or their products left in scratch"
        );
    }
}

/// A key tagged with a zero-weight distribution would encrypt with `u = 0`.
pub fn test_glwe_encrypt_pk_zero_ephemeral<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    encrypt_zero_pk_with_ephemeral(params, module, Distribution::TernaryProb(0.0));
}

/// A NaN probability draws only zero ephemerals.
pub fn test_glwe_encrypt_pk_nan_ephemeral<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    encrypt_zero_pk_with_ephemeral(params, module, Distribution::BinaryProb(f64::NAN));
}

fn encrypt_zero_pk_with_ephemeral<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
    dist: Distribution,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptPk<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: module.n().into(),
        base2k: params.base2k.into(),
        k: (params.base2k * 3 + 1).into(),
        rank: 2_usize.into(),
    })
    .unwrap();
    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    let mut pk: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&infos);
    *pk.dist_mut() = dist;
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_pk_tmp_bytes(&infos, &infos));
    module.glwe_encrypt_zero_pk(
        &mut ct,
        &pk,
        &infos,
        &mut Source::new([0u8; 32]),
        &mut Source::new([0u8; 32]),
        &mut scratch.borrow(),
    );
}

/// A public key has one entry per rank, so rank 0 has none.
pub fn test_glwe_public_key_rank_zero<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
{
    let _: GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> =
        module.glwe_public_key_alloc(params.base2k.into(), (params.base2k * 3 + 1).into(), Rank(0));
}
