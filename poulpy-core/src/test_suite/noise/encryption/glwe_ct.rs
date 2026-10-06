use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{
        ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpPPolAlloc, SvpPrepare,
        VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftAlloc, VecZnxDftBytesOf, VecZnxDftZero, VecZnxFillUniformSource, VecZnxIdftApplyTmpA, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Module, PrepareHint, ReaderFrom, Ring, ScratchOwned, SvpPPolToBackendMut, SvpPPolToBackendRef, ToOwnedDeep,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, WriterTo, ZnxView,
    },
    source::Source,
    test_suite::{TestParams, scalar_znx_backend_mut, scalar_znx_backend_ref, vec_znx_backend_mut, vec_znx_backend_ref},
};

use crate::layouts::GLWESecretSampling;
use crate::test_suite::noise::glwe_noise_checked;
use crate::{
    GLWECompressedEncryptSk, GLWEDecrypt, GLWEEncryptPk, GLWEEncryptSk, GLWEMaskFill, GLWENoise, GLWENormalize,
    GLWEPublicKeyCompressedGenerate, GLWEPublicKeyGenerate, GLWESub, GetDistribution, GetDistributionMut, Noise,
    ScalarZnxFillDistribution, VecZnxAddNoise, VecZnxBigAddNoise,
    dist::Distribution,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWEPrepared, GLWEPreparedFactory, GLWEPublicKey,
        GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, Rank,
        compressed::{GLWECompressed, GLWEDecompress, GLWEPublicKeyDecompress},
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

/// Encryption over caller-set masks equals `glwe_encrypt_sk` drawing the same masks.
pub fn test_glwe_encrypt_sk_with_mask<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE> + GLWEMaskFill<BE> + GLWESecretPreparedFactory<BE> + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    for rank in 1_usize..3 {
        let n: usize = module.n();
        let glwe_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: (base2k * 4 + 1).into(),
            rank: rank.into(),
        };
        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: (base2k * 2 + 1).into(),
        });
        module.vec_znx_fill_uniform_source(
            base2k,
            pt.k().as_usize(),
            &mut vec_znx_backend_mut::<BE>(&mut pt.data),
            0,
            &mut Source::new([1u8; 32]),
        );
        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&glwe_infos);
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([2u8; 32]));
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_sk_tmp_bytes(&glwe_infos));

        let mut want: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_encrypt_sk(
            &mut want,
            &pt,
            &sk_prepared,
            &mut Source::new([3u8; 32]),
            &mut Source::new([4u8; 32]),
            &mut scratch.borrow(),
        );
        let mut have: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        module.fill_glwe_mask_from_source(&mut have, &mut Source::new([4u8; 32]));
        module.glwe_encrypt_sk_with_mask(
            &mut have,
            &pt,
            &sk_prepared,
            &mut Source::new([3u8; 32]),
            &mut scratch.borrow(),
        );
        assert!(have == want, "encryption with supplied mask differs for rank={rank}");
    }
}

pub fn test_glwe_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>
        + GLWESub<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k_ct: usize = base2k * 4 + 1;
    let k_pt: usize = base2k * 2 + 1;

    for rank in 1_usize..3 {
        let n: usize = module.n();

        let glwe_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        };

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
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        assert_canonical(&ct);
        assert_eq!(
            ct.noise(),
            Some(crate::ComponentNoise::from_secret_at(sk.dist, ct.k(), ct.rank().as_usize()))
        );

        let noise_have: f64 = glwe_noise_checked(module, &ct, &pt_want, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let noise_want: f64 = DEFAULT_SIGMA_XE.log2() - (k_ct as f64) + 0.5;

        assert!(
            noise_have <= noise_want,
            "noise_have: {noise_have} > noise_want: {noise_want}"
        );

        // The products of the masks and the secret would reveal it from the scratch.
        crate::test_suite::assert_wipes_scratch::<BE>(module.glwe_encrypt_sk_tmp_bytes(&glwe_infos), |scratch| {
            module.glwe_encrypt_sk(&mut ct, &pt_want, &sk_prepared, &mut source_xe, &mut source_xa, scratch)
        });
        let mut pt_have: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&pt_infos);
        crate::test_suite::assert_wipes_scratch::<BE>(module.glwe_decrypt_tmp_bytes(&ct), |scratch| {
            module.glwe_decrypt(&ct, &mut pt_have, &sk_prepared, scratch)
        });
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

        let glwe_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        };

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
            &mut source_xe,
            &mut scratch.borrow(),
        );

        let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        module.decompress_glwe(&mut ct, &ct_compressed);
        assert_eq!(ct.noise(), ct_compressed.noise());
        assert_canonical(&ct);
        assert_eq!(
            ct.noise(),
            Some(crate::ComponentNoise::from_secret_at(sk.dist, ct.k(), ct.rank().as_usize()))
        );

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
            &mut source_xe_want,
            &mut scratch.borrow(),
        );
        let mut ct_zero: GLWECompressed<BE::OwnedBuf, BE::ZnxWord> = module.glwe_compressed_alloc_from_infos(&glwe_infos);
        let mut source_xe_have: Source = Source::new([2u8; 32]);
        module.glwe_compressed_encrypt_zero_sk(
            &mut ct_zero,
            &sk_prepared,
            seed_xa,
            &mut source_xe_have,
            &mut scratch.borrow(),
        );
        assert!(
            ct_zero == ct_compressed,
            "zero encryption differs from encrypting a zero plaintext"
        );
        assert!(source_xe_have.new_seed() == source_xe_want.new_seed(), "values differ");
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

        let glwe_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ct.into(),
            rank: rank.into(),
        };

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

        module.glwe_encrypt_zero_sk(&mut ct, &sk_prepared, &mut source_xe, &mut source_xa, &mut scratch.borrow());
        assert_canonical(&ct);
        assert_eq!(
            ct.noise(),
            Some(crate::ComponentNoise::from_secret_at(sk.dist, ct.k(), ct.rank().as_usize()))
        );

        // Reproduce the error independently at the ciphertext's partial-limb
        // precision. This detects placement above its least significant bit.
        let mut error_expected = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.vec_znx_add_noise(
            base2k,
            k_ct,
            &mut vec_znx_backend_mut::<BE>(&mut error_expected.data),
            0,
            Noise::ENCRYPTION,
            &mut Source::new([1u8; 32]),
        );
        module.glwe_normalize_assign(&mut error_expected, &mut scratch.borrow());
        let mut error_actual = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.glwe_decrypt(&ct, &mut error_actual, &sk_prepared, &mut scratch.borrow());
        assert!(error_actual == error_expected, "decrypted fresh noise differs");

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

    let infos = GLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        k: (3 * base2k + 1).into(),
        rank: 1_usize.into(),
    };

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
    module.glwe_public_key_generate(&mut pk, &sk_prepared, &mut source_xe, &mut source_xa, &mut scratch.borrow());

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
        &mut source_xu,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let mut ct_zero: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    module.glwe_encrypt_zero_pk(
        &mut ct_zero,
        &pk_prepared,
        &mut source_xu,
        &mut source_xe,
        &mut scratch.borrow(),
    );

    // Hash the public-key distribution and first entry, then the message and zero ciphertexts.
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
        + GLWEPublicKeyCompressedGenerate<BE>
        + GLWEPublicKeyDecompress
        + GLWEDecompress<Backend = BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>
        + GLWESub<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k_ct: usize = base2k * 4 + 1;

    for law in [
        Distribution::TernaryProb(0.5),
        Distribution::TernaryProb(2.0 / 3.0),
        Distribution::BinaryProb(0.5),
    ] {
        let binary = matches!(law, Distribution::BinaryProb(_));
        for rank in 1_usize..3 {
            let n: usize = module.n();
            let layout = |k: usize| GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k.into(),
                rank: rank.into(),
            };
            let glwe_infos = layout(k_ct);

            // A public key more precise than the output needs scratch past the output's width; a
            // decompressed key encrypts as a generated one.
            for (k_pk, compressed) in [
                (k_ct, false),
                (k_ct + 1, false),
                (k_ct + base2k, false),
                (k_ct, true),
                (k_ct + 1, true),
                (k_ct + base2k, true),
            ] {
                let pk_infos = layout(k_pk);

                // Pool independent encryptions so one polynomial's sampling variation
                // does not decide the test. Every random role uses a distinct child stream.
                const TRIALS: usize = 64;
                let mut trial_seeds = Source::new([0u8; 32]);
                let mut noise_variance = 0.0;
                let mut modeled_variance = 0.0;
                let mut coefficient_zero = 0.0;
                for _ in 0..TRIALS {
                    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
                    let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_plaintext_alloc_from_infos(&glwe_infos);

                    let mut role_source = || Source::new(trial_seeds.new_seed());
                    let mut source_xs = role_source();
                    let mut source_xe = role_source();
                    let mut source_xa = role_source();
                    let mut source_xu = role_source();
                    let mask_seed = trial_seeds.new_seed();

                    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                        module
                            .glwe_noise_tmp_bytes(&glwe_infos)
                            .max(module.glwe_public_key_generate_tmp_bytes(&pk_infos))
                            .max(module.glwe_public_key_compressed_generate_tmp_bytes(&pk_infos))
                            .max(module.glwe_public_key_prepare_tmp_bytes(&pk_infos)),
                    );
                    let mut enc_scratch: ScratchOwned<BE> =
                        ScratchOwned::alloc(module.glwe_encrypt_pk_tmp_bytes(&glwe_infos, &pk_infos));

                    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&glwe_infos);
                    if binary {
                        module.glwe_secret_fill_binary_prob(&mut sk, 0.5, &mut source_xs);
                    } else {
                        if let Distribution::TernaryProb(p) = law {
                            module.glwe_secret_fill_ternary_prob(&mut sk, p, &mut source_xs);
                        }
                    }

                    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
                    module.glwe_secret_prepare(&mut sk_prepared, &sk);

                    let mut pk: GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_public_key_alloc_from_infos(&pk_infos);
                    if compressed {
                        let mut pk_compressed = module.glwe_public_key_compressed_alloc_from_infos(&pk_infos);
                        module.glwe_public_key_compressed_generate(
                            &mut pk_compressed,
                            &sk_prepared,
                            mask_seed,
                            &mut source_xe,
                            &mut scratch.borrow(),
                        );
                        let mut bytes = Vec::new();
                        pk_compressed.write_to(&mut bytes).unwrap();
                        let mut restored = module.glwe_public_key_compressed_alloc_from_infos(&pk_infos);
                        restored.read_from(&mut bytes.as_slice()).unwrap();
                        assert!(restored == pk_compressed);
                        module.decompress_glwe_public_key(&mut pk, &restored);
                        assert!(pk.dist() == sk_prepared.dist());
                    } else {
                        module.glwe_public_key_generate(
                            &mut pk,
                            &sk_prepared,
                            &mut source_xe,
                            &mut source_xa,
                            &mut scratch.borrow(),
                        );
                    }

                    let mut bytes = Vec::new();
                    pk.write_to(&mut bytes).unwrap();
                    let mut restored = module.glwe_public_key_alloc_from_infos(&pk_infos);
                    restored.read_from(&mut bytes.as_slice()).unwrap();
                    assert!(restored == pk);
                    pk = restored;

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
                        &mut source_xu,
                        &mut source_xe,
                        &mut enc_scratch.borrow(),
                    );
                    assert_canonical(&ct);

                    let factor = <BE::Ring as Ring>::CYCLOTOMIC_ORDER_FACTOR as usize / 2;
                    let model = |ct: &GLWE<BE::OwnedBuf, BE::ZnxWord>| {
                        let metadata = ct.noise().unwrap();
                        assert_eq!(metadata.precision(), ct.k());
                        metadata.weighted_phase_noise(n, n * factor * factor).variance_at(0u32.into())
                    };
                    let stats = glwe_noise_checked(module, &ct, &pt_want, &sk_prepared, &mut scratch.borrow());
                    noise_variance += stats.second_moment();
                    modeled_variance += model(&ct);
                    let mut phase = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
                    module.glwe_decrypt(&ct, &mut phase, &sk_prepared, &mut scratch.borrow());
                    module.glwe_sub_assign(&mut phase, &pt_want);
                    module.glwe_normalize_assign(&mut phase, &mut scratch.borrow());
                    let mut residual = vec![0i64; n];
                    phase.decode_vec_i64(&mut residual, ct.k());
                    coefficient_zero += (residual[0] as f64 * (-(k_ct as f64)).exp2()).powi(2);

                    // Encryption of zero follows its own metadata.
                    module.glwe_encrypt_zero_pk(
                        &mut ct,
                        &pk_prepared,
                        &mut source_xu,
                        &mut source_xe,
                        &mut enc_scratch.borrow(),
                    );
                    assert_canonical(&ct);
                    let zero = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
                    let stats = glwe_noise_checked(module, &ct, &zero, &sk_prepared, &mut scratch.borrow());
                    noise_variance += stats.second_moment();
                    modeled_variance += model(&ct);
                }
                let ratio = noise_variance / modeled_variance;
                let ring_factor = <BE::Ring as Ring>::CYCLOTOMIC_ORDER_FACTOR / 2;
                if ring_factor == 2 {
                    assert!(
                        // The model counts two encryptions per trial, coefficient zero one.
                        coefficient_zero <= 2.5 * modeled_variance / 2.0,
                        "CI coefficient-zero second moment exceeds metadata: {}",
                        2.0 * coefficient_zero / modeled_variance
                    );
                }
                // CI uses a bound covering coefficient zero, twice the average product weight.
                // Rounding uses a 1/4 bound although its variance approaches 1/12.
                // For binary CI, the all-ones product has coefficients 2*n-1 at zero
                // and 2*(n-c) elsewhere. Its average squared bias is about 4*n^2/3,
                // versus the model's (4*n)^2 bound, so the ratio can approach 1/12.
                let lower = if ring_factor == 2 && binary && k_pk > k_ct {
                    0.06
                } else if ring_factor == 2 {
                    0.18
                } else if k_pk == k_ct {
                    0.85
                } else if binary {
                    0.15
                } else {
                    0.25
                };
                assert!(
                    ratio >= lower && ratio <= 1.15,
                    "empirical/model second moment={ratio}, binary={binary}, rank={rank}, key gap={}",
                    k_pk - k_ct
                );
            }
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
    let glwe_infos = layout(base2k * 4 + 1);
    for (label, pk_layout) in [
        (
            "radix mismatch",
            GLWELayout {
                base2k: (base2k - 1).into(),
                ..glwe_infos
            },
        ),
        ("unprepared public key", glwe_infos),
        ("imprecise public key", layout(base2k * 3 + 1)),
    ] {
        let mut ct = module.glwe_alloc_from_infos(&glwe_infos);
        ct.noise = Some(crate::ComponentNoise::from_secret_at(
            Distribution::TernaryProb(0.5),
            glwe_infos.k,
            1,
        ));
        ct.canonical = false;
        let before = ct.to_owned_deep();
        let pt = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        let pk = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_pk_tmp_bytes(&glwe_infos, &pk_layout));
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            module.glwe_encrypt_pk(
                &mut ct,
                &pt,
                &pk,
                &mut Source::new([0u8; 32]),
                &mut Source::new([0u8; 32]),
                &mut scratch.borrow(),
            );
        }));
        assert!(rejected.is_err(), "{label} must be rejected");
        assert!(ct == before, "{label} changed the destination");
        assert!(!ct.is_canonical(), "{label} changed the canonical flag");
        if label == "imprecise public key" {
            std::panic::resume_unwind(rejected.unwrap_err());
        }
    }
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
        + VecZnxBigAddNoise<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = base2k * 3 + 1;
    let n: usize = module.n();

    for rank in 1_usize..3 {
        let infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

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
            &mut Source::new([2u8; 32]),
            &mut Source::new([3u8; 32]),
            &mut scratch.borrow(),
        );

        let (mut xe, mut xa) = (Source::new([2u8; 32]), Source::new([3u8; 32]));
        let keys_want: Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> = (0..rank)
            .map(|_| {
                let mut key: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
                module.glwe_encrypt_zero_sk(&mut key, &sk_prepared, &mut xe, &mut xa, &mut scratch.borrow());
                module.glwe_normalize_assign(&mut key, &mut scratch.borrow());
                key.noise = Some(crate::ComponentNoise::from_secret_at(*sk.dist(), infos.k, rank));
                key
            })
            .collect();
        for (l, key) in keys_want.iter().enumerate() {
            assert!(pk.at(l).to_owned_deep() == key.to_owned_deep(), "rank={rank} entry={l}");
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
            module.vec_znx_big_add_noise(
                base2k,
                infos.k.as_usize(),
                &mut big.to_backend_mut(),
                0,
                Noise::ENCRYPTION,
                &mut source_xe,
            );
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
        // Equal key and output precisions: the key error times the ephemeral,
        // plus one fresh draw per component.
        let ring_factor = <BE::Ring as Ring>::CYCLOTOMIC_ORDER_FACTOR as usize / 2;
        let sigma2 = DEFAULT_SIGMA_XE.powi(2);
        let inherited = sigma2 * (rank * n * ring_factor * ring_factor) as f64 * 0.5;
        want.noise = Some(
            crate::ComponentNoise::from_secret_at(*sk.dist(), infos.k, rank).with_components(
                (0..=rank)
                    .map(|i| crate::FreshNoiseEstimate::new(if i == 0 { inherited + sigma2 } else { sigma2 }, infos.k))
                    .collect(),
            ),
        );
        assert!(ct == want, "rank={rank}");

        // Changing the ephemeral law would invalidate the recorded collective
        // secret/noise model. Reject it before changing coefficients or provenance.
        *pk_prepared.dist_mut() = Distribution::BinaryProb(0.5);
        let before = ct.to_owned_deep();
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            module.glwe_encrypt_pk(
                &mut ct,
                &pt,
                &pk_prepared,
                &mut Source::new([5u8; 32]),
                &mut Source::new([6u8; 32]),
                &mut scratch.borrow(),
            );
        }));
        assert!(rejected.is_err());
        assert!(ct == before, "invalid public-key provenance must not mutate the destination");
        *pk_prepared.dist_mut() = *sk.dist();

        // The ephemerals and their products would decrypt `ct` from the scratch.
        crate::test_suite::assert_wipes_scratch::<BE>(module.glwe_encrypt_pk_tmp_bytes(&infos, &infos), |scratch| {
            module.glwe_encrypt_pk(
                &mut ct,
                &pt,
                &pk_prepared,
                &mut Source::new([5u8; 32]),
                &mut Source::new([6u8; 32]),
                scratch,
            )
        });
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
    let infos = GLWELayout {
        n: module.n().into(),
        base2k: params.base2k.into(),
        k: (params.base2k * 3 + 1).into(),
        rank: 2_usize.into(),
    };
    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&infos);
    let mut pk: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&infos);
    *pk.dist_mut() = dist;
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_pk_tmp_bytes(&infos, &infos));
    module.glwe_encrypt_zero_pk(
        &mut ct,
        &pk,
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
