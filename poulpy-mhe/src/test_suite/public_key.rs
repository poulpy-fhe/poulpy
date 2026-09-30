//! Collective public key over three parties: a ciphertext encrypted
//! under the finalized key decrypts under the ideal secret.

use poulpy_core::{
    DEFAULT_SIGMA_XE, Distribution, EncryptionLayout, GLWEEncryptPk, GLWENoise,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWEPublicKey, GLWEPublicKeyPreparedFactory, GLWESecretPrepared,
        GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign, VecZnxFillUniformSource},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
    test_suite::vec_znx_backend_mut,
};

use super::fixtures::{BASE2K, K, PARTIES, RANK, SEEDS, collective_public_key, ideal_secret, party_secrets};
use crate::{api::GLWEPublicKeyProtocol, layouts::MHEModuleAlloc};

pub fn test_glwe_public_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEPublicKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAddScalarAssign<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEEncryptPk<BE>
        + GLWENoise<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let parties = party_secrets(module);
    let sk_ideal = ideal_secret(module, &parties);
    let pk_prepared = collective_public_key(module, &parties, &layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_pk_tmp_bytes(&layout, &layout)
            .max(module.glwe_noise_tmp_bytes(&layout)),
    );

    let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    module.vec_znx_fill_uniform_source(
        BASE2K.as_usize(),
        K.as_usize(),
        &mut vec_znx_backend_mut::<BE>(pt.data_mut()),
        0,
        &mut Source::new([30u8; 32]),
    );
    let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_encrypt_pk(
        &mut ct,
        &pt,
        &pk_prepared,
        &enc_infos,
        &mut Source::new([31u8; 32]),
        &mut Source::new([32u8; 32]),
        &mut scratch.borrow(),
    );

    // Sum_l u_l e_l over rank entries whose errors sum PARTIES errors, plus <e_1, s> with ideal-secret variance PARTIES / 2.
    let n = module.n() as f64;
    let rank = RANK.as_usize() as f64;
    let variance = 2.0 * rank * n * 0.5 * PARTIES as f64 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
    let bound = variance.sqrt().log2() - K.as_usize() as f64 + 1.25_f64.log2();
    let noise: f64 = module.glwe_noise(&ct, &pt, &sk_ideal, &mut scratch.borrow()).std().log2();
    assert!(noise <= bound, "noise {noise} above bound {bound}");
}

/// Finalizing with a non-samplable distribution panics.
pub fn test_glwe_public_key_finalize_dist_none<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let share = module.glwe_public_key_share_alloc_from_infos(&layout);
    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_public_key_finalize_tmp_bytes());
    module.glwe_public_key_finalize(&mut pk, &share, Distribution::NONE, &mut scratch.borrow());
}

/// Entries sharing a seed share their masks, which makes encryption rank 1 in the ephemerals; a fresh share has zero seeds.
pub fn test_glwe_public_key_finalize_shared_seed<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let share = module.glwe_public_key_share_alloc_from_infos(&layout);
    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_public_key_finalize_tmp_bytes());
    module.glwe_public_key_finalize(&mut pk, &share, Distribution::TernaryProb(0.5), &mut scratch.borrow());
}

/// Sharing under a secret without a samplable distribution panics.
pub fn test_glwe_public_key_gen_secret_none<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let sk: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    let mut res = module.glwe_public_key_share_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_public_key_gen_tmp_bytes(&layout));
    module.glwe_public_key_gen(
        &mut res,
        &sk,
        SEEDS[0],
        &enc_infos,
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

/// Invalid shapes fail at the protocol boundary with an exact static message.
pub fn test_glwe_public_key_gen_shape_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: ModuleNew<BE> + MHEModuleAlloc<BE> + GLWEPublicKeyProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let small_module = Module::<BE>::new((module.n() / 2) as u64);
    let expected = [
        "invalid share: secret degree differs from the share's",
        "invalid share: secret rank differs from the share's",
        "invalid layout: degree differs from the module's",
    ];
    for (case, expected) in expected.into_iter().enumerate() {
        super::fixtures::assert_panics_with(expected, || {
            if case == 2 {
                module.glwe_public_key_gen_tmp_bytes(&GLWELayout {
                    n: (module.n() / 2).into(),
                    ..layout
                });
                return;
            }
            let sk = if case == 0 {
                small_module.glwe_secret_prepared_alloc(RANK)
            } else {
                module.glwe_secret_prepared_alloc(Rank(1))
            };
            let mut res = module.glwe_public_key_share_alloc_from_infos(&layout);
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_public_key_gen_tmp_bytes(&layout));
            module.glwe_public_key_gen(
                &mut res,
                &sk,
                SEEDS[0],
                &enc_infos,
                &mut Source::new([10u8; 32]),
                &mut scratch.borrow(),
            );
        });
    }
}
