//! Aggregation, normalization and finalization of every PAT shape over
//! three parties: the finalized aggregate decrypts under the ideal
//! secret, and finalizing before or after normalization gives the same result.

use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GGLWECompressedEncryptSk, GGLWEEncryptSk, GGLWENoise, GLWECompressedEncryptSk, GLWENoise,
    layouts::{
        GGLWE, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWESecret,
        GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos, ModuleCoreAlloc, TorusPrecision,
        compressed::{GGLWECompressedSeedMut, GLWECompressedSeedMut},
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned, WriterTo, ZnxView, ZnxViewMut},
    source::Source,
};

use super::fixtures::{BASE2K, K, PARTIES, RANK, SEEDS, gglwe_layout, ideal_secret, party_secrets, secret_from_seed};
use crate::{
    api::{PatAggregate, PatFinalize, PatNormalize},
    layouts::MHEModuleAlloc,
};

/// Bound on the log2 noise of the sum of `PARTIES` fresh encryptions at precision `k`.
pub(crate) fn aggregate_noise_bound(k: usize) -> f64 {
    (DEFAULT_SIGMA_XE * (PARTIES as f64).sqrt()).log2() - k as f64 + 0.5
}

pub fn test_glwe_pat_compressed_ops<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + PatAggregate<BE>
        + PatNormalize<BE>
        + PatFinalize<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWECompressedEncryptSk<BE>
        + GLWENoise<BE>,
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
    let pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
        n: layout.n,
        base2k: BASE2K,
        k: K,
    });
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&layout)
            .max(module.pat_normalize_tmp_bytes())
            .max(module.pat_finalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.glwe_pat_compressed_alloc_from_infos(&layout);
    let mut share = module.glwe_pat_compressed_alloc_from_infos(&layout);
    for (i, (_, sk)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_compressed_encrypt_sk(dst, &pt, sk, SEEDS[0], &enc_infos, &mut source_xe, &mut scratch.borrow());
        if i > 0 {
            module.glwe_pat_compressed_aggregate_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_pat_compressed_finalize(&mut lazy, &acc, &mut scratch.borrow());
    assert!(lazy.is_canonical());
    assert!(!acc.is_canonical());
    let noise: f64 = module.glwe_noise(&lazy, &pt, &sk_ideal, &mut scratch.borrow()).std().log2();
    assert!(noise <= aggregate_noise_bound(K.as_usize()), "noise {noise} above bound");

    module.glwe_pat_compressed_normalize_assign(&mut acc, &mut scratch.borrow());
    assert!(acc.is_canonical());
    let mut eager: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_pat_compressed_finalize(&mut eager, &acc, &mut scratch.borrow());
    assert!(eager.is_canonical());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

pub fn test_gglwe_pat_compressed_ops<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + PatAggregate<BE>
        + PatNormalize<BE>
        + PatFinalize<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWECompressedEncryptSk<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let parties = party_secrets(module);
    let sk_ideal = ideal_secret(module, &parties);
    let (pt, _) = secret_from_seed(module, [7u8; 32]);
    let pt_want = scaled(module, &pt, PARTIES as i64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .gglwe_compressed_encrypt_sk_tmp_bytes(&layout)
            .max(module.pat_normalize_tmp_bytes())
            .max(module.pat_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.gglwe_pat_compressed_alloc_from_infos(&layout);
    let mut share = module.gglwe_pat_compressed_alloc_from_infos(&layout);
    for (i, (_, sk)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.gglwe_compressed_encrypt_sk(
            dst,
            pt.data(),
            sk,
            SEEDS[0],
            &enc_infos,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        if i > 0 {
            module.gglwe_pat_compressed_aggregate_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.gglwe_pat_compressed_finalize(&mut lazy, &acc, &mut scratch.borrow());
    assert!(!acc.is_canonical());
    assert_gglwe_noise(module, &lazy, &pt_want, &sk_ideal, &mut scratch);

    module.gglwe_pat_compressed_normalize_assign(&mut acc, &mut scratch.borrow());
    assert!(acc.is_canonical());
    let mut eager: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.gglwe_pat_compressed_finalize(&mut eager, &acc, &mut scratch.borrow());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

/// Unseeded shares under one common key with independent masks: the aggregate
/// encrypts the sum of the plaintexts under that key.
pub fn test_gglwe_pat_ops<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + PatAggregate<BE>
        + PatNormalize<BE>
        + PatFinalize<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEEncryptSk<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let (pt, _) = secret_from_seed(module, [7u8; 32]);
    let pt_want = scaled(module, &pt, PARTIES as i64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .gglwe_encrypt_sk_tmp_bytes(&layout)
            .max(module.pat_normalize_tmp_bytes())
            .max(module.pat_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.gglwe_pat_alloc_from_infos(&layout);
    let mut share = module.gglwe_pat_alloc_from_infos(&layout);
    for i in 0..PARTIES {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        let mut source_xa = Source::new([20 + i as u8; 32]);
        module.gglwe_encrypt_sk(
            dst,
            pt.data(),
            &sk,
            &enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        if i > 0 {
            module.gglwe_pat_aggregate_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.gglwe_pat_finalize(&mut lazy, &acc, &mut scratch.borrow());
    assert!(!acc.is_canonical());
    assert_gglwe_noise(module, &lazy, &pt_want, &sk, &mut scratch);

    module.gglwe_pat_normalize_assign(&mut acc, &mut scratch.borrow());
    assert!(acc.is_canonical());
    let mut eager: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.gglwe_pat_finalize(&mut eager, &acc, &mut scratch.borrow());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

/// Aggregating shares drawn under different seeds panics.
pub fn test_pat_aggregate_seed_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + PatAggregate<BE>,
{
    let mut a = module.glwe_pat_compressed_alloc(BASE2K, K, RANK);
    let mut b = module.glwe_pat_compressed_alloc(BASE2K, K, RANK);
    *a.seed_mut() = SEEDS[0];
    *b.seed_mut() = SEEDS[1];
    module.glwe_pat_compressed_aggregate_assign(&mut a, &b);
}

/// Aggregating GGLWE shares drawn under different seeds panics.
pub fn test_gglwe_pat_compressed_aggregate_seed_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + PatAggregate<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.gglwe_pat_compressed_alloc_from_infos(&layout);
    let mut b = module.gglwe_pat_compressed_alloc_from_infos(&layout);
    b.seed_mut()[0] = SEEDS[1];
    module.gglwe_pat_compressed_aggregate_assign(&mut a, &b);
}

/// Aggregating PATs of different layouts panics.
pub fn test_pat_aggregate_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + PatAggregate<BE>,
{
    let mut a = module.glwe_pat_compressed_alloc(BASE2K, K, RANK);
    let b = module.glwe_pat_compressed_alloc(BASE2K, TorusPrecision(K.0 + BASE2K.0), RANK);
    module.glwe_pat_compressed_aggregate_assign(&mut a, &b);
}

/// Finalizing a PAT into a target of a different layout panics.
pub fn test_pat_finalize_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE> + PatFinalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let pat = module.glwe_pat_compressed_alloc(BASE2K, K, RANK);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: TorusPrecision(K.0 + BASE2K.0),
        rank: RANK,
    });
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.pat_finalize_tmp_bytes());
    module.glwe_pat_compressed_finalize(&mut res, &pat, &mut scratch.borrow());
}

fn scaled<BE>(module: &Module<BE>, sk: &GLWESecret<AlignedBuf, i64>, factor: i64) -> GLWESecret<AlignedBuf, i64>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
{
    let mut res: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(RANK);
    for col in 0..RANK.as_usize() {
        for (r, x) in res.data_mut().at_mut(col, 0).iter_mut().zip(sk.data().at(col, 0)) {
            *r = factor * x;
        }
    }
    res
}

fn assert_gglwe_noise<BE, S>(
    module: &Module<BE>,
    ct: &GGLWE<AlignedBuf, i64>,
    pt_want: &GLWESecret<AlignedBuf, i64>,
    sk: &S,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GGLWENoise<BE>,
    S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let bound = aggregate_noise_bound(ct.k().as_usize());
    for row in 0..ct.dnum().as_usize() {
        for col in 0..ct.rank_in().as_usize() {
            let noise: f64 = module
                .gglwe_noise(ct, row, col, &pt_want.data().to_ref(), sk, &mut scratch.borrow())
                .std()
                .log2();
            assert!(noise <= bound, "row {row} col {col}: noise {noise} above bound");
        }
    }
}
