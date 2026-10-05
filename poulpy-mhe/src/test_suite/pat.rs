//! Aggregation and finalization of every PAT shape over three parties: the
//! finalized aggregate decrypts under the ideal secret.

use poulpy_core::{
    DEFAULT_SIGMA_XE, Distribution, EncryptionMetadata, GGLWECompressedEncryptSk, GGLWENoise, GLWECompressedEncryptSk,
    GLWEEncryptPk, GLWENoise,
    layouts::{
        GGLWE, GGLWEAtViewRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext,
        GLWEPlaintextLayout, GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos,
        ModuleCoreAlloc, TorusPrecision,
        compressed::{GGLWECompressedSeedMut, GLWECompressedSeedMut, GLWECompressedToBackendMut},
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxFillUniformSource},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchOwned, vec_znx_backend_mut, vec_znx_backend_ref},
    source::Source,
};

use super::fixtures::{
    BASE2K, K, PARTIES, RANK, SEEDS, collective_public_key, gglwe_layout, ideal_secret, party_messages, party_secrets, secret_sum,
};
use crate::{
    api::{GGLWEPatCompressedOps, GGLWEPatOps, GLWEPatCompressedOps, GLWEPublicKeyMHEProtocol},
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
        + GLWEPatCompressedOps<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxFillUniformSource<BE>
        + VecZnxAddAssign<BE>
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

    let parties = party_secrets(module);
    let sk_ideal = ideal_secret(module, &parties);
    let pt_layout = GLWEPlaintextLayout {
        n: layout.n,
        base2k: BASE2K,
        k: K,
    };
    let pts: Vec<GLWEPlaintext<AlignedBuf, i64>> = (0..PARTIES)
        .map(|i| {
            let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&pt_layout);
            module.vec_znx_fill_uniform_source(
                BASE2K.as_usize(),
                K.as_usize(),
                &mut vec_znx_backend_mut::<BE>(pt.data_mut()),
                0,
                &mut Source::new([30 + i as u8; 32]),
            );
            pt
        })
        .collect();
    let mut pt_want: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&pt_layout);
    for pt in &pts {
        module.vec_znx_add_assign(
            &mut vec_znx_backend_mut::<BE>(pt_want.data_mut()),
            0,
            &vec_znx_backend_ref::<BE>(pt.data()),
            0,
        );
    }
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&layout)
            .max(module.glwe_pat_compressed_finalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.glwe_pat_compressed_alloc_from_infos(&layout);
    let mut share = module.glwe_pat_compressed_alloc_from_infos(&layout);
    for (i, (_, sk)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_compressed_encrypt_sk(dst, &pts[i], sk, SEEDS[0], &mut source_xe, &mut scratch.borrow());
        if i == 1 {
            // Backend views own their metadata snapshot, like their seeds and precision.
            let metadata = {
                let mut view = poulpy_core::layouts::compressed::GLWECompressedViewMut::<BE>::from_inner(
                    GLWECompressedToBackendMut::<BE>::to_backend_mut(&mut acc),
                );
                module.glwe_pat_compressed_aggregate_assign(&mut view, &share);
                super::fixtures::assert_collective_metadata(&view, i + 1);
                view.encryption_metadata()
            };
            super::fixtures::assert_collective_metadata(&acc, 1);
            GLWECompressedToBackendMut::<BE>::set_encryption_metadata(&mut acc, metadata);
        } else if i > 1 {
            module.glwe_pat_compressed_aggregate_assign(&mut acc, &share);
        }
    }

    // Reject inconsistent provenance before changing coefficients or metadata.
    let before = acc.clone();
    let mut mismatched = share.clone();
    GLWECompressedToBackendMut::<BE>::set_encryption_metadata(
        &mut mismatched,
        Some(EncryptionMetadata::from_secret_at(Distribution::BinaryProb(0.5), K)),
    );
    super::fixtures::assert_panics_with("invalid aggregation: secret distributions differ", || {
        module.glwe_pat_compressed_aggregate_assign(&mut acc, &mismatched);
    });
    assert!(acc == before);

    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_pat_compressed_finalize(&mut res, &acc, &mut scratch.borrow());
    super::fixtures::assert_collective_metadata(&res, PARTIES);
    assert!(res.is_canonical());
    let noise: f64 = module
        .glwe_noise(&res, &pt_want, &sk_ideal, &mut scratch.borrow())
        .std()
        .log2();
    assert!(noise <= aggregate_noise_bound(K.as_usize()), "noise {noise} above bound");
}

pub fn test_gglwe_pat_compressed_ops<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GGLWEPatCompressedOps<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAddScalarAssign<BE>
        + GGLWECompressedEncryptSk<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);

    let parties = party_secrets(module);
    let sk_ideal = ideal_secret(module, &parties);
    let messages = party_messages(module);
    let pt_want = secret_sum(module, &messages);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .gglwe_compressed_encrypt_sk_tmp_bytes(&layout)
            .max(module.gglwe_pat_compressed_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.gglwe_pat_compressed_alloc_from_infos(&layout);
    let mut share = module.gglwe_pat_compressed_alloc_from_infos(&layout);
    for (i, (_, sk)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.gglwe_compressed_encrypt_sk(dst, messages[i].0.data(), sk, SEEDS[0], &mut source_xe, &mut scratch.borrow());
        if i > 0 {
            module.gglwe_pat_compressed_aggregate_assign(&mut acc, &share);
        }
    }

    let mut res: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.gglwe_pat_compressed_finalize(&mut res, &acc, &mut scratch.borrow());
    super::fixtures::assert_collective_metadata(&res, PARTIES);
    assert_gglwe_noise(module, &res, &pt_want, &sk_ideal, &mut scratch);
}

/// Unseeded shares of public-key encryptions under the collective key: the
/// aggregate encrypts the sum of the plaintexts under the ideal secret.
pub fn test_gglwe_pat_ops<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GGLWEPatOps<BE>
        + GLWEPublicKeyMHEProtocol<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxFillUniformSource<BE>
        + VecZnxAddAssign<BE>
        + GLWEEncryptPk<BE>
        + GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let k = layout.k();
    let pk_layout = GLWELayout {
        n: layout.n,
        base2k: BASE2K,
        k,
        rank: RANK,
    };
    let pt_layout = GLWEPlaintextLayout {
        n: layout.n,
        base2k: BASE2K,
        k,
    };

    let parties = party_secrets(module);
    let sk_ideal = ideal_secret(module, &parties);
    let pk = collective_public_key(module, &parties, &pk_layout);
    let (dnum, rank_in) = (layout.dnum.as_usize(), layout.rank_in.as_usize());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_pk_tmp_bytes(&pk_layout, &pk_layout)
            .max(module.gglwe_pat_finalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&pk_layout)),
    );

    let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&pt_layout);
    let mut pts_want: Vec<GLWEPlaintext<AlignedBuf, i64>> = (0..dnum * rank_in)
        .map(|_| module.glwe_plaintext_alloc_from_infos(&pt_layout))
        .collect();
    let mut acc = module.gglwe_pat_alloc_from_infos(&layout);
    let mut share = module.gglwe_pat_alloc_from_infos(&layout);
    for i in 0..PARTIES {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xm = Source::new([30 + i as u8; 32]);
        let mut source_xu = Source::new([20 + i as u8; 32]);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        let mut metadata = None;
        {
            let mut dst_be = GGLWEToBackendMut::<BE>::to_backend_mut(dst);
            for row in 0..dnum {
                for col in 0..rank_in {
                    module.vec_znx_fill_uniform_source(
                        BASE2K.as_usize(),
                        k.as_usize(),
                        &mut vec_znx_backend_mut::<BE>(pt.data_mut()),
                        0,
                        &mut source_xm,
                    );
                    let mut cell = dst_be.at_view_mut(row, col);
                    module.glwe_encrypt_pk(&mut cell, &pt, &pk, &mut source_xu, &mut source_xe, &mut scratch.borrow());
                    metadata = cell.encryption_metadata();
                    module.vec_znx_add_assign(
                        &mut vec_znx_backend_mut::<BE>(pts_want[row * rank_in + col].data_mut()),
                        0,
                        &vec_znx_backend_ref::<BE>(pt.data()),
                        0,
                    );
                }
            }
        }
        GGLWEToBackendMut::<BE>::set_encryption_metadata(dst, metadata);
        if i == 1 {
            // Backend borrows stand in for the owned PATs.
            let metadata = {
                let mut view = acc.to_backend_mut();
                module.gglwe_pat_aggregate_assign(&mut view, &share.to_backend_ref());
                view.encryption_metadata()
            };
            GGLWEToBackendMut::<BE>::set_encryption_metadata(&mut acc, metadata);
        } else if i > 1 {
            module.gglwe_pat_aggregate_assign(&mut acc, &share);
        }
    }

    let mut res: GGLWE<AlignedBuf, i64> = module.gglwe_alloc_from_infos(&layout);
    module.gglwe_pat_finalize(&mut res, &acc, &mut scratch.borrow());
    super::fixtures::assert_collective_metadata(&res, PARTIES);
    // The public key test variance, times PARTIES independent encryptions.
    let n = module.n() as f64;
    let variance = 2.0 * RANK.as_usize() as f64 * n * 0.5 * (PARTIES * PARTIES) as f64 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
    let bound = variance.sqrt().log2() - k.as_usize() as f64 + 1.25_f64.log2();
    for row in 0..dnum {
        for col in 0..rank_in {
            let noise: f64 = module
                .glwe_noise(
                    &res.at_view(row, col),
                    &pts_want[row * rank_in + col],
                    &sk_ideal,
                    &mut scratch.borrow(),
                )
                .std()
                .log2();
            assert!(noise <= bound, "row {row} col {col}: noise {noise} above bound {bound}");
        }
    }
}

/// Aggregating shares drawn under different seeds panics.
pub fn test_pat_aggregate_seed_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPatCompressedOps<BE>,
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
    Module<BE>: MHEModuleAlloc<BE> + GGLWEPatCompressedOps<BE>,
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
    Module<BE>: MHEModuleAlloc<BE> + GLWEPatCompressedOps<BE>,
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
    Module<BE>: MHEModuleAlloc<BE> + GLWEPatCompressedOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let pat = module.glwe_pat_compressed_alloc(BASE2K, K, RANK);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: TorusPrecision(K.0 + BASE2K.0),
        rank: RANK,
    });
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_pat_compressed_finalize_tmp_bytes());
    module.glwe_pat_compressed_finalize(&mut res, &pat, &mut scratch.borrow());
}

/// Every entry of `ct` decrypts to `pt_want` under `sk` within [`aggregate_noise_bound`].
pub(crate) fn assert_gglwe_noise<BE, C, S>(
    module: &Module<BE>,
    ct: &C,
    pt_want: &GLWESecret<AlignedBuf, i64>,
    sk: &S,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GGLWENoise<BE>,
    C: GGLWEToBackendRef<BE> + GGLWEInfos,
    S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let bound = aggregate_noise_bound(ct.k().as_usize());
    assert_gglwe_noise_within(module, ct, &pt_want.data().to_ref(), sk, bound, scratch);
}

/// Every entry `(row, col)` of `ct` decrypts under `sk` to column `col` of
/// `pt_want` with log2 noise at most `bound`.
pub(crate) fn assert_gglwe_noise_within<BE, C, S>(
    module: &Module<BE>,
    ct: &C,
    pt_want: &ScalarZnx<&[u8], i64>,
    sk: &S,
    bound: f64,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GGLWENoise<BE>,
    C: GGLWEToBackendRef<BE> + GGLWEInfos,
    S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    for row in 0..ct.dnum().as_usize() {
        for col in 0..ct.rank_in().as_usize() {
            let noise: f64 = module
                .gglwe_noise(ct, row, col, pt_want, sk, &mut scratch.borrow())
                .std()
                .log2();
            assert!(noise <= bound, "row {row} col {col}: noise {noise} above bound {bound}");
        }
    }
}
