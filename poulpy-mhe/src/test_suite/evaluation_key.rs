//! Collective evaluation keys over three parties: the finalized aggregate is a
//! key of the ideal secrets.

use poulpy_core::{
    GGLWENoise,
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{
        Degree, GGLWELayout, GLWEAutomorphismKey, GLWEInfos, GLWESecret, GLWESecretLayout, GLWESecretPrepared, GLWESwitchingKey,
        GLWESwitchingKeyDegrees, ModuleCoreAlloc, Rank,
    },
    test_suite::with_scratch,
};
use poulpy_hal::{
    AlignedBuf,
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign, VecZnxAutomorphism},
    layouts::{
        GaloisElement, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnxAsVecZnxBackendMut, ScalarZnxAsVecZnxBackendRef,
        ScratchOwned,
    },
    source::Source,
};

use super::{
    fixtures::{P, PARTIES, SEEDS, Secret, gglwe_layout, ideal_secret, party_secrets, secret_from_seed, secret_sum},
    pat::assert_gglwe_noise,
};
use crate::{
    api::{GLWEAutomorphismKeyMHEProtocol, GLWESwitchingKeyMHEProtocol},
    layouts::MHEModuleAlloc,
};

pub fn test_glwe_switching_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWESwitchingKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);

    let parties_in: Vec<Secret<BE>> = (0..PARTIES).map(|i| secret_from_seed(module, [150 + i as u8; 32])).collect();
    let parties_out = party_secrets(module);
    let pt_want = secret_sum(module, &parties_in);
    let sk_out_ideal = ideal_secret(module, &parties_out);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&layout));

    let mut acc = module.glwe_switching_key_share_alloc_from_infos(&layout);
    let mut share = module.glwe_switching_key_share_alloc_from_infos(&layout);
    for (i, ((sk_in, _), (sk_out, _))) in parties_in.iter().zip(&parties_out).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_switching_key_share_gen_tmp_bytes(&layout), |s| {
            module.mhe_glwe_switching_key_share_gen(dst, sk_in, sk_out, SEEDS[0], &mut source_xe, s)
        });
        if i > 0 {
            module.mhe_glwe_switching_key_share_aggregate(&mut acc, &share);
        }
    }

    let mut res: GLWESwitchingKey<AlignedBuf, i64> = module.glwe_switching_key_alloc_from_infos(&layout);
    with_scratch::<BE, _>(module.mhe_glwe_switching_key_share_finalize_tmp_bytes(), |s| {
        module.mhe_glwe_switching_key_share_finalize(&mut res, &acc, s)
    });
    super::fixtures::assert_collective_metadata(&res, PARTIES);
    let n = Degree(module.n() as u32);
    assert_eq!((*res.input_degree(), *res.output_degree()), (n, n));
    assert_gglwe_noise(module, &res, &pt_want, &sk_out_ideal, &mut scratch);
}

pub fn test_glwe_automorphism_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEAutomorphismKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + GaloisElement
        + VecZnxAddScalarAssign<BE>
        + VecZnxAutomorphism<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);

    let parties = party_secrets(module);
    let pt_want = secret_sum(module, &parties);
    let sk_out = automorphism_inv_prepared(module, &pt_want, P);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&layout));

    let mut acc = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    let mut share = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    for (i, (sk, _)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_automorphism_key_share_gen_tmp_bytes(&layout), |s| {
            module.mhe_glwe_automorphism_key_share_gen(dst, P, sk, SEEDS[0], &mut source_xe, s)
        });
        if i > 0 {
            module.mhe_glwe_automorphism_key_share_aggregate(&mut acc, &share);
        }
    }

    let mut res: GLWEAutomorphismKey<AlignedBuf, i64> = module.glwe_automorphism_key_alloc_from_infos(&layout);
    with_scratch::<BE, _>(module.mhe_glwe_automorphism_key_share_finalize_tmp_bytes(), |s| {
        module.mhe_glwe_automorphism_key_share_finalize(&mut res, &acc, s)
    });
    super::fixtures::assert_collective_metadata(&res, PARTIES);
    assert_eq!(res.p(), P);
    assert_gglwe_noise(module, &res, &pt_want, &sk_out, &mut scratch);
}

/// Aggregating switching key shares of different degrees panics.
pub fn test_mhe_glwe_switching_key_share_aggregate_rejects_degree_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESwitchingKeyMHEProtocol<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_switching_key_share_alloc_from_infos(&layout);
    let mut b = module.glwe_switching_key_share_alloc_from_infos(&layout);
    b.input_degree = Degree(module.n() as u32);
    let before = a.clone();
    super::fixtures::assert_panics_with("invalid aggregation: degrees differ", || {
        module.mhe_glwe_switching_key_share_aggregate(&mut a, &b);
    });
    assert!(a == before);
}

/// Aggregating switching key shares of different output degrees panics.
pub fn test_mhe_glwe_switching_key_share_aggregate_rejects_out_degree_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESwitchingKeyMHEProtocol<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_switching_key_share_alloc_from_infos(&layout);
    let mut b = module.glwe_switching_key_share_alloc_from_infos(&layout);
    b.output_degree = Degree(module.n() as u32);
    let before = a.clone();
    super::fixtures::assert_panics_with("invalid aggregation: degrees differ", || {
        module.mhe_glwe_switching_key_share_aggregate(&mut a, &b);
    });
    assert!(a == before);
}

/// Aggregating automorphism key shares of different Galois elements panics.
pub fn test_mhe_glwe_automorphism_key_share_aggregate_rejects_p_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEAutomorphismKeyMHEProtocol<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    let mut b = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    b.p = P;
    let before = a.clone();
    super::fixtures::assert_panics_with("invalid aggregation: Galois elements differ", || {
        module.mhe_glwe_automorphism_key_share_aggregate(&mut a, &b);
    });
    assert!(a == before);
}

/// `sigma_{p^-1}(sk)` prepared: the secret core's automorphism key encrypts under.
fn automorphism_inv_prepared<BE>(
    module: &Module<BE>,
    sk: &GLWESecret<AlignedBuf, i64>,
    p: i64,
) -> GLWESecretPrepared<AlignedBuf, BE>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretPreparedFactory<BE> + GaloisElement + VecZnxAutomorphism<BE>,
{
    let mut sigma: GLWESecret<AlignedBuf, i64> = sk.clone();
    {
        let src = ScalarZnxAsVecZnxBackendRef::<BE>::as_vec_znx_backend(sk.data());
        let mut dst = ScalarZnxAsVecZnxBackendMut::<BE>::as_vec_znx_backend_mut(sigma.data_mut());
        for col in 0..sk.rank().as_usize() {
            module.vec_znx_automorphism(module.galois_element_inv(p), &mut dst, col, &src, col);
        }
    }
    let mut prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(sk.rank());
    module.glwe_secret_prepare(&mut prepared, &sigma);
    prepared
}

/// Generating under a secret of another rank than the key's, or of a degree above the module's, panics before any draw,
/// and a key whose input degree and rank differ from its output's is generated.
pub fn test_mhe_glwe_switching_key_share_gen_rejects_shape<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: ModuleNew<BE>
        + MHEModuleAlloc<BE>
        + GLWESwitchingKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);

    let (sk, _) = secret_from_seed(module, [100u8; 32]);
    let rank_one = module.glwe_secret_alloc(Rank(1));
    let large_module = Module::<BE>::new((module.n() * 2) as u64);
    let large_sk = large_module.glwe_secret_alloc(layout.rank_in);
    for (sk_in, sk_out, expected) in [
        (&rank_one, &sk, "invalid share: input secret rank differs from the key's"),
        (&sk, &rank_one, "invalid share: output secret rank differs from the key's"),
        (&large_sk, &sk, "invalid share: unsupported input secret degree"),
        (&sk, &large_sk, "invalid share: unsupported output secret degree"),
    ] {
        let mut share = module.glwe_switching_key_share_alloc_from_infos(&layout);
        let before = share.clone();
        let mut source_xe = Source::new([10u8; 32]);
        super::fixtures::assert_panics_with(expected, || {
            with_scratch::<BE, _>(module.mhe_glwe_switching_key_share_gen_tmp_bytes(&layout), |s| {
                module.mhe_glwe_switching_key_share_gen(&mut share, sk_in, sk_out, SEEDS[0], &mut source_xe, s)
            });
        });
        assert!(share == before);
        assert_eq!(source_xe.next_i64(), Source::new([10u8; 32]).next_i64());
    }

    let mut sk_in = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
        n: (module.n() / 2).into(),
        rank: Rank(1),
    });
    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut Source::new([150u8; 32]));
    let mut key_layout = layout;
    key_layout.rank_in = Rank(1);
    let mut share = module.glwe_switching_key_share_alloc_from_infos(&key_layout);
    with_scratch::<BE, _>(module.mhe_glwe_switching_key_share_gen_tmp_bytes(&key_layout), |s| {
        module.mhe_glwe_switching_key_share_gen(&mut share, &sk_in, &sk, SEEDS[0], &mut Source::new([10u8; 32]), s)
    });
    assert_eq!(*share.input_degree(), Degree((module.n() / 2) as u32));
    assert_eq!(*share.output_degree(), Degree(module.n() as u32));
}

/// Generating under a secret of another degree than the key's, into a key whose input and output ranks differ, or for
/// an even Galois element, panics before any draw.
pub fn test_mhe_glwe_automorphism_key_share_gen_rejects_shape<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEAutomorphismKeyMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);

    let (sk, _) = secret_from_seed(module, [100u8; 32]);
    let small_sk = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
        n: (module.n() / 2).into(),
        rank: layout.rank_out,
    });
    let mixed_ranks = GGLWELayout {
        rank_in: Rank(1),
        ..layout
    };
    for (key_layout, p, secret, expected) in [
        (layout, P, &small_sk, "invalid share: secret degree differs from the key's"),
        (
            mixed_ranks,
            P,
            &sk,
            "invalid share: automorphism key input and output ranks differ",
        ),
        (layout, 2, &sk, "invalid share: Galois element must be odd"),
    ] {
        let mut share = module.glwe_automorphism_key_share_alloc_from_infos(&key_layout);
        let before = share.clone();
        let mut source_xe = Source::new([10u8; 32]);
        super::fixtures::assert_panics_with(expected, || {
            with_scratch::<BE, _>(module.mhe_glwe_automorphism_key_share_gen_tmp_bytes(&key_layout), |s| {
                module.mhe_glwe_automorphism_key_share_gen(&mut share, p, secret, SEEDS[0], &mut source_xe, s)
            });
        });
        assert!(share == before);
        assert_eq!(source_xe.next_i64(), Source::new([10u8; 32]).next_i64());
    }
}
