//! Collective evaluation keys over three parties: the finalized aggregate is a
//! key of the ideal secrets.

use poulpy_core::{
    EncryptionLayout, GGLWENoise,
    layouts::{
        Degree, GLWEAutomorphismKey, GLWEInfos, GLWESecret, GLWESecretLayout, GLWESecretPrepared, GLWESecretPreparedFactory,
        GLWESecretSampling, GLWESwitchingKey, GLWESwitchingKeyDegrees, ModuleCoreAlloc, Rank,
    },
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
    api::{GLWEAutomorphismKeyProtocol, GLWESwitchingKeyProtocol},
    layouts::MHEModuleAlloc,
};

pub fn test_glwe_switching_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWESwitchingKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let parties_in: Vec<Secret<BE>> = (0..PARTIES).map(|i| secret_from_seed(module, [150 + i as u8; 32])).collect();
    let parties_out = party_secrets(module);
    let pt_want = secret_sum(module, &parties_in);
    let sk_out_ideal = ideal_secret(module, &parties_out);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_switching_key_gen_tmp_bytes(&layout)
            .max(module.glwe_switching_key_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.glwe_switching_key_share_alloc_from_infos(&layout);
    let mut share = module.glwe_switching_key_share_alloc_from_infos(&layout);
    for (i, ((sk_in, _), (sk_out, _))) in parties_in.iter().zip(&parties_out).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_switching_key_gen(
            dst,
            sk_in,
            sk_out,
            SEEDS[0],
            &enc_infos,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        if i > 0 {
            module.glwe_switching_key_aggregate(&mut acc, &share);
        }
    }

    let mut res: GLWESwitchingKey<AlignedBuf, i64> = module.glwe_switching_key_alloc_from_infos(&layout);
    module.glwe_switching_key_finalize(&mut res, &acc, &mut scratch.borrow());
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
        + GLWEAutomorphismKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + GaloisElement
        + VecZnxAddScalarAssign<BE>
        + VecZnxAutomorphism<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let parties = party_secrets(module);
    let pt_want = secret_sum(module, &parties);
    let sk_out = automorphism_inv_prepared(module, &pt_want, P);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_automorphism_key_gen_tmp_bytes(&layout)
            .max(module.glwe_automorphism_key_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    let mut share = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    for (i, (sk, _)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_automorphism_key_gen(dst, P, sk, SEEDS[0], &enc_infos, &mut source_xe, &mut scratch.borrow());
        if i > 0 {
            module.glwe_automorphism_key_aggregate(&mut acc, &share);
        }
    }

    let mut res: GLWEAutomorphismKey<AlignedBuf, i64> = module.glwe_automorphism_key_alloc_from_infos(&layout);
    module.glwe_automorphism_key_finalize(&mut res, &acc, &mut scratch.borrow());
    assert_eq!(res.p(), P);
    assert_gglwe_noise(module, &res, &pt_want, &sk_out, &mut scratch);
}

/// Aggregating switching key shares of different degrees panics.
pub fn test_glwe_switching_key_degree_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESwitchingKeyProtocol<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_switching_key_share_alloc_from_infos(&layout);
    let mut b = module.glwe_switching_key_share_alloc_from_infos(&layout);
    b.input_degree = Degree(module.n() as u32);
    module.glwe_switching_key_aggregate(&mut a, &b);
}

/// Aggregating switching key shares of different output degrees panics.
pub fn test_glwe_switching_key_out_degree_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESwitchingKeyProtocol<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_switching_key_share_alloc_from_infos(&layout);
    let mut b = module.glwe_switching_key_share_alloc_from_infos(&layout);
    b.output_degree = Degree(module.n() as u32);
    module.glwe_switching_key_aggregate(&mut a, &b);
}

/// Aggregating automorphism key shares of different Galois elements panics.
pub fn test_glwe_automorphism_key_p_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEAutomorphismKeyProtocol<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    let mut b = module.glwe_automorphism_key_share_alloc_from_infos(&layout);
    b.p = P;
    module.glwe_automorphism_key_aggregate(&mut a, &b);
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

/// Reject malformed key shares without rejecting supported degree and rank changes.
pub fn test_glwe_evaluation_key_share_shape_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: ModuleNew<BE>
        + MHEModuleAlloc<BE>
        + GLWESwitchingKeyProtocol<BE>
        + GLWEAutomorphismKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let (sk, _) = secret_from_seed(module, [100u8; 32]);
    let rank_one = module.glwe_secret_alloc(Rank(1));
    let small_sk = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
        n: (module.n() / 2).into(),
        rank: layout.rank_out,
    });
    let large_module = Module::<BE>::new((module.n() * 2) as u64);
    let large_sk = large_module.glwe_secret_alloc(layout.rank_in);
    let expected = [
        "invalid share: input secret rank differs from the key's",
        "invalid share: output secret rank differs from the key's",
        "invalid share: secret degree differs from the key's",
        "invalid share: automorphism key input and output ranks differ",
        "invalid share: Galois element must be odd",
        "invalid share: unsupported input secret degree",
        "invalid share: unsupported output secret degree",
    ];
    for (case, expected) in expected.into_iter().enumerate() {
        super::fixtures::assert_panics_with(expected, || {
            let mut source_xe = Source::new([10u8; 32]);
            if matches!(case, 0 | 1 | 5 | 6) {
                let mut share = module.glwe_switching_key_share_alloc_from_infos(&layout);
                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_switching_key_gen_tmp_bytes(&layout));
                let sk_in = match case {
                    0 => &rank_one,
                    5 => &large_sk,
                    _ => &sk,
                };
                let sk_out = match case {
                    1 => &rank_one,
                    6 => &large_sk,
                    _ => &sk,
                };
                module.glwe_switching_key_gen(
                    &mut share,
                    sk_in,
                    sk_out,
                    SEEDS[0],
                    &enc_infos,
                    &mut source_xe,
                    &mut scratch.borrow(),
                );
            } else {
                let mut key_layout = layout;
                if case == 3 {
                    key_layout.rank_in = Rank(1);
                }
                let mut share = module.glwe_automorphism_key_share_alloc_from_infos(&key_layout);
                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_automorphism_key_gen_tmp_bytes(&key_layout));
                module.glwe_automorphism_key_gen(
                    &mut share,
                    if case == 4 { 2 } else { P },
                    if case == 2 { &small_sk } else { &sk },
                    SEEDS[0],
                    &enc_infos,
                    &mut source_xe,
                    &mut scratch.borrow(),
                );
            }
        });
    }

    // A smaller input degree and a different input rank are supported by core.
    let mut sk_in = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
        n: (module.n() / 2).into(),
        rank: Rank(1),
    });
    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut Source::new([150u8; 32]));
    let mut key_layout = layout;
    key_layout.rank_in = Rank(1);
    let mut share = module.glwe_switching_key_share_alloc_from_infos(&key_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_switching_key_gen_tmp_bytes(&key_layout));
    module.glwe_switching_key_gen(
        &mut share,
        &sk_in,
        &sk,
        SEEDS[0],
        &enc_infos,
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
    assert_eq!(*share.input_degree(), Degree((module.n() / 2) as u32));
    assert_eq!(*share.output_degree(), Degree(module.n() as u32));
}
