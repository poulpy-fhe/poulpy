//! Collective tensor key over three parties: every entry of the finalized
//! key decrypts under the ideal secret to the ideal secret's tensor.

use poulpy_core::{
    DEFAULT_SIGMA_XE, Distribution, GGLWENoise, GetDistributionMut,
    api::{GLWEPublicKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling, GLWESecretTensorFromGLWESecret},
    layouts::{
        Base2K, Dnum, Dsize, GLWELayout, GLWEPublicKeyPrepared, GLWESecret, GLWESecretAlloc, GLWESecretLayout, GLWESecretTensor,
        GLWESecretTensorAlloc, GLWETensorKey, GLWETensorKeyAlloc, GLWETensorKeyLayout, LWEInfos, TorusPrecision, k_digit_of,
    },
    test_suite::with_scratch,
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use super::{
    fixtures::{
        BASE2K, DNUM, DSIZE, PARTIES, RANK, assert_short_scratch_panics, collective_public_key, ideal_secret, party_secrets,
        secret_from_seed, secret_sum,
    },
    pat::assert_gglwe_noise_within,
};
use crate::{
    api::{MHEGLWEPublicKeyProtocol, MHEGLWETensorKeyProtocol},
    layouts::MHEGLWETensorKeyShareAlloc,
};

pub fn test_glwe_tensor_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEGLWETensorKeyShareAlloc<BE>
        + MHEGLWETensorKeyProtocol<BE>
        + MHEGLWEPublicKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretTensorFromGLWESecret<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + VecZnxAddScalarAssign<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    for (dnum, dsize) in [(DNUM, DSIZE), (Dnum(2), Dsize(2))] {
        let layout = tensor_key_layout(module, dnum, dsize);
        // A public key more precise than the share exercises the share scratch query for real.
        let pk_layout = public_key_layout(module, TorusPrecision(layout.k().0 + BASE2K.0));

        let parties = party_secrets(module);
        let sk_ideal = ideal_secret(module, &parties);
        let mut pk = collective_public_key(module, &parties, &pk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_secret_tensor_from_glwe_secret_tmp_bytes(RANK)
                .max(module.gglwe_noise_tmp_bytes(&layout)),
        );
        let mut pt_want: GLWESecretTensor<AlignedBuf, i64> = module.glwe_secret_tensor_alloc(RANK);
        module.glwe_secret_tensor_from_glwe_secret(&mut pt_want, &secret_sum(module, &parties), &mut scratch.borrow());

        let mut acc = module.mhe_glwe_tensor_key_share_alloc_from_infos(&layout);
        let mut share = module.mhe_glwe_tensor_key_share_alloc_from_infos(&layout);
        for (i, (sk, _)) in parties.iter().enumerate() {
            let dst = if i == 0 { &mut acc } else { &mut share };
            let mut source_xu = Source::new([20 + i as u8; 32]);
            let mut source_xe = Source::new([10 + i as u8; 32]);
            let gen_bytes = module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout);
            if i == 0 {
                let before = dst.clone();
                assert_short_scratch_panics::<BE>(gen_bytes, "invalid share: insufficient scratch", |s| {
                    module.mhe_glwe_tensor_key_share_gen(dst, sk, &pk, &mut source_xu, &mut source_xe, s)
                });
                assert!(*dst == before);
                assert_eq!(source_xu.next_i64(), Source::new([20u8; 32]).next_i64());
                assert_eq!(source_xe.next_i64(), Source::new([10u8; 32]).next_i64());
                (source_xu, source_xe) = (Source::new([20u8; 32]), Source::new([10u8; 32]));
            }
            // The secret components and their encryptions would be left in the scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(gen_bytes, |scratch| {
                module.mhe_glwe_tensor_key_share_gen(dst, sk, &pk, &mut source_xu, &mut source_xe, scratch)
            });
            if i > 0 {
                module.mhe_glwe_tensor_key_share_aggregate(&mut acc, &share);
            }
        }

        let mut res: GLWETensorKey<AlignedBuf, i64> = module.glwe_tensor_key_alloc_from_infos(&layout);
        let finalize_bytes = module.mhe_glwe_tensor_key_share_finalize_tmp_bytes();
        let before = res.clone();
        assert_short_scratch_panics::<BE>(finalize_bytes, "invalid finalization: insufficient scratch", |s| {
            module.mhe_glwe_tensor_key_share_finalize(&mut res, &acc, s)
        });
        assert!(res == before);
        with_scratch::<BE, _>(finalize_bytes, |s| {
            module.mhe_glwe_tensor_key_share_finalize(&mut res, &acc, s)
        });
        super::fixtures::assert_collective_metadata(&res, PARTIES);
        // Each party's pk encryption of zero adds 2 * rank * n * 0.5 * PARTIES * sigma^2 + sigma^2.
        let n = module.n() as f64;
        let parties_f = PARTIES as f64;
        let variance = 2.0 * RANK.as_usize() as f64 * n * 0.5 * parties_f * parties_f * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE
            + parties_f * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
        let sigma2 = DEFAULT_SIGMA_XE.powi(2);
        let rank_n = RANK.as_usize() as f64 * n;
        let secret_second = parties_f * 0.5;
        let inherited =
            rank_n * 0.5 * parties_f * sigma2 * (2.0 * (layout.k().as_usize() as f64 - pk_layout.k.as_usize() as f64)).exp2();
        let per_share = super::fixtures::expected_pk_variance(
            inherited,
            (1.0 + rank_n * secret_second) * sigma2,
            1.0 + rank_n * secret_second,
            rank_n * 0.5 * (1.0 + rank_n * secret_second),
            layout.k().as_usize(),
            pk_layout.k.as_usize(),
        );
        let expected_fresh = parties_f * per_share;
        super::fixtures::assert_fresh_noise(&res, expected_fresh, layout.k());
        assert!(expected_fresh > parties_f * sigma2);
        let bound = 0.5 * variance.log2() - layout.k().as_usize() as f64 + 0.5;
        assert_gglwe_noise_within(module, &res, &pt_want.data().to_ref(), &sk_ideal, bound, &mut scratch);

        *pk.dist_mut() = Distribution::BinaryProb(0.5);
        let before = share.clone();
        let mut source_xu = Source::new([20u8; 32]);
        let mut source_xe = Source::new([10u8; 32]);
        super::fixtures::assert_panics_with(
            "invalid public key: ephemeral distribution differs from its secret provenance",
            || {
                with_scratch::<BE, _>(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout), |s| {
                    module.mhe_glwe_tensor_key_share_gen(&mut share, &parties[0].0, &pk, &mut source_xu, &mut source_xe, s)
                });
            },
        );
        assert!(share == before);
        assert_eq!(source_xu.next_i64(), Source::new([20u8; 32]).next_i64());
        assert_eq!(source_xe.next_i64(), Source::new([10u8; 32]).next_i64());
    }
}

/// Sharing under a public key less precise than the share panics.
pub fn test_mhe_glwe_tensor_key_share_gen_rejects_pk_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWETensorKeyShareAlloc<BE>
        + MHEGLWETensorKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = tensor_key_layout(module, DNUM, DSIZE);
    let pk_layout = public_key_layout(module, TorusPrecision(layout.k().0 - BASE2K.0));

    let (sk, _) = secret_from_seed(module, [100u8; 32]);
    let pk: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
    let mut res = module.mhe_glwe_tensor_key_share_alloc_from_infos(&layout);
    let before = res.clone();
    let mut source_a = Source::new([20u8; 32]);
    let mut source_b = Source::new([10u8; 32]);
    super::fixtures::assert_panics_with("invalid share: public key less precise than the share", || {
        with_scratch::<BE, _>(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout), |s| {
            module.mhe_glwe_tensor_key_share_gen(&mut res, &sk, &pk, &mut source_a, &mut source_b, s)
        })
    });
    assert!(res == before);
    assert_eq!(source_a.next_i64(), Source::new([20u8; 32]).next_i64());
    assert_eq!(source_b.next_i64(), Source::new([10u8; 32]).next_i64());
}

/// Sharing with a secret whose degree differs from the key's panics.
pub fn test_mhe_glwe_tensor_key_share_gen_rejects_secret_degree<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWETensorKeyShareAlloc<BE>
        + MHEGLWETensorKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = tensor_key_layout(module, DNUM, DSIZE);
    let pk_layout = public_key_layout(module, layout.k());

    let sk_layout = GLWESecretLayout {
        n: (module.n() / 2).into(),
        rank: RANK,
    };
    let mut sk: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc_from_infos(&sk_layout);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new([100u8; 32]));
    let pk: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
    let mut res = module.mhe_glwe_tensor_key_share_alloc_from_infos(&layout);
    let before = res.clone();
    let mut source_a = Source::new([20u8; 32]);
    let mut source_b = Source::new([10u8; 32]);
    super::fixtures::assert_panics_with("invalid share: secret degree differs from the key's", || {
        with_scratch::<BE, _>(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout), |s| {
            module.mhe_glwe_tensor_key_share_gen(&mut res, &sk, &pk, &mut source_a, &mut source_b, s)
        })
    });
    assert!(res == before);
    assert_eq!(source_a.next_i64(), Source::new([20u8; 32]).next_i64());
    assert_eq!(source_b.next_i64(), Source::new([10u8; 32]).next_i64());
}

pub(crate) fn tensor_key_layout<BE: Backend>(module: &Module<BE>, dnum: Dnum, dsize: Dsize) -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: module.n().into(),
        base2k: BASE2K,
        dnum,
        k_aux: TorusPrecision(dsize.0 * BASE2K.0 + module.log_n() as u32),
        rank: RANK,
        k_digit: k_digit_of(BASE2K, dsize),
    }
}

/// A public key layout at precision `k`, the tensor key's rank and radix.
fn public_key_layout<BE: Backend>(module: &Module<BE>, k: TorusPrecision) -> GLWELayout {
    GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k,
        rank: RANK,
    }
}

/// Sharing under a public key of another degree or radix, or of an encapsulated distribution, panics before any draw.
pub fn test_mhe_glwe_tensor_key_share_gen_rejects_pk_shape<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWETensorKeyShareAlloc<BE>
        + MHEGLWETensorKeyProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = tensor_key_layout(module, DNUM, DSIZE);

    let (sk, _) = secret_from_seed(module, [100u8; 32]);
    let expected = [
        "invalid share: public key degree differs from the key's",
        "invalid share: public key radix differs from the key's",
        "invalid share: public key needs a samplable distribution",
    ];
    for (case, expected) in expected.into_iter().enumerate() {
        let mut pk_layout = public_key_layout(module, layout.k());
        let mut pk = if case == 0 {
            module.glwe_public_key_prepared_alloc_from_infos(&GLWELayout {
                n: (module.n() / 2).into(),
                ..pk_layout
            })
        } else {
            if case == 1 {
                pk_layout.base2k = Base2K(BASE2K.0 / 2);
            }
            module.glwe_public_key_prepared_alloc_from_infos(&pk_layout)
        };
        if case == 2 {
            *pk.dist_mut() = Distribution::ENCAPSULATED("private distribution label");
        }
        let mut res = module.mhe_glwe_tensor_key_share_alloc_from_infos(&layout);
        let before = res.clone();
        let mut source_a = Source::new([20u8; 32]);
        let mut source_b = Source::new([10u8; 32]);
        super::fixtures::assert_panics_with(expected, || {
            with_scratch::<BE, _>(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout), |s| {
                module.mhe_glwe_tensor_key_share_gen(&mut res, &sk, &pk, &mut source_a, &mut source_b, s)
            });
        });
        assert!(res == before);
        assert_eq!(source_a.next_i64(), Source::new([20u8; 32]).next_i64());
        assert_eq!(source_b.next_i64(), Source::new([10u8; 32]).next_i64());
    }
}
