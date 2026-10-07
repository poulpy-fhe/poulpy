//! Collective tensor key over three parties: every entry of the finalized
//! key decrypts under the ideal secret to the ideal secret's tensor.

use poulpy_core::{
    DEFAULT_SIGMA_XE, Distribution, GGLWENoise, GetDistributionMut,
    layouts::{
        Base2K, Dnum, Dsize, GLWELayout, GLWEPublicKeyPrepared, GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretLayout,
        GLWESecretPreparedFactory, GLWESecretSampling, GLWESecretTensor, GLWESecretTensorFactory, GLWETensorKey,
        GLWETensorKeyLayout, LWEInfos, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use super::{
    fixtures::{
        BASE2K, DNUM, DSIZE, PARTIES, RANK, collective_public_key, ideal_secret, party_secrets, secret_from_seed, secret_sum,
    },
    pat::assert_gglwe_noise_within,
};
use crate::{
    api::{GLWEPublicKeyMHEProtocol, GLWETensorKeyMHEProtocol},
    layouts::MHEModuleAlloc,
};

pub fn test_glwe_tensor_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWETensorKeyMHEProtocol<BE>
        + GLWEPublicKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretTensorFactory<BE>
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
        let pk = collective_public_key(module, &parties, &pk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_secret_tensor_prepare_tmp_bytes(RANK)
                .max(module.mhe_glwe_tensor_key_share_finalize_tmp_bytes())
                .max(module.gglwe_noise_tmp_bytes(&layout)),
        );
        let mut pt_want: GLWESecretTensor<AlignedBuf, i64> = module.glwe_secret_tensor_alloc(RANK);
        module.glwe_secret_tensor_prepare(&mut pt_want, &secret_sum(module, &parties), &mut scratch.borrow());

        let mut acc = module.glwe_tensor_key_share_alloc_from_infos(&layout);
        let mut share = module.glwe_tensor_key_share_alloc_from_infos(&layout);
        for (i, (sk, _)) in parties.iter().enumerate() {
            let dst = if i == 0 { &mut acc } else { &mut share };
            let mut source_xu = Source::new([20 + i as u8; 32]);
            let mut source_xe = Source::new([10 + i as u8; 32]);
            // The secret components and their encryptions would be left in the scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(
                module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout),
                |scratch| module.mhe_glwe_tensor_key_share_gen(dst, sk, &pk, &mut source_xu, &mut source_xe, scratch),
            );
            if i > 0 {
                module.mhe_glwe_tensor_key_share_aggregate(&mut acc, &share);
            }
        }

        let mut res: GLWETensorKey<AlignedBuf, i64> = module.glwe_tensor_key_alloc_from_infos(&layout);
        module.mhe_glwe_tensor_key_share_finalize(&mut res, &acc, &mut scratch.borrow());
        // Each party's pk encryption of zero adds 2 * rank * n * 0.5 * PARTIES * sigma^2 + sigma^2.
        let n = module.n() as f64;
        let parties_f = PARTIES as f64;
        let variance = 2.0 * RANK.as_usize() as f64 * n * 0.5 * parties_f * parties_f * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE
            + parties_f * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
        let bound = 0.5 * variance.log2() - layout.k().as_usize() as f64 + 0.5;
        assert_gglwe_noise_within(module, &res, &pt_want.data().to_ref(), &sk_ideal, bound, &mut scratch);
    }
}

/// Sharing under a public key less precise than the share panics.
pub fn test_glwe_tensor_key_pk_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWETensorKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = tensor_key_layout(module, DNUM, DSIZE);
    let pk_layout = public_key_layout(module, TorusPrecision(layout.k().0 - BASE2K.0));
    let (sk, _) = secret_from_seed(module, [100u8; 32]);
    let pk: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
    let mut res = module.glwe_tensor_key_share_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout));
    module.mhe_glwe_tensor_key_share_gen(
        &mut res,
        &sk,
        &pk,
        &mut Source::new([20u8; 32]),
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

/// Sharing with a secret whose degree differs from the key's panics.
pub fn test_glwe_tensor_key_secret_degree<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWETensorKeyMHEProtocol<BE>
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
    let mut res = module.glwe_tensor_key_share_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout));
    module.mhe_glwe_tensor_key_share_gen(
        &mut res,
        &sk,
        &pk,
        &mut Source::new([20u8; 32]),
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

fn tensor_key_layout<BE: Backend>(module: &Module<BE>, dnum: Dnum, dsize: Dsize) -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: module.n().into(),
        base2k: BASE2K,
        dnum,
        k_aux: TorusPrecision(dsize.0 * BASE2K.0 + module.log_n() as u32),
        rank: RANK,
        dsize,
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

/// Public key layout mismatches are rejected before core encryption.
pub fn test_glwe_tensor_key_pk_shape_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWETensorKeyMHEProtocol<BE>
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
        super::fixtures::assert_panics_with(expected, || {
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
            let mut res = module.glwe_tensor_key_share_alloc_from_infos(&layout);
            let mut scratch: ScratchOwned<BE> =
                ScratchOwned::alloc(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(&layout, &pk_layout));
            module.mhe_glwe_tensor_key_share_gen(
                &mut res,
                &sk,
                &pk,
                &mut Source::new([20u8; 32]),
                &mut Source::new([10u8; 32]),
                &mut scratch.borrow(),
            );
        });
    }
}
