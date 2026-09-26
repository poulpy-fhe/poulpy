//! Collective tensor key over three parties: every entry of the finalized
//! key decrypts under the ideal secret to the ideal secret's tensor, and
//! finalizing before or after normalization gives the same result.

use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GGLWENoise,
    layouts::{
        GLWELayout, GLWEPublicKeyPrepared, GLWEPublicKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWESecretTensor, GLWESecretTensorFactory, GLWETensorKey, GLWETensorKeyLayout, LWEInfos, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned, WriterTo},
    source::Source,
};

use super::{
    fixtures::{
        BASE2K, DNUM, DSIZE, PARTIES, RANK, collective_public_key, ideal_secret, party_secrets, secret_from_seed, secret_sum,
    },
    pat::assert_gglwe_noise_within,
};
use crate::{
    api::{GLWEPublicKeyShare, GLWETensorKeyShare, PatAggregate, PatFinalize, PatNormalize},
    layouts::MHEModuleAlloc,
};

pub fn test_glwe_tensor_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWETensorKeyShare<BE>
        + GLWEPublicKeyShare<BE>
        + PatAggregate<BE>
        + PatNormalize<BE>
        + PatFinalize<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretTensorFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = tensor_key_layout(module);
    let pk_layout = public_key_layout(module, layout.k());
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let parties = party_secrets(module);
    let sk_ideal = ideal_secret(module, &parties);
    let pk = collective_public_key(module, &parties, &pk_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_tensor_key_share_tmp_bytes(&layout, &pk_layout)
            .max(module.glwe_secret_tensor_prepare_tmp_bytes(RANK))
            .max(module.pat_normalize_tmp_bytes())
            .max(module.pat_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );
    let mut pt_want: GLWESecretTensor<AlignedBuf, i64> = module.glwe_secret_tensor_alloc(RANK);
    module.glwe_secret_tensor_prepare(&mut pt_want, &secret_sum(module, &parties), &mut scratch.borrow());

    let mut acc = module.gglwe_pat_alloc_from_infos(&layout);
    let mut share = module.gglwe_pat_alloc_from_infos(&layout);
    for (i, (sk, _)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        dst.set_canonical(false);
        let mut source_xu = Source::new([20 + i as u8; 32]);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_tensor_key_share(
            dst,
            sk,
            &pk,
            &enc_infos,
            &mut source_xu,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        assert!(dst.is_canonical());
        if i > 0 {
            module.gglwe_pat_aggregate_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GLWETensorKey<AlignedBuf, i64> = module.glwe_tensor_key_alloc_from_infos(&layout);
    module.gglwe_pat_finalize(&mut lazy, &acc, &mut scratch.borrow());
    // Each party's pk encryption of zero adds (rank + 1) * n * 0.5 * PARTIES * sigma^2 + sigma^2.
    let n = module.n() as f64;
    let parties_f = PARTIES as f64;
    let variance = (RANK.as_usize() as f64 + 1.0) * n * 0.5 * parties_f * parties_f * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE
        + parties_f * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
    let bound = 0.5 * variance.log2() - layout.k().as_usize() as f64 + 0.5;
    assert_gglwe_noise_within(module, &lazy, &pt_want.data().to_ref(), &sk_ideal, bound, &mut scratch);

    module.gglwe_pat_normalize_assign(&mut acc, &mut scratch.borrow());
    assert!(acc.is_canonical());
    let mut eager: GLWETensorKey<AlignedBuf, i64> = module.glwe_tensor_key_alloc_from_infos(&layout);
    module.gglwe_pat_finalize(&mut eager, &acc, &mut scratch.borrow());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

/// Sharing under a public key less precise than the share panics.
pub fn test_glwe_tensor_key_pk_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWETensorKeyShare<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = tensor_key_layout(module);
    let pk_layout = public_key_layout(module, TorusPrecision(layout.k().0 - BASE2K.0));
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let (sk, _) = secret_from_seed(module, [100u8; 32]);
    let pk: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
    let mut res = module.gglwe_pat_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_tensor_key_share_tmp_bytes(&layout, &pk_layout));
    module.glwe_tensor_key_share(
        &mut res,
        &sk,
        &pk,
        &enc_infos,
        &mut Source::new([20u8; 32]),
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

fn tensor_key_layout<BE: Backend>(module: &Module<BE>) -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: module.n().into(),
        base2k: BASE2K,
        dnum: DNUM,
        k_aux: TorusPrecision(DSIZE.0 * BASE2K.0 + module.log_n() as u32),
        rank: RANK,
        dsize: DSIZE,
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
