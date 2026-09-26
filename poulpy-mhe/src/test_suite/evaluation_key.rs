//! Collective evaluation keys over three parties: the finalized aggregate is a
//! key of the ideal secrets, and finalizing before or after normalization gives
//! the same result.

use poulpy_core::{
    EncryptionLayout, GGLWENoise,
    layouts::{
        Degree, GLWEAutomorphismKey, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling,
        GLWESwitchingKey, GLWESwitchingKeyDegrees, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism},
    layouts::{
        GaloisElement, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnxAsVecZnxBackendMut, ScalarZnxAsVecZnxBackendRef,
        ScratchOwned, WriterTo,
    },
    source::Source,
};

use super::{
    fixtures::{P, PARTIES, RANK, SEEDS, Secret, gglwe_layout, ideal_secret, party_secrets, secret_from_seed, secret_sum},
    pat::assert_gglwe_noise,
};
use crate::{
    api::{GLWEAutomorphismKeyShare, GLWESwitchingKeyShare},
    layouts::MHEModuleAlloc,
};

pub fn test_glwe_switching_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>:
        MHEModuleAlloc<BE> + GLWESwitchingKeyShare<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE> + GGLWENoise<BE>,
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
            .glwe_switching_key_share_tmp_bytes(&layout)
            .max(module.glwe_switching_key_share_normalize_tmp_bytes())
            .max(module.glwe_switching_key_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.glwe_switching_key_pat_compressed_alloc_from_infos(&layout);
    let mut share = module.glwe_switching_key_pat_compressed_alloc_from_infos(&layout);
    for (i, ((sk_in, _), (sk_out, _))) in parties_in.iter().zip(&parties_out).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_switching_key_share(
            dst,
            sk_in,
            sk_out,
            SEEDS[0],
            &enc_infos,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        assert!(dst.is_canonical());
        if i > 0 {
            module.glwe_switching_key_share_aggregate_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GLWESwitchingKey<AlignedBuf, i64> = module.glwe_switching_key_alloc_from_infos(&layout);
    module.glwe_switching_key_finalize(&mut lazy, &acc, &mut scratch.borrow());
    let n = Degree(module.n() as u32);
    assert_eq!((*lazy.input_degree(), *lazy.output_degree()), (n, n));
    assert_gglwe_noise(module, &lazy, &pt_want, &sk_out_ideal, &mut scratch);

    module.glwe_switching_key_share_normalize_assign(&mut acc, &mut scratch.borrow());
    assert!(acc.is_canonical());
    let mut eager: GLWESwitchingKey<AlignedBuf, i64> = module.glwe_switching_key_alloc_from_infos(&layout);
    module.glwe_switching_key_finalize(&mut eager, &acc, &mut scratch.borrow());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

pub fn test_glwe_automorphism_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEAutomorphismKeyShare<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + GaloisElement
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
            .glwe_automorphism_key_share_tmp_bytes(&layout)
            .max(module.glwe_automorphism_key_share_normalize_tmp_bytes())
            .max(module.glwe_automorphism_key_finalize_tmp_bytes())
            .max(module.gglwe_noise_tmp_bytes(&layout)),
    );

    let mut acc = module.glwe_automorphism_key_pat_compressed_alloc_from_infos(&layout);
    let mut share = module.glwe_automorphism_key_pat_compressed_alloc_from_infos(&layout);
    for (i, (sk, _)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_automorphism_key_share(dst, P, sk, SEEDS[0], &enc_infos, &mut source_xe, &mut scratch.borrow());
        assert!(dst.is_canonical());
        if i > 0 {
            module.glwe_automorphism_key_share_aggregate_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GLWEAutomorphismKey<AlignedBuf, i64> = module.glwe_automorphism_key_alloc_from_infos(&layout);
    module.glwe_automorphism_key_finalize(&mut lazy, &acc, &mut scratch.borrow());
    assert_eq!(lazy.p(), P);
    assert_gglwe_noise(module, &lazy, &pt_want, &sk_out, &mut scratch);

    module.glwe_automorphism_key_share_normalize_assign(&mut acc, &mut scratch.borrow());
    assert!(acc.is_canonical());
    let mut eager: GLWEAutomorphismKey<AlignedBuf, i64> = module.glwe_automorphism_key_alloc_from_infos(&layout);
    module.glwe_automorphism_key_finalize(&mut eager, &acc, &mut scratch.borrow());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

/// Aggregating switching key shares of different degrees panics.
pub fn test_glwe_switching_key_degree_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWESwitchingKeyShare<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_switching_key_pat_compressed_alloc_from_infos(&layout);
    let mut b = module.glwe_switching_key_pat_compressed_alloc_from_infos(&layout);
    b.input_degree = Degree(module.n() as u32);
    module.glwe_switching_key_share_aggregate_assign(&mut a, &b);
}

/// Aggregating automorphism key shares of different Galois elements panics.
pub fn test_glwe_automorphism_key_p_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEAutomorphismKeyShare<BE>,
{
    let layout = gglwe_layout(module);
    let mut a = module.glwe_automorphism_key_pat_compressed_alloc_from_infos(&layout);
    let mut b = module.glwe_automorphism_key_pat_compressed_alloc_from_infos(&layout);
    b.p = P;
    module.glwe_automorphism_key_share_aggregate_assign(&mut a, &b);
}

/// `sigma_{p^-1}(sk)` prepared: the secret core's automorphism key encrypts under.
pub(crate) fn automorphism_inv_prepared<BE>(
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
        for col in 0..RANK.as_usize() {
            module.vec_znx_automorphism(module.galois_element_inv(p), &mut dst, col, &src, col);
        }
    }
    let mut prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    module.glwe_secret_prepare(&mut prepared, &sigma);
    prepared
}
