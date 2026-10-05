//! Collective CKKS refresh over three parties: both share parts carry independent
//! ordinary noise, and the refreshed ciphertext retains the plaintext integers.

use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, EncryptionLayout, GLWEAdd, GLWEDecrypt, GLWEEncryptSk, GLWEMaskInnerProduct,
    GLWENormalize, GLWEShift, GLWESub, NoiseInfos,
    layouts::{
        Base2K, GLWE, GLWELayout, GLWEPlaintext, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign, VecZnxFillUniformSource},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
    test_suite::vec_znx_backend_mut,
};

use crate::test_suite::fixtures::{
    BASE2K, K, K_OUT, LOG_BOUND, LOG_MESSAGE, PARTIES, SEED_XE, SEEDS, bounded_integers, encrypt_integers, glwe_layout_at,
    ideal_secret, party_secrets, plaintext_integers, secret_from_seed,
};
use crate::{
    api::GLWEShareToEncMHEProtocol,
    ckks::{api::CKKSRefreshMHEProtocol, layouts::MHECKKSModuleAlloc},
};

pub fn test_ckks_refresh<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHECKKSModuleAlloc<BE>
        + CKKSRefreshMHEProtocol<BE>
        + GLWEShareToEncMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEAdd<BE>
        + GLWESub<BE>
        + GLWEShift<BE>
        + GLWEMaskInnerProduct<BE>
        + GLWENormalize<BE>
        + VecZnxFillUniformSource<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    for (k_out, sigma_s2e) in [(K, DEFAULT_SIGMA_XE), (K_OUT, 2.0 * DEFAULT_SIGMA_XE)] {
        let in_layout = glwe_layout_at(module, K);
        let out_layout = glwe_layout_at(module, k_out);
        // S2E must use its output grid even when the caller describes the input grid.
        // Its configurable sigma must not change E2S's fixed ordinary noise.
        let enc_infos = NoiseInfos::new(K.as_usize(), sigma_s2e, 6.0 * sigma_s2e).unwrap();
        let parties = party_secrets(module);
        let sk = ideal_secret(module, &parties);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_encrypt_sk_tmp_bytes(&in_layout)
                .max(module.mhe_ckks_refresh_share_finalize_tmp_bytes(&out_layout))
                .max(module.mhe_glwe_share_to_enc_share_finalize_tmp_bytes())
                .max(module.glwe_decrypt_tmp_bytes(&out_layout))
                .max(module.glwe_mask_inner_product_tmp_bytes(&in_layout))
                .max(module.glwe_shift_tmp_bytes(in_layout.size()))
                .max(module.glwe_normalize_tmp_bytes()),
        );
        let m = bounded_integers(module.n(), LOG_MESSAGE, [30u8; 32]);
        let ct = encrypt_integers(module, &in_layout, &m, &sk, &mut scratch);

        let mut acc = module.ckks_refresh_share_alloc_from_infos(&in_layout, &out_layout);
        let mut share = module.ckks_refresh_share_alloc_from_infos(&in_layout, &out_layout);
        for (i, (_, sk_i)) in parties.iter().enumerate() {
            let dst = if i == 0 { &mut acc } else { &mut share };
            dst.e2s.inner.set_canonical(false);
            let mut source_xm = Source::new([60 + i as u8; 32]);
            let mut source_xe = Source::new([80 + i as u8; 32]);
            // The party's integers and inner product would be left in the scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(
                module.mhe_ckks_refresh_share_gen_tmp_bytes(&in_layout, &out_layout),
                |scratch| {
                    module.mhe_ckks_refresh_share_gen(
                        dst,
                        &ct,
                        sk_i,
                        LOG_BOUND,
                        SEEDS[0],
                        &enc_infos,
                        &mut source_xm,
                        &mut source_xe,
                        scratch,
                    )
                },
            );
            assert!(dst.e2s.inner.is_canonical());

            // Recover only the mask, without replaying either noise sampler.
            let mut mask: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&in_layout);
            module.vec_znx_fill_uniform_source(
                BASE2K.as_usize(),
                LOG_BOUND,
                &mut vec_znx_backend_mut::<BE>(mask.data_mut()),
                0,
                &mut Source::new([60 + i as u8; 32]),
            );
            module.glwe_rsh(K.as_usize() - LOG_BOUND, &mut mask, &mut scratch.borrow());
            let mask_integers = plaintext_integers(&mask);
            let mut product: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&in_layout);
            let mut residual: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&in_layout);
            module.glwe_mask_inner_product(&mut product, &ct, sk_i, &mut scratch.borrow());
            module.glwe_sub(&mut residual, &dst.e2s.inner, &product);
            module.glwe_add_assign(&mut residual, &mask);
            module.glwe_normalize_assign(&mut residual, &mut scratch.borrow());
            let e2s_noise = plaintext_integers(&residual);
            let e2s_mean = assert_ordinary_noise(&e2s_noise, DEFAULT_SIGMA_XE.powi(2), DEFAULT_BOUND_XE.ceil() as i64 + 1);

            let mut encrypted_mask: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
            module.mhe_glwe_share_to_enc_share_finalize(&mut encrypted_mask, &dst.s2e, &mut scratch.borrow());
            let mut decrypted_mask: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&out_layout);
            module.glwe_decrypt(&encrypted_mask, &mut decrypted_mask, sk_i, &mut scratch.borrow());
            let s2e_noise: Vec<i64> = plaintext_integers(&decrypted_mask)
                .iter()
                .zip(&mask_integers)
                .map(|(got, want)| got - want)
                .collect();
            let s2e_mean = assert_ordinary_noise(&s2e_noise, sigma_s2e.powi(2), (6.0 * sigma_s2e).ceil() as i64 + 1);
            let covariance = e2s_noise
                .iter()
                .zip(&s2e_noise)
                .map(|(&a, &b)| (a as f64 - e2s_mean) * (b as f64 - s2e_mean))
                .sum::<f64>()
                / module.n() as f64;
            assert!(
                covariance.abs() < 0.2 * DEFAULT_SIGMA_XE * sigma_s2e,
                "E2S and S2E noise must be independent, covariance {covariance}"
            );
            if i > 0 {
                module.mhe_ckks_refresh_share_aggregate(&mut acc, &share);
            }
        }

        let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
        module.mhe_ckks_refresh_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
        assert!(res.is_canonical());
        // The integer-preserving raise retains both the input and E2S errors.
        let bound =
            ((PARTIES + 1) as f64 * DEFAULT_BOUND_XE + PARTIES as f64 * 6.0 * sigma_s2e).ceil() as i64 + 2 * PARTIES as i64 + 1;
        let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&out_layout);
        module.glwe_decrypt(&res, &mut pt, &sk, &mut scratch.borrow());
        let errors: Vec<i64> = plaintext_integers(&pt).iter().zip(&m).map(|(got, want)| got - want).collect();
        let variance = (PARTIES + 1) as f64 * DEFAULT_SIGMA_XE.powi(2) + PARTIES as f64 * sigma_s2e.powi(2);
        assert_ordinary_noise(&errors, variance, bound);
    }
}

/// Checks the scale and bottom bit without prescribing the sampler's random stream.
fn assert_ordinary_noise(errors: &[i64], expected_variance: f64, bound: i64) -> f64 {
    assert!(errors.iter().all(|e| e.abs() <= bound), "ordinary noise exceeds {bound}");
    let mean = errors.iter().map(|&e| e as f64).sum::<f64>() / errors.len() as f64;
    let variance = errors.iter().map(|&e| (e as f64 - mean).powi(2)).sum::<f64>() / errors.len() as f64;
    assert!(
        (0.6 * expected_variance..1.4 * expected_variance).contains(&variance),
        "ordinary noise variance {variance}, expected {expected_variance}"
    );
    let odd = errors.iter().filter(|&&e| e & 1 != 0).count();
    assert!(
        odd > errors.len() / 4 && odd < 3 * errors.len() / 4,
        "ordinary noise must reach the bottom bit, {odd} odd coefficients"
    );
    mean
}

/// Finalizing a ciphertext whose radix differs from the output's panics.
pub fn test_ckks_refresh_finalize_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHECKKSModuleAlloc<BE> + CKKSRefreshMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let in_layout = glwe_layout_at(module, K);
    let ct_layout = GLWELayout {
        base2k: Base2K(BASE2K.0 + 1),
        ..in_layout
    };
    let out_layout = glwe_layout_at(module, K_OUT);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let share = module.ckks_refresh_share_alloc_from_infos(&in_layout, &out_layout);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ckks_refresh_share_finalize_tmp_bytes(&out_layout));
    module.mhe_ckks_refresh_share_finalize(&mut res, &ct, &share, &mut scratch.borrow());
}

/// Finalizing into an output less precise than the ciphertext panics.
pub fn test_ckks_refresh_finalize_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHECKKSModuleAlloc<BE> + CKKSRefreshMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let out_layout = glwe_layout_at(module, K);
    let in_layout = glwe_layout_at(module, K_OUT);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&in_layout);
    let share = module.ckks_refresh_share_alloc_from_infos(&in_layout, &out_layout);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ckks_refresh_share_finalize_tmp_bytes(&out_layout));
    module.mhe_ckks_refresh_share_finalize(&mut res, &ct, &share, &mut scratch.borrow());
}

/// A bound outside the ciphertext precision panics.
pub fn test_ckks_refresh_bound<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHECKKSModuleAlloc<BE> + CKKSRefreshMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let in_layout = glwe_layout_at(module, K);
    let out_layout = glwe_layout_at(module, K_OUT);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&in_layout);
    let mut share = module.ckks_refresh_share_alloc_from_infos(&in_layout, &out_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_ckks_refresh_share_gen_tmp_bytes(&in_layout, &out_layout));
    module.mhe_ckks_refresh_share_gen(
        &mut share,
        &ct,
        &sk,
        K.as_usize(),
        SEEDS[0],
        &EncryptionLayout::new_from_default_sigma(out_layout).unwrap(),
        &mut Source::new([60u8; 32]),
        &mut Source::new(SEED_XE),
        &mut scratch.borrow(),
    );
}
