//! Collective refresh over three parties: a ciphertext refreshed to a larger
//! precision decrypts under the ideal secret within the input, flood and fresh
//! noise bounds.

use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, EncryptionLayout, GLWEAdd, GLWEDecrypt, GLWEEncryptSk, GLWENormalize,
    layouts::{Base2K, GLWE, GLWELayout, GLWEPlaintext, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc},
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use super::fixtures::{
    BASE2K, K, K_OUT, LOG_BOUND, LOG_MESSAGE, PARTIES, SEEDS, assert_flooded_integers, bounded_integers, encrypt_integers,
    glwe_layout_at, ideal_secret, integer_flood_infos, party_secrets, plaintext_integers,
};
use crate::{api::GLWERefreshMHEProtocol, layouts::MHEModuleAlloc};

pub fn test_glwe_refresh<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWERefreshMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    for sigma in [1024.0, 4096.0] {
        let in_layout = glwe_layout_at(module, K);
        let flood = integer_flood_infos(in_layout, sigma);
        let out_layout = glwe_layout_at(module, K_OUT);
        let enc_infos = EncryptionLayout::new_from_default_sigma(out_layout).unwrap();
        let parties = party_secrets(module);
        let sk = ideal_secret(module, &parties);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_encrypt_sk_tmp_bytes(&in_layout)
                .max(module.mhe_glwe_refresh_share_gen_tmp_bytes(&in_layout, &out_layout))
                .max(module.mhe_glwe_refresh_share_finalize_tmp_bytes(&out_layout))
                .max(module.glwe_decrypt_tmp_bytes(&out_layout)),
        );
        let m = bounded_integers(module.n(), LOG_MESSAGE, [30u8; 32]);
        let ct = encrypt_integers(module, &in_layout, &m, &sk, &mut scratch);

        let mut acc = module.glwe_refresh_share_alloc_from_infos(&in_layout, &out_layout);
        let mut share = module.glwe_refresh_share_alloc_from_infos(&in_layout, &out_layout);
        for (i, (_, sk_i)) in parties.iter().enumerate() {
            let dst = if i == 0 { &mut acc } else { &mut share };
            dst.e2s.inner.set_canonical(false);
            let mut source_xm = Source::new([60 + i as u8; 32]);
            let mut source_xe = Source::new([80 + i as u8; 32]);
            module.mhe_glwe_refresh_share_gen(
                dst,
                &ct,
                sk_i,
                LOG_BOUND,
                SEEDS[0],
                &flood,
                &enc_infos,
                &mut source_xm,
                &mut source_xe,
                &mut scratch.borrow(),
            );
            assert!(dst.e2s.inner.is_canonical());
            if i > 0 {
                module.mhe_glwe_refresh_share_aggregate(&mut acc, &share);
            }
        }

        let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
        module.mhe_glwe_refresh_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
        assert!(res.is_canonical());
        // The integer-preserving raise must retain E2S flooding at its input-frame magnitude.
        // Each S2E part also adds fresh output-frame encryption noise.
        let bound = ((PARTIES + 1) as f64 * DEFAULT_BOUND_XE + PARTIES as f64 * 6.0 * sigma).ceil() as i64 + PARTIES as i64;
        let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&out_layout);
        module.glwe_decrypt(&res, &mut pt, &sk, &mut scratch.borrow());
        assert_flooded_integers(
            &plaintext_integers(&pt),
            &m,
            sigma,
            (PARTIES + 1) as f64 * DEFAULT_SIGMA_XE.powi(2),
            bound,
        );
    }
}

/// Finalizing a ciphertext whose radix differs from the output's panics.
pub fn test_glwe_refresh_finalize_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWERefreshMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let in_layout = glwe_layout_at(module, K);
    let ct_layout = GLWELayout {
        base2k: Base2K(BASE2K.0 + 1),
        ..in_layout
    };
    let out_layout = glwe_layout_at(module, K_OUT);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let share = module.glwe_refresh_share_alloc_from_infos(&in_layout, &out_layout);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_refresh_share_finalize_tmp_bytes(&out_layout));
    module.mhe_glwe_refresh_share_finalize(&mut res, &ct, &share, &mut scratch.borrow());
}

/// Finalizing into an output less precise than the ciphertext panics.
pub fn test_glwe_refresh_finalize_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWERefreshMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let out_layout = glwe_layout_at(module, K);
    let in_layout = glwe_layout_at(module, K_OUT);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&in_layout);
    let share = module.glwe_refresh_share_alloc_from_infos(&in_layout, &out_layout);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_refresh_share_finalize_tmp_bytes(&out_layout));
    module.mhe_glwe_refresh_share_finalize(&mut res, &ct, &share, &mut scratch.borrow());
}
