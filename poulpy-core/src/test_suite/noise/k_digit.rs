//! Gadget products with digits of any bit width, on and off limb boundaries.
//!
//! Each sweep runs one product per digit width and checks the measured noise
//! against the layout's noise model, which reads the width in bits: the whole
//! output noise, and the gadget term alone against the input's phase.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSource, VecZnxRotateAssign},
    layouts::{Module, ScalarZnx, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::api::GLWESecretSampling;
use crate::layouts::prepared::{GGLWEPreparedToBackendRef, GGSWPreparedToBackendRef};
use crate::test_suite::noise::glwe_noise_checked;
use crate::test_suite::poisoned_scratch;
use crate::{
    GGSWEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, GLWEKeyswitch, GLWENoise, GLWENormalize,
    GLWESwitchingKeyEncryptSk,
    api::{GGSWPreparedFactory, GLWESecretPreparedFactory, GLWESwitchingKeyPreparedFactory},
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGSW, GGSWLayout, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESwitchingKey, GLWESwitchingKeyLayout, ModuleCoreAlloc,
        prepared::{GGSWPrepared, GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::{GGLWENoiseModel, GGSWNoiseModel},
};

/// Digit widths swept: every width from 2 bits to three limbs.
fn k_digits(base2k: usize) -> impl Iterator<Item = usize> {
    2..=3 * base2k
}

/// Key switch with digits of every width from 2 bits to three limbs, from radix
/// `base2k - 1` through a key at `base2k` to radix `base2k - 2`.
pub fn test_glwe_keyswitch_k_digit<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSource<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let key_base2k: usize = params.base2k;
    let (in_base2k, out_base2k) = (key_base2k - 1, key_base2k - 2);
    let k_in: usize = 4 * in_base2k + 1;
    let n: usize = module.n();
    let rank: usize = 1;
    for k_digit in k_digits(key_base2k) {
        let dnum: usize = k_in.div_ceil(k_digit);
        // Wide enough that the output rounding stays below the gadget noise.
        let k_out: usize = dnum * k_digit + k_digit + module.log_n() + key_base2k;
        let glwe_in_infos = GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k_in.into(),
            rank: rank.into(),
        };
        let glwe_out_infos = GLWELayout {
            n: n.into(),
            base2k: out_base2k.into(),
            k: k_out.into(),
            rank: rank.into(),
        };
        let ksk_infos = GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: key_base2k.into(),
            dnum: dnum.into(),
            k_aux: (k_digit + module.log_n()).into(),
            k_digit: k_digit.into(),
            k_offset: crate::layouts::TorusPrecision(0),
            rank_in: rank.into(),
            rank_out: rank.into(),
        };

        let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
        let mut glwe_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut glwe_out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
        let mut pt_out: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

        let mut source_xs: Source = Source::new([1u8; 32]);
        let mut source_xe: Source = Source::new([2u8; 32]);
        let mut source_xa: Source = Source::new([3u8; 32]);
        module.vec_znx_fill_uniform_source(
            in_base2k,
            k_in,
            &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
            0,
            &mut source_xa,
        );

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module.glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                | module.glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                | module.glwe_noise_tmp_bytes(&glwe_out_infos)
                | module.glwe_decrypt_tmp_bytes(&glwe_in_infos)
                | module.glwe_switching_key_prepare_tmp_bytes(&ksk_infos)
                | module.glwe_normalize_tmp_bytes(),
        );

        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);
        let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
        let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

        module.glwe_switching_key_encrypt_sk(
            &mut ksk,
            &sk_in,
            &sk_out,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        );
        module.glwe_encrypt_sk(
            &mut glwe_in,
            &pt_in,
            &sk_in_prepared,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
        module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());
        module.glwe_keyswitch(
            &mut glwe_out,
            &glwe_in,
            &ksk_prepared.to_backend_ref(),
            &mut poisoned_scratch::<BE>(module.glwe_keyswitch_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ksk_infos)).borrow(),
        );

        let model: f64 = ksk_infos.log2_std_noise_keyswitch(
            &glwe_out_infos,
            &glwe_in_infos,
            0.5,
            0.5,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            0f64,
        );
        module.glwe_normalize(&mut pt_out, &pt_in, &mut scratch.borrow());
        let have = glwe_noise_checked(module, &glwe_out, &pt_out, &sk_out_prepared, &mut scratch.borrow())
            .std()
            .log2();
        assert!(have <= model + 1.0, "k_digit={k_digit}: noise {have} > model {model} + 1");

        // The switch alone, against the input's phase: the input noise cancels
        // and what is left is the gadget term the digit width sets.
        let mut phase: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
        module.glwe_decrypt(&glwe_in, &mut phase, &sk_in_prepared, &mut scratch.borrow());
        let gadget_have = glwe_noise_checked(module, &glwe_out, &phase, &sk_out_prepared, &mut scratch.borrow())
            .std()
            .log2();
        let gadget_model: f64 = ksk_infos.log2_std_noise_keyswitch(
            &glwe_out_infos,
            &glwe_in_infos,
            0.5,
            0.5,
            0f64,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            0f64,
        );
        assert!(
            (gadget_have - gadget_model).abs() <= 1.0,
            "k_digit={k_digit}: gadget noise {gadget_have} vs model {gadget_model}"
        );
    }
}

/// External product with a GGSW of digits of every width from 2 bits to three limbs.
pub fn test_glwe_external_product_k_digit<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxFillUniformSource<BE>
        + GLWEExternalProduct<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + VecZnxRotateAssign<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let key_base2k: usize = params.base2k;
    let (in_base2k, out_base2k) = (key_base2k - 1, key_base2k - 2);
    let k_in: usize = 4 * in_base2k + 1;
    let n: usize = module.n();
    for rank in 1_usize..3 {
        for k_digit in k_digits(key_base2k) {
            let dnum: usize = k_in.div_ceil(k_digit);
            let k_out: usize = dnum * k_digit + k_digit + module.log_n() + key_base2k;
            let glwe_in_infos = GLWELayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            };
            let glwe_out_infos = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };
            let ggsw_infos = GGSWLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum.into(),
                k_aux: (k_digit + module.log_n()).into(),
                k_digit: k_digit.into(),
                rank: rank.into(),
            };

            let mut ggsw: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_infos);
            let mut glwe_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
            let mut glwe_out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
            let mut pt_ggsw: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);
            let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
            let mut pt_out: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([4u8; 32]);
            let mut source_xe: Source = Source::new([5u8; 32]);
            let mut source_xa: Source = Source::new([6u8; 32]);
            module.vec_znx_fill_uniform_source(
                in_base2k,
                k_in,
                &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
                0,
                &mut source_xa,
            );
            let k: usize = 1;
            pt_ggsw.raw_mut()[k] = 1; // X^k

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module.ggsw_encrypt_sk_tmp_bytes(&ggsw_infos)
                    | module.glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                    | module.glwe_noise_tmp_bytes(&glwe_out_infos)
                    | module.glwe_decrypt_tmp_bytes(&glwe_in_infos)
                    | module.ggsw_prepare_tmp_bytes(&ggsw_infos)
                    | module.glwe_normalize_tmp_bytes(),
            );
            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.ggsw_encrypt_sk(
                &mut ggsw,
                &pt_ggsw,
                &sk_prepared,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            module.glwe_encrypt_sk(
                &mut glwe_in,
                &pt_in,
                &sk_prepared,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            let mut ggsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw);
            module.ggsw_prepare(&mut ggsw_prepared, &ggsw, &mut scratch.borrow());
            module.glwe_external_product(
                &mut glwe_out,
                &glwe_in,
                &ggsw_prepared.to_backend_ref(),
                &mut poisoned_scratch::<BE>(module.glwe_external_product_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ggsw_infos))
                    .borrow(),
            );

            module.vec_znx_rotate_assign(
                k as i64,
                &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
                0,
                &mut scratch.borrow(),
            );
            module.glwe_normalize(&mut pt_out, &pt_in, &mut scratch.borrow());

            let var_sigma: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
            let model: f64 = ggsw_infos.log2_std_noise_external_product(
                &glwe_out_infos,
                &glwe_in_infos,
                0.5,
                1f64 / n as f64,
                var_sigma,
                0f64,
                var_sigma,
                0f64,
            );
            let have = glwe_noise_checked(module, &glwe_out, &pt_out, &sk_prepared, &mut scratch.borrow())
                .std()
                .log2();
            assert!(have <= model + 1.0, "k_digit={k_digit}: noise {have} > model {model} + 1");

            // The product alone, against X^k times the input's phase.
            let mut phase: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
            module.glwe_decrypt(&glwe_in, &mut phase, &sk_prepared, &mut scratch.borrow());
            module.vec_znx_rotate_assign(
                k as i64,
                &mut vec_znx_backend_mut::<BE>(&mut phase.data),
                0,
                &mut scratch.borrow(),
            );
            let gadget_have = glwe_noise_checked(module, &glwe_out, &phase, &sk_prepared, &mut scratch.borrow())
                .std()
                .log2();
            let gadget_model: f64 = ggsw_infos.log2_std_noise_external_product(
                &glwe_out_infos,
                &glwe_in_infos,
                0.5,
                1f64 / n as f64,
                0f64,
                0f64,
                var_sigma,
                0f64,
            );
            assert!(
                (gadget_have - gadget_model).abs() <= 1.0,
                "k_digit={k_digit}: gadget noise {gadget_have} vs model {gadget_model}"
            );
        }
    }
}
