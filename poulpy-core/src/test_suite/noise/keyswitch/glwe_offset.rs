//! Key switches with keys whose rows carry a row offset.

use std::panic::{AssertUnwindSafe, catch_unwind};

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssign, VecZnxDftAlloc, VecZnxFillUniformSource, VecZnxIdftNormalize,
        VecZnxIdftNormalizeTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes,
    },
    layouts::{Module, ScratchOwned, VecZnxDftToBackendMut},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};

use crate::layouts::GLWESecretSampling;
use crate::test_suite::noise::glwe_noise_checked;
use crate::{
    GGLWEKeyswitch, GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWENormalize, GLWEShift,
    GLWESwitchingKeyEncryptSk,
    encryption::DEFAULT_SIGMA_XE,
    gadget::gglwe_product_accumulation_output_size,
    layouts::{
        GGLWEInfos, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey,
        GLWESwitchingKeyLayout, GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, TorusPrecision,
        prepared::{GGLWEPreparedToBackendRef, GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::GGLWENoiseModel,
};

struct Case {
    in_base2k: usize,
    k_offset: usize,
    k_digit: usize,
}

/// The digit and offset shapes the tests sweep at radix `base2k`.
fn cases(base2k: usize, k_in: usize) -> Vec<Case> {
    let mut cases = Vec::new();
    for in_base2k in [base2k, base2k - 1] {
        // Aligned and unaligned offsets.
        for k_offset in [base2k, 2 * base2k + 5] {
            // One-limb digits, two-limb digits, and one digit covering the input.
            for k_digit in [base2k, 2 * base2k, k_in] {
                cases.push(Case {
                    in_base2k,
                    k_offset,
                    k_digit,
                });
            }
        }
    }
    cases
}

/// A switch with an offset key equals the input lowered by `k_offset` bits with
/// `glwe_rsh` and switched with an offset-free key: both read the input digits as
/// balanced integers, so their phases differ only by the two keys' noise.
pub fn test_glwe_keyswitch_row_offset<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSource<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEKeyswitch<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let log_n: usize = module.log_n();
    let k_in: usize = 2 * base2k + 3;
    let (rank_in, rank_out) = (2_usize, 1_usize);

    for case in cases(base2k, k_in) {
        let Case {
            in_base2k,
            k_offset,
            k_digit,
        } = case;
        let dnum: usize = k_in.div_ceil(k_digit);
        let ksk_infos = GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (k_digit + log_n).into(),
            k_digit: k_digit.into(),
            k_offset: k_offset.into(),
            rank_in: rank_in.into(),
            rank_out: rank_out.into(),
        };
        // The offset-free key of the reference switch, over the lowered input.
        let k_wide: usize = k_in + k_offset;
        let ref_infos = GLWESwitchingKeyLayout {
            dnum: k_wide.div_ceil(base2k).into(),
            k_aux: (base2k + log_n).into(),
            k_digit: base2k.into(),
            k_offset: TorusPrecision(0),
            ..ksk_infos
        };
        let in_infos = GLWELayout {
            n: n.into(),
            base2k: in_base2k.into(),
            k: k_in.into(),
            rank: rank_in.into(),
        };
        let wide_infos = GLWELayout {
            k: k_wide.into(),
            ..in_infos
        };
        let out_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: (k_offset + dnum * k_digit).into(),
            rank: rank_out.into(),
        };

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                .max(module.glwe_switching_key_encrypt_sk_tmp_bytes(&ref_infos))
                .max(module.glwe_encrypt_sk_tmp_bytes(&in_infos))
                .max(module.glwe_keyswitch_tmp_bytes(&out_infos, &in_infos, &ksk_infos))
                .max(module.glwe_keyswitch_tmp_bytes(&out_infos, &wide_infos, &ref_infos))
                .max(module.glwe_shift_tmp_bytes(wide_infos.size()))
                .max(module.glwe_decrypt_tmp_bytes(&out_infos)),
        );
        let (mut source_xs, mut source_xe, mut source_xa) =
            (Source::new([1u8; 32]), Source::new([2u8; 32]), Source::new([3u8; 32]));

        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank_in.into());
        module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);
        let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
        let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank_out.into());
        module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

        let prepared = |infos: &GLWESwitchingKeyLayout,
                        scratch: &mut ScratchOwned<BE>,
                        source_xe: &mut Source,
                        source_xa: &mut Source|
         -> GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> {
            let mut key: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(infos);
            module.glwe_switching_key_encrypt_sk(&mut key, &sk_in, &sk_out, source_xe, source_xa, &mut scratch.arena());
            assert_eq!(key.k_offset(), infos.k_offset);
            let mut key_prepared = module.glwe_switching_key_prepared_alloc_from_infos(&key);
            module.glwe_switching_key_prepare(&mut key_prepared, &key, &mut scratch.borrow());
            key_prepared
        };
        let ksk = prepared(&ksk_infos, &mut scratch, &mut source_xe, &mut source_xa);
        let ksk_ref = prepared(&ref_infos, &mut scratch, &mut source_xe, &mut source_xa);

        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&in_infos);
        module.vec_znx_fill_uniform_source(
            pt_in.base2k().into(),
            pt_in.k().as_usize(),
            &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
            0,
            &mut source_xa,
        );
        let mut glwe_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&in_infos);
        module.glwe_encrypt_sk(
            &mut glwe_in,
            &pt_in,
            &sk_in_prepared,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut glwe_out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&out_infos);
        module.glwe_keyswitch(&mut glwe_out, &glwe_in, &ksk.to_backend_ref(), &mut scratch.borrow());

        let mut glwe_wide: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&wide_infos);
        module.glwe_copy(&mut glwe_wide, &glwe_in, &mut scratch.borrow());
        module.glwe_rsh(k_offset, &mut glwe_wide, &mut scratch.borrow());
        let mut glwe_ref: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&out_infos);
        module.glwe_keyswitch(&mut glwe_ref, &glwe_wide, &ksk_ref.to_backend_ref(), &mut scratch.borrow());

        let mut pt_ref: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&out_infos);
        module.glwe_decrypt(&glwe_ref, &mut pt_ref, &sk_out_prepared, &mut scratch.borrow());

        let var_key = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
        let var = |infos: &GLWESwitchingKeyLayout, input: &GLWELayout| {
            infos.var_noise_keyswitch(&out_infos, input, 0.5, 0.5, 0.0, var_key, var_key)
        };
        // The output sits at the precision the offset key's rows reach, so the final
        // normalization of each switch rounds there, limb boundary or not: at most half
        // an ulp per component, folded by the output secret.
        let res_k = out_infos.k.as_usize() as f64;
        let var_round = 2.0 * (1.0 + (rank_out * n) as f64 * 0.5) * (-2.0 * res_k).exp2() / 4.0;
        let noise_max: f64 = 0.5 * (var(&ksk_infos, &in_infos) + var(&ref_infos, &wide_infos) + var_round).log2() + 1.0;
        let noise_have = glwe_noise_checked(module, &glwe_out, &pt_ref, &sk_out_prepared, &mut scratch.borrow())
            .std()
            .log2();
        assert!(
            noise_have <= noise_max,
            "in_base2k={in_base2k} k_offset={k_offset} k_digit={k_digit}: noise_have {noise_have} > noise_max {noise_max}"
        );
    }
}

/// `glwe_keyswitch_dft_add` over several inputs, finished by one inverse transform
/// with their bodies summed limb by limb and lowered by the key's offset, equals
/// the sum of the single switches up to the rounding of the final normalization.
/// Every scratch is the advertised size, exactly.
pub fn test_glwe_keyswitch_dft_add<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSource<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWEAdd<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + GLWENormalize<BE>
        + VecZnxDftAlloc<BE>
        + poulpy_hal::api::VecZnxDftAddAssign<BE>
        + VecZnxAddAssign<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let log_n: usize = module.log_n();
    let k_in: usize = 2 * base2k + 3;
    let terms: usize = 3;
    let (rank_in, rank_out) = (1_usize, 1_usize);

    for case in cases(base2k, k_in).into_iter().filter(|case| case.in_base2k == base2k) {
        let Case { k_offset, k_digit, .. } = case;
        let dnum: usize = k_in.div_ceil(k_digit);
        let ksk_infos = GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (k_digit + log_n).into(),
            k_digit: k_digit.into(),
            k_offset: k_offset.into(),
            rank_in: rank_in.into(),
            rank_out: rank_out.into(),
        };
        let in_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_in.into(),
            rank: rank_in.into(),
        };
        let out_infos = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: (k_offset + dnum * k_digit).into(),
            rank: rank_out.into(),
        };
        let acc_size: usize = gglwe_product_accumulation_output_size::<BE, _, _, _>(&out_infos, &in_infos, &ksk_infos, terms);

        let mut setup: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                .max(module.glwe_encrypt_sk_tmp_bytes(&in_infos))
                .max(module.glwe_keyswitch_tmp_bytes(&out_infos, &in_infos, &ksk_infos)),
        );
        let (mut source_xs, mut source_xe, mut source_xa) =
            (Source::new([4u8; 32]), Source::new([5u8; 32]), Source::new([6u8; 32]));
        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank_in.into());
        module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);
        let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
        let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank_out.into());
        module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);
        let mut key: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
        module.glwe_switching_key_encrypt_sk(&mut key, &sk_in, &sk_out, &mut source_xe, &mut source_xa, &mut setup.arena());
        let mut ksk: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> = module.glwe_switching_key_prepared_alloc_from_infos(&key);
        module.glwe_switching_key_prepare(&mut ksk, &key, &mut setup.borrow());

        let inputs: Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> = (0..terms)
            .map(|_| {
                let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&in_infos);
                module.vec_znx_fill_uniform_source(base2k, k_in, &mut vec_znx_backend_mut::<BE>(&mut pt.data), 0, &mut source_xa);
                let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&in_infos);
                module.glwe_encrypt_sk(
                    &mut ct,
                    &pt,
                    &sk_in_prepared,
                    &mut source_xe,
                    &mut source_xa,
                    &mut setup.borrow(),
                );
                ct
            })
            .collect();

        // The sum of the single switches.
        let mut want: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&out_infos);
        let mut single: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&out_infos);
        for (t, input) in inputs.iter().enumerate() {
            if t == 0 {
                module.glwe_keyswitch(&mut want, input, &ksk.to_backend_ref(), &mut setup.borrow());
            } else {
                module.glwe_keyswitch(&mut single, input, &ksk.to_backend_ref(), &mut setup.borrow());
                module.glwe_add_assign(&mut want, &single);
            }
        }
        module.glwe_normalize_assign(&mut want, &mut setup.borrow());

        // The accumulated products, each with exactly the scratch it advertises.
        let mut acc = module.vec_znx_dft_alloc(n, rank_out + 1, acc_size);
        let mut scratch_dft: ScratchOwned<BE> =
            ScratchOwned::alloc(module.glwe_keyswitch_dft_tmp_bytes(acc_size, &in_infos, &ksk_infos));
        let mut scratch_add: ScratchOwned<BE> =
            ScratchOwned::alloc(module.glwe_keyswitch_dft_add_tmp_bytes(acc_size, &in_infos, &ksk_infos));
        for (t, input) in inputs.iter().enumerate() {
            if t == 0 {
                module.glwe_keyswitch_dft(
                    &mut acc.to_backend_mut(),
                    input,
                    &ksk.to_backend_ref(),
                    &mut scratch_dft.arena(),
                );
            } else {
                module.glwe_keyswitch_dft_add(
                    &mut acc.to_backend_mut(),
                    input,
                    &ksk.to_backend_ref(),
                    &mut scratch_add.arena(),
                );
            }
        }
        // The bodies, summed limb by limb so that their sum keeps its integer part, then lowered.
        let mut bodies = module.vec_znx_alloc(n, 1, inputs[0].size());
        for input in &inputs {
            module.vec_znx_add_assign(
                &mut vec_znx_backend_mut::<BE>(&mut bodies),
                0,
                &vec_znx_backend_ref::<BE>(&input.data),
                0,
            );
        }
        let mut lowered = module.vec_znx_alloc(n, 1, acc_size);
        let mut finish: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .vec_znx_normalize_tmp_bytes()
                .max(module.vec_znx_idft_normalize_tmp_bytes(out_infos.size(), acc_size)),
        );
        module.vec_znx_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut lowered),
            base2k,
            acc_size * base2k,
            -(k_offset as i64),
            0,
            &vec_znx_backend_ref::<BE>(&bodies),
            base2k,
            0,
            &mut finish.arena(),
        );
        let mut have: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&out_infos);
        let res_k = have.k().as_usize();
        {
            have.set_canonical(true);
            let mut have_data = vec_znx_backend_mut::<BE>(&mut have.data);
            let lowered_ref = vec_znx_backend_ref::<BE>(&lowered);
            for col in 0..rank_out + 1 {
                let addend = (col == 0).then_some((&lowered_ref, 0));
                module.vec_znx_idft_normalize(
                    &mut have_data,
                    base2k,
                    res_k,
                    0,
                    col,
                    &mut acc.to_backend_mut(),
                    col,
                    base2k,
                    addend,
                    &mut finish.arena(),
                );
            }
        }

        let mut pt_want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&out_infos);
        crate::GLWEDecrypt::glwe_decrypt(module, &want, &mut pt_want, &sk_out_prepared, &mut setup.borrow());
        let noise_have = glwe_noise_checked(module, &have, &pt_want, &sk_out_prepared, &mut setup.borrow())
            .std()
            .log2();
        // Only roundings differ: one per single switch for the sum, one for the
        // accumulated product, each at most half an ulp per component, folded by
        // the output secret.
        let var_round = (terms + 1) as f64 * (1.0 + (rank_out * n) as f64 * 0.5) * (-2.0 * res_k as f64).exp2() / 4.0;
        let noise_max = 0.5 * var_round.log2() + 1.0;
        assert!(
            noise_have <= noise_max,
            "k_offset={k_offset} k_digit={k_digit}: noise_have {noise_have} > noise_max {noise_max}"
        );
    }
}

/// The operations that cannot take a key with a row offset refuse one, and a
/// switch with one refuses an input it cannot decompose whole, and takes an
/// output wider than its rows, as an offset-free switch does. A GGLWE switched
/// into a result whose rows sit elsewhere is refused.
pub fn test_glwe_keyswitch_row_offset_refusals<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSource<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let k_in: usize = 2 * base2k + 3;
    let k_offset: usize = base2k + 5;
    let dnum: usize = k_in.div_ceil(base2k);
    let offset_infos = GLWESwitchingKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: dnum.into(),
        k_aux: (base2k + module.log_n()).into(),
        k_digit: base2k.into(),
        k_offset: k_offset.into(),
        rank_in: 1_usize.into(),
        rank_out: 1_usize.into(),
    };
    let free_infos = GLWESwitchingKeyLayout {
        k_offset: TorusPrecision(0),
        ..offset_infos
    };
    let glwe_infos = |k: usize| GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1_usize.into(),
    };
    let (in_infos, out_infos) = (glwe_infos(k_in), glwe_infos(k_offset + dnum * base2k));
    let wide_infos = glwe_infos(k_in + base2k);
    let past_rows_infos = glwe_infos(k_offset + dnum * base2k + base2k);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_switching_key_encrypt_sk_tmp_bytes(&offset_infos)
            .max(module.glwe_encrypt_sk_tmp_bytes(&wide_infos))
            .max(module.glwe_keyswitch_tmp_bytes(&out_infos, &wide_infos, &offset_infos))
            .max(module.glwe_keyswitch_tmp_bytes(&in_infos, &in_infos, &offset_infos))
            .max(module.glwe_keyswitch_tmp_bytes(&past_rows_infos, &in_infos, &offset_infos))
            .max(module.gglwe_keyswitch_tmp_bytes(&offset_infos, &offset_infos, &offset_infos))
            .max(module.gglwe_keyswitch_tmp_bytes(&free_infos, &offset_infos, &free_infos)),
    );
    let (mut source_xs, mut source_xe, mut source_xa) = (Source::new([7u8; 32]), Source::new([8u8; 32]), Source::new([9u8; 32]));
    let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(1_usize.into());
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(1_usize.into());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut key_offset: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&offset_infos);
    module.glwe_switching_key_encrypt_sk(
        &mut key_offset,
        &sk,
        &sk,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.arena(),
    );
    let mut key_free: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&free_infos);
    module.glwe_switching_key_encrypt_sk(&mut key_free, &sk, &sk, &mut source_xe, &mut source_xa, &mut scratch.arena());
    let mut ksk_offset: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
        module.glwe_switching_key_prepared_alloc_from_infos(&key_offset);
    module.glwe_switching_key_prepare(&mut ksk_offset, &key_offset, &mut scratch.borrow());
    let mut ksk_free: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> = module.glwe_switching_key_prepared_alloc_from_infos(&key_free);
    module.glwe_switching_key_prepare(&mut ksk_free, &key_free, &mut scratch.borrow());

    let mut ct: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&in_infos);
    let mut wide: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&wide_infos);
    let pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&wide_infos);
    module.glwe_encrypt_sk(
        &mut wide,
        &pt,
        &sk_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let mut out: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&out_infos);

    let mut past_rows: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&past_rows_infos);
    module.glwe_keyswitch(&mut past_rows, &ct, &ksk_offset.to_backend_ref(), &mut scratch.borrow());

    let refused = |what: &str, message: &str, op: &mut dyn FnMut()| {
        let err = catch_unwind(AssertUnwindSafe(op)).expect_err(what);
        let have = crate::test_suite::decryption::panic_message(&*err);
        assert!(have.contains(message), "{what}: panicked with {have:?}, expected {message:?}");
    };
    refused(
        "glwe_keyswitch_assign accepted a key with a row offset",
        "keys with a row offset",
        &mut || module.glwe_keyswitch_assign(&mut ct, &ksk_offset.to_backend_ref(), &mut scratch.borrow()),
    );
    refused(
        "glwe_keyswitch accepted an input past the offset key's max_input_k",
        "max_input_k",
        &mut || module.glwe_keyswitch(&mut out, &wide, &ksk_offset.to_backend_ref(), &mut scratch.borrow()),
    );
    let mut res_offset: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&offset_infos);
    refused(
        "gglwe_keyswitch accepted a key with a row offset",
        "keys with a row offset",
        &mut || {
            module.gglwe_keyswitch(
                &mut res_offset,
                &key_offset,
                &ksk_offset.to_backend_ref(),
                &mut scratch.borrow(),
            )
        },
    );
    let mut res_free: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&free_infos);
    refused(
        "gglwe_keyswitch wrote rows with an offset into a result without one",
        "res k_offset",
        &mut || module.gglwe_keyswitch(&mut res_free, &key_offset, &ksk_free.to_backend_ref(), &mut scratch.borrow()),
    );
}
