//! The operations on a small ciphertext: one raised to a wider precision,
//! held as an integer polynomial at the bottom of the Torus.
//!
//! A plaintext product followed by a division keeps it at the same width; an
//! automorphism needs no key; and its mask times row 0 of a one-digit
//! switching key is its key switch ([`glwe_keyswitch_small_accumulate`]).

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism, VecZnxBigNormalize, VecZnxCopy,
        VecZnxDftAlloc, VecZnxFillUniformSource, VecZnxIdftApply, VecZnxRshAdd,
    },
    layouts::{CnvPVecRToBackendRef, Module, ScratchOwned, VecZnxDftToBackendMut, ZnxView, ZnxViewMut, galois_element},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::{
    GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEMulPlain, GLWENoise, GLWEShift,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos, ModuleCoreAlloc,
        prepared::GLWESecretPrepared,
    },
    reference::keyswitching::small::{
        glwe_automorphism_keyless, glwe_keyswitch_small_acc_size, glwe_keyswitch_small_accumulate, glwe_keyswitch_small_finish,
        glwe_secret_automorphism, glwe_small_switching_key_encrypt_sk, glwe_small_switching_key_prepare,
        glwe_small_switching_key_prepared_alloc,
    },
    test_suite::noise::glwe_noise_checked,
};

pub fn test_glwe_small_switch<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSource<BE>
        + VecZnxAutomorphism<BE>
        + VecZnxCopy<BE>
        + VecZnxRshAdd<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApply<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + crate::GLWENormalize<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWEMulPlain<BE>
        + GLWEShift<BE>
        + GLWECopy<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    // A two-limb ciphertext raised into four limbs, with one limb of guard below its
    // last bit; the switch lands it on a seven-limb destination.
    let (k_fresh, small_size, wide_size): (usize, usize, usize) = (2 * base2k, 4, 7);
    let guard: usize = base2k;
    let layout = |k: usize| GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1u32.into(),
    };
    let small_layout = layout(small_size * base2k);
    let wide_layout = layout(wide_size * base2k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 24);

    let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(1u32.into());
    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
    let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(1u32.into());
    module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);
    let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(1u32.into());
    module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
    let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(1u32.into());
    module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

    // A fresh narrow ciphertext, raised: copied to the small width, then shifted so that
    // its last bit sits `guard` bits above the bottom.
    let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&layout(k_fresh));
    module.vec_znx_fill_uniform_source(
        base2k,
        k_fresh,
        &mut vec_znx_backend_mut::<BE>(&mut pt.data),
        0,
        &mut source_xa,
    );
    let mut fresh: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout(k_fresh));
    module.glwe_encrypt_sk(
        &mut fresh,
        &pt,
        &sk_in_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let mut small: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&small_layout);
    module.glwe_copy(&mut small, &fresh, &mut scratch.borrow());
    module.glwe_rsh(small_size * base2k - k_fresh - guard, &mut small, &mut scratch.borrow());
    let mut small_pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&small_layout);
    module.glwe_decrypt(&small, &mut small_pt, &sk_in_prepared, &mut scratch.borrow());

    // Level-conserving product: by `2^shift * X^3` then a division by `2^shift`, at the
    // same width. It must be the ciphertext of the same integers times `X^3`.
    let shift: usize = base2k + 3;
    let mut monomial: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&layout(2 * base2k));
    monomial.data.at_mut(0, 0)[3] = 1 << 3;
    let mut product: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&small_layout);
    // The offset of a plain product is the plaintext's width; a smaller one divides.
    module.glwe_mul_plain(2 * base2k - shift, &mut product, &small, &monomial, &mut scratch.borrow());

    // A slot rotation, with no key: the ciphertext is then under the rotated secret.
    let p: i64 = galois_element(5, 2 * n as i64);
    let mut rotated: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&small_layout);
    glwe_automorphism_keyless(module, &mut rotated, p, &product);
    let mut sk_rotated: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(1u32.into());
    glwe_secret_automorphism(module, &mut sk_rotated, p, &sk_in);

    // The key: the rotated secret at the destination's last bit. Its noise is scaled by the
    // mask, an integer of three limbs here, so it keeps four limbs below that bit.
    let mut key: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout((wide_size + 4) * base2k));
    glwe_small_switching_key_encrypt_sk(
        module,
        &mut key,
        ((wide_size * base2k) as u32).into(),
        &sk_rotated,
        &sk_out_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    // The switch: the key prepared once, the masked part accumulated in the transform
    // domain, then brought back with the body.
    let mut prepared = glwe_small_switching_key_prepared_alloc(module, &key);
    glwe_small_switching_key_prepare(module, &mut prepared, &key, &mut scratch.borrow());
    let mut acc = module.vec_znx_dft_alloc(n, 2, glwe_keyswitch_small_acc_size(key.size()));
    let mut switched: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&wide_layout);
    glwe_keyswitch_small_accumulate(
        module,
        &mut acc.to_backend_mut(),
        true,
        &rotated,
        &prepared.to_backend_ref(),
        &mut scratch.borrow(),
    );
    glwe_keyswitch_small_finish(
        module,
        &mut switched,
        &mut acc.to_backend_mut(),
        &rotated,
        &mut scratch.borrow(),
    );

    // Expected: the raised integers times `X^3`, rotated, on the last limbs of the destination.
    let mut want: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&wide_layout);
    let order: usize = 2 * n;
    let p_pos: usize = p.rem_euclid(order as i64) as usize;
    for limb in 0..small_size {
        let src = small_pt.data.at(0, limb);
        let dst = want.data.at_mut(0, wide_size - small_size + limb);
        for (i, &value) in src.iter().enumerate() {
            // times X^3, then X -> X^p
            let e = ((i + 3) * p_pos) % order;
            if e < n { dst[e] = value } else { dst[e - n] = -value }
        }
    }
    assert_eq!(want.data.size(), wide_size);

    // In units of the last bit of the destination: rounding, and next to nothing from the key.
    let noise = glwe_noise_checked(module, &switched, &want, &sk_out_prepared, &mut scratch.borrow())
        .std()
        .log2()
        + (wide_size * base2k) as f64;
    println!("DBG glwe_small_switch noise={noise:.2}");
    assert!(noise <= 3.0, "small switch noise: {noise} bits above the last bit");
}
