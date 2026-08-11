//! Minimal end-to-end CKKS walkthrough: the shortest path from keys to a
//! verified homomorphic result.
//!
//! ```bash
//! cargo run -p poulpy-cpu-ref --features enable-ckks --example ckks_hello
//! ```
//!
//! Computes `z = (a + b) * b` on encrypted slot vectors:
//!
//! 1. keygen (secret key + tensor/relinearization key)
//! 2. encode + encrypt `a` and `b`
//! 3. homomorphic add, multiply (with relinearization), and rescale
//! 4. decrypt + decode, then compare against the cleartext result
//!
//! For a deeper example (Chebyshev polynomial evaluation in BSGS form), see
//! `ckks_poly2.rs`.

use anyhow::Result;
use poulpy_ckks::prelude::*;
use poulpy_core::layouts::GLWESecretSampling;
use poulpy_core::{
    EncryptionLayout, GLWETensorKeyEncryptSk,
    layouts::{
        GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, ModuleCoreAlloc, Rank, prepared::GLWESecretPreparedFactory,
    },
};
use poulpy_cpu_ref::NTT4x30Ref;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};

type BackendImpl = NTT4x30Ref;

const N: usize = 1024;
/// Complex slot count.
const M: usize = N / 2;
const BASE2K: usize = 52;
/// Ciphertext torus width: 3 limbs of 52 bits.
const CT_K: usize = 156;
/// Encoding scale: values are encoded at `2^45` precision.
const LOG_DELTA: usize = 45;
/// Secret-key Hamming weight.
const HW: usize = 192;

fn glwe_layout() -> EncryptionLayout<GLWELayout> {
    EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: CT_K.into(),
        rank: Rank(1),
    })
    .unwrap()
}

fn tsk_layout() -> EncryptionLayout<GLWETensorKeyLayout> {
    EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k_aux: BASE2K.into(),
        rank: Rank(1),
        dsize: 1usize.into(),
        dnum: CT_K.div_ceil(BASE2K).into(),
    })
    .unwrap()
}

fn main() -> Result<()> {
    // ── 1. setup: module, secret key, relinearization key, scratch ──────────
    let module = Module::<BackendImpl>::new(N as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_layout());
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, HW, &mut source_xs);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout());
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    // One arena sized for every op this example runs.
    let ct_infos = module.ckks_ciphertext_alloc_from_glwe_infos(&glwe_layout());
    let meta = CKKSMeta {
        log_delta: LOG_DELTA,
        log_sparsity: 0,
    };
    let mut scratch = ScratchOwned::<BackendImpl>::alloc(module.ckks_all_ops_tmp_bytes(&ct_infos, &tsk_layout(), &ct_infos));

    let mut tsk = module.glwe_tensor_key_alloc_from_infos(&tsk_layout());
    module.glwe_tensor_key_encrypt_sk(
        &mut tsk,
        &sk_raw,
        &tsk_layout(),
        &mut source_xa,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout());
    module.prepare_tensor_key(&mut tsk_prepared, &tsk, &mut scratch.borrow());

    // ── 2. encode + encrypt two slot vectors ────────────────────────────────
    let a_re: Vec<f64> = (0..M).map(|i| i as f64 / M as f64 - 0.5).collect();
    let b_re: Vec<f64> = (0..M).map(|i| 0.25 + 0.5 * (i as f64 / M as f64)).collect();
    let zeros = vec![0.0f64; M];

    let mut encrypt = |re: &[f64]| -> Result<CKKSCiphertext<Vec<u8>, i64>> {
        let mut pt = module.ckks_pt_vec_alloc(BASE2K.into(), CT_K.into());
        pt.set_meta(meta);
        module.ckks_encode_reim_into(&mut pt, re, &zeros, &mut scratch.borrow())?;
        let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
        module.ckks_encrypt_sk(
            &mut ct,
            &pt,
            &sk,
            &glwe_layout(),
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        )?;
        Ok(ct)
    };
    let mut ct = encrypt(&a_re)?;
    let ct_b = encrypt(&b_re)?;
    println!("fresh ct: log_delta={} log_budget={}", ct.log_delta(), ct.log_budget());

    // ── 3. evaluate z = (a + b) * b, then rescale ───────────────────────────
    module.ckks_add_assign(&mut ct, &ct_b, &mut scratch.borrow())?;
    module.ckks_mul_assign(&mut ct, &ct_b, &tsk_prepared, &mut scratch.borrow())?;
    println!("after mul: log_delta={} log_budget={}", ct.log_delta(), ct.log_budget());
    // Trade 5 bits of headroom for 5 extra fraction bits under the scale — the
    // closest analogue of an RNS "rescale" in this base-2^base2k model.
    module.ckks_div_pow2_assign(&mut ct, 5)?;

    // ── 4. decrypt + decode + verify ────────────────────────────────────────
    let mut pt_out = module.ckks_plaintext_alloc_from_infos(&ct);
    module.ckks_decrypt(&mut pt_out, &ct, &sk, &mut scratch.borrow())?;
    let (mut have_re, mut have_im) = (vec![0.0f64; M], vec![0.0f64; M]);
    module.ckks_decode_reim_into(&pt_out, &mut have_re, &mut have_im, &mut scratch.borrow())?;

    // `div_pow2` divides the encrypted value by 2^5; undo it in the reference.
    let want: Vec<f64> = a_re.iter().zip(&b_re).map(|(a, b)| (a + b) * b / 32.0).collect();
    let max_err = want.iter().zip(&have_re).map(|(w, h)| (w - h).abs()).fold(0.0f64, f64::max);
    println!("max |want - have| = {max_err:.3e}");
    assert!(max_err < 1e-9, "homomorphic result diverges: {max_err:.3e}");
    println!("ok: z = (a + b) * b verified on {M} slots");
    Ok(())
}
