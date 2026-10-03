//! Runs every CKKS bootstrapping preset on the AVX-512 IFMA backend and prints,
//! per preset, the first four slots before and after bootstrapping and the
//! output precision against the encoded reference vector.
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo run --release -p poulpy-cpu-avx512 \
//!     --features enable-ifma,enable-ckks --example ckks_bootstrapping_presets [preset_name...]
//! ```
//!
//! Preset names select a subset; no argument runs them all.

use std::time::Instant;

use anyhow::Result;
use poulpy_ckks::{
    api::CKKSBootstrappingOps,
    layouts::{BootstrappingContext, CKKSCiphertextOwned},
    prelude::*,
    presets::bootstrapping::all,
    test_suite::helpers::{PrecisionStats, ckks_spec, precision_stats, test_vector_1},
};
use poulpy_core::{
    EncryptionLayout,
    layouts::{GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos, ModuleCoreAlloc},
};
use poulpy_cpu_avx512::NTT3x42Ifma;
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};

type BE = NTT3x42Ifma;

/// Plaintext bits above the scale kept when decrypting, enough to measure precision.
const DECRYPT_LOG_BUDGET: usize = 8;

fn main() -> Result<()> {
    let names: Vec<String> = std::env::args().skip(1).collect();
    for preset in all()?
        .into_iter()
        .filter(|p| names.is_empty() || names.iter().any(|n| n == p.name()))
    {
        let n = preset.n();
        let base2k = preset.base2k();
        let plan = preset.plan();
        let input_layout = preset.input_layout();
        let bootstrap_layout = preset.bootstrap_layout();
        let keys_layout = *preset.keys_layout();
        let module = Module::<BE>::new(n as u64);

        println!("bootstrap_layout: {}", bootstrap_layout.k());
        println!("keys_layout: {}", keys_layout.automorphism_key.k());

        let scratch_size = {
            let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&bootstrap_layout);
            ct.set_meta(bootstrap_layout.meta);
            let eval_mod = plan.eval_mod().coeffs_meta;
            module.ckks_all_ops_with_atk_tmp_bytes(
                &ct,
                &keys_layout.tensor_key,
                &keys_layout.automorphism_key,
                &ckks_spec(n, base2k, eval_mod.log_delta(), eval_mod.log_budget()),
            )
        };
        let context = BootstrappingContext::<BE, f64>::compile(
            &module,
            base2k.into(),
            plan,
            &mut ScratchOwned::<BE>::alloc(scratch_size).borrow(),
        )?;
        let boot_scratch = module.ckks_bootstrap_tmp_bytes(&bootstrap_layout, &input_layout, &context, &keys_layout);
        let mut scratch = ScratchOwned::<BE>::alloc(scratch_size.max(boot_scratch));

        let mut sk_raw = module.glwe_secret_alloc_from_infos(&bootstrap_layout.glwe_layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut Source::new([0; 32]));
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&bootstrap_layout.glwe_layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);

        print!("generate_keys:");
        let now = Instant::now();
        let keys = context
            .generate_keys(
                &module,
                &sk_raw,
                &keys_layout,
                &mut Source::new([7; 32]),
                &mut Source::new([2; 32]),
                &mut Source::new([1; 32]),
                &mut scratch.borrow(),
            )?
            .prepare(&module, &mut scratch.borrow());
        println!(" {:?}", now.elapsed());

        let (want_re, want_im) = test_vector_1::<f64>(n / 2);
        let mut pt = module.ckks_pt_vec_alloc(base2k.into(), input_layout.k());
        pt.set_meta(input_layout.meta());
        module.ckks_encode_reim_into(&mut pt, &want_re, &want_im, &mut scratch.borrow())?;
        let mut input = module.ckks_ciphertext_alloc_from_glwe_infos(&input_layout);
        module.ckks_encrypt_sk(
            &mut input,
            &pt,
            &sk,
            &EncryptionLayout::new_from_default_sigma(input_layout.glwe_layout)?,
            &mut Source::new([4; 32]),
            &mut Source::new([3; 32]),
            &mut scratch.borrow(),
        )?;

        let mut output = module.ckks_ciphertext_alloc_from_glwe_infos(&bootstrap_layout);
        output.set_k(preset.bootstrap_k().into());

        print!("ckks_bootstrap:");
        let now = Instant::now();
        module.ckks_bootstrap(&mut output, &input, &context, &keys, &mut scratch.borrow())?;
        println!(" {:?}", now.elapsed());

        let (in_re, in_im) = decrypt(&module, &input, &sk, &mut scratch)?;
        let (out_re, out_im) = decrypt(&module, &output, &sk, &mut scratch)?;

        println!(
            "== {} (N=2^{}, base2k={}, k: {} -> {}, advertised {} bits)",
            preset.name(),
            preset.log_n(),
            base2k,
            input.k().as_usize(),
            output.k().as_usize(),
            preset.log2_precision()
        );
        for i in 0..4 {
            println!(
                "slot {i}: want {:+.8} {:+.8}i | before {:+.8} {:+.8}i | after {:+.8} {:+.8}i",
                want_re[i], want_im[i], in_re[i], in_im[i], out_re[i], out_im[i]
            );
        }
        let log_delta = preset.log_delta();
        print_stats("before re", &precision_stats(&in_re, &want_re, log_delta));
        print_stats("before im", &precision_stats(&in_im, &want_im, log_delta));
        print_stats("after  re", &precision_stats(&out_re, &want_re, log_delta));
        print_stats("after  im", &precision_stats(&out_im, &want_im, log_delta));
    }
    Ok(())
}

fn decrypt(
    module: &Module<BE>,
    ct: &CKKSCiphertextOwned<BE>,
    sk: &GLWESecretPrepared<AlignedBuf, BE>,
    scratch: &mut ScratchOwned<BE>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let log_budget = ct.log_budget().min(DECRYPT_LOG_BUDGET);
    let mut pt = module.ckks_pt_vec_alloc(ct.base2k(), (ct.log_delta() + log_budget).into());
    pt.set_meta(CKKSMeta {
        log_sparsity: 0,
        log_delta: ct.log_delta(),
        slots: SlotsKind::Complex,
    });
    module.ckks_decrypt(&mut pt, ct, sk, &mut scratch.borrow())?;
    let m = ct.n().as_usize() / 2;
    let (mut re, mut im) = (vec![0.0; m], vec![0.0; m]);
    module.ckks_decode_reim_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
    Ok((re, im))
}

fn print_stats(label: &str, s: &PrecisionStats) {
    println!(
        "{label}: min {:.2}b avg {:.2}b max {:.2}b | worst slot {} err {:.3e}",
        s.min_log2_prec, s.avg_log2_prec, s.max_log2_prec, s.worst_idx, s.worst_err
    );
}
