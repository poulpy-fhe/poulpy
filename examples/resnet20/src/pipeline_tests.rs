//! A convolution in the last SlotsToCoeffs factor of an S2C-first bootstrap with real slots: slots, the factor with the
//! kernels folded in, the raise, compared with the cleartext.
//!
//! A host client holds the secret, generates the keys and encrypts; the server runs the layer on `Server`, the CPU or,
//! with the `cuda` feature, the GPU.

use poulpy_ckks::{
    CKKSInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{CKKSBootstrappingOps, CKKSCopyOps, CKKSDFTOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncryptOps, CKKSPow2Ops},
    layouts::{BootstrappingContextPrepared, CKKSCiphertextAlloc, CKKSPlaintextAlloc, EvalModRing},
    presets::bootstrapping::{BootstrappingPreset, n16_d35_k685_p19_s2c_ffr_ci},
};
use poulpy_core::{
    api::{GLWESecretPreparedFactory, GLWESecretSampling, TransferInto},
    layouts::{GLWELayout, GLWESecretAlloc, LWEInfos, Rank, SetK},
};
use poulpy_cpu_portable::NTT4x30Portable as Host;
use poulpy_hal::{
    api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CoeffLayout, MergedConv,
    network::{Layouts, ServerBackend as Server},
    reference::conv2d,
    slots_to_blocks,
};

fn arena<B: Backend>(bytes: usize) -> ScratchOwned<B>
where
    ScratchOwned<B>: ScratchOwnedAlloc<B>,
{
    ScratchOwned::<B>::alloc(bytes)
}

/// The index of `i` among `bits`-bit integers read backwards.
fn bit_reverse(i: usize, bits: u32) -> usize {
    i.reverse_bits() >> (usize::BITS - bits)
}

/// Runs a convolution from `input` to `output` in the last S2C factor of `preset`'s bootstrap on random values and
/// weights, and returns the largest error of the bootstrapped slots against the cleartext convolution, zero outside
/// the output samples.
fn convolve_in_bootstrap(preset: &BootstrappingPreset, input: CoeffLayout, output: CoeffLayout) -> f64 {
    let (n, base2k) = (preset.n(), preset.base2k());
    let (client, module) = (Module::<Host>::new(n as u64), Module::<Server>::new(n as u64));
    let boot = preset.bootstrap_layout();
    let ct_layout = |k: usize| GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: Rank(1),
    };
    let in_k = preset.input_layout().k().as_usize();
    let level0_k = preset.log_delta() + preset.log_msg_ratio();
    let keys_layout = preset.keys_layout();
    let scales = preset.scales();

    // Client: the secret and the keys of the bootstrap.
    let (mut source_xs, mut source_xa, mut source_xe) = (Source::new([0; 32]), Source::new([1; 32]), Source::new([2; 32]));
    let mut sk_raw = client.glwe_secret_alloc_from_infos(&boot.glwe_layout);
    client.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_secret_hamming_weight(), &mut source_xs);
    let mut sk = client.glwe_secret_prepared_alloc_from_infos(&boot.glwe_layout);
    client.glwe_secret_prepare(&mut sk, &sk_raw);
    let bytes = keys_layout.generate_tmp_bytes(&client);
    let key_set = preset
        .plan()
        .generate_keys(
            &client,
            &sk_raw,
            keys_layout,
            &mut source_xs,
            &mut source_xe,
            &mut source_xa,
            &mut arena::<Host>(bytes).borrow(),
        )
        .unwrap();

    // Server: the context and the prepared keys.
    let bytes = BootstrappingContextPrepared::<Server, f64>::compile_unprepared_tmp_bytes(&module, preset.plan()).unwrap();
    let unprepared = BootstrappingContextPrepared::<Server, f64>::compile_unprepared(
        &module,
        base2k.into(),
        preset.plan(),
        &mut arena::<Server>(bytes).borrow(),
    )
    .unwrap();
    let bytes = unprepared.prepare_tmp_bytes(&module);
    let context = unprepared.prepare(&module, &mut arena::<Server>(bytes).borrow()).unwrap();
    let bytes = keys_layout.prepare_tmp_bytes(&module);
    let keys = key_set.transfer_and_prepare(&module, &mut arena::<Server>(bytes).borrow());

    // The convolution.
    let (c, m) = (input.channels(), output.channels());
    let mut source = Source::new([3; 32]);
    let tensor: Vec<f64> = (0..c * input.height() * input.width())
        .map(|_| source.next_f64(-1.0, 1.0))
        .collect();
    let weights: Vec<f64> = (0..m * c * 9).map(|_| source.next_f64(-0.5, 0.5)).collect();
    let mut client_scratch = arena::<Host>(
        client
            .ckks_encrypt_sk_tmp_bytes(&boot)
            .max(CKKSEncodingHostOps::<Host, f64>::ckks_encoding_host_tmp_bytes(&client, n / 2))
            .max(client.ckks_decrypt_tmp_bytes(&boot, &boot))
            .max(client.ckks_copy_tmp_bytes(&boot, &boot)),
    );
    let bytes = MergedConv::new_tmp_bytes(&module, &context, base2k).unwrap();
    let conv = MergedConv::new(
        &module,
        &context,
        &input,
        &output,
        &weights,
        3,
        1,
        scales.s2c_log_delta,
        &mut arena::<Server>(bytes).borrow(),
    )
    .unwrap();

    // Client: coefficient `p` of the layout goes in the slot SlotsToCoeffs sends to it.
    let slots = n / 2;
    let log_slots = slots.ilog2();
    let coeffs_in = input.pack(&tensor).unwrap();
    let re: Vec<f64> = (0..slots).map(|j| coeffs_in[bit_reverse(j, log_slots)]).collect();
    let mut pt = client.ckks_plaintext_alloc_from_slots(slots, base2k.into(), in_k.into());
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(preset.log_delta()),
        slots: SlotsKind::Real,
        log_sparsity: 0,
    });
    client
        .ckks_encode_slots_host_into(&mut pt, &re, &vec![0.0; slots], &mut client_scratch.borrow())
        .unwrap();
    let mut client_ct = client.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout(in_k));
    client
        .ckks_encrypt_sk(
            &mut client_ct,
            &pt,
            &sk,
            &mut source_xe,
            &mut source_xa,
            &mut client_scratch.borrow(),
        )
        .unwrap();
    let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout(in_k));
    client_ct.transfer_into(&mut *ct);
    ct.set_meta(client_ct.meta());

    // Server: SlotsToCoeffs up to its last factor, `2·x` through the split decoding matrix, then the convolution, into
    // the raise width.
    let start = std::time::Instant::now();
    let s2c_bytes = keys_layout
        .rotation_key_layouts()
        .iter()
        .map(|atk| {
            module
                .ckks_dft_tmp_bytes(&ct, &ct, context.slots_to_coeffs(), atk)
                .max(conv.apply_tmp_bytes(&module, &ct, &keys_layout.tensor_key, atk))
        })
        .fold(module.ckks_double_tmp_bytes(ct.size()), usize::max);
    let mut blocks = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout(in_k));
    let mut scratch = arena::<Server>(s2c_bytes);
    module.ckks_double_into(&mut blocks, &ct, &mut scratch.borrow()).unwrap();
    slots_to_blocks(&module, &mut blocks, &context, &keys.rotation_keys, &mut scratch.borrow()).unwrap();
    let mut level0 = module.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout(level0_k));
    conv.apply(&module, &mut level0, &blocks, &keys.rotation_keys, &mut scratch.borrow())
        .unwrap();

    // The raise through EvalMod, keeping only the output samples.
    let mut keep = vec![false; n];
    for o in 0..output.channels() {
        for i in 0..output.height() {
            for j in 0..output.width() {
                keep[output.index(o, i, j)] = true;
            }
        }
    }
    let bytes = context.real_raise_mask_tmp_bytes(&module).unwrap();
    let mask = context
        .real_raise_mask(&module, &keep, &mut arena::<Server>(bytes).borrow())
        .unwrap();
    let mut out = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
    let bytes = module.ckks_bootstrap_real_raise_tmp_bytes(&out, &level0, &context, keys_layout);
    module
        .ckks_bootstrap_real_raise(
            &mut out,
            &level0,
            &context,
            Some(&mask),
            &keys,
            &mut arena::<Server>(bytes).borrow(),
        )
        .unwrap();
    module.synchronize();
    println!("server layer: {:.1} ms", start.elapsed().as_secs_f64() * 1e3);

    // Client: decrypt a copy narrowed to 60 bits above the scale, host decoding reading at most 127 bits.
    let mut client_out = client.ckks_ciphertext_alloc_from_glwe_infos(&boot);
    client_out.set_k(out.k());
    out.transfer_into(&mut *client_out);
    client_out.set_meta(out.meta());
    let mut narrow = client.ckks_ciphertext_alloc_from_glwe_infos(&ct_layout(preset.log_delta() + 60));
    client
        .ckks_copy(&mut narrow, &client_out, &mut client_scratch.borrow())
        .unwrap();
    let mut pt_out = client.ckks_plaintext_alloc_from_infos(&narrow);
    client
        .ckks_decrypt(&mut pt_out, &narrow, &sk, &mut client_scratch.borrow())
        .unwrap();
    let (mut re, mut im) = (vec![0.0; slots], vec![0.0; slots]);
    client
        .ckks_decode_slots_host_into(&pt_out, &mut re, &mut im, &mut client_scratch.borrow())
        .unwrap();
    let mut coeffs_out = vec![0.0; n];
    for j in 0..slots {
        let p = bit_reverse(j, log_slots);
        coeffs_out[p] = re[j];
        coeffs_out[slots + p] = im[j];
    }
    let have = output.unpack(&coeffs_out).unwrap();
    let want = conv2d(&tensor, &weights, c, m, input.height(), input.width(), 3, 1);
    let outside = (0..n).filter(|&p| !keep[p]).fold(0.0f64, |e, p| e.max(coeffs_out[p].abs()));
    have.iter().zip(&want).fold(outside, |e, (h, w)| e.max((h - w).abs()))
}

#[test]
#[ignore = "a full logN16 bootstrap; opt in with --ignored, on the GPU with --features cuda"]
fn layer1_convolution_in_the_last_s2c_factor() {
    let preset = n16_d35_k685_p19_s2c_ffr_ci()
        .unwrap()
        .with_slots_to_coeffs(&[(4, 4), (5, 128), (6, 4096)])
        .unwrap()
        .with_eval_mod_ring(EvalModRing::Standard)
        .unwrap();
    let stage1 = Layouts::new(preset.n()).unwrap().stage1;
    let err = convolve_in_bootstrap(&preset, stage1.clone(), stage1);
    println!(
        "layer-1 convolution in the bootstrap, N = 2^16: max error 2^{:.2}",
        err.log2()
    );
    assert!(err < (-10.0f64).exp2(), "max error 2^{:.2}", err.log2());
}
