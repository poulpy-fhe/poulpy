//! The encrypted ResNet-20 on the S2C-first bootstrap of `n16_d35_k685_p19_s2c_ffr_ci`, with real slots. Stages 2 and 3
//! fit half the slots: their bootstraps run on `Z[X^2]`, with transforms of one layer less.
//!
//! An activation lives in the real slots between two layers. A layer runs SlotsToCoeffs up to its last factor, then that
//! factor with the layer's convolution folded in, the same on the residual, the bias, the raise, masked to the layer's
//! output samples, and the ReLU on the slots. The client encrypts the image in the real slots, as an activation, and
//! decrypts the logits at level 0 from the head's coefficients.
//!
//! Every activation is stored normalized: a ReLU of divisor `D` computes `y + y·sign(y)` on `y = x / D`, so the
//! stored value is the activation times `2 / D`, and the next convolution's weights carry `D / 2` over its own
//! divisor.

use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSBootstrappingOps, CKKSCopyOps, CKKSDFTOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncryptOps,
        CKKSPow2Ops,
    },
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeySet, BootstrappingKeys, BootstrappingKeysPrepared, CKKSCiphertextAlloc,
        CKKSCiphertextOwned, CKKSPlaintextAlloc, CKKSPlaintextOwned, EvalModRing, RealRaiseMaskPrepared,
    },
    presets::bootstrapping::{BootstrappingPreset, PresetScales, n16_d35_k685_p19_s2c_ffr_ci},
};
use poulpy_core::{
    api::{GLWESecretPreparedFactory, GLWESecretSampling, TransferInto},
    layouts::{BackendGLWESecret, BackendGLWESecretPrepared, GLWELayout, GLWESecretAlloc, LWEInfos, Rank, SetK},
};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::{
    api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CoeffLayout, MergedConv, Relu,
    model::{Conv, ResNet20, log_divisor},
    slots_to_blocks,
};

/// The client's backend, which holds the secret.
pub type Host = NTT4x30Portable;
#[cfg(not(feature = "enable-cuda"))]
pub type ServerBackend = NTT4x30Portable;
#[cfg(feature = "enable-cuda")]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Cuda;

const LOG_DELTA_W: usize = 20;

/// The transforms of stages 2 and 3, on 14 layers.
const SUBRING_C2S: &[(usize, usize)] = &[(3, 1), (3, 1024), (3, 128), (3, 16), (2, 2)];
const SUBRING_S2C: &[(usize, usize)] = &[(4, 4), (4, 64), (6, 2048)];

/// Levels of the ReLU: four per sign stage and the product by its input.
const RELU_LEVELS: usize = 9;

/// The narrowest scales that keep the accuracy of the cleartext polynomial network, tuned on the wall clock.
const SCALES: PresetScales = PresetScales {
    log_delta: 28,
    log_msg_ratio: 6,
    c2s_guard_bits: 4,
    c2s_log_delta: 34,
    s2c_log_delta: 16,
    eval_mod_log_delta: 41,
    eval_mod_coeffs_log_delta: 30,
};

/// The integer part of a raised coefficient, a sum of 33 uniforms, reaches 14 with probability `2^-69`.
const EVAL_MOD_INTERVAL: usize = 14;
const EVAL_MOD_DEGREE: usize = 26;
/// One level fewer needs a degree that costs more time and precision than the level saves.
const EVAL_MOD_SQUARINGS: usize = 3;

/// A narrower key outruns the extra row up to this cap on a gadget digit, in limbs.
const MAX_DIGIT_LIMBS: usize = 8;

fn arena<B: Backend>(bytes: usize) -> ScratchOwned<B>
where
    ScratchOwned<B>: ScratchOwnedAlloc<B>,
{
    ScratchOwned::<B>::alloc(bytes)
}

/// The bootstrap and the widths around the convolutions.
#[derive(Clone)]
pub struct Params {
    /// The bootstrap of each gap, by its `log2`: the full ring, then `Z[X^2]`.
    pub presets: [BootstrappingPreset; 2],
    /// Scale of the last SlotsToCoeffs factor, with the kernels folded in: the bits it spends.
    pub log_delta_w: usize,
    /// Width of the SlotsToCoeffs input.
    pub in_k: usize,
    /// Width of a layer's output, the raise input.
    pub level0_k: usize,
}

impl Params {
    pub fn new() -> Result<Self> {
        let preset = n16_d35_k685_p19_s2c_ffr_ci()?
            .with_slots_to_coeffs(&[(4, 4), (5, 128), (6, 4096)])?
            .with_scales(SCALES)?
            .with_eval_mod_ring(EvalModRing::Standard)?
            .with_eval_mod_polynomial(EVAL_MOD_INTERVAL, EVAL_MOD_DEGREE, EVAL_MOD_SQUARINGS)?
            .with_input_headroom(LOG_DELTA_W.saturating_sub(SCALES.s2c_log_delta))?;
        // What a layer spends past the bootstrap: the ReLU's nine levels and the three factors of SlotsToCoeffs, the last
        // one holding the convolution.
        let preset = preset.with_output_k(preset.input_k() + RELU_LEVELS * preset.log_delta())?;
        let preset = preset.with_dsizes(MAX_DIGIT_LIMBS, preset.dense_to_sparse_dsize())?;
        let subring = preset.with_log_gap(1, SUBRING_C2S, SUBRING_S2C)?;
        ensure!(
            (subring.input_k(), subring.bootstrap_k()) == (preset.input_k(), preset.bootstrap_k()),
            "the two bootstraps must share their widths"
        );
        let in_k = preset.input_k();
        let level0_k = preset.log_delta() + preset.log_msg_ratio();
        Ok(Self {
            presets: [preset, subring],
            log_delta_w: LOG_DELTA_W,
            in_k,
            level0_k,
        })
    }

    pub fn n(&self) -> usize {
        self.presets[0].n()
    }

    pub fn log_delta(&self) -> usize {
        self.presets[0].log_delta()
    }

    /// A ciphertext of degree `N` and width `k`.
    pub fn layout(&self, k: usize) -> GLWELayout {
        GLWELayout {
            n: self.n().into(),
            base2k: self.presets[0].base2k().into(),
            k: k.into(),
            rank: Rank(1),
        }
    }

    fn slots_meta(&self) -> GLWEPlaintextMeta {
        GLWEPlaintextMeta {
            scale: Scale::Log(self.log_delta()),
            slots: SlotsKind::Real,
            log_sparsity: 0,
        }
    }
}

/// Where each activation of the network sits in the coefficients.
pub struct Layouts {
    /// The image, three channels of [`Self::stage1`].
    pub image: CoeffLayout,
    /// Stage 1, 16 channels of 32 x 32, one per coset.
    pub stage1: CoeffLayout,
    /// Stage 2, 32 channels of 16 x 16, as a layer reads them and as a layer writes them.
    pub in2: CoeffLayout,
    pub out2: CoeffLayout,
    /// Stage 3, 64 channels of 8 x 8, as a layer reads them and as a layer writes them.
    pub in3: CoeffLayout,
    pub out3: CoeffLayout,
    /// The ten logits, at the even coefficients 0 to 18.
    pub logits: CoeffLayout,
}

impl Layouts {
    pub fn new(n: usize) -> Result<Self> {
        const ROW_STRIDE: usize = 36;
        const SIDE_BY_SIDE: usize = 18;
        // Two grids side by side on the even cosets, the second one half up at a layer's output: the terms of a
        // channel read through the wrong grid land off the samples, and the real slots bring it back at its input.
        let grid = |side: usize, step: usize, cosets: usize, output: bool| {
            let half = if output { n / 2 } else { 0 };
            let bases = (0..2 * cosets)
                .map(|q| 2 * (q % cosets) + (q / cosets) * (step * SIDE_BY_SIDE + half))
                .collect();
            CoeffLayout::new(n, side, side, ROW_STRIDE, step, bases)
        };
        Ok(Self {
            image: CoeffLayout::new(n, 32, 32, ROW_STRIDE, 16, (0..3).collect())?,
            stage1: CoeffLayout::new(n, 32, 32, ROW_STRIDE, 16, (0..16).collect())?,
            in2: grid(16, 32, 16, false)?,
            out2: grid(16, 32, 16, true)?,
            in3: grid(8, 64, 32, false)?,
            out3: grid(8, 64, 32, true)?,
            logits: CoeffLayout::new(n, 1, 1, ROW_STRIDE, 512, (0..10).map(|c| 2 * c).collect())?,
        })
    }
}

/// Slot `s` of `slots` holds the coefficient of this index.
fn bit_reverse(s: usize, slots: usize) -> usize {
    s.reverse_bits() >> (usize::BITS - slots.trailing_zeros())
}

/// The samples of `layout`.
fn samples(layout: &CoeffLayout) -> Vec<bool> {
    let mut keep = vec![false; layout.n()];
    for q in 0..layout.channels() {
        for i in 0..layout.height() {
            for j in 0..layout.width() {
                keep[layout.index(q, i, j)] = true;
            }
        }
    }
    keep
}

/// The client: the secret, the keys, the image in, the logits out.
pub struct Client {
    module: Module<Host>,
    sk_raw: BackendGLWESecret<Host>,
    sk: BackendGLWESecretPrepared<Host>,
    scratch: ScratchOwned<Host>,
    source_xs: Source,
    source_xe: Source,
    source_xa: Source,
}

impl Client {
    /// The client of the secret and the randomness of `seed`, one child seed per role: whoever holds `seed` decrypts.
    pub fn new(params: &Params, seed: [u8; 32]) -> Self {
        let mut root = Source::new(seed);
        let preset = &params.presets[0];
        let n = preset.n();
        let module = Module::<Host>::new(n as u64);
        let boot = preset.bootstrap_layout();
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&boot.glwe_layout);
        module.glwe_secret_fill_ternary_hw(
            &mut sk_raw,
            preset.dense_secret_hamming_weight(),
            &mut Source::new(root.new_seed()),
        );
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&boot.glwe_layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let scratch = arena::<Host>(
            module
                .ckks_encrypt_sk_tmp_bytes(&boot)
                .max(CKKSEncodingHostOps::<Host, f64>::ckks_encoding_host_tmp_bytes(&module, n))
                .max(module.ckks_decrypt_tmp_bytes(&boot, &boot))
                .max(module.ckks_copy_tmp_bytes(&boot, &boot)),
        );
        Self {
            module,
            sk_raw,
            sk,
            scratch,
            source_xs: Source::new(root.new_seed()),
            source_xe: Source::new(root.new_seed()),
            source_xa: Source::new(root.new_seed()),
        }
    }

    /// The bootstrapping keys of each gap.
    pub fn keys(&mut self, params: &Params) -> Result<Vec<BootstrappingKeySet<<Host as Backend>::OwnedBuf, i64>>> {
        params
            .presets
            .iter()
            .map(|preset| {
                let bytes = preset.keys_layout().generate_tmp_bytes(&self.module);
                preset.plan().generate_keys(
                    &self.module,
                    &self.sk_raw,
                    preset.keys_layout(),
                    &mut self.source_xs,
                    &mut self.source_xe,
                    &mut self.source_xa,
                    &mut arena::<Host>(bytes).borrow(),
                )
            })
            .collect()
    }

    /// `image`, `[3][32][32]`, in the real slots SlotsToCoeffs sends to the coefficients of `layout`.
    pub fn encrypt_image(&mut self, params: &Params, layout: &CoeffLayout, image: &[f64]) -> Result<CKKSCiphertextOwned<Host>> {
        let coeffs = layout.pack(image)?;
        let slots = layout.n() / 2;
        ensure!(
            coeffs[slots..].iter().all(|&c| c == 0.0),
            "real slots carry the lower half of the coefficients"
        );
        let re: Vec<f64> = (0..slots).map(|s| coeffs[bit_reverse(s, slots)]).collect();
        let mut pt = self
            .module
            .ckks_plaintext_alloc_from_slots(slots, params.presets[0].base2k().into(), params.in_k.into());
        pt.set_meta(params.slots_meta());
        self.module
            .ckks_encode_slots_host_into(&mut pt, &re, &vec![0.0; slots], &mut self.scratch.borrow())?;
        let mut ct = self.module.ckks_ciphertext_alloc_from_glwe_infos(&params.layout(params.in_k));
        self.module.ckks_encrypt_sk(
            &mut ct,
            &pt,
            &self.sk,
            &mut self.source_xe,
            &mut self.source_xa,
            &mut self.scratch.borrow(),
        )?;
        Ok(ct)
    }

    /// The tensor in the slots of `ct` that SlotsToCoeffs sends to the coefficients of `layout`.
    pub fn decrypt_slots(&mut self, params: &Params, ct: &CKKSCiphertextOwned<Host>, layout: &CoeffLayout) -> Result<Vec<f64>> {
        // The host decoder reads at most 127 bits: 60 above the scale.
        let mut narrow = self
            .module
            .ckks_ciphertext_alloc_from_glwe_infos(&params.layout(ct.log_delta() + 60));
        self.module.ckks_copy(&mut narrow, ct, &mut self.scratch.borrow())?;
        let mut pt = self.module.ckks_plaintext_alloc_from_infos(&narrow);
        pt.set_meta(narrow.meta());
        self.module
            .ckks_decrypt(&mut pt, &narrow, &self.sk, &mut self.scratch.borrow())?;
        let half = layout.n() / 2;
        let (mut re, mut im) = (vec![0.0; half], vec![0.0; half]);
        self.module
            .ckks_decode_slots_host_into(&pt, &mut re, &mut im, &mut self.scratch.borrow())?;
        // The slots repeat with the period of the layout's subring.
        let gap = 1 << layout.log_gap();
        let slots = half / gap;
        let mut coeffs = vec![0.0; layout.n()];
        for (j, &v) in re[..slots].iter().enumerate() {
            coeffs[gap * bit_reverse(j, slots)] = v;
        }
        layout.unpack(&coeffs)
    }

    /// The tensor in the coefficients of `ct`, laid out as `layout`.
    pub fn decrypt_coeffs(&mut self, ct: &CKKSCiphertextOwned<Host>, layout: &CoeffLayout) -> Result<Vec<f64>> {
        let mut pt = self.module.ckks_plaintext_alloc_from_infos(ct);
        pt.set_meta(ct.meta());
        self.module.ckks_decrypt(&mut pt, ct, &self.sk, &mut self.scratch.borrow())?;
        let mut coeffs = vec![0.0; layout.n()];
        self.module
            .ckks_decode_coeffs_host_into(&pt, &mut coeffs, &mut self.scratch.borrow())?;
        layout.unpack(&coeffs)
    }
}

/// One layer at level 0: `conv(x) + shortcut(skip) + bias`, then, but for the head, the masked raise.
struct Layer {
    /// The ReLU that follows, `stem`, `layerS.B.r1` or `layerS.B.r2`, or `head`.
    name: String,
    conv: MergedConv,
    shortcut: Option<MergedConv>,
    bias: CKKSPlaintextOwned<ServerBackend>,
    mask: Option<RealRaiseMaskPrepared<ServerBackend>>,
    /// `log2` of the gap of the bootstrap that follows.
    raise_gap: usize,
}

/// Time spent per stage of an inference, the server synchronized at each boundary.
#[derive(Default, Debug, Clone, Copy)]
pub struct Timings {
    pub s2c: Duration,
    pub conv: Duration,
    pub raise: Duration,
    pub relu: Duration,
}

impl Timings {
    pub fn total(&self) -> Duration {
        self.s2c + self.conv + self.raise + self.relu
    }
}

/// The server's ciphertexts, allocated once: a recording binds them.
struct Buffers {
    /// The image, in real slots.
    image: CKKSCiphertextOwned<ServerBackend>,
    /// The input of a block, then of the head, short of the last SlotsToCoeffs factor, which also reads it for the
    /// residual, allocated at the SlotsToCoeffs input width.
    x_blocks: CKKSCiphertextOwned<ServerBackend>,
    /// The middle activation of a block, in the same state.
    h_blocks: CKKSCiphertextOwned<ServerBackend>,
    level0: CKKSCiphertextOwned<ServerBackend>,
    skip: CKKSCiphertextOwned<ServerBackend>,
    reduced: CKKSCiphertextOwned<ServerBackend>,
    activated: CKKSCiphertextOwned<ServerBackend>,
    narrow: CKKSCiphertextOwned<ServerBackend>,
}

/// What an inference writes.
struct State {
    scratch: ScratchOwned<ServerBackend>,
    bufs: Buffers,
}

/// The server: the compiled bootstrap, the evaluation keys and the encoded network.
pub struct Server {
    params: Params,
    module: Module<ServerBackend>,
    /// The bootstrap of each gap, by its `log2`.
    contexts: Vec<BootstrappingContextPrepared<ServerBackend, f64>>,
    keys: Vec<BootstrappingKeysPrepared<<ServerBackend as Backend>::OwnedBuf, ServerBackend>>,
    stem: Layer,
    blocks: Vec<[Layer; 2]>,
    head: Layer,
    relu: Relu<ServerBackend>,
    /// Allocates the copies handed back to the client.
    host: Module<Host>,
    state: RefCell<State>,
    stage_sync: bool,
}

/// The weights of `conv` times `factor`, and its bias over `divisor`.
fn scaled(conv: &Conv, factor: f64, divisor: f64) -> (Vec<f64>, Vec<f64>) {
    (
        conv.weights.iter().map(|w| w * factor).collect(),
        conv.bias.iter().map(|b| b / divisor).collect(),
    )
}

/// The `1 x 1` shortcut of `channels` input channels into `out` channels: `factor` times input `q` into output
/// `q + offset`.
fn shortcut_weights(channels: usize, out: usize, offset: usize, factor: f64) -> Vec<f64> {
    let mut w = vec![0.0; out * channels];
    for q in 0..channels {
        w[(q + offset) * channels + q] = factor;
    }
    w
}

/// Where [`Server::infer_traced`] probes a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The ReLU's input, in the coefficients at level 0.
    Input,
    /// The ReLU's output, in the slots.
    Activation,
}

/// A probe of [`Server::infer_traced`]: the ReLU's name, the stage, the copy and its layout.
pub type Probe<'a> = &'a mut dyn FnMut(&str, Stage, &CKKSCiphertextOwned<Host>, &CoeffLayout);

/// The layout `ct` was allocated with, which a transfer copies whatever its width.
fn allocated<C: CKKSCtBounds>(ct: &C) -> GLWELayout {
    GLWELayout {
        n: ct.n(),
        base2k: ct.base2k(),
        k: (ct.max_size() * ct.base2k().as_usize()).into(),
        rank: ct.rank(),
    }
}

/// A copy of `ct` on the client.
fn to_host(module: &Module<Host>, ct: &CKKSCiphertextOwned<ServerBackend>) -> CKKSCiphertextOwned<Host> {
    let mut res = module.ckks_ciphertext_alloc_from_glwe_infos(&allocated(ct));
    res.set_k(ct.k());
    ct.transfer_into(&mut *res);
    res.set_meta(ct.meta());
    res
}

impl Server {
    /// Compiles the bootstrap, prepares the client's keys and encodes `model` laid out as `layouts`.
    pub fn new(
        params: Params,
        model: &ResNet20,
        layouts: &Layouts,
        key_sets: &[BootstrappingKeySet<<Host as Backend>::OwnedBuf, i64>],
    ) -> Result<Self> {
        let preset = &params.presets[0];
        let (n, base2k, log_delta) = (preset.n(), preset.base2k(), preset.log_delta());
        let module = Module::<ServerBackend>::new(n as u64);
        let contexts = params
            .presets
            .iter()
            .map(|preset| {
                let bytes =
                    BootstrappingContextPrepared::<ServerBackend, f64>::compile_unprepared_tmp_bytes(&module, preset.plan())?;
                let unprepared = BootstrappingContextPrepared::<ServerBackend, f64>::compile_unprepared(
                    &module,
                    base2k.into(),
                    preset.plan(),
                    &mut arena::<ServerBackend>(bytes).borrow(),
                )?;
                let bytes = unprepared.prepare_tmp_bytes(&module);
                unprepared.prepare(&module, &mut arena::<ServerBackend>(bytes).borrow())
            })
            .collect::<Result<Vec<_>>>()?;
        let keys: Vec<_> = key_sets
            .iter()
            .zip(&params.presets)
            .map(|(key_set, preset)| {
                let bytes = preset.keys_layout().prepare_tmp_bytes(&module);
                key_set.transfer_and_prepare(&module, &mut arena::<ServerBackend>(bytes).borrow())
            })
            .collect();

        let conv_bytes = contexts
            .iter()
            .map(|context| MergedConv::new_tmp_bytes(&module, context, base2k))
            .try_fold(0, |max, bytes| Ok::<_, anyhow::Error>(max.max(bytes?)))?;
        let mut enc = arena::<ServerBackend>(conv_bytes);
        let mut conv = |log_gap: usize, input: &CoeffLayout, output: &CoeffLayout, weights: &[f64], kernel, stride| {
            MergedConv::new(
                &module,
                &contexts[log_gap],
                input,
                output,
                weights,
                kernel,
                stride,
                params.log_delta_w,
                &mut enc.borrow(),
            )
        };
        let mask_bytes = contexts
            .iter()
            .map(|context| context.real_raise_mask_tmp_bytes(&module))
            .try_fold(0, |max, bytes| Ok::<_, anyhow::Error>(max.max(bytes?)))?;
        let mut aux = arena::<ServerBackend>(
            CKKSEncodingHostOps::<ServerBackend, f64>::ckks_encoding_host_tmp_bytes(&module, n).max(mask_bytes),
        );
        let mut finish = |name: String, conv: MergedConv, shortcut, bias: &[f64], raise_gap: Option<usize>| -> Result<Layer> {
            let output = conv.output().clone();
            let coeffs = output.pack(
                &bias
                    .iter()
                    .flat_map(|&b| std::iter::repeat_n(b, output.height() * output.width()))
                    .collect::<Vec<_>>(),
            )?;
            let max = coeffs.iter().fold(1.0f64, |m, b| m.max(b.abs()));
            let mut pt =
                module.ckks_plaintext_alloc(n.into(), base2k.into(), (log_delta + 1 + max.log2().ceil() as usize).into());
            pt.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Real,
                log_sparsity: 0,
            });
            module.ckks_encode_coeffs_host_into(&mut pt, &coeffs, &mut aux.borrow())?;
            let mask = raise_gap
                .map(|gap| contexts[gap].real_raise_mask(&module, &samples(&output), &mut aux.borrow()))
                .transpose()?;
            Ok(Layer {
                name,
                conv,
                shortcut,
                bias: pt,
                mask,
                raise_gap: raise_gap.unwrap_or(0),
            })
        };
        let divisor = |name: &str| log_divisor(name).map(|d| f64::from(1u32 << d));

        // The stem reads the image as it is.
        let d_stem = divisor("stem")?;
        let (w, b) = scaled(&model.stem, 1.0 / d_stem, d_stem);
        let stem = finish(
            "stem".into(),
            conv(0, &layouts.image, &layouts.stage1, &w, 3, 1)?,
            None,
            &b,
            Some(0),
        )?;
        // A stored activation is the value times `2 / D`.
        let mut sigma = d_stem / 2.0;
        let mut blocks = Vec::new();
        for (i, block) in model.blocks.iter().enumerate() {
            let (stage, first) = (i / 3, i % 3 == 0);
            let (input, output, in_gap, out_gap) = match (stage, first) {
                (0, _) => (&layouts.stage1, &layouts.stage1, 0, 0),
                (1, true) => (&layouts.stage1, &layouts.out2, 0, 1),
                (1, false) => (&layouts.in2, &layouts.out2, 1, 1),
                (2, true) => (&layouts.in2, &layouts.out3, 1, 1),
                _ => (&layouts.in3, &layouts.out3, 1, 1),
            };
            let middle = output.folded();
            let (d1, d2) = (
                divisor(&format!("{}.r1", block.name))?,
                divisor(&format!("{}.r2", block.name))?,
            );
            let (w1, b1) = scaled(&block.conv1, sigma / d1, d1);
            let first_layer = finish(
                format!("{}.r1", block.name),
                conv(in_gap, input, output, &w1, 3, block.stride)?,
                None,
                &b1,
                Some(out_gap),
            )?;
            let (w2, b2) = scaled(&block.conv2, d1 / 2.0 / d2, d2);
            let (channels, out) = (block.conv1.in_channels, block.conv2.out_channels);
            let offset = if block.stride == 1 { 0 } else { (out - channels) / 2 };
            let skip = shortcut_weights(channels, out, offset, sigma / d2);
            let shortcut = conv(in_gap, input, output, &skip, 1, block.stride)?;
            let second_layer = finish(
                format!("{}.r2", block.name),
                conv(out_gap, &middle, output, &w2, 3, 1)?,
                Some(shortcut),
                &b2,
                Some(out_gap),
            )?;
            blocks.push([first_layer, second_layer]);
            sigma = d2 / 2.0;
        }
        // Average pooling and the linear layer as one product from the 8 x 8 grid: a 15 x 15 kernel at stride 8 whose
        // lower-right quadrant reads the grid.
        let (classes, channels) = (model.fc_bias.len(), model.fc_weight.len() / model.fc_bias.len());
        let mut head_weights = vec![0.0; classes * channels * 225];
        for c in 0..classes {
            for q in 0..channels {
                for dk in 0..8 {
                    for dl in 0..8 {
                        head_weights[((c * channels + q) * 15 + dk + 7) * 15 + dl + 7] =
                            model.fc_weight[c * channels + q] * sigma / 64.0;
                    }
                }
            }
        }
        let head = finish(
            "head".into(),
            conv(1, &layouts.in3, &layouts.logits, &head_weights, 15, 8)?,
            None,
            &model.fc_bias,
            None,
        )?;
        drop((enc, aux));
        let relu = Relu::new(
            &module,
            base2k.into(),
            log_delta,
            &mut arena::<ServerBackend>(CKKSEncodingHostOps::<ServerBackend, f64>::ckks_encoding_host_tmp_bytes(
                &module, n,
            ))
            .borrow(),
        )?;

        let boot = preset.bootstrap_layout();
        let alloc = |k: usize| module.ckks_ciphertext_alloc_from_glwe_infos(&params.layout(k));
        let bufs = Buffers {
            image: alloc(params.in_k),
            // SlotsToCoeffs runs in place from its input width.
            x_blocks: alloc(params.in_k),
            h_blocks: alloc(params.in_k),
            level0: alloc(params.level0_k),
            skip: alloc(params.level0_k),
            reduced: module.ckks_ciphertext_alloc_from_glwe_infos(&boot),
            activated: module.ckks_ciphertext_alloc_from_glwe_infos(&boot),
            narrow: alloc(params.in_k),
        };

        // One arena for the largest stage.
        let (x, level0) = (&bufs.x_blocks, &bufs.level0);
        let tensor_key = &preset.keys_layout().tensor_key;
        let mut bytes = module
            .ckks_double_tmp_bytes(x.size())
            .max(module.ckks_copy_tmp_bytes(&bufs.narrow, &bufs.activated))
            .max(module.ckks_add_tmp_bytes(level0.size()))
            .max(module.ckks_add_pt_vec_tmp_bytes(level0.size()))
            .max(relu.apply_tmp_bytes(&module, &bufs.reduced, tensor_key));
        for (context, preset) in contexts.iter().zip(&params.presets) {
            let keys_layout = preset.keys_layout();
            bytes = keys_layout
                .rotation_key_layouts()
                .iter()
                .map(|atk| module.ckks_dft_tmp_bytes(x, x, context.slots_to_coeffs(), atk))
                .fold(bytes, usize::max)
                .max(module.ckks_bootstrap_real_raise_tmp_bytes(&bufs.reduced, level0, context, keys_layout));
        }
        for layer in std::iter::once(&stem)
            .chain(blocks.iter().flatten())
            .chain(std::iter::once(&head))
        {
            for c in std::iter::once(&layer.conv).chain(layer.shortcut.as_ref()) {
                let keys_layout = params.presets[c.log_gap()].keys_layout();
                for atk in keys_layout.rotation_key_layouts() {
                    bytes = bytes.max(c.apply_tmp_bytes(&module, x, &keys_layout.tensor_key, &atk));
                }
            }
        }

        let state = RefCell::new(State {
            scratch: arena::<ServerBackend>(bytes),
            bufs,
        });
        Ok(Self {
            host: Module::<Host>::new(n as u64),
            params,
            module,
            contexts,
            keys,
            stem,
            blocks,
            head,
            relu,
            state,
            stage_sync: true,
        })
    }

    /// Whether [`Self::infer`] waits for the device at each stage boundary, which [`Timings`] needs and which costs
    /// about 8%: the host stops queueing while the device runs.
    pub fn set_stage_sync(&mut self, on: bool) {
        self.stage_sync = on;
    }

    /// `image` into the input buffer, in place: a recording binds that buffer.
    fn write_image(&self, state: &mut State, image: &CKKSCiphertextOwned<Host>) {
        image.transfer_into(&mut *state.bufs.image);
        state.bufs.image.set_meta(image.meta());
    }

    /// The logits of the encrypted `image`, in the coefficients of a level-0 ciphertext, and the time per stage.
    pub fn infer(&self, image: &CKKSCiphertextOwned<Host>) -> Result<(CKKSCiphertextOwned<Host>, Timings)> {
        self.infer_traced(image, None)
    }

    /// [`Self::infer`], handing `probe` a copy of each layer's output: the ReLU's name, whether the copy is the
    /// activation (in slots) or its input (in coefficients at level 0), the copy and its layout.
    pub fn infer_traced(
        &self,
        image: &CKKSCiphertextOwned<Host>,
        mut probe: Option<Probe<'_>>,
    ) -> Result<(CKKSCiphertextOwned<Host>, Timings)> {
        let state = &mut *self.state.borrow_mut();
        self.write_image(state, image);
        let mut look = |layer: &Layer, stage: Stage, ct: &CKKSCiphertextOwned<ServerBackend>| {
            if let Some(probe) = probe.as_mut() {
                probe(&layer.name, stage, &to_host(&self.host, ct), layer.conv.output());
            }
        };
        let t = self.run(state, self.stage_sync, &mut look)?;
        Ok((to_host(&self.host, &state.bufs.level0), t))
    }

    /// The whole circuit, from the image buffer to the logits in `level0`: device work only unless `look` reads or
    /// `stage_sync` waits.
    fn run(
        &self,
        state: &mut State,
        stage_sync: bool,
        look: &mut dyn FnMut(&Layer, Stage, &CKKSCiphertextOwned<ServerBackend>),
    ) -> Result<Timings> {
        let State { scratch, bufs } = state;
        let eval = Eval {
            params: &self.params,
            module: &self.module,
            contexts: &self.contexts,
            keys: &self.keys,
            relu: &self.relu,
            stage_sync,
        };
        let (stem, head) = (&self.stem, &self.head);
        let mut t = Timings::default();
        eval.to_blocks(&bufs.image, 0, &mut bufs.narrow, &mut bufs.x_blocks, scratch, &mut t)?;
        eval.layer(stem, &bufs.x_blocks, None, &mut bufs.level0, &mut bufs.skip, scratch, &mut t)?;
        look(stem, Stage::Input, &bufs.level0);
        eval.activate(stem, bufs, scratch, &mut t)?;
        look(stem, Stage::Activation, &bufs.activated);
        eval.to_blocks(
            &bufs.activated,
            stem.raise_gap,
            &mut bufs.narrow,
            &mut bufs.x_blocks,
            scratch,
            &mut t,
        )?;
        for [first, second] in self.blocks.iter() {
            eval.layer(first, &bufs.x_blocks, None, &mut bufs.level0, &mut bufs.skip, scratch, &mut t)?;
            look(first, Stage::Input, &bufs.level0);
            eval.activate(first, bufs, scratch, &mut t)?;
            look(first, Stage::Activation, &bufs.activated);
            eval.to_blocks(
                &bufs.activated,
                first.raise_gap,
                &mut bufs.narrow,
                &mut bufs.h_blocks,
                scratch,
                &mut t,
            )?;
            eval.layer(
                second,
                &bufs.h_blocks,
                Some(&bufs.x_blocks),
                &mut bufs.level0,
                &mut bufs.skip,
                scratch,
                &mut t,
            )?;
            look(second, Stage::Input, &bufs.level0);
            eval.activate(second, bufs, scratch, &mut t)?;
            look(second, Stage::Activation, &bufs.activated);
            eval.to_blocks(
                &bufs.activated,
                second.raise_gap,
                &mut bufs.narrow,
                &mut bufs.x_blocks,
                scratch,
                &mut t,
            )?;
        }
        eval.layer(head, &bufs.x_blocks, None, &mut bufs.level0, &mut bufs.skip, scratch, &mut t)?;
        Ok(t)
    }
}

/// The whole circuit recorded as one graph on the device: an inference writes the image, replays the graph and reads
/// the logits, with no orchestration on the host.
#[cfg(feature = "enable-cuda")]
pub struct RecordedInference<'a> {
    server: &'a Server,
    replay: poulpy_gpu_portable::Replay<'a, poulpy_gpu_portable::Cuda>,
}

#[cfg(feature = "enable-cuda")]
impl Server {
    /// Runs the circuit on `image`, then records it. The recording binds the server's buffers and arenas, so it lives
    /// as long as this borrow, and the plain [`Self::infer`] must not run while it does.
    ///
    /// # Errors
    ///
    /// As [`Self::infer`], for the first run: it uploads what the device builds on first use, a host write a recording
    /// must not see.
    pub fn record(&self, image: &CKKSCiphertextOwned<Host>) -> Result<RecordedInference<'_>> {
        self.infer(image)?;
        self.module.synchronize();
        let replay = poulpy_gpu_portable::Replay::record(move || {
            self.run(&mut self.state.borrow_mut(), false, &mut |_, _, _| {})
                .expect("an inference on valid buffers does not fail");
        });
        Ok(RecordedInference { server: self, replay })
    }
}

#[cfg(feature = "enable-cuda")]
impl RecordedInference<'_> {
    /// The logits of the encrypted `image`, in the coefficients of a level-0 ciphertext.
    pub fn infer(&mut self, image: &CKKSCiphertextOwned<Host>) -> CKKSCiphertextOwned<Host> {
        self.server.write_image(&mut self.server.state.borrow_mut(), image);
        self.replay.replay();
        to_host(&self.server.host, &self.server.state.borrow().bufs.level0)
    }

    /// Whether inferences run the recorded graph rather than the circuit again.
    pub fn is_graph(&self) -> bool {
        self.replay.is_graph()
    }

    /// Kernel launches of the circuit.
    pub fn launches(&self) -> usize {
        self.replay.launches()
    }

    /// Device bytes only the recording keeps reserved.
    pub fn retained_bytes(&self) -> u64 {
        self.replay.retained_bytes()
    }

    /// What the circuit did besides launching kernels: a graph needs no read, write or sync.
    pub fn host_events(&self) -> poulpy_gpu_portable::HostEvents {
        self.replay.host_events()
    }

    /// Runs the circuit again and panics unless it issues the launches recorded.
    pub fn check(&mut self) {
        self.replay.check();
    }
}

/// What every stage reads.
struct Eval<'a> {
    params: &'a Params,
    module: &'a Module<ServerBackend>,
    contexts: &'a [BootstrappingContextPrepared<ServerBackend, f64>],
    keys: &'a [BootstrappingKeysPrepared<<ServerBackend as Backend>::OwnedBuf, ServerBackend>],
    relu: &'a Relu<ServerBackend>,
    stage_sync: bool,
}

impl Eval<'_> {
    /// Charges the time since `start` to `slot`, once the server is done.
    fn lap(&self, start: Instant, slot: &mut Duration) {
        if self.stage_sync {
            self.module.synchronize();
        }
        *slot += start.elapsed();
    }

    /// `res = conv(x) + shortcut(skip) + bias` at level 0, from `x` and `skip` short of the last SlotsToCoeffs factor.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        layer: &Layer,
        x: &CKKSCiphertextOwned<ServerBackend>,
        skip: Option<&CKKSCiphertextOwned<ServerBackend>>,
        res: &mut CKKSCiphertextOwned<ServerBackend>,
        tmp: &mut CKKSCiphertextOwned<ServerBackend>,
        scratch: &mut ScratchOwned<ServerBackend>,
        t: &mut Timings,
    ) -> Result<()> {
        let start = Instant::now();
        let (module, keys) = (self.module, self.keys);
        res.set_k(self.params.level0_k.into());
        let conv_keys = &keys[layer.conv.log_gap()].rotation_keys;
        layer.conv.apply(module, res, x, conv_keys, &mut scratch.borrow())?;
        if let (Some(shortcut), Some(skip)) = (&layer.shortcut, skip) {
            tmp.set_k(self.params.level0_k.into());
            let shortcut_keys = &keys[shortcut.log_gap()].rotation_keys;
            shortcut.apply(module, tmp, skip, shortcut_keys, &mut scratch.borrow())?;
            module.ckks_add_assign(res, tmp, &mut scratch.borrow())?;
        }
        module.ckks_add_pt_vec_assign(res, &layer.bias, &mut scratch.borrow())?;
        self.lap(start, &mut t.conv);
        Ok(())
    }

    /// The masked raise of `bufs.level0` and the ReLU on the slots, into `bufs.activated`.
    fn activate(
        &self,
        layer: &Layer,
        bufs: &mut Buffers,
        scratch: &mut ScratchOwned<ServerBackend>,
        t: &mut Timings,
    ) -> Result<()> {
        let module = self.module;
        let boot_k = self.params.presets[0].bootstrap_layout().k();
        let start = Instant::now();
        bufs.reduced.set_k(boot_k);
        module.ckks_bootstrap_real_raise(
            &mut bufs.reduced,
            &bufs.level0,
            &self.contexts[layer.raise_gap],
            layer.mask.as_ref(),
            &self.keys[layer.raise_gap],
            &mut scratch.borrow(),
        )?;
        self.lap(start, &mut t.raise);
        let start = Instant::now();
        bufs.activated.set_k(boot_k);
        self.relu.apply(
            module,
            &mut bufs.activated,
            &bufs.reduced,
            self.keys[0].tensor_key(),
            &mut scratch.borrow(),
        )?;
        self.lap(start, &mut t.relu);
        Ok(())
    }

    /// SlotsToCoeffs of `slots` into `blocks`, from the input width, short of its last factor, at the gap of `slots`.
    fn to_blocks(
        &self,
        slots: &CKKSCiphertextOwned<ServerBackend>,
        log_gap: usize,
        narrow: &mut CKKSCiphertextOwned<ServerBackend>,
        blocks: &mut CKKSCiphertextOwned<ServerBackend>,
        scratch: &mut ScratchOwned<ServerBackend>,
        t: &mut Timings,
    ) -> Result<()> {
        let module = self.module;
        let start = Instant::now();
        ensure!(
            slots.k().as_usize() >= self.params.in_k,
            "the activation leaves {} bits, SlotsToCoeffs takes {}",
            slots.k().as_usize(),
            self.params.in_k
        );
        module.ckks_copy(narrow, slots, &mut scratch.borrow())?;
        blocks.set_k(self.params.in_k.into());
        module.ckks_double_into(blocks, &*narrow, &mut scratch.borrow())?;
        slots_to_blocks(
            module,
            blocks,
            &self.contexts[log_gap],
            &self.keys[log_gap].rotation_keys,
            &mut scratch.borrow(),
        )?;
        self.lap(start, &mut t.s2c);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Layouts;
    use crate::kernel::kernel_polynomials;

    /// Every product of the network: its layouts chain and no term lands on a sample it does not compute.
    #[test]
    fn every_layer_has_its_kernel_polynomials() {
        let l = Layouts::new(1 << 16).unwrap();
        assert_eq!((l.out2.folded(), l.out3.folded()), (l.in2.clone(), l.in3.clone()));
        let products = [
            (&l.image, &l.stage1, 3, 1),
            (&l.stage1, &l.stage1, 3, 1),
            (&l.stage1, &l.out2, 3, 2),
            (&l.stage1, &l.out2, 1, 2),
            (&l.in2, &l.out2, 3, 1),
            (&l.in2, &l.out3, 3, 2),
            (&l.in2, &l.out3, 1, 2),
            (&l.in3, &l.out3, 3, 1),
            (&l.stage1, &l.stage1, 1, 1),
            (&l.in2, &l.out2, 1, 1),
            (&l.in3, &l.out3, 1, 1),
            (&l.in3, &l.logits, 15, 8),
        ];
        for (input, output, kernel, stride) in products {
            let (c, m) = (input.channels(), output.channels());
            // Option A for the 1 x 1 shortcuts, every weight elsewhere; the head reads the grid from its quadrant.
            let weights: Vec<f64> = (0..m * c * kernel * kernel)
                .map(|t| {
                    let (o, q, tap) = (t / (c * kernel * kernel), t / (kernel * kernel) % c, t % (kernel * kernel));
                    let keep = match kernel {
                        1 if stride == 1 => o == q,
                        1 => o == q + c / 2,
                        15 => tap / 15 >= 7 && tap % 15 >= 7,
                        _ => true,
                    };
                    if keep { 1.0 + (t % 7) as f64 } else { 0.0 }
                })
                .collect();
            let log_gap = input.log_gap();
            let polys = kernel_polynomials(input, output, &weights, kernel, stride, 64, log_gap).unwrap();
            let in_subring = |(e, &c): (usize, &f64)| c == 0.0 || e.is_multiple_of(1 << log_gap);
            assert!(polys.iter().all(|poly| poly.iter().enumerate().all(in_subring)));
        }
    }
}
