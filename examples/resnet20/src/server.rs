//! The encrypted network. An activation lives in the real slots between two layers, stored over the divisor `D` of its
//! ReLU. A layer runs three steps on it:
//!
//! 1. SlotsToCoeffs, short of its last factor;
//! 2. that factor with the layer's convolution folded in, the bias and, on a block's second layer, the block's
//!    shortcut, which leaves `y = SPAN · x / D` in the coefficients at level 0, at a message ratio of 1 bit;
//! 3. the raise, masked to the layer's output samples, which leaves `exp(iπ·y)` in the slots, and the trigonometric
//!    polynomial of the ReLU on it.
//!
//! The client encrypts the image in the real slots, as an activation, and decrypts the logits at level 0 from the
//! head's coefficients.

use std::cell::RefCell;

use anyhow::{Context as _, Result, ensure};
use poulpy_ckks::{
    CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSBootstrappingOps, CKKSDFTOps, CKKSEncodingHostOps, CKKSPow2Ops},
    layouts::{
        BootstrappingKeys, BootstrappingKeysPrepared, CKKSCiphertextAlloc, CKKSCiphertextOwned, CKKSPlaintextAlloc,
        CoeffsMaskPrepared,
    },
};
use poulpy_core::{
    api::TransferInto,
    layouts::{GLWEInfos, LWEInfos, SetGLWEPlaintextInfos, SetK},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
};

use crate::{
    Context, Ct, Host, KeySet, Params, Pt, ServerBackend,
    layout::CoeffLayout,
    merged::MergedConv,
    model::{Conv, RELU_AT_ZERO, ResNet20, SPAN, Window, divisor, shortcut_weights},
    params::LOG_DELTA_W,
    relu::Relu,
};

type Keys = BootstrappingKeysPrepared<<ServerBackend as Backend>::OwnedBuf, ServerBackend>;

/// A convolution with its bias, at level 0.
struct Product {
    conv: MergedConv,
    bias: Pt,
}

/// What a layer does with the shortcut of its block.
enum Skip {
    /// The stem and the head belong to no block.
    None,
    /// A block's first layer leaves the block's shortcut of its input beside its own output.
    Save(MergedConv),
    /// A block's second layer adds it.
    Add,
}

/// One layer before a ReLU: the stem or a convolution of a block.
struct Layer {
    product: Product,
    skip: Skip,
    /// The stage of the output.
    stage: usize,
}

/// What the raise and the ReLU of a layer read, which its output layout alone decides: one per stage.
struct Activation {
    /// Keeps the output samples through the raise. The real slots already keep the lower half: a layout that fills it
    /// has none.
    mask: Option<CoeffsMaskPrepared<ServerBackend>>,
    /// The ReLU's value at zero on the slots of the output samples.
    relu_offset: Pt,
}

/// The server's ciphertexts, allocated once: a recording binds them.
struct Buffers {
    /// The image, in real slots.
    image: Ct,
    /// The input of a layer, short of the last SlotsToCoeffs factor, allocated at the SlotsToCoeffs input width.
    blocks: Ct,
    /// The output of a layer, in the coefficients.
    level0: Ct,
    /// The shortcut of a block, from its first layer to its second.
    skip: Ct,
    /// The raise, then CoeffsToSlots, on the slots, at the bootstrap layout.
    raised: Ct,
    /// EvalMod: the unit circle, in the slots.
    reduced: Ct,
    /// The ReLU: the activation, in the real slots.
    activated: Ct,
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
    context: Context,
    keys: Keys,
    /// The stem, then the two layers of each block.
    layers: Vec<Layer>,
    head: Product,
    /// By stage.
    activations: Vec<Activation>,
    relu: Relu,
    state: RefCell<State>,
}

/// Encodes the network on the server.
struct Encoder<'a> {
    module: &'a Module<ServerBackend>,
    context: &'a Context,
    params: &'a Params,
    /// Scratch of a merged factor.
    factor: ScratchOwned<ServerBackend>,
    /// Scratch of the host encodings and of a mask.
    host: ScratchOwned<ServerBackend>,
}

impl<'a> Encoder<'a> {
    fn new(module: &'a Module<ServerBackend>, context: &'a Context, params: &'a Params) -> Result<Self> {
        let slots = module.n() / 2;
        Ok(Self {
            module,
            context,
            params,
            factor: ScratchOwned::alloc(MergedConv::new_tmp_bytes(module, context, LOG_DELTA_W)?),
            host: ScratchOwned::alloc(
                CKKSEncodingHostOps::<ServerBackend, f64>::ckks_encoding_host_tmp_bytes(module, slots)
                    .max(context.coeffs_mask_tmp_bytes(module)?),
            ),
        })
    }

    fn conv(&mut self, input: &CoeffLayout, output: &CoeffLayout, weights: &[f64], window: Window) -> Result<MergedConv> {
        MergedConv::new(
            self.module,
            self.context,
            input,
            output,
            weights,
            window,
            LOG_DELTA_W,
            &mut self.factor.borrow(),
        )
    }

    /// `bias`, one value per channel of `output`, on its samples.
    fn bias(&mut self, output: &CoeffLayout, bias: &[f64]) -> Result<Pt> {
        let plane = output.height() * output.width();
        let coeffs = output.pack(&bias.iter().flat_map(|&b| std::iter::repeat_n(b, plane)).collect::<Vec<_>>())?;
        let (preset, max) = (self.params.preset(), bias.iter().fold(1.0f64, |m, b| m.max(b.abs())));
        // The sign and the integer bits of the largest bias, above the scale.
        let k = preset.log_delta() + 1 + max.log2().ceil() as usize;
        let mut pt = self
            .module
            .ckks_plaintext_alloc(self.module.n().into(), preset.base2k().into(), k.into());
        pt.set_meta(self.params.activation_meta());
        self.module
            .ckks_encode_coeffs_host_into(&mut pt, &coeffs, &mut self.host.borrow())?;
        Ok(pt)
    }

    /// `conv` at stride `stride` from `input` to `output`, on an input stored over `d_in`, for a ReLU of divisor `d_out`:
    /// it writes `SPAN / d_out` times the convolution of the values.
    fn product(
        &mut self,
        input: &CoeffLayout,
        output: &CoeffLayout,
        conv: &Conv,
        stride: usize,
        d_in: f64,
        d_out: f64,
    ) -> Result<Product> {
        let weights: Vec<f64> = conv.weights.iter().map(|w| w * SPAN * d_in / d_out).collect();
        let bias: Vec<f64> = conv.bias.iter().map(|b| b * SPAN / d_out).collect();
        Ok(Product {
            conv: self.conv(input, output, &weights, conv.window(stride))?,
            bias: self.bias(output, &bias)?,
        })
    }

    fn activation(&mut self, output: &CoeffLayout) -> Result<Activation> {
        let (preset, slots) = (self.params.preset(), self.module.n() / 2);
        let mask = (output.len() < slots)
            .then(|| {
                self.context
                    .coeffs_mask(self.module, &output.occupied(), &mut self.host.borrow())
            })
            .transpose()?;
        let re = output.pack_slots(&vec![RELU_AT_ZERO; output.len()])?;
        let mut relu_offset =
            self.module
                .ckks_plaintext_alloc_from_slots(slots, preset.base2k().into(), (preset.log_delta() + 1).into());
        relu_offset.set_meta(self.params.activation_meta());
        self.module
            .ckks_encode_slots_host_into(&mut relu_offset, &re, &vec![0.0; slots], &mut self.host.borrow())?;
        Ok(Activation { mask, relu_offset })
    }
}

/// The bootstrap of `params`, compiled and prepared on `module`.
fn compile(module: &Module<ServerBackend>, params: &Params) -> Result<Context> {
    let preset = params.preset();
    let bytes = Context::compile_unprepared_tmp_bytes(module, preset.plan())?;
    let unprepared = Context::compile_unprepared(
        module,
        preset.base2k().into(),
        preset.plan(),
        &mut ScratchOwned::alloc(bytes).borrow(),
    )?;
    let bytes = unprepared.prepare_tmp_bytes(module);
    unprepared.prepare(module, &mut ScratchOwned::alloc(bytes).borrow())
}

impl Server {
    /// Compiles the bootstrap, prepares the client's keys and encodes `model`.
    pub fn new(params: &Params, model: &ResNet20, key_set: KeySet) -> Result<Self> {
        let preset = params.preset();
        let module = Module::<ServerBackend>::new(preset.n() as u64);
        let context = compile(&module, params)?;
        let keys = {
            let bytes = preset.keys_layout().prepare_tmp_bytes(&module);
            key_set.transfer_and_prepare(&module, &mut ScratchOwned::alloc(bytes).borrow())?
        };
        drop(key_set);

        let layouts = params.layouts();
        let mut encoder = Encoder::new(&module, &context, params)?;
        let activations = layouts
            .stages
            .iter()
            .map(|stage| encoder.activation(stage))
            .collect::<Result<Vec<_>>>()?;
        // The stem reads the image as it is; every other layer an activation stored over the divisor before it.
        let mut d_in = divisor(0);
        let mut layers = vec![Layer {
            product: encoder.product(&layouts.image, &layouts.stages[0], &model.stem, 1, 1.0, d_in)?,
            skip: Skip::None,
            stage: 0,
        }];
        let mut stage = 0;
        for (b, block) in model.blocks.iter().enumerate() {
            let (d1, d2) = (divisor(2 * b + 1), divisor(2 * b + 2));
            let input = &layouts.stages[stage];
            stage += usize::from(block.stride > 1);
            let output = &layouts.stages[stage];
            let shortcut = encoder.conv(
                input,
                output,
                &shortcut_weights(input.channels(), output.channels(), SPAN * d_in / d2),
                Window {
                    kernel: 1,
                    stride: block.stride,
                    padding: 0,
                },
            )?;
            layers.push(Layer {
                product: encoder.product(input, output, &block.conv1, block.stride, d_in, d1)?,
                skip: Skip::Save(shortcut),
                stage,
            });
            layers.push(Layer {
                product: encoder.product(output, output, &block.conv2, 1, d1, d2)?,
                skip: Skip::Add,
                stage,
            });
            d_in = d2;
        }
        // Average pooling and the linear layer as one product: a window over the whole grid of the last stage.
        let last = &layouts.stages[stage];
        let (side, samples) = (last.height(), (last.height() * last.width()) as f64);
        let pooled: Vec<f64> = model
            .fc_weight
            .iter()
            .flat_map(|w| std::iter::repeat_n(w * d_in / samples, samples as usize))
            .collect();
        let pool = Window {
            kernel: side,
            stride: side,
            padding: 0,
        };
        let head = Product {
            conv: encoder.conv(last, &layouts.logits, &pooled, pool)?,
            bias: encoder.bias(&layouts.logits, &model.fc_bias)?,
        };
        let relu = Relu::new(
            &module,
            preset.base2k().into(),
            preset.log_delta(),
            &mut encoder.host.borrow(),
        )?;
        drop(encoder);

        let boot = preset.bootstrap_layout();
        let alloc = |k: usize| module.ckks_ciphertext_alloc(preset.base2k().into(), k.into());
        let bufs = Buffers {
            image: alloc(params.in_k()),
            blocks: alloc(params.in_k()),
            level0: alloc(params.logits_k()),
            skip: alloc(params.level0_k()),
            raised: module.ckks_ciphertext_alloc_from_glwe_infos(&boot),
            reduced: module.ckks_ciphertext_alloc_from_glwe_infos(&boot),
            activated: module.ckks_ciphertext_alloc_from_glwe_infos(&boot),
        };

        // One arena for the largest step, whatever the key that runs it.
        let keys_layout = preset.keys_layout();
        let tensor_key = &keys_layout.tensor_key;
        let (blocks, raised) = (&bufs.blocks, &bufs.raised);
        let run = context.run(GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            ..bufs.level0.meta()
        });
        let mut bytes = <Module<ServerBackend> as CKKSPow2Ops<ServerBackend>>::ckks_double_tmp_bytes(&module, blocks.size())
            .max(module.ckks_mod_raise_tmp_bytes(raised, &bufs.level0, &context, keys_layout))
            .max(module.ckks_bootstrap_eval_mod_tmp_bytes(&bufs.reduced, raised, &run, keys_layout));
        for atk in keys_layout.rotation_key_layouts() {
            bytes = bytes
                .max(relu.apply_tmp_bytes(&module, &bufs.reduced, tensor_key, atk))
                .max(module.ckks_dft_tmp_bytes(blocks, blocks, context.slots_to_coeffs(), atk))
                .max(module.ckks_dft_tmp_bytes(raised, raised, context.coeffs_to_slots(), atk));
            for (conv, skip) in layers
                .iter()
                .map(|layer| (&layer.product.conv, &layer.skip))
                .chain([(&head.conv, &Skip::None)])
            {
                let shortcut = match skip {
                    Skip::Save(shortcut) => Some(shortcut),
                    _ => None,
                };
                bytes = bytes.max(conv.apply_tmp_bytes(&module, shortcut, blocks, tensor_key, atk));
            }
            for mask in activations.iter().filter_map(|activation| activation.mask.as_ref()) {
                bytes = bytes.max(module.ckks_dft_tmp_bytes(raised, raised, &context.masked(mask)?, atk));
            }
        }

        let state = RefCell::new(State {
            scratch: ScratchOwned::alloc(bytes),
            bufs,
        });
        Ok(Self {
            params: params.clone(),
            module,
            context,
            keys,
            layers,
            head,
            activations,
            relu,
            state,
        })
    }

    /// `image` into the input buffer, in place: a recording binds that buffer.
    fn write_image(&self, state: &mut State, image: &CKKSCiphertextOwned<Host>) -> Result<()> {
        let buf = &mut state.bufs.image;
        ensure!(
            image.glwe_layout() == buf.glwe_layout()
                && image.max_size() == buf.max_size()
                && image.meta() == self.params.activation_meta(),
            "the image must be encrypted as `Client::encrypt_image` does"
        );
        image.transfer_into(&mut **buf);
        Ok(())
    }

    /// The logits of the encrypted `image`, in the coefficients of a level-0 ciphertext.
    ///
    /// # Errors
    ///
    /// An `image` of another layout or metadata than [`Client::encrypt_image`](crate::Client::encrypt_image) gives it.
    pub fn infer(&self, image: &CKKSCiphertextOwned<Host>) -> Result<CKKSCiphertextOwned<Host>> {
        let state = &mut *self.state.borrow_mut();
        self.write_image(state, image)?;
        self.run(state)?;
        Ok(state.bufs.level0.to_host_owned::<ServerBackend>())
    }

    /// The whole circuit, from the image buffer to the logits in `level0`: device work only.
    fn run(&self, state: &mut State) -> Result<()> {
        let State { scratch, bufs } = state;
        self.to_blocks(&bufs.image, &mut bufs.blocks, scratch)?;
        for layer in &self.layers {
            let activation = &self.activations[layer.stage];
            self.layer(&layer.product, &layer.skip, self.params.level0_k(), bufs, scratch)?;
            self.raise(activation, bufs, scratch)?;
            self.relu.apply(
                &self.module,
                &mut bufs.activated,
                &bufs.reduced,
                &activation.relu_offset,
                self.keys.tensor_key().context("the keys hold no tensor key")?,
                &self.keys.rotation_keys,
                &mut scratch.borrow(),
            )?;
            self.to_blocks(&bufs.activated, &mut bufs.blocks, scratch)?;
        }
        self.layer(&self.head, &Skip::None, self.params.logits_k(), bufs, scratch)
    }

    /// SlotsToCoeffs of the activation `slots` into `blocks`, from its input width, short of its last factor.
    fn to_blocks(&self, slots: &Ct, blocks: &mut Ct, scratch: &mut ScratchOwned<ServerBackend>) -> Result<()> {
        let in_k = self.params.in_k();
        ensure!(
            slots.k().as_usize() >= in_k,
            "the activation leaves {} bits, SlotsToCoeffs takes {in_k}",
            slots.k().as_usize()
        );
        blocks.set_k(in_k.into());
        // The real-slot decoding matrix halves its input.
        self.module.ckks_double_into(blocks, slots, &mut scratch.borrow())?;
        Ok(self.module.ckks_slots_to_coeffs_assign(
            blocks,
            &self.context.slots_to_coeffs().head(),
            &self.keys.rotation_keys,
            &mut scratch.borrow(),
        )?)
    }

    /// `level0 = conv(blocks) + bias` at width `output_k`, with the shortcut of the block as `skip` says.
    fn layer(
        &self,
        product: &Product,
        skip: &Skip,
        output_k: usize,
        bufs: &mut Buffers,
        scratch: &mut ScratchOwned<ServerBackend>,
    ) -> Result<()> {
        let (module, keys) = (&self.module, &self.keys.rotation_keys);
        let Buffers {
            blocks,
            level0,
            skip: saved,
            ..
        } = bufs;
        level0.set_k(output_k.into());
        let beside = match skip {
            Skip::Save(shortcut) => {
                saved.set_k(self.params.level0_k().into());
                Some((shortcut, &mut *saved))
            }
            _ => None,
        };
        product
            .conv
            .apply(module, level0, beside, blocks, keys, &mut scratch.borrow())?;
        if matches!(skip, Skip::Add) {
            module.ckks_add_assign(level0, &*saved, &mut scratch.borrow())?;
        }
        module.ckks_add_pt_vec_assign(level0, &product.bias, &mut scratch.borrow())?;
        Ok(())
    }

    /// The masked raise of `level0` into `reduced`: the output samples on the unit circle, in the slots.
    fn raise(&self, activation: &Activation, bufs: &mut Buffers, scratch: &mut ScratchOwned<ServerBackend>) -> Result<()> {
        let (module, context, keys) = (&self.module, &self.context, &self.keys);
        let Buffers {
            level0, raised, reduced, ..
        } = bufs;
        let scratch = &mut scratch.borrow();
        // The raise leaves `level0` unspecified, so its metadata is read first.
        let run = context.run(GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            ..level0.meta()
        });
        let boot_k = self.params.preset().bootstrap_k();
        raised.set_k(boot_k.into());
        reduced.set_k(boot_k.into());
        module.ckks_mod_raise(raised, level0, context, keys, scratch)?;
        match &activation.mask {
            Some(mask) => module.ckks_coeffs_to_slots_assign(raised, &context.masked(mask)?, &keys.rotation_keys, scratch)?,
            None => module.ckks_coeffs_to_slots_assign(raised, context, &keys.rotation_keys, scratch)?,
        }
        raised.set_slots(SlotsKind::Real);
        module.ckks_bootstrap_eval_mod(reduced, raised, &run, keys, scratch)?;
        Ok(())
    }
}

/// The whole circuit recorded as one graph on the device: an inference writes the image, replays the graph and reads
/// the logits, with no orchestration on the host. It is the one way the GPU runs the network.
#[cfg(feature = "enable-cuda")]
pub struct RecordedInference<'a> {
    server: &'a Server,
    /// The recording: its launches, what it retains and what the host did besides.
    pub replay: poulpy_gpu_portable::Replay<'a, poulpy_gpu_portable::Cuda>,
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
        poulpy_hal::api::ModuleSynchronize::synchronize(&self.module);
        let replay = poulpy_gpu_portable::Replay::record(move || {
            self.run(&mut self.state.borrow_mut())
                .expect("an inference on valid buffers does not fail");
        });
        Ok(RecordedInference { server: self, replay })
    }
}

#[cfg(feature = "enable-cuda")]
impl RecordedInference<'_> {
    /// The logits of the encrypted `image`, in the coefficients of a level-0 ciphertext.
    ///
    /// # Errors
    ///
    /// As [`Server::infer`], before the replay.
    pub fn infer(&mut self, image: &CKKSCiphertextOwned<Host>) -> Result<CKKSCiphertextOwned<Host>> {
        self.server.write_image(&mut self.server.state.borrow_mut(), image)?;
        self.replay.replay();
        Ok(self.server.state.borrow().bufs.level0.to_host_owned::<ServerBackend>())
    }
}

#[cfg(test)]
mod tests {
    use super::Server;
    use crate::{
        Client, Params,
        model::{ResNet20, Variant, max_error, test_images},
    };

    #[test]
    #[ignore = "a full encrypted inference; opt in with --ignored, on the GPU with --features enable-cuda"]
    fn the_first_image_matches_the_cleartext_network() {
        let params = Params::new().unwrap();
        let model = ResNet20::load().unwrap();
        let (image, _) = &test_images(1).unwrap()[0];
        let mut client = Client::new(&params, [0; 32]);
        let server = Server::new(&params, &model, client.keys().unwrap()).unwrap();
        let out = server.infer(&client.encrypt_image(image).unwrap()).unwrap();
        let logits = client.decrypt_logits(&out).unwrap();
        let err = max_error(&logits, &model.forward(image, Variant::Polynomial));
        println!(
            "encrypted logits against the cleartext polynomial network: max error 2^{:.2}",
            err.log2()
        );
        assert!(err < (-5.0f64).exp2(), "the encrypted logits are off by more than 2^-5");
    }
}
