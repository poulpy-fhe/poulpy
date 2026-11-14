//! A ResNet-20 for CIFAR-10: its weights with the batch norms folded into the convolutions, the
//! approximate ReLU of the encrypted evaluator and the forward pass in the clear.

use std::{borrow::Cow, collections::HashMap, fs};

use anyhow::{Context, Result, ensure};

/// Side of an image, and bytes of a record of the CIFAR-10 binary format: a label and three channels.
const SIDE: usize = 32;
const RECORD: usize = 1 + 3 * SIDE * SIDE;

/// Samples between the starts of two rows of every tensor of the network. Stage 1 fills its rows, so its convolutions
/// read past a row's end into the next row, where the network was trained on zero padding: 0.13% on the test set,
/// [`Variant::Wrapped`] against [`Variant::Trained`].
pub(crate) const ROW_STRIDE: usize = SIDE;

/// The window of a convolution, as PyTorch states it: output `(i, j)` reads the inputs
/// `(stride * i + dk - padding, stride * j + dl - padding)` for `dk` and `dl` below `kernel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Window {
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
}

/// The convolution of window `window`, on `[channel][row][column]` tensors and `[out][in][kernel][kernel]` weights. A
/// channel sits on rows of `row_stride` samples: what lies outside the grid reads zero, and a read past a row's end the
/// row after it when `row_stride` leaves no room, as the coefficients of the encrypted network do.
pub(crate) fn conv2d(input: &[f64], weights: &[f64], height: usize, width: usize, row_stride: usize, window: Window) -> Vec<f64> {
    let Window { kernel, stride, padding } = window;
    let in_channels = input.len() / (height * width);
    let out_channels = weights.len() / (in_channels * kernel * kernel);
    let (out_height, out_width) = (
        (height + 2 * padding - kernel) / stride + 1,
        (width + 2 * padding - kernel) / stride + 1,
    );
    // Each channel on its rows of `row_stride` samples, between the zeros a window reaches: a read is one index.
    let margin = padding * (row_stride + 1);
    let mut rows = vec![0.0; in_channels * (height * row_stride + 2 * margin)];
    for (channel, rows) in input
        .chunks_exact(height * width)
        .zip(rows.chunks_exact_mut(height * row_stride + 2 * margin))
    {
        for (row, samples) in channel.chunks_exact(width).enumerate() {
            rows[margin + row * row_stride..][..width].copy_from_slice(samples);
        }
    }
    let mut res = vec![0.0; out_channels * out_height * out_width];
    let mut acc = vec![0.0; out_width];
    for (o, res) in res.chunks_exact_mut(out_height * out_width).enumerate() {
        for (c, rows) in rows.chunks_exact(height * row_stride + 2 * margin).enumerate() {
            let taps = &weights[(o * in_channels + c) * kernel * kernel..][..kernel * kernel];
            for (i, res) in res.chunks_exact_mut(out_width).enumerate() {
                acc.fill(0.0);
                for (t, &w) in taps.iter().enumerate() {
                    let first = &rows[(stride * i + t / kernel) * row_stride + t % kernel..];
                    // Contiguous reads vectorize: 17 of the 19 convolutions.
                    if stride == 1 {
                        for (acc, x) in acc.iter_mut().zip(first) {
                            *acc += w * x;
                        }
                    } else {
                        for (acc, x) in acc.iter_mut().zip(first.iter().step_by(stride)) {
                            *acc += w * x;
                        }
                    }
                }
                for (res, acc) in res.iter_mut().zip(&acc) {
                    *res += acc;
                }
            }
        }
    }
    res
}

/// The share of the period a divisor's interval spans: `[-1, 1]` over the divisor maps to the phases `[-SPAN/2, SPAN/2]`
/// of the unit circle, which leaves the rest of the period for the two ends of the ReLU to join.
pub const SPAN: f64 = 0.9;

/// `(re, im)` of the coefficients of `z(a) = sum_l c_l a^l`, a minimax fit by reweighted least squares of `z(a) + conj(z(a))
/// + RELU_AT_ZERO` to the ReLU on the phases `a = exp(iπ·SPAN·u)`, `u` in `[-1, 1]`. The sum vanishes at `a = 1`.
pub(crate) const RELU_TRIG: [(f64, f64); 16] = [
    (0.138076357856, 0.000000000000),
    (-0.114287281818, -0.175201301161),
    (0.001625738389, 0.085180273184),
    (-0.014122593422, -0.054175521796),
    (0.001487199233, 0.038008668471),
    (-0.005943910980, -0.027866144580),
    (0.001283080536, 0.020825542483),
    (-0.003515028717, -0.015644111282),
    (0.001035555780, 0.011703990804),
    (-0.002340644338, -0.008659127610),
    (0.000730091536, 0.006292050187),
    (-0.001540921670, -0.004448294742),
    (0.000337294069, 0.003003201057),
    (-0.000872554687, -0.001847916103),
    (-0.000039990966, 0.000887702528),
    (-0.001912390801, -0.000222727208),
];

/// The ReLU of zero under [`RELU_TRIG`].
pub(crate) const RELU_AT_ZERO: f64 = 0.003299254007;

/// The largest activation magnitude before each ReLU, in the order the network evaluates them (the stem, then the two
/// of each block), over 20,505 CIFAR-10 training images.
const MAX_ACTIVATION: [f64; 19] = [
    7.55, 17.48, 11.69, 12.79, 10.40, 16.13, 11.10, 13.84, 11.41, 10.19, 11.49, 11.50, 14.26, 12.61, 19.02, 12.33, 22.59, 13.33,
    51.82,
];

/// What a divisor leaves above the largest activation seen, before it rounds to a power of two.
const HEADROOM: f64 = 1.3;

/// The divisor of ReLU `relu`, half the power of two at or above 1.3 times its largest activation: its input
/// over the divisor lies in `[-1, 1]`, but for the outliers. An input past its divisor wraps around the unit circle:
/// that input comes out wrong, and nothing diverges.
pub fn divisor(relu: usize) -> f64 {
    f64::from(1u32 << ((HEADROOM * MAX_ACTIVATION[relu]).log2().ceil() as u32 - 1))
}

/// The approximate ReLU of `u`, periodic, close to `max(u, 0)` on `[-1, 1]`: a polynomial in `a = exp(iπ·SPAN·u)`,
/// by Horner's rule in `a`.
pub fn relu(u: f64) -> f64 {
    let (sin, cos) = (std::f64::consts::PI * SPAN * u).sin_cos();
    let (re, _) = RELU_TRIG.iter().rev().fold((0.0, 0.0), |(re, im), c| {
        (re * cos - im * sin + c.0, re * sin + im * cos + c.1)
    });
    RELU_AT_ZERO + 2.0 * re
}

/// The weight archive in `weights/`.
const MANIFEST: &[u8] = include_bytes!("../weights/manifest.json");
const PAYLOAD: &[u8] = include_bytes!("../weights/weights.f32.bin");

/// The tensors of the weight archive, by name.
struct Weights(HashMap<String, (Vec<usize>, Vec<f64>)>);

impl Weights {
    fn load() -> Result<Self> {
        let manifest: serde_json::Value = serde_json::from_slice(MANIFEST)?;
        let mut tensors = HashMap::new();
        for tensor in manifest["tensors"].as_array().context("the manifest lists no tensors")? {
            let name = tensor["name"].as_str().context("a tensor without a name")?;
            let shape = tensor["shape"]
                .as_array()
                .context("a tensor without a shape")?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<Vec<_>>>()
                .context("a shape of non-integers")?;
            let (offset, len) = (
                tensor["offset_bytes"].as_u64().context("a tensor without an offset")? as usize,
                tensor["length"].as_u64().context("a tensor without a length")? as usize,
            );
            ensure!(
                shape.iter().product::<usize>() == len,
                "the shape of {name} does not hold its length"
            );
            let bytes = PAYLOAD
                .get(offset..offset + 4 * len)
                .context("a tensor past the weight file")?;
            let values = bytes
                .chunks_exact(4)
                .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect();
            tensors.insert(name.to_string(), (shape, values));
        }
        Ok(Self(tensors))
    }

    fn get(&self, name: &str) -> Result<&(Vec<usize>, Vec<f64>)> {
        self.0.get(name).with_context(|| format!("the archive holds no {name}"))
    }

    /// The convolution `conv` with the batch norm `bn` folded in: `w * s[o]` and `beta - mean * s` with
    /// `s = gamma / sqrt(var + eps)`.
    fn folded(&self, conv: &str, bn: &str) -> Result<Conv> {
        let (shape, w) = self.get(&format!("{conv}.weight"))?;
        ensure!(shape.len() == 4 && shape[2] == shape[3], "{conv} is not a square convolution");
        let param = |p: &str| self.get(&format!("{bn}.{p}")).map(|(_, v)| v);
        let (gamma, beta, mean, var) = (
            param("weight")?,
            param("bias")?,
            param("running_mean")?,
            param("running_var")?,
        );
        let scale: Vec<f64> = gamma.iter().zip(var).map(|(g, v)| g / (v + 1e-5).sqrt()).collect();
        let per_out = w.len() / shape[0];
        Ok(Conv {
            weights: w.iter().enumerate().map(|(t, w)| w * scale[t / per_out]).collect(),
            bias: (0..shape[0]).map(|o| beta[o] - mean[o] * scale[o]).collect(),
            out_channels: shape[0],
            in_channels: shape[1],
            kernel: shape[2],
        })
    }
}

/// A convolution with its batch norm folded in, `weights` as `[out][in][kernel][kernel]`.
pub(crate) struct Conv {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: Vec<f64>,
    pub(crate) out_channels: usize,
    pub(crate) in_channels: usize,
    kernel: usize,
}

impl Conv {
    /// The window at stride `stride`: the padding keeps the grid at stride 1.
    pub(crate) fn window(&self, stride: usize) -> Window {
        Window {
            kernel: self.kernel,
            stride,
            padding: self.kernel / 2,
        }
    }
}

/// A residual block: `relu(conv2(relu(conv1(x))) + shortcut(x))`, the shortcut the identity or, at stride 2, Option
/// A (every other sample, the channels zero-padded on both sides).
pub(crate) struct Block {
    pub(crate) stride: usize,
    pub(crate) conv1: Conv,
    pub(crate) conv2: Conv,
}

/// The network, with its batch norms folded.
pub struct ResNet20 {
    pub(crate) stem: Conv,
    pub(crate) blocks: Vec<Block>,
    /// `[class][channel]`.
    pub(crate) fc_weight: Vec<f64>,
    pub(crate) fc_bias: Vec<f64>,
}

/// The network a forward pass runs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// As trained: zero padding and the exact ReLU.
    Trained,
    /// On the rows of the encrypted network, which wrap in stage 1, with the exact ReLU.
    Wrapped,
    /// On the same rows with `D * relu(x / D)`, the divisor of each ReLU: what the encrypted network computes.
    Polynomial,
}

impl ResNet20 {
    /// The network of the weight archive built into this crate.
    pub fn load() -> Result<Self> {
        let w = Weights::load()?;
        let mut blocks = Vec::new();
        for stage in 1..=3 {
            for b in 0..3 {
                let name = format!("layer{stage}.{b}");
                blocks.push(Block {
                    stride: if stage > 1 && b == 0 { 2 } else { 1 },
                    conv1: w.folded(&format!("{name}.conv1"), &format!("{name}.bn1"))?,
                    conv2: w.folded(&format!("{name}.conv2"), &format!("{name}.bn2"))?,
                });
            }
        }
        Ok(Self {
            stem: w.folded("conv1", "bn1")?,
            blocks,
            fc_weight: w.get("linear.weight")?.1.clone(),
            fc_bias: w.get("linear.bias")?.1.clone(),
        })
    }

    /// The logits of `image`, `[3][32][32]` normalized.
    pub fn forward(&self, image: &[f64], variant: Variant) -> Vec<f64> {
        self.forward_traced(image, variant, &mut |_, _| {})
    }

    /// [`Self::forward`], handing `trace` each ReLU's index, in evaluation order, and input.
    pub fn forward_traced(&self, image: &[f64], variant: Variant, trace: &mut dyn FnMut(usize, &[f64])) -> Vec<f64> {
        let mut relus = 0;
        let mut activate = |x: Vec<f64>| -> Vec<f64> {
            trace(relus, &x);
            let d = divisor(relus);
            relus += 1;
            match variant {
                Variant::Trained | Variant::Wrapped => x.into_iter().map(|v| v.max(0.0)).collect(),
                Variant::Polynomial => x.into_iter().map(|v| d * relu(v / d)).collect(),
            }
        };
        let conv = |c: &Conv, x: &[f64], side: usize, stride: usize| {
            let window = c.window(stride);
            // Rows the window cannot read across are the zero padding of the training.
            let row_stride = if variant == Variant::Trained {
                side + window.padding
            } else {
                ROW_STRIDE
            };
            let mut y = conv2d(x, &c.weights, side, side, row_stride, window);
            let plane = y.len() / c.out_channels;
            for (plane, bias) in y.chunks_exact_mut(plane).zip(&c.bias) {
                plane.iter_mut().for_each(|v| *v += bias);
            }
            y
        };
        let mut x = activate(conv(&self.stem, image, SIDE, 1));
        let mut side = SIDE;
        for block in &self.blocks {
            let h = activate(conv(&block.conv1, &x, side, block.stride));
            let out_side = side / block.stride;
            let mut y = conv(&block.conv2, &h, out_side, 1);
            for (v, s) in y.iter_mut().zip(shortcut(&x, block.conv1.in_channels, side, block.stride)) {
                *v += s;
            }
            x = activate(y);
            side = out_side;
        }
        let channels = x.len() / (side * side);
        let pooled: Vec<f64> = x
            .chunks_exact(side * side)
            .map(|c| c.iter().sum::<f64>() / (side * side) as f64)
            .collect();
        (0..self.fc_bias.len())
            .map(|class| {
                self.fc_bias[class]
                    + (0..channels)
                        .map(|q| self.fc_weight[class * channels + q] * pooled[q])
                        .sum::<f64>()
            })
            .collect()
    }
}

/// The identity, or at stride 2 Option A: every other sample in both directions, the channels doubled with zeros on
/// both sides.
fn shortcut(x: &[f64], channels: usize, side: usize, stride: usize) -> Vec<f64> {
    if stride == 1 {
        return x.to_vec();
    }
    let out = side / 2;
    let mut res = vec![0.0; 2 * channels * out * out];
    for q in 0..channels {
        for i in 0..out {
            for j in 0..out {
                res[((q + channels / 2) * out + i) * out + j] = x[(q * side + 2 * i) * side + 2 * j];
            }
        }
    }
    res
}

/// The weights of the shortcut as a `1 x 1` convolution from `channels` to `out` channels: `factor` times input `q`
/// into output `q + (out - channels) / 2`, the identity or Option A.
pub(crate) fn shortcut_weights(channels: usize, out: usize, factor: f64) -> Vec<f64> {
    let mut w = vec![0.0; out * channels];
    for q in 0..channels {
        w[(q + (out - channels) / 2) * channels + q] = factor;
    }
    w
}

/// The first ten images of the CIFAR-10 test batch, in its binary format.
const SAMPLE: &[u8] = include_bytes!("../data/test_images.bin");

/// The first `limit` images of the CIFAR-10 test batch, `[3][32][32]` normalized as the training did, with their labels:
/// from `test_batch.bin` at `$CIFAR10_TEST_BATCH`, or else from the ten built into this crate.
pub fn test_images(limit: usize) -> Result<Vec<(Vec<f64>, usize)>> {
    const MEAN: [f64; 3] = [0.4914, 0.4822, 0.4465];
    const STD: [f64; 3] = [0.2023, 0.1994, 0.2010];
    let raw = match std::env::var_os("CIFAR10_TEST_BATCH") {
        Some(path) => Cow::Owned(fs::read(&path).with_context(|| format!("reading {}", path.display()))?),
        None => {
            ensure!(
                limit <= SAMPLE.len() / RECORD,
                "ten images are built in: the crate's README gets the others"
            );
            Cow::Borrowed(SAMPLE)
        }
    };
    ensure!(raw.len().is_multiple_of(RECORD), "not a CIFAR-10 binary batch");
    Ok(raw
        .chunks_exact(RECORD)
        .take(limit)
        .map(|record| {
            let image = record[1..]
                .iter()
                .enumerate()
                .map(|(t, &p)| (f64::from(p) / 255.0 - MEAN[t / (SIDE * SIDE)]) / STD[t / (SIDE * SIDE)])
                .collect();
            (image, usize::from(record[0]))
        })
        .collect())
}

/// The index of the largest logit.
pub fn argmax(logits: &[f64]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0
}

/// The largest gap between `have` and `want`.
pub fn max_error(have: &[f64], want: &[f64]) -> f64 {
    have.iter().zip(want).fold(0.0f64, |e, (h, w)| e.max((h - w).abs()))
}
