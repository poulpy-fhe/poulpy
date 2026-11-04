//! A ResNet-20 for CIFAR-10: its weights with the batch norms folded into the convolutions, the
//! approximate ReLU of the encrypted evaluator and the forward pass in the clear.

use std::{collections::HashMap, fs};

use anyhow::{Context, Result, ensure};

use crate::reference::conv2d;

/// The two odd Chebyshev stages of a composite sign approximation on `[-1, 1]`, fitted with the Remez algorithm of
/// Lattigo.
pub const SIGN_STAGES: [[f64; 16]; 2] = [
    [
        0.0,
        0.820341946651,
        0.0,
        -0.274108809591,
        0.0,
        0.165343263898,
        0.0,
        -0.119117784621,
        0.0,
        0.093810087268,
        0.0,
        -0.078181361866,
        0.0,
        0.067989005078,
        0.0,
        -0.387755475663,
    ],
    [
        0.0,
        1.263568252469,
        0.0,
        -0.396096345236,
        0.0,
        0.209696958377,
        0.0,
        -0.123355609980,
        0.0,
        0.073080583151,
        0.0,
        -0.041454802075,
        0.0,
        0.021480414179,
        0.0,
        -0.010865648854,
    ],
];

/// The largest activation magnitude before each ReLU over 20,505 CIFAR-10 training images; the divisor of a ReLU is the
/// power of two at or above 1.3 times it. An input past its divisor diverges: half of those images, ranged on the other
/// half, need 1.1, and the accuracy does not move up to 1.3.
const MAX_ACTIVATION: [(&str, f64); 19] = [
    ("stem", 7.55),
    ("layer1.0.r1", 17.48),
    ("layer1.0.r2", 11.69),
    ("layer1.1.r1", 12.79),
    ("layer1.1.r2", 10.40),
    ("layer1.2.r1", 16.13),
    ("layer1.2.r2", 11.10),
    ("layer2.0.r1", 13.84),
    ("layer2.0.r2", 11.41),
    ("layer2.1.r1", 10.19),
    ("layer2.1.r2", 11.49),
    ("layer2.2.r1", 11.50),
    ("layer2.2.r2", 14.26),
    ("layer3.0.r1", 12.61),
    ("layer3.0.r2", 19.02),
    ("layer3.1.r1", 12.33),
    ("layer3.1.r2", 22.59),
    ("layer3.2.r1", 13.33),
    ("layer3.2.r2", 51.82),
];

/// `log2` of the divisor of the ReLU named `name` (`stem`, `layerS.B.r1` or `layerS.B.r2`): its input over the
/// divisor lies in `[-1, 1]`.
pub fn log_divisor(name: &str) -> Result<u32> {
    let (_, max) = MAX_ACTIVATION
        .iter()
        .find(|(n, _)| *n == name)
        .with_context(|| format!("no range for the ReLU {name}"))?;
    Ok((1.3 * max).log2().ceil() as u32)
}

/// `sign(y)` for `y` in `[-1, 1]`, the composite of [`SIGN_STAGES`].
pub fn sign(y: f64) -> f64 {
    SIGN_STAGES.iter().fold(y, |y, stage| chebyshev(stage, y))
}

/// `sum_k c_k T_k(y)` by Clenshaw's recurrence.
fn chebyshev(coeffs: &[f64], y: f64) -> f64 {
    let (mut b1, mut b2) = (0.0, 0.0);
    for &c in coeffs.iter().skip(1).rev() {
        (b1, b2) = (2.0 * y * b1 - b2 + c, b1);
    }
    y * b1 - b2 + coeffs[0]
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
pub struct Conv {
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel: usize,
}

/// A residual block: `relu(conv2(relu(conv1(x))) + shortcut(x))`, the shortcut the identity or, at stride 2, Option
/// A (every other sample, the channels zero-padded on both sides).
pub struct Block {
    pub name: String,
    pub stride: usize,
    pub conv1: Conv,
    pub conv2: Conv,
}

/// The network, with its batch norms folded.
pub struct ResNet20 {
    pub stem: Conv,
    pub blocks: Vec<Block>,
    /// `[class][channel]`.
    pub fc_weight: Vec<f64>,
    pub fc_bias: Vec<f64>,
}

/// The ReLU of the forward pass.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Relu {
    Exact,
    /// `(x + x * sign(x / D)) / 2`, with the composite sign and the divisor of each ReLU.
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
                    name,
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
    pub fn forward(&self, image: &[f64], relu: Relu) -> Result<Vec<f64>> {
        self.forward_traced(image, relu, &mut |_, _| {})
    }

    /// [`Self::forward`], handing `trace` each ReLU's name and input.
    pub fn forward_traced(&self, image: &[f64], relu: Relu, trace: &mut dyn FnMut(&str, &[f64])) -> Result<Vec<f64>> {
        let mut activate = |x: Vec<f64>, name: &str| -> Result<Vec<f64>> {
            trace(name, &x);
            Ok(match relu {
                Relu::Exact => x.into_iter().map(|v| v.max(0.0)).collect(),
                Relu::Polynomial => {
                    let d = f64::from(1u32 << log_divisor(name)?);
                    x.into_iter().map(|v| 0.5 * (v + v * sign(v / d))).collect()
                }
            })
        };
        let conv = |c: &Conv, x: &[f64], side: usize, stride: usize| {
            let mut y = conv2d(x, &c.weights, c.in_channels, c.out_channels, side, side, c.kernel, stride);
            let plane = y.len() / c.out_channels;
            for (t, v) in y.iter_mut().enumerate() {
                *v += c.bias[t / plane];
            }
            y
        };
        let mut x = activate(conv(&self.stem, image, 32, 1), "stem")?;
        let mut side = 32;
        for block in &self.blocks {
            let h = activate(conv(&block.conv1, &x, side, block.stride), &format!("{}.r1", block.name))?;
            let out_side = side / block.stride;
            let mut y = conv(&block.conv2, &h, out_side, 1);
            for (v, s) in y.iter_mut().zip(shortcut(&x, block.conv1.in_channels, side, block.stride)) {
                *v += s;
            }
            x = activate(y, &format!("{}.r2", block.name))?;
            side = out_side;
        }
        let channels = x.len() / (side * side);
        let pooled: Vec<f64> = x
            .chunks_exact(side * side)
            .map(|c| c.iter().sum::<f64>() / (side * side) as f64)
            .collect();
        Ok((0..self.fc_bias.len())
            .map(|class| {
                self.fc_bias[class]
                    + (0..channels)
                        .map(|q| self.fc_weight[class * channels + q] * pooled[q])
                        .sum::<f64>()
            })
            .collect())
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

/// The first ten images of the CIFAR-10 test batch, in its binary format.
const SAMPLE: &[u8] = include_bytes!("../data/test_images.bin");

/// The first `limit` images of the CIFAR-10 test batch, `[3][32][32]` normalized as the training did, with their labels:
/// from `test_batch.bin` at `$CIFAR10_TEST_BATCH`, or else from the ten built into this crate.
pub fn test_images(limit: usize) -> Result<Vec<(Vec<f64>, usize)>> {
    const MEAN: [f64; 3] = [0.4914, 0.4822, 0.4465];
    const STD: [f64; 3] = [0.2023, 0.1994, 0.2010];
    let read;
    let raw = match std::env::var_os("CIFAR10_TEST_BATCH") {
        Some(path) => {
            read = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            &read[..]
        }
        None => {
            ensure!(
                limit <= SAMPLE.len() / 3073,
                "ten images are built in: the crate's README gets the others"
            );
            SAMPLE
        }
    };
    ensure!(raw.len().is_multiple_of(3073), "not a CIFAR-10 binary batch");
    Ok(raw
        .chunks_exact(3073)
        .take(limit)
        .map(|record| {
            let image = record[1..]
                .iter()
                .enumerate()
                .map(|(t, &p)| (f64::from(p) / 255.0 - MEAN[t / 1024]) / STD[t / 1024])
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
