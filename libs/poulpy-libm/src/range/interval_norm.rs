//! Large-interval normalization from Cheon et al., ePrint 2022/280.

use anyhow::{Context, Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive};
use poulpy_core::layouts::Base2K;
use poulpy_hal::{
    AlignedBuf,
    layouts::{HostBytesBackend, Module, Ring},
};

use poulpy_ckks::{
    CKKSInfos, CKKSLayout,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

use crate::plan::encode_constant;

/// Compression factor `L` (experimental, from the reference).
pub const COMPRESSION_L: f64 = 2.45;

/// Prepared interval-normalization constants.
pub struct IntervalNorm<P> {
    pub(crate) consts: Vec<P>,
    pub(crate) coeff_log_delta: usize,
}

impl<Rg: Ring> IntervalNorm<CKKSPlaintext<AlignedBuf, i64, Rg>> {
    /// Builds the constants for compressing `[−max, max]`.
    pub fn from_max<F>(max: f64, base2k: Base2K, coeff_meta: CKKSLayout, module: &Module<HostBytesBackend>) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive,
        CKKSPlaintext<AlignedBuf, i64, Rg>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(max.is_finite() && max > 0.0, "interval_norm: max must be positive and finite");
        let n = (max.log2() / COMPRESSION_L.log2()).ceil().max(1.0) as usize;
        let c: Vec<F> = (0..n)
            .map(|i| {
                let den = 27.0 * COMPRESSION_L.powi(2 * (n - 1 - i) as i32);
                F::from_f64(4.0 / den).ok_or_else(|| anyhow!("interval_norm: c_{i} not representable"))
            })
            .collect::<Result<_>>()?;
        let consts = c
            .into_iter()
            .map(|c| encode_constant(module, base2k, coeff_meta, c).context("interval_norm"))
            .collect::<Result<_>>()?;
        Ok(Self {
            consts,
            coeff_log_delta: coeff_meta.log_delta(),
        })
    }
}

impl<P> IntervalNorm<P> {
    /// Per-step constants.
    pub fn consts(&self) -> &[P] {
        &self.consts
    }

    /// Number of compression steps `ceil(log_L(max))`.
    pub fn steps(&self) -> usize {
        self.consts.len()
    }

    /// Scale the constants were encoded at.
    pub fn coeff_log_delta(&self) -> usize {
        self.coeff_log_delta
    }

    /// Consumed modulus bits.
    pub fn consumed_bits(&self, log_delta: usize) -> usize {
        2 * self.steps() * log_delta
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        2 * self.steps()
    }

    /// Maps prepared plaintexts.
    pub fn map_plaintexts<Q>(self, f: impl FnMut(&P) -> Q) -> IntervalNorm<Q> {
        IntervalNorm {
            consts: self.consts.iter().map(f).collect(),
            coeff_log_delta: self.coeff_log_delta,
        }
    }
}
