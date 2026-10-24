//! Large-interval normalization from Cheon et al., ePrint 2022/280.

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive};
use poulpy_core::layouts::{Base2K, LWEInfos};
use poulpy_hal::layouts::{HostBytesBackend, Module};

use poulpy_ckks::{
    CKKSInfos, CKKSLayout, SetCKKSInfos,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

/// Compression factor `L` (experimental, from the reference).
pub const COMPRESSION_L: f64 = 2.45;

/// Prepared interval-normalization constants.
pub struct IntervalNorm<P> {
    /// Per-step constants.
    pub consts: P,
    /// Number of compression steps `⌈log_L(Max)⌉`.
    pub n: usize,
    /// Scale the constants were encoded at.
    pub coeff_log_delta: usize,
}

impl IntervalNorm<CKKSPlaintext<Vec<u8>, i64>> {
    /// Builds the constants for compressing `[−max, max]`.
    pub fn from_max<F>(
        max: f64,
        base2k: Base2K,
        coeff_meta: CKKSLayout,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            max.is_finite() && max > 0.0,
            "interval_norm: max must be positive and finite"
        );
        let n = (max.log2() / COMPRESSION_L.log2()).ceil().max(1.0) as usize;
        let c: Vec<F> = (0..n)
            .map(|i| {
                let den = 27.0 * COMPRESSION_L.powi(2 * (n - 1 - i) as i32);
                F::from_f64(4.0 / den)
                    .ok_or_else(|| anyhow!("interval_norm: c_{i} not representable"))
            })
            .collect::<Result<_>>()?;
        let mut consts = module.ckks_pt_coeffs_alloc(n, base2k, coeff_meta.k());
        consts.set_meta(coeff_meta.meta());
        consts
            .encode_host_floats(&c)
            .map_err(|e| anyhow!("interval_norm: {e}"))?;
        Ok(Self {
            consts,
            n,
            coeff_log_delta: coeff_meta.meta.log_delta,
        })
    }
}

impl<P> IntervalNorm<P> {
    /// Consumed modulus bits.
    pub fn consumed_bits(&self, log_delta: usize) -> usize {
        2 * self.n * log_delta
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        2 * self.n
    }

    /// Maps prepared plaintexts.
    pub fn map_plaintexts<Q>(self, f: impl FnOnce(&P) -> Q) -> IntervalNorm<Q> {
        IntervalNorm {
            consts: f(&self.consts),
            n: self.n,
            coeff_log_delta: self.coeff_log_delta,
        }
    }
}
