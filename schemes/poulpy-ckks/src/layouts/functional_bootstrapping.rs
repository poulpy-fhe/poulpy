//! Encoded lookup tables for CKKS functional bootstrapping.
//!
//! Table entries are indexed by integer messages `0..p`. A general LUT uses
//! trigonometric Hermite interpolation after EvalMod; a binary LUT uses a
//! cosine approximation with its output map folded in and avoids EvalMod.
//! Both forms plug into an S2C-first bootstrapping recipe.

use std::num::NonZeroUsize;

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst};
use poulpy_core::api::TransferInto;
use poulpy_core::layouts::Base2K;
use poulpy_core::layouts::{GLWEPlaintextAlloc, ModuleCoreAlloc};
use poulpy_hal::layouts::{Backend, HostBytesBackend, HostStaged, Module, Standard};

use crate::{
    CKKSInfos, CoeffsMeta, SetCKKSInfos, SlotsKind,
    eval_lut::{cos_hermite_binary, trig_hermite_lut},
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextOwned, CKKSPlaintextVecHostCodec, CKKSScalar},
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
};

/// A lookup table encoded as plaintext polynomial coefficients.
///
/// Inputs contain integer messages `m`.
/// Functional bootstrapping expects an input satisfying
/// `log_budget - s2c_consumed_bits == ceil(log2(p))`, where `p` is the table size.
/// Its output scale is the input scale plus `ceil(log2(p))`.
pub struct EncodedLut<P> {
    kind: EncodedLutKind<P>,
    log_msg_ratio: usize,
    message_modulus: usize,
}

pub(crate) enum EncodedLutKind<P> {
    General(ComplexBSGSPolynomial<P>),
    Binary {
        cos: BSGSPolynomial<P>,
        /// Absent without interval reduction.
        chain: Option<DoubleAngleChain<P>>,
    },
}

/// The double-angle steps of a binary LUT, at least one.
pub(crate) struct DoubleAngleChain<P> {
    /// One constant per step: subtracted from the square in every step but
    /// the last, then `f1`.
    consts: P,
    steps: NonZeroUsize,
    /// Whether the last square is subtracted from `f1` instead of added to it.
    negate: bool,
}

impl<P> DoubleAngleChain<P> {
    pub(crate) fn consts(&self) -> &P {
        &self.consts
    }

    pub(crate) fn steps(&self) -> NonZeroUsize {
        self.steps
    }

    pub(crate) fn negate(&self) -> bool {
        self.negate
    }

    fn map<Q>(&self, f: impl FnOnce(&P) -> Q) -> DoubleAngleChain<Q> {
        DoubleAngleChain {
            consts: f(&self.consts),
            steps: self.steps,
            negate: self.negate,
        }
    }
}

impl EncodedLut<CKKSPlaintextOwned<HostBytesBackend>> {
    /// Encodes a table of at least two entries with period `p = table.len()` and degree `p - 1`.
    ///
    /// Configure the plan with [`super::BootstrappingPlan::with_functional_bootstrap`].
    /// Evaluation returns `table[m mod p]`.
    /// This form requires EvalMod at evaluation time.
    pub fn general<F>(
        host_module: &Module<HostBytesBackend>,
        table: &[F],
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
        strategy: SplitStrategy,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst,
    {
        ensure!(table.len() >= 2, "LUT length must be at least two");
        let log_msg_ratio = table_log_msg_ratio(table.len())?;
        let bsgs = trig_hermite_lut(table)?.encode_bsgs_with(host_module, base2k, coeffs_meta, strategy)?;
        Ok(Self {
            kind: EncodedLutKind::General(bsgs),
            log_msg_ratio,
            message_modulus: table.len(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Encodes the binary table `[f0, f1]` using a cosine approximation.
    ///
    /// `degree` controls the approximation polynomial, `k_interval` is its
    /// initial interpolation interval, and `log_interval_reduction` is the
    /// number of double-angle reduction steps applied after evaluation.
    /// The map from the cosine to `[f0, f1]` is folded into the polynomial
    /// and the double-angle constants, so it consumes no budget.
    pub fn binary<F>(
        host_module: &Module<HostBytesBackend>,
        f0: F,
        f1: F,
        degree: usize,
        k_interval: usize,
        log_interval_reduction: usize,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
        strategy: SplitStrategy,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst,
    {
        let binary = cos_hermite_binary(f0, f1, degree, k_interval, log_interval_reduction)?;
        let cos = <Polynomial<F> as EncodeBSGS>::encode_bsgs_with(&binary.poly, host_module, base2k, coeffs_meta, strategy)?;
        let chain = match NonZeroUsize::new(binary.consts.len()) {
            None => None,
            Some(steps) => {
                let mut consts = host_module.ckks_pt_coeffs_alloc(steps.get(), base2k, coeffs_meta.k);
                let mut meta = coeffs_meta.meta;
                meta.slots = SlotsKind::Real;
                consts.set_meta(meta);
                consts
                    .encode_host_floats(&binary.consts)
                    .map_err(|e| anyhow!("double-angle constants: {e}"))?;
                Some(DoubleAngleChain {
                    consts,
                    steps,
                    negate: binary.negate,
                })
            }
        };
        Ok(Self {
            kind: EncodedLutKind::Binary { cos, chain },
            log_msg_ratio: 1,
            message_modulus: 2,
        })
    }

    /// Uploads every encoded coefficient to `module`'s backend while
    /// preserving the LUT kind and message-ratio metadata.
    pub fn transfer_to<BE>(&self, module: &Module<BE>) -> EncodedLut<CKKSPlaintextOwned<BE>>
    where
        BE: Backend<Ring = Standard> + HostStaged,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
    {
        self.map(|pt| {
            let mut inner = module.glwe_plaintext_alloc_from_infos(&pt.inner);
            pt.inner.transfer_into(&mut inner);
            CKKSPlaintext::from_inner(inner, pt.meta())
        })
    }
}

impl<P> EncodedLut<P> {
    fn map<Q>(&self, mut f: impl FnMut(&P) -> Q) -> EncodedLut<Q> {
        let kind = match &self.kind {
            EncodedLutKind::General(series) => EncodedLutKind::General(series.map_baby_steps_ref(&mut f)),
            EncodedLutKind::Binary { cos, chain } => EncodedLutKind::Binary {
                cos: cos.map_baby_steps_ref(&mut f),
                chain: chain.as_ref().map(|chain| chain.map(&mut f)),
            },
        };
        EncodedLut {
            kind,
            log_msg_ratio: self.log_msg_ratio,
            message_modulus: self.message_modulus,
        }
    }

    pub(crate) fn general_series(&self) -> Option<&ComplexBSGSPolynomial<P>> {
        match &self.kind {
            EncodedLutKind::General(series) => Some(series),
            EncodedLutKind::Binary { .. } => None,
        }
    }

    /// Returns `ceil(log2(p))`, where `p` is the number of table entries.
    pub fn log_msg_ratio(&self) -> usize {
        self.log_msg_ratio
    }

    /// The exact message modulus (number of table entries).
    pub fn message_modulus(&self) -> usize {
        self.message_modulus
    }

    /// Multiplicative budget consumed at the given ciphertext scale.
    pub fn consumed_bits(&self, log_delta: usize) -> usize
    where
        P: CKKSInfos,
    {
        let coeff_log_delta = self.coeffs().log_delta();
        match &self.kind {
            EncodedLutKind::General(series) => series.re.consumed_bits(log_delta, coeff_log_delta),
            EncodedLutKind::Binary { cos, chain } => {
                cos.consumed_bits(log_delta, coeff_log_delta) + chain.as_ref().map_or(0, |chain| chain.steps().get()) * log_delta
            }
        }
    }

    pub(crate) fn requires_eval_mod(&self) -> bool {
        matches!(&self.kind, EncodedLutKind::General(_))
    }

    pub(crate) fn coeffs(&self) -> &P {
        match &self.kind {
            EncodedLutKind::General(series) => &series.re.baby_steps()[0],
            EncodedLutKind::Binary { cos, .. } => &cos.baby_steps()[0],
        }
    }

    pub(crate) fn kind(&self) -> &EncodedLutKind<P> {
        &self.kind
    }
}

fn table_log_msg_ratio(len: usize) -> Result<usize> {
    ensure!(len > 0, "LUT length must be nonzero");
    let binary_span = len
        .checked_next_power_of_two()
        .ok_or_else(|| anyhow!("LUT length {len} is too large"))?;
    Ok(binary_span.ilog2() as usize)
}

#[cfg(test)]
mod tests {
    use super::{EncodedLut, table_log_msg_ratio};
    use crate::{CoeffsMeta, polynomial::SplitStrategy};
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    #[test]
    fn general_lut_preserves_exact_modulus_and_degree() {
        let module = Module::<HostBytesBackend>::new(16);
        for p in [2, 3, 4, 5, 6, 7, 8, 9, 17, 33] {
            let table: Vec<_> = (0..p).map(|m| ((m * m + 3 * m + 2) % 7) as f64).collect();
            let lut = EncodedLut::general(
                &module,
                &table,
                32usize.into(),
                CoeffsMeta::from_delta_budget(55, 32),
                SplitStrategy::MinDepth,
            )
            .unwrap();
            assert_eq!(lut.message_modulus(), p);
            let series = lut.general_series().unwrap();
            assert_eq!(series.re.degree(), p - 1);
            assert_eq!(series.im.degree(), p - 1);
            let transferred = lut.transfer_to(&module);
            assert_eq!(transferred.message_modulus(), p);
        }
    }

    #[test]
    fn message_ratio_is_derived_from_table_length() {
        assert_eq!(table_log_msg_ratio(1).unwrap(), 0);
        assert_eq!(table_log_msg_ratio(2).unwrap(), 1);
        assert_eq!(table_log_msg_ratio(3).unwrap(), 2);
        assert_eq!(table_log_msg_ratio(4).unwrap(), 2);
        assert_eq!(table_log_msg_ratio(5).unwrap(), 3);
        assert_eq!(table_log_msg_ratio(6).unwrap(), 3);
        assert_eq!(table_log_msg_ratio(7).unwrap(), 3);
        assert_eq!(table_log_msg_ratio(8).unwrap(), 3);
        assert_eq!(table_log_msg_ratio(9).unwrap(), 4);
        assert!(table_log_msg_ratio(0).is_err());
        assert!(table_log_msg_ratio(usize::MAX).is_err());
    }
}
