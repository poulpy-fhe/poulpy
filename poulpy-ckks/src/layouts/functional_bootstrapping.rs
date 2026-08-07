use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::{
    CoeffsMeta, SetCKKSInfos,
    eval_lut::{cos_hermite_binary, trig_hermite_lut},
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
};

pub struct EncodedLut<P> {
    kind: EncodedLutKind<P>,
    log_msg_ratio: usize,
}

pub(crate) enum EncodedLutKind<P> {
    General(ComplexBSGSPolynomial<P>),
    Binary {
        cos: BSGSPolynomial<P>,
        affine: P,
        log_interval_reduction: usize,
    },
}

impl EncodedLut<CKKSPlaintext<Vec<u8>>> {
    pub fn general<F>(
        host_module: &Module<HostBytesBackend>,
        table: &[F],
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
        strategy: SplitStrategy,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
    {
        let log_msg_ratio = table_log_msg_ratio(table.len())?;
        let bsgs = trig_hermite_lut(table).encode_bsgs_with(host_module, base2k, coeffs_meta, strategy)?;
        Ok(Self {
            kind: EncodedLutKind::General(bsgs),
            log_msg_ratio,
        })
    }

    #[allow(clippy::too_many_arguments)]
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
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
    {
        let (cos_poly, affine) = cos_hermite_binary(f0, f1, degree, k_interval, log_interval_reduction)?;
        let cos = <Polynomial<F> as EncodeBSGS>::encode_bsgs_with(&cos_poly, host_module, base2k, coeffs_meta, strategy)?;
        let mut affine_pt = host_module.ckks_pt_coeffs_alloc(2, base2k, coeffs_meta.k);
        affine_pt.set_meta(coeffs_meta.meta);
        affine_pt.encode_host_floats(&affine).map_err(|e| anyhow!("affine: {e}"))?;
        Ok(Self {
            kind: EncodedLutKind::Binary {
                cos,
                affine: affine_pt,
                log_interval_reduction,
            },
            log_msg_ratio: 1,
        })
    }
}

impl<P> EncodedLut<P> {
    pub fn map<Q>(&self, mut f: impl FnMut(&P) -> Q) -> EncodedLut<Q> {
        let kind = match &self.kind {
            EncodedLutKind::General(series) => EncodedLutKind::General(series.map_baby_steps_ref(&mut f)),
            EncodedLutKind::Binary {
                cos,
                affine,
                log_interval_reduction,
            } => EncodedLutKind::Binary {
                cos: cos.map_baby_steps_ref(&mut f),
                affine: f(affine),
                log_interval_reduction: *log_interval_reduction,
            },
        };
        EncodedLut {
            kind,
            log_msg_ratio: self.log_msg_ratio,
        }
    }

    pub fn general_series(&self) -> Option<&ComplexBSGSPolynomial<P>> {
        match &self.kind {
            EncodedLutKind::General(series) => Some(series),
            EncodedLutKind::Binary { .. } => None,
        }
    }

    pub fn log_msg_ratio(&self) -> usize {
        self.log_msg_ratio
    }

    pub fn consumed_bits(&self, log_delta: usize, coeff_log_delta: usize) -> usize {
        match &self.kind {
            EncodedLutKind::General(series) => series.re.consumed_bits(log_delta, coeff_log_delta),
            EncodedLutKind::Binary {
                cos,
                log_interval_reduction,
                ..
            } => cos.consumed_bits(log_delta, coeff_log_delta) + log_interval_reduction * log_delta + log_delta,
        }
    }

    pub(crate) fn kind(&self) -> &EncodedLutKind<P> {
        &self.kind
    }
}

fn table_log_msg_ratio(len: usize) -> Result<usize> {
    ensure!(len.is_power_of_two(), "LUT length must be a nonzero power of two, got {len}");
    Ok(len.ilog2() as usize)
}

#[cfg(test)]
mod tests {
    use super::table_log_msg_ratio;

    #[test]
    fn message_ratio_is_derived_from_table_length() {
        assert_eq!(table_log_msg_ratio(4).unwrap(), 2);
        assert_eq!(table_log_msg_ratio(8).unwrap(), 3);
        assert!(table_log_msg_ratio(0).is_err());
        assert!(table_log_msg_ratio(3).is_err());
    }
}
