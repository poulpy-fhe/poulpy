use anyhow::{Result, anyhow};
use num_traits::{Float, FloatConst};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::{
    CoeffsMeta, SetCKKSInfos,
    eval_lut::{cos_hermite_binary, trig_hermite_lut},
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
};

pub enum EncodedLut<P> {
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
        let bsgs = trig_hermite_lut(table).encode_bsgs_with(host_module, base2k, coeffs_meta, strategy)?;
        Ok(Self::General(bsgs))
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
        Ok(Self::Binary {
            cos,
            affine: affine_pt,
            log_interval_reduction,
        })
    }
}

impl<P> EncodedLut<P> {
    pub fn map<Q>(&self, mut f: impl FnMut(&P) -> Q) -> EncodedLut<Q> {
        match self {
            Self::General(bsgs) => EncodedLut::General(bsgs.map_baby_steps_ref(&mut f)),
            Self::Binary {
                cos,
                affine,
                log_interval_reduction,
            } => EncodedLut::Binary {
                cos: cos.map_baby_steps_ref(&mut f),
                affine: f(affine),
                log_interval_reduction: *log_interval_reduction,
            },
        }
    }

    pub fn general_series(&self) -> Option<&ComplexBSGSPolynomial<P>> {
        match self {
            Self::General(bsgs) => Some(bsgs),
            Self::Binary { .. } => None,
        }
    }

    pub fn consumed_bits(&self, log_delta: usize, coeff_log_delta: usize) -> usize {
        match self {
            Self::General(bsgs) => bsgs.re.consumed_bits(log_delta, coeff_log_delta),
            Self::Binary {
                cos,
                log_interval_reduction,
                ..
            } => cos.consumed_bits(log_delta, coeff_log_delta) + log_interval_reduction * log_delta + log_delta,
        }
    }
}
