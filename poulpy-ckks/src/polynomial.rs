use std::fmt::Debug;

use anyhow::{Result, anyhow};
use poulpy_core::layouts::{Base2K, GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive};

use crate::{
    CKKSInfos, CKKSMeta,
    api::BSGSPolynomialInfos,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

pub use poulpy_core::layouts::{
    BSGSPolynomial, Basis, DEFAULT_SPLIT_STRATEGY, Parity, Polynomial, SplitStrategy, split_degree,
};

/// CKKS encoding of a [`Polynomial`] into a [`BSGSPolynomial`] of plaintexts.
pub trait EncodeBSGS {
    /// Decomposes and encodes using [`DEFAULT_SPLIT_STRATEGY`].
    fn encode_bsgs(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
    ) -> Result<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>>;

    /// Decomposes and encodes using an explicit [`SplitStrategy`].
    fn encode_bsgs_with(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
        strategy: SplitStrategy,
    ) -> Result<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>>;
}

impl<F> EncodeBSGS for Polynomial<F>
where
    F: Float + FloatConst + FromPrimitive + Debug + CKKSScalar,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
{
    fn encode_bsgs(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
    ) -> Result<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>> {
        self.encode_bsgs_with(module, base2k, coeff_meta, DEFAULT_SPLIT_STRATEGY)
    }

    fn encode_bsgs_with(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
        strategy: SplitStrategy,
    ) -> Result<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>> {
        let mut step_idx = 0usize;
        self.decompose_bsgs_with(strategy, |baby_coeffs| {
            let mut pt = module.ckks_pt_coeffs_alloc(baby_coeffs.len(), base2k, coeff_meta);
            pt.encode_host_floats(baby_coeffs)
                .map_err(|e| anyhow!("encode_bsgs: step {step_idx}: {e}"))?;
            step_idx += 1;
            Ok(pt)
        })
    }
}

impl<BE: Backend, C> BSGSPolynomialInfos<BE> for BSGSPolynomial<C>
where
    C: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    type Coeffs = C;

    fn degree(&self) -> usize {
        BSGSPolynomial::degree(self)
    }

    fn baby_steps(&self) -> usize {
        BSGSPolynomial::baby_steps(self).len()
    }

    fn baby_step(&self, i: usize) -> &Self::Coeffs {
        BSGSPolynomial::baby_step(self, i)
    }

    fn basis(&self) -> Basis {
        BSGSPolynomial::basis(self)
    }

    fn parity(&self) -> Parity {
        BSGSPolynomial::parity(self)
    }
}
