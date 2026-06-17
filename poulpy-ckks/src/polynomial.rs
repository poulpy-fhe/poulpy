use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive};

use crate::{
    CKKSMeta,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

pub use poulpy_core::layouts::{BSGSPolynomial, Basis, DEFAULT_SPLIT_STRATEGY, Parity, Polynomial, SplitStrategy, split_degree};

/// Adaptive-precision split of a Chebyshev polynomial: low-degree terms at the
/// working scale, high-degree terms compensated by `2^drop` and encoded `drop`
/// bits below. See
/// [`EncodeBSGS::encode_bsgs_adaptive`].
pub struct AdaptiveBSGS<C> {
    pub low: BSGSPolynomial<C>,
    pub high: BSGSPolynomial<C>,
    pub drop: usize,
}

impl<C> AdaptiveBSGS<C> {
    /// Rebuilds by mapping borrowed baby-step coefficients of both branches.
    pub fn map_baby_steps_ref<D>(&self, mut f: impl FnMut(&C) -> D) -> AdaptiveBSGS<D> {
        AdaptiveBSGS {
            low: self.low.map_baby_steps_ref(&mut f),
            high: self.high.map_baby_steps_ref(&mut f),
            drop: self.drop,
        }
    }
}

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

    /// Splits a Chebyshev polynomial at degree `split` for modulus-preserving
    /// adaptive evaluation: low branch `[0, split)` at `coeff_meta`, high branch
    /// `[split, degree]` multiplied by `2^drop` and encoded `drop` bits below.
    /// Errors unless the basis is Chebyshev, `0 < split <= degree`, and
    /// `0 < drop < coeff_meta.log_delta`.
    fn encode_bsgs_adaptive(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
        split: usize,
        drop: usize,
        strategy: SplitStrategy,
    ) -> Result<AdaptiveBSGS<CKKSPlaintext<Vec<u8>>>>;
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

    fn encode_bsgs_adaptive(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
        split: usize,
        drop: usize,
        strategy: SplitStrategy,
    ) -> Result<AdaptiveBSGS<CKKSPlaintext<Vec<u8>>>> {
        let degree = self.degree();
        ensure!(
            self.basis == Basis::Chebyshev,
            "encode_bsgs_adaptive: requires the Chebyshev basis"
        );
        ensure!(
            split > 0 && split <= degree,
            "encode_bsgs_adaptive: split must be in (0, degree={degree}]"
        );
        ensure!(
            drop > 0 && drop < coeff_meta.log_delta,
            "encode_bsgs_adaptive: drop must be in (0, coeff_meta.log_delta={})",
            coeff_meta.log_delta
        );

        // Low branch is truncated (shallower). High branch keeps the full degree
        // with low terms zeroed, and its small high-degree coefficients are
        // compensated before lower-scale encoding so their integer precision stays
        // comparable to the full-scale branch.
        let compensation = F::from_f64(2.0).expect("f64 → scalar").powi(drop as i32);
        let mut low_coeffs = self.coeffs.clone();
        low_coeffs.truncate(split);
        let mut high_coeffs = self.coeffs.clone();
        for c in high_coeffs.iter_mut().take(split) {
            *c = F::zero();
        }
        for c in high_coeffs.iter_mut().skip(split) {
            *c = *c * compensation;
        }

        let high_meta = CKKSMeta {
            log_delta: coeff_meta.log_delta - drop,
            log_budget: coeff_meta.log_budget,
            log_sparsity: coeff_meta.log_sparsity,
        };
        let low = Polynomial::new_with_parity(self.basis, low_coeffs, self.parity)
            .encode_bsgs_with(module, base2k, coeff_meta, strategy)?;
        let high = Polynomial::new_with_parity(self.basis, high_coeffs, self.parity)
            .encode_bsgs_with(module, base2k, high_meta, strategy)?;
        Ok(AdaptiveBSGS { low, high, drop })
    }
}

/// A plaintext polynomial with complex coefficients `re[k] + i·im[k]`.
///
/// `re[k]`/`im[k]` are the real/imaginary parts of the degree-`k` term
/// (monomial basis) or of `Tₖ(x)` (Chebyshev basis).
pub struct ComplexPolynomial<F> {
    pub basis: Basis,
    pub re: Vec<F>,
    pub im: Vec<F>,
}

/// A complex polynomial decomposed for BSGS evaluation.
///
/// `re`/`im` share an identical baby-step schedule (same degree split and
/// parity), so the two halves align step-for-step during evaluation.
pub struct ComplexBSGSPolynomial<C> {
    pub re: BSGSPolynomial<C>,
    pub im: BSGSPolynomial<C>,
}

impl<F> ComplexPolynomial<F>
where
    F: Float + FloatConst + FromPrimitive + Debug + CKKSScalar,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
{
    /// Constructs a complex polynomial, padding `re`/`im` to equal length.
    pub fn new(basis: Basis, mut re: Vec<F>, mut im: Vec<F>) -> Self {
        let len = re.len().max(im.len());
        re.resize(len, F::zero());
        im.resize(len, F::zero());
        Self { basis, re, im }
    }

    /// Combined parity: index `k` is present if `re[k] != 0 || im[k] != 0`.
    fn combined_parity(&self) -> Parity {
        let present = |k: usize| self.re[k] != F::zero() || self.im[k] != F::zero();
        if (0..self.re.len()).all(|k| k.is_multiple_of(2) || !present(k)) {
            Parity::Even
        } else if (0..self.re.len()).all(|k| !k.is_multiple_of(2) || !present(k)) {
            Parity::Odd
        } else {
            Parity::Full
        }
    }

    /// Encodes both parts into a [`ComplexBSGSPolynomial`] using
    /// [`DEFAULT_SPLIT_STRATEGY`].
    pub fn encode_bsgs(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
    ) -> Result<ComplexBSGSPolynomial<CKKSPlaintext<Vec<u8>>>> {
        self.encode_bsgs_with(module, base2k, coeff_meta, DEFAULT_SPLIT_STRATEGY)
    }

    /// Encodes both parts with a shared parity and `strategy`, yielding two
    /// `BSGSPolynomial`s with identical baby-step structure.
    pub fn encode_bsgs_with(
        &self,
        module: &Module<HostBytesBackend>,
        base2k: Base2K,
        coeff_meta: CKKSMeta,
        strategy: SplitStrategy,
    ) -> Result<ComplexBSGSPolynomial<CKKSPlaintext<Vec<u8>>>> {
        let parity = self.combined_parity();
        let re_poly = Polynomial::new_with_parity(self.basis, self.re.clone(), parity);
        let im_poly = Polynomial::new_with_parity(self.basis, self.im.clone(), parity);
        let re = re_poly.encode_bsgs_with(module, base2k, coeff_meta, strategy)?;
        let im = im_poly.encode_bsgs_with(module, base2k, coeff_meta, strategy)?;
        Ok(ComplexBSGSPolynomial { re, im })
    }
}

impl<C> ComplexBSGSPolynomial<C> {
    /// Rebuilds by mapping borrowed baby-step coefficients of both parts.
    pub fn map_baby_steps_ref<D>(&self, mut f: impl FnMut(&C) -> D) -> ComplexBSGSPolynomial<D> {
        ComplexBSGSPolynomial {
            re: self.re.map_baby_steps_ref(&mut f),
            im: self.im.map_baby_steps_ref(&mut f),
        }
    }
}
