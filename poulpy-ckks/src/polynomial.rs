use std::fmt::Debug;

use anyhow::{Result, anyhow};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive};

use crate::{
    CKKSMeta,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

pub use poulpy_core::layouts::{
    BSGSPolynomial, Basis, DEFAULT_SPLIT_STRATEGY, Parity, Polynomial, SplitStrategy, change_of_basis, evaluate_coeffs,
    split_degree,
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

/// A plaintext polynomial with complex coefficients `re[k] + i·im[k]`.
///
/// `re[k]`/`im[k]` are the real/imaginary parts of the degree-`k` term
/// (monomial basis) or of `Tₖ(x)` (Chebyshev basis).
pub struct ComplexPolynomial<F> {
    pub basis: Basis,
    pub re: Vec<F>,
    pub im: Vec<F>,
    /// Lower/upper bounds of the approximation interval `[a, b]`, expressed in the
    /// [`change_of_basis`] variable (Chebyshev maps `[a, b]→[-1, 1]`; Monomial is
    /// identity). Defaults to `[-1, 1]`; set via [`ComplexPolynomial::with_interval`].
    pub a: F,
    pub b: F,
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
    /// Constructs a complex polynomial, padding `re`/`im` to equal length. The
    /// interval defaults to `[-1, 1]`; set it with [`Self::with_interval`].
    pub fn new(basis: Basis, mut re: Vec<F>, mut im: Vec<F>) -> Self {
        let len = re.len().max(im.len());
        re.resize(len, F::zero());
        im.resize(len, F::zero());
        let one = F::one();
        Self {
            basis,
            re,
            im,
            a: -one,
            b: one,
        }
    }

    /// Sets the approximation interval `[a, b]` (see the [`a`](Self::a)/[`b`](Self::b)
    /// fields).
    pub fn with_interval(mut self, a: F, b: F) -> Self {
        self.a = a;
        self.b = b;
        self
    }

    /// The approximation interval `[a, b]`.
    pub fn interval(&self) -> (F, F) {
        (self.a, self.b)
    }

    /// Affine change of basis `(u, w)` mapping `x ∈ [a, b]` to the coefficient
    /// variable (`y = u·x + w`); see [`change_of_basis`].
    pub fn change_of_basis(&self) -> (F, F) {
        change_of_basis(self.basis, self.a, self.b)
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
        let re_poly = Polynomial::new_with_parity(self.basis, self.re.clone(), parity).with_interval(self.a, self.b);
        let im_poly = Polynomial::new_with_parity(self.basis, self.im.clone(), parity).with_interval(self.a, self.b);
        let re = re_poly.encode_bsgs_with(module, base2k, coeff_meta, strategy)?;
        let im = im_poly.encode_bsgs_with(module, base2k, coeff_meta, strategy)?;
        Ok(ComplexBSGSPolynomial { re, im })
    }
}

impl<F> ComplexPolynomial<F>
where
    F: Float,
{
    /// Evaluates both components at real `x`, returning `(re, im)`.
    ///
    /// Mirrors the bare per-component evaluation the homomorphic circuit
    /// performs (no interval remapping); see [`Polynomial::evaluate`].
    pub fn evaluate(&self, x: F) -> (F, F) {
        (
            evaluate_coeffs(self.basis, &self.re, x),
            evaluate_coeffs(self.basis, &self.im, x),
        )
    }
}

impl<C> ComplexBSGSPolynomial<C> {
    /// The approximation interval `[a, b]` (shared by both `re`/`im` halves).
    pub fn interval(&self) -> (f64, f64) {
        self.re.interval()
    }

    /// Affine change of basis `(u, w)` mapping `x ∈ [a, b]` to the coefficient
    /// variable (`y = u·x + w`); see [`change_of_basis`].
    pub fn change_of_basis(&self) -> (f64, f64) {
        self.re.change_of_basis()
    }

    /// Rebuilds by mapping borrowed baby-step coefficients of both parts.
    pub fn map_baby_steps_ref<D>(&self, mut f: impl FnMut(&C) -> D) -> ComplexBSGSPolynomial<D> {
        ComplexBSGSPolynomial {
            re: self.re.map_baby_steps_ref(&mut f),
            im: self.im.map_baby_steps_ref(&mut f),
        }
    }
}
