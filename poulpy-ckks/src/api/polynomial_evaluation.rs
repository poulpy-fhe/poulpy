use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Polynomial evaluation basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// Standard monomial basis: {1, X, X², …}
    Monomial,
    /// Chebyshev first-kind basis: {T₀(X), T₁(X), T₂(X), …}
    Chebyshev,
}

/// Symmetry class of a polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    /// No symmetry: all powers may be non-zero.
    Full,
    /// Even polynomial: only even-degree coefficients are non-zero.
    Even,
    /// Odd polynomial: only odd-degree coefficients are non-zero.
    Odd,
}

pub trait BSGSPolynomialInfos<BE: Backend> {
    type Coeffs: GLWEToBackendRef<BE> + CKKSCtBounds;
    fn degree(&self) -> usize;
    fn baby_steps(&self) -> usize;
    fn baby_step(&self, i: usize) -> &Self::Coeffs;
    fn basis(&self) -> Basis;
    fn parity(&self) -> Parity;
}

pub trait BabyStep<BE: Backend> {
    type Value: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
    fn degree(&self) -> usize;
    fn get(&self) -> &Self::Value;
    fn get_mut(&mut self) -> &mut Self::Value;
}

pub trait PowerBasisHelper<BE: Backend, A> {
    fn get(&self, power: usize) -> Result<&A>;
}

pub trait PolynomialEvaluation<BE: Backend> {
    fn ckks_eval_baby_step<R, C, A, G>(
        &self,
        res: &mut R,
        coeffs: &C,
        parity: Parity,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>;

    fn ckks_eval_giant_steps<R, B, A, G, T>(
        &self,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, T>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: BSGSPolynomialInfos<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}
