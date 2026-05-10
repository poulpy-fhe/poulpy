use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef,
};

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos};

/// Polynomial evaluation basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// Standard monomial basis: {1, X, X², …}
    Monomial,
    /// Chebyshev first-kind basis: {T₀(X), T₁(X), T₂(X), …}
    Chebyshev,
}

pub trait BSGSPolynomialInfos<BE: Backend> {
    type Coeffs: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos;

    fn degree(&self) -> usize;
    fn baby_steps(&self) -> usize;
    fn baby_degree(&self, i: usize) -> usize;
    fn baby_step(&self, i: usize) -> &Self::Coeffs;

    fn basis(&self) -> Basis;
}

pub trait PowerBasisHelper<BE: Backend, A> {
    fn get(&self, power: usize) -> Result<&A>;
}

pub trait PolynomialEvaluation<BE: Backend> {
    fn ckks_eval_poly_const_coeffs<R, B, A, G, T>(
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
