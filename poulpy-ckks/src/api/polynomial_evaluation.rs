use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
};

use crate::{CKKSCtBounds, SetCKKSInfos};

pub use poulpy_core::layouts::{BSGSPolynomialInfos, BabyStep, Basis, Parity, PowerBasisHelper};

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
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
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
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
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
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}
