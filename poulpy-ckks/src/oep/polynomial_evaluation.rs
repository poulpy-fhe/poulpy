use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, PowerBasisHelper},
    layouts::CKKSCiphertextOwned,
    polynomial::ComplexBSGSPolynomial,
};

/// # Safety
///
/// Implementations must satisfy the contracts of the polynomial-evaluation
/// API, including the invariants of the underlying add/mul/copy kernels.
pub unsafe trait CKKSPolynomialEvaluationImpl:
    Backend + super::CKKSCopyImpl + super::CKKSMulImpl + super::CKKSPow2Impl + super::CKKSSubImpl
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_impl<R, B, A, G, H>(
        module: &Module<Self>,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<Self>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<Self> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<Self, A>,
        H: GetTensorKey<Self>;

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_impl<R, C, A, G, H>(
        module: &Module<Self>,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<Self> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<Self, A>,
        H: GetTensorKey<Self>;

    fn ckks_eval_poly_real_const_coeffs_impl<R, S, B, H>(
        module: &Module<Self>,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<Self> + CKKSCtBounds,
        B: BSGSPolynomialInfos<Self>,
        B::Coeffs: CKKSCtBounds,
        H: GetTensorKey<Self>,
        CKKSCiphertextOwned<Self>: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::oep::derived::polynomial_evaluation::ckks_eval_poly_real_const_coeffs_derived(module, dst, src, bsgs, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs_impl<R, S, C, H>(
        module: &Module<Self>,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<Self> + CKKSCtBounds,
        C: GLWEToBackendRef<Self> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<Self>,
        CKKSCiphertextOwned<Self>: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::oep::derived::polynomial_evaluation::ckks_eval_poly_complex_const_coeffs_derived(
            module, dst, src, poly, tsk, scratch,
        )
    }
}

#[macro_export]
macro_rules! impl_ckks_polynomial_evaluation_reference {
 ($be:ty) => {
 unsafe impl $crate::oep::CKKSPolynomialEvaluationImpl for $be {
fn ckks_eval_poly_real_const_coeffs_from_power_basis_impl<R, B, A, G, H>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> $crate::CKKSResult<()>
    where
        R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos + ::poulpy_core::layouts::SetBSGSMeta,
        B: $crate::api::BSGSPolynomialInfos<Self>,
        B::Coeffs: $crate::CKKSCtBounds,
        A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: $crate::api::PowerBasisHelper<Self, A>,
        H: ::poulpy_core::layouts::GetTensorKey<Self>,
    {
        $crate::reference::polynomial_evaluation::PolynomialEvaluationReference::ckks_eval_poly_real_const_coeffs_from_power_basis_reference::<R, B, A, G, H>(module, res, poly, power_basis, tsk, scratch)
    }
fn ckks_eval_poly_complex_const_coeffs_from_power_basis_impl<R, C, A, G, H>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        poly: &$crate::polynomial::ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> $crate::CKKSResult<()>
    where
        R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos + ::poulpy_core::layouts::SetBSGSMeta,
        C: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + poulpy_core::layouts::BSGSMeta + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
        A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: $crate::api::PowerBasisHelper<Self, A>,
        H: ::poulpy_core::layouts::GetTensorKey<Self>,
    {
        $crate::reference::polynomial_evaluation::PolynomialEvaluationReference::ckks_eval_poly_complex_const_coeffs_from_power_basis_reference::<R, C, A, G, H>(module,
            res,
            poly,
            power_basis,
            tsk,
            scratch,
        )
    }
 }
 };
}
pub use crate::impl_ckks_polynomial_evaluation_reference;
