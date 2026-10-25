use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GetTensorKey};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, PowerBasisHelper},
    layouts::CKKSCiphertextOwned,
    polynomial::ComplexBSGSPolynomial,
};

/// Backend contract of [`CKKSPolynomialEvaluationOps`](crate::api::CKKSPolynomialEvaluationOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSPolynomialEvaluationImpl:
    Backend
    + super::CKKSCopyImpl
    + super::CKKSMulImpl
    + super::CKKSPow2Impl
    + super::CKKSSubImpl
    + poulpy_core::oep::GLWENormalizeImpl
{
    /// Reproduces
    /// [`ckks_eval_poly_real_const_coeffs_from_power_basis_reference`](crate::reference::polynomial_evaluation::PolynomialEvaluationReference::ckks_eval_poly_real_const_coeffs_from_power_basis_reference)
    /// bit for bit; free in kernels and scratch use. Its scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        module: &Module<Self>,
        res: &mut R,
        power_basis: &G,
        poly: &B,
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

    /// Default body: `ckks_eval_poly_real_const_coeffs_derived`, the folded input of the polynomial's input transform
    /// (an allocated copy through `ckks_copy` for the identity, `ckks_square_into` for the square forms,
    /// `ckks_square_into`, `ckks_double_into` and `ckks_sub_one_assign` for the Chebyshev `T2` forms), a power basis of
    /// it populated to the polynomial's degree, log split and parity through the selected operations, then
    /// [`Self::ckks_eval_poly_real_const_coeffs_from_power_basis`]. The two times-input transforms end with
    /// `ckks_mul_assign` of `dst` by `src`.
    fn ckks_eval_poly_real_const_coeffs<R, S, B, H>(
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
}

/// Backend contract of [`CKKSComplexPolynomialEvaluationOps`](crate::api::CKKSComplexPolynomialEvaluationOps):
/// complex-coefficient polynomial evaluation (`re + i·im`), standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSComplexPolynomialEvaluationImpl: CKKSPolynomialEvaluationImpl + super::CKKSImagImpl {
    /// Reproduces
    /// [`ckks_eval_poly_complex_const_coeffs_from_power_basis_reference`](crate::reference::polynomial_evaluation::PolynomialEvaluationReference::ckks_eval_poly_complex_const_coeffs_from_power_basis_reference)
    /// bit for bit; free in kernels and scratch use. Its scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
        module: &Module<Self>,
        res: &mut R,
        power_basis: &G,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<Self> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<Self, A>,
        H: GetTensorKey<Self>;

    /// Default body: `ckks_eval_poly_complex_const_coeffs_derived`. Real and imaginary parts with different input
    /// transforms are a refusal. Then the folded input of the polynomial's input transform (an allocated copy through
    /// `ckks_copy` for the identity, `ckks_square_into` for the square forms, `ckks_square_into`, `ckks_double_into`
    /// and `ckks_sub_one_assign` for the Chebyshev `T2` forms), a power basis of it populated to the polynomial's
    /// degree, log split and parity through the selected operations, taken from the real part, then
    /// [`Self::ckks_eval_poly_complex_const_coeffs_from_power_basis`]. The two times-input transforms end with
    /// `ckks_mul_assign` of `dst` by `src`.
    fn ckks_eval_poly_complex_const_coeffs<R, S, C, H>(
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

/// Selects the reference implementation of `CKKSPolynomialEvaluationImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_polynomial_evaluation_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSPolynomialEvaluationImpl for $be {
            fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                power_basis: &G,
                poly: &B,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::SetBSGSMeta,
                B: $crate::api::BSGSPolynomialInfos<Self>,
                B::Coeffs: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + poulpy_core::layouts::BSGSMeta,
                G: $crate::api::PowerBasisHelper<Self, A>,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::polynomial_evaluation::PolynomialEvaluationReference::ckks_eval_poly_real_const_coeffs_from_power_basis_reference::<R, B, A, G, H>(
                    module, res, power_basis, poly, tsk, scratch,
                )
            }
        }
    };
}

/// Selects the reference implementation of `CKKSComplexPolynomialEvaluationImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_complex_polynomial_evaluation_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSComplexPolynomialEvaluationImpl for $be {
            fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                power_basis: &G,
                poly: &$crate::polynomial::ComplexBSGSPolynomial<C>,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::SetBSGSMeta,
                C: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::GLWEInfos
                    + poulpy_core::layouts::BSGSMeta
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + poulpy_core::layouts::BSGSMeta,
                G: $crate::api::PowerBasisHelper<Self, A>,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::polynomial_evaluation::PolynomialEvaluationReference::ckks_eval_poly_complex_const_coeffs_from_power_basis_reference::<R, C, A, G, H>(
                    module, res, power_basis, poly, tsk, scratch,
                )
            }
        }
    };
}

pub use crate::{impl_ckks_complex_polynomial_evaluation_reference_full, impl_ckks_polynomial_evaluation_reference_full};
