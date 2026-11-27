use crate::CKKSResult as Result;
use crate::layouts::CKKSMaskedConstants;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GetTensorKey};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

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
pub unsafe trait CKKSPolynomialEvaluationImpl<Rg: Ring = Standard>:
    Backend
    + super::CKKSFactoredPolynomialEvaluationImpl<Rg>
    + super::CKKSCopyImpl<Rg>
    + super::CKKSMulImpl<Rg>
    + super::CKKSPow2Impl<Rg>
    + super::CKKSSubImpl<Rg>
    + poulpy_core::oep::GLWENormalizeImpl<Rg>
{
    /// Compiles the baby steps, prepared giant products and output fold into a factored program.
    /// The selected factored evaluator runs it within the shared BSGS scratch bound.
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        module: &Module<Self>,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<Self, Rg>,
        B::Coeffs: CKKSCtBounds + CKKSMaskedConstants<Self, Rg>,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<Self, A, Rg>,
        H: GetTensorKey<Self, Rg>,
    {
        crate::oep::derived::polynomial_evaluation::ckks_eval_poly_real_const_coeffs_from_power_basis_derived(
            module,
            res,
            power_basis,
            poly,
            tsk,
            scratch,
        )
    }

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
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        B: BSGSPolynomialInfos<Self, Rg>,
        B::Coeffs: CKKSCtBounds + CKKSMaskedConstants<Self, Rg>,
        H: GetTensorKey<Self, Rg>,
        CKKSCiphertextOwned<Self, Rg>: GLWEToBackendMut<Self, Rg> + GLWEToBackendRef<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
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
pub unsafe trait CKKSComplexPolynomialEvaluationImpl: Backend + CKKSPolynomialEvaluationImpl {
    /// Compiles the baby steps, prepared giant products and output fold into a factored program.
    /// The selected factored evaluator runs it within the shared BSGS scratch bound.
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
        C: GLWEToBackendRef<Self> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<Self>,
        A: GLWEToBackendRef<Self> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<Self, A>,
        H: GetTensorKey<Self>,
    {
        crate::oep::derived::polynomial_evaluation::ckks_eval_poly_complex_const_coeffs_from_power_basis_derived(
            module,
            res,
            power_basis,
            poly,
            tsk,
            scratch,
        )
    }

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
        C: GLWEToBackendRef<Self>
            + GLWEInfos
            + poulpy_core::layouts::BSGSMeta
            + CKKSCtBounds
            + IntPolyInfos
            + CKKSMaskedConstants<Self>,
        H: GetTensorKey<Self>,
        CKKSCiphertextOwned<Self>: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::oep::derived::polynomial_evaluation::ckks_eval_poly_complex_const_coeffs_derived(
            module, dst, src, poly, tsk, scratch,
        )
    }
}

/// Selects the derived real BSGS operations on one or both rings.
#[macro_export]
macro_rules! impl_ckks_polynomial_evaluation_derived_full {
    ($be:ty) => {
        $crate::impl_ckks_polynomial_evaluation_derived_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_polynomial_evaluation_derived_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSPolynomialEvaluationImpl<$ring> for $be {}
    };
}

/// Selects the derived complex BSGS operations on the standard ring.
#[macro_export]
macro_rules! impl_ckks_complex_polynomial_evaluation_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSComplexPolynomialEvaluationImpl for $be {}
    };
}

/// Selects the real BSGS defaults, forwarding to [`impl_ckks_polynomial_evaluation_derived_full!`](crate::impl_ckks_polynomial_evaluation_derived_full).
#[macro_export]
macro_rules! impl_ckks_polynomial_evaluation_reference_full {
    ($($args:tt)*) => { $crate::impl_ckks_polynomial_evaluation_derived_full!($($args)*); };
}

/// Selects the complex BSGS defaults, forwarding to [`impl_ckks_complex_polynomial_evaluation_derived_full!`](crate::impl_ckks_complex_polynomial_evaluation_derived_full).
#[macro_export]
macro_rules! impl_ckks_complex_polynomial_evaluation_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_complex_polynomial_evaluation_derived_full!($be);
    };
}

pub use crate::{
    impl_ckks_complex_polynomial_evaluation_derived_full, impl_ckks_complex_polynomial_evaluation_reference_full,
    impl_ckks_polynomial_evaluation_derived_full, impl_ckks_polynomial_evaluation_reference_full,
};
