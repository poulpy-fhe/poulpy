use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSPreparedLeft, CKKSPreparedRight},
    oep::derived::dot_product::{
        ckks_dot_product_pt_const_derived, ckks_dot_product_pt_const_tmp_bytes_derived, ckks_dot_product_pt_vec_derived,
        ckks_dot_product_pt_vec_tmp_bytes_derived,
    },
};

/// Backend contract of [`CKKSDotProductOps`](crate::api::CKKSDotProductOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSDotProductImpl: Backend + super::CKKSMulImpl + super::CKKSAddImpl {
    /// Scratch bound of [`Self::ckks_dot_product_ct`] and [`Self::ckks_dot_product_prepared`]; may report less than the
    /// reference, never less than the body takes. `a` and `b` describe the widest input pair, and prepared operands are
    /// passed as such, their infos reporting the ciphertexts they were prepared from.
    fn ckks_dot_product_ct_tmp_bytes<R: CKKSCtBounds, A: CKKSCtBounds, B: CKKSCtBounds, T: GGLWEInfos>(
        module: &Module<Self>,
        n: usize,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize;

    /// Default body: `ckks_dot_product_pt_vec_tmp_bytes_derived`, the bytes of a `res`-shaped temporary plus the larger
    /// of the selected `ckks_mul_pt_vec_tmp_bytes` and `ckks_add_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_dot_product_pt_vec`], whatever the number of terms; may report less than the reference, never less
    /// than the body takes.
    fn ckks_dot_product_pt_vec_tmp_bytes<R: CKKSCtBounds, A: CKKSCtBounds, P: CKKSInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize {
        ckks_dot_product_pt_vec_tmp_bytes_derived(module, res, a, b)
    }

    /// Default body: `ckks_dot_product_pt_const_tmp_bytes_derived`, the bytes of a `res`-shaped temporary plus the
    /// larger of the selected `ckks_mul_pt_const_tmp_bytes` and `ckks_add_tmp_bytes` at `res.size()`. Scratch bound of
    /// [`Self::ckks_dot_product_pt_const`], whatever the number of terms; may report less than the reference, never
    /// less than the body takes.
    fn ckks_dot_product_pt_const_tmp_bytes<R: CKKSCtBounds, A: CKKSCtBounds, P: CKKSInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize {
        ckks_dot_product_pt_const_tmp_bytes_derived(module, res, a, b)
    }

    /// Reproduces
    /// [`ckks_dot_product_ct_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_ct_reference) bit
    /// for bit, within [`Self::ckks_dot_product_ct_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dot_product_ct<Dst, A, B, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &[&A],
        b: &[&B],
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        B: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`ckks_dot_product_prepared_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_reference)
    /// bit for bit, within [`Self::ckks_dot_product_ct_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dot_product_prepared<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &[&CKKSPreparedLeft<Self>],
        b: &[&CKKSPreparedRight<Self>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<Self>;

    /// Default body: `ckks_dot_product_pt_vec_derived`. An empty or length-mismatched pair of slices is a refusal. Then
    /// `ckks_mul_pt_vec_into` of the first pair into `dst`, and for each later pair `ckks_mul_pt_vec_into` into a
    /// `dst`-shaped scratch temporary followed by `ckks_add_assign` into `dst`.
    fn ckks_dot_product_pt_vec<Dst: Data, D: Data, E>(
        module: &Module<Self>,
        dst: &mut CKKSCiphertext<Dst, Self::ZnxWord, Self::Ring>,
        a: &[&CKKSCiphertext<D, Self::ZnxWord, Self::Ring>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<D, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self> + GLWEInfos,
        E: GLWEToBackendRef<Self> + CKKSCtBounds + IntPolyInfos,
    {
        ckks_dot_product_pt_vec_derived(module, dst, a, b, scratch)
    }

    /// Default body: `ckks_dot_product_pt_const_derived`. An empty slice, a length mismatch between the ciphertexts and
    /// the plaintexts or one between the ciphertexts and `pt_coeffs` is a refusal. Then `ckks_mul_pt_const_into` of the
    /// first pair at its coefficient into `dst`, and for each later pair `ckks_mul_pt_const_into` at its coefficient
    /// into a `dst`-shaped scratch temporary followed by `ckks_add_assign` into `dst`.
    fn ckks_dot_product_pt_const<Dst: Data, D: Data, E>(
        module: &Module<Self>,
        dst: &mut CKKSCiphertext<Dst, Self::ZnxWord, Self::Ring>,
        a: &[&CKKSCiphertext<D, Self::ZnxWord, Self::Ring>],
        b: &[&E],
        pt_coeffs: &[usize],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<D, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self> + GLWEInfos,
        E: GLWEToBackendRef<Self> + CKKSCtBounds + IntPolyInfos,
    {
        ckks_dot_product_pt_const_derived(module, dst, a, b, pt_coeffs, scratch)
    }

    /// Scratch bound of [`Self::ckks_dot_product_prepared_pt`]; may report less than the reference, never less than the
    /// body takes. It does not grow with the number of terms. `a` and `b` are the prepared operands, whose infos report
    /// their sources, or those sources.
    fn ckks_dot_product_prepared_pt_tmp_bytes<R: CKKSCtBounds, A: GLWEInfos, P: GLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize;

    /// Reproduces
    /// [`ckks_dot_product_prepared_pt_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_pt_reference)
    /// bit for bit, within [`Self::ckks_dot_product_prepared_pt_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dot_product_prepared_pt<Dst>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &[&CKKSPreparedLeft<Self>],
        b: &[&CKKSPreparedRight<Self>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

/// Selects the reference implementation of `CKKSDotProductImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_dot_product_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSDotProductImpl for $be {
            fn ckks_dot_product_ct_tmp_bytes<
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
                B: $crate::CKKSCtBounds,
                T: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                n: usize,
                res: &R,
                a: &A,
                b: &B,
                tsk: &T,
            ) -> usize {
                $crate::reference::CKKSDotProductReference::ckks_dot_product_ct_tmp_bytes_reference(module, n, res, a, b, tsk)
            }

            fn ckks_dot_product_ct<Dst, A, B, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &[&A],
                b: &[&B],
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::CKKSDotProductReference::ckks_dot_product_ct_reference(module, dst, a, b, tsk, scratch)
            }

            fn ckks_dot_product_prepared<Dst, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &[&$crate::layouts::CKKSPreparedLeft<Self>],
                b: &[&$crate::layouts::CKKSPreparedRight<Self>],
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_reference(module, dst, a, b, tsk, scratch)
            }

            fn ckks_dot_product_prepared_pt_tmp_bytes<
                R: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &P,
            ) -> usize {
                $crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_pt_tmp_bytes_reference(module, res, a, b)
            }

            fn ckks_dot_product_prepared_pt<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &[&$crate::layouts::CKKSPreparedLeft<Self>],
                b: &[&$crate::layouts::CKKSPreparedRight<Self>],
                galois: &[i64],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_pt_reference(
                    module, dst, a, b, galois, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_dot_product_reference_full;
