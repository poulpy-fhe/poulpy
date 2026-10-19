use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSPreparedLeft, CKKSPreparedRight},
    oep::derived::dot_product::{
        ckks_dot_product_pt_const, ckks_dot_product_pt_const_tmp_bytes, ckks_dot_product_pt_vec,
        ckks_dot_product_pt_vec_tmp_bytes,
    },
};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures. See `poulpy-hal/docs/backend_safety_contract.md`.
pub unsafe trait CKKSDotProductImpl: Backend + super::CKKSMulImpl + super::CKKSAddImpl {
    fn ckks_dot_product_ct_tmp_bytes_impl<R: CKKSCtBounds, A: CKKSCtBounds, B: CKKSCtBounds, T: GGLWEInfos>(
        module: &Module<Self>,
        n: usize,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize;

    /// Scratch of [`Self::ckks_dot_product_pt_vec_impl`]: a `res`-shaped term and the product or addition.
    fn ckks_dot_product_pt_vec_tmp_bytes_impl<R: CKKSCtBounds, A: CKKSCtBounds, P: CKKSInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize {
        ckks_dot_product_pt_vec_tmp_bytes(module, res, a, b)
    }

    /// Scratch of [`Self::ckks_dot_product_pt_const_impl`]: a `res`-shaped term and the product or addition.
    fn ckks_dot_product_pt_const_tmp_bytes_impl<R: CKKSCtBounds, A: CKKSCtBounds, P: CKKSInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize {
        ckks_dot_product_pt_const_tmp_bytes(module, res, a, b)
    }

    fn ckks_dot_product_ct_impl<Dst, A, B, H>(
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

    fn ckks_dot_product_prepared_impl<Dst, H>(
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

    /// `ckks_mul_pt_vec_into` of the first pair, then each later pair's product added to it.
    fn ckks_dot_product_pt_vec_impl<Dst: Data, D: Data, E>(
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
        ckks_dot_product_pt_vec(module, dst, a, b, scratch)
    }

    /// `ckks_mul_pt_const_into` of the first pair, then each later pair's product added to it.
    fn ckks_dot_product_pt_const_impl<Dst: Data, D: Data, E>(
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
        ckks_dot_product_pt_const(module, dst, a, b, pt_coeffs, scratch)
    }
}

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_dot_product_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSDotProductImpl for $be {
            fn ckks_dot_product_ct_tmp_bytes_impl<
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

            fn ckks_dot_product_ct_impl<Dst, A, B, H>(
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

            fn ckks_dot_product_prepared_impl<Dst, H>(
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
        }
    };
}
pub use crate::impl_ckks_dot_product_reference;
