use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard, VecZnxDftBackendMut};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
    oep::derived::dot_product::ckks_dot_product_prepared_tmp_bytes_derived,
};

/// Backend contract of [`CKKSDotProductOps`](crate::api::CKKSDotProductOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSDotProductImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::ckks_dot_product`]; may report less than the reference, never less than the body
    /// takes. It covers `n` input pairs, `a` and `b` describing the widest pair (the internal products are carved at
    /// `max(a.k, b.k)`).
    fn ckks_dot_product_tmp_bytes<R: CKKSCtBounds, A: CKKSCtBounds, B: CKKSCtBounds, T: GGLWEInfos>(
        module: &Module<Self>,
        n: usize,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize;

    /// Scratch bound of [`Self::ckks_dot_product_pt`], whatever the number of terms, as
    /// [`ckks_dot_product_pt_tmp_bytes_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_pt_tmp_bytes_reference)
    /// counts it; may report less than the reference, never less than the body takes.
    fn ckks_dot_product_pt_tmp_bytes<R: CKKSCtBounds, A: CKKSCtBounds, P: CKKSInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize;

    /// Reproduces
    /// [`ckks_dot_product_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_reference) bit
    /// for bit, within [`Self::ckks_dot_product_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dot_product<Dst, A, B, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &[&A],
        b: &[&B],
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        H: GetTensorKey<Self, Rg>;

    /// Default body: `ckks_dot_product_prepared_tmp_bytes_derived`, [`Self::ckks_dot_product_tmp_bytes`]. Scratch
    /// bound of [`Self::ckks_dot_product_prepared`] over `n` pairs, `a` and `b` describing the widest pair, prepared
    /// operands passed as such, their infos reporting the ciphertexts they were prepared from; may report less than the
    /// reference, never less than the body takes.
    fn ckks_dot_product_prepared_tmp_bytes<R: CKKSCtBounds, A: CKKSCtBounds, B: CKKSCtBounds, T: GGLWEInfos>(
        module: &Module<Self>,
        n: usize,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize {
        ckks_dot_product_prepared_tmp_bytes_derived(module, n, res, a, b, tsk)
    }

    /// Reproduces
    /// [`ckks_dot_product_prepared_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_reference)
    /// bit for bit, within [`Self::ckks_dot_product_prepared_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dot_product_prepared<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &[&CKKSLeftOperandPrepared<Self, Rg>],
        b: &[&CKKSRightOperandPrepared<Self, Rg>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<Self, Rg>;

    /// Reproduces
    /// [`ckks_dot_product_pt_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_pt_reference)
    /// bit for bit, within [`Self::ckks_dot_product_pt_tmp_bytes`], and refuses as it does; free in kernels and scratch
    /// use.
    fn ckks_dot_product_pt<Dst, A, B>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &[&A],
        b: &[&B],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<Self, Rg> + CKKSCtBounds + IntPolyInfos;

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
        a: &[&CKKSLeftOperandPrepared<Self, Rg>],
        b: &[&CKKSRightOperandPrepared<Self, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos;

    /// Scratch bound of [`Self::ckks_dot_product_prepared_pt_add`]; may report less than the reference, never less than
    /// the body takes. `a` and `b` are the prepared operands, whose infos report their sources, or those sources. It does
    /// not grow with the number of terms.
    fn ckks_dot_product_prepared_pt_add_tmp_bytes<R: CKKSCtBounds, A: GLWEInfos, P: GLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize;

    /// Reproduces
    /// [`ckks_dot_product_prepared_pt_add_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_pt_add_reference)
    /// bit for bit, within [`Self::ckks_dot_product_prepared_pt_add_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dot_product_prepared_pt_add<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &[&CKKSLeftOperandPrepared<Self, Rg>],
        b: &[&CKKSRightOperandPrepared<Self, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos;

    /// Scratch bound of [`Self::ckks_dot_product_prepared_pt_dft`]; may report less than the reference, never less than
    /// the body takes. It does not grow with the number of terms.
    fn ckks_dot_product_prepared_pt_dft_tmp_bytes<R: CKKSCtBounds, A: GLWEInfos, P: GLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize;

    /// Reproduces
    /// [`ckks_dot_product_prepared_pt_dft_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_pt_dft_reference)
    /// bit for bit through `idft`, within [`Self::ckks_dot_product_prepared_pt_dft_tmp_bytes`]; free in kernels and
    /// scratch use.
    fn ckks_dot_product_prepared_pt_dft<R>(
        module: &Module<Self>,
        acc: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<Self, Rg>],
        b: &[&CKKSRightOperandPrepared<Self, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: CKKSCtBounds;

    /// Scratch bound of [`Self::ckks_dot_product_prepared_pt_dft_add`]; may report less than the reference, never less
    /// than the body takes. It does not grow with the number of terms.
    fn ckks_dot_product_prepared_pt_dft_add_tmp_bytes<R: CKKSCtBounds, A: GLWEInfos, P: GLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize;

    /// Reproduces
    /// [`ckks_dot_product_prepared_pt_dft_add_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_pt_dft_add_reference)
    /// bit for bit through `idft`, within [`Self::ckks_dot_product_prepared_pt_dft_add_tmp_bytes`]; free in kernels and
    /// scratch use.
    fn ckks_dot_product_prepared_pt_dft_add<R>(
        module: &Module<Self>,
        acc: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<Self, Rg>],
        b: &[&CKKSRightOperandPrepared<Self, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: CKKSCtBounds;

    /// Scratch bound of [`Self::ckks_dot_product_prepared_pt_from_dft`]; may report less than the reference, never
    /// less than the body takes.
    fn ckks_dot_product_prepared_pt_from_dft_tmp_bytes<R: CKKSCtBounds, A: GLWEInfos, P: GLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &P,
    ) -> usize;

    /// Reproduces
    /// [`ckks_dot_product_prepared_pt_from_dft_reference`](crate::reference::CKKSDotProductReference::ckks_dot_product_prepared_pt_from_dft_reference)
    /// bit for bit, within [`Self::ckks_dot_product_prepared_pt_from_dft_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dot_product_prepared_pt_from_dft<Dst>(
        module: &Module<Self>,
        res: &mut Dst,
        acc: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        a: &CKKSLeftOperandPrepared<Self, Rg>,
        b: &CKKSRightOperandPrepared<Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos;
}

/// Selects the reference implementation of `CKKSDotProductImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_dot_product_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_dot_product_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_dot_product_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSDotProductImpl<$ring> for $be {
            fn ckks_dot_product_tmp_bytes<
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
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_tmp_bytes_reference(
                    module, n, res, a, b, tsk,
                )
            }

            fn ckks_dot_product<Dst, A, B, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &[&A],
                b: &[&B],
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + $crate::CKKSCtBounds,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + $crate::CKKSCtBounds,
                H: ::poulpy_core::layouts::GetTensorKey<Self, $ring>,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_reference(
                    module, dst, a, b, tsk, scratch,
                )
            }

            fn ckks_dot_product_pt_tmp_bytes<R: $crate::CKKSCtBounds, A: $crate::CKKSCtBounds, P: $crate::CKKSInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &P,
            ) -> usize {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_pt_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn ckks_dot_product_pt<Dst, A, B>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &[&A],
                b: &[&B],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + $crate::CKKSCtBounds,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring>
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_pt_reference(
                    module, dst, a, b, scratch,
                )
            }

            fn ckks_dot_product_prepared<Dst, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &[&$crate::layouts::CKKSLeftOperandPrepared<Self, $ring>],
                b: &[&$crate::layouts::CKKSRightOperandPrepared<Self, $ring>],
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetTensorKey<Self, $ring>,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_reference(
                    module, dst, a, b, tsk, scratch,
                )
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
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn ckks_dot_product_prepared_pt<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &[&$crate::layouts::CKKSLeftOperandPrepared<Self, $ring>],
                b: &[&$crate::layouts::CKKSRightOperandPrepared<Self, $ring>],
                galois: &[i64],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_reference(
                    module, dst, a, b, galois, scratch,
                )
            }

            fn ckks_dot_product_prepared_pt_add_tmp_bytes<
                R: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &P,
            ) -> usize {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_add_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn ckks_dot_product_prepared_pt_add<R>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &[&$crate::layouts::CKKSLeftOperandPrepared<Self, $ring>],
                b: &[&$crate::layouts::CKKSRightOperandPrepared<Self, $ring>],
                galois: &[i64],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_add_reference(
                    module, res, a, b, galois, scratch,
                )
            }

            fn ckks_dot_product_prepared_pt_dft_tmp_bytes<
                R: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &P,
            ) -> usize {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_dft_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn ckks_dot_product_prepared_pt_dft<R>(
                module: &::poulpy_hal::layouts::Module<Self>,
                acc: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
                res: &R,
                a: &[&$crate::layouts::CKKSLeftOperandPrepared<Self, $ring>],
                b: &[&$crate::layouts::CKKSRightOperandPrepared<Self, $ring>],
                galois: &[i64],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: $crate::CKKSCtBounds,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_dft_reference(
                    module, acc, res, a, b, galois, scratch,
                )
            }

            fn ckks_dot_product_prepared_pt_dft_add_tmp_bytes<
                R: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &P,
            ) -> usize {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_dft_add_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn ckks_dot_product_prepared_pt_dft_add<R>(
                module: &::poulpy_hal::layouts::Module<Self>,
                acc: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
                res: &R,
                a: &[&$crate::layouts::CKKSLeftOperandPrepared<Self, $ring>],
                b: &[&$crate::layouts::CKKSRightOperandPrepared<Self, $ring>],
                galois: &[i64],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: $crate::CKKSCtBounds,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_dft_add_reference(
                    module, acc, res, a, b, galois, scratch,
                )
            }

            fn ckks_dot_product_prepared_pt_from_dft_tmp_bytes<
                R: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &P,
            ) -> usize {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_from_dft_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn ckks_dot_product_prepared_pt_from_dft<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut Dst,
                acc: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self, $ring>,
                a: &$crate::layouts::CKKSLeftOperandPrepared<Self, $ring>,
                b: &$crate::layouts::CKKSRightOperandPrepared<Self, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::CKKSDotProductReference::<Self, $ring>::ckks_dot_product_prepared_pt_from_dft_reference(
                    module, res, acc, a, b, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_dot_product_reference_full;
