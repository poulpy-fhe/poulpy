use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
    GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef,
};

/// Backend hook for GLWE external products.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEExternalProductImpl: Backend {
    /// Scratch bound of [`Self::glwe_external_product_dft`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    /// Reproduces
    /// [`glwe_external_product_dft_reference`](crate::reference::external_product::glwe::GLWEExternalProductDftReference::glwe_external_product_dft_reference)
    /// in the integers under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_external_product_internal_tmp_bytes`]: after inverse transform and rounding its output is bit for
    /// bit the reference's, whatever the DFT representation; free in kernels and scratch use.
    ///
    /// It fills the full DFT output at the key radix, normalizing a noncanonical input whose radix must match the
    /// key.
    fn glwe_external_product_dft<'r, A>(
        module: &Module<Self>,
        res_dft: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'r, Self>,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        A: GLWEToBackendRef<Self>;

    /// Scratch bound of [`Self::glwe_external_product`] and [`Self::glwe_external_product_assign`]; may report less
    /// than the reference, never less than the body takes.
    fn glwe_external_product_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;

    /// Reproduces
    /// [`glwe_external_product_reference`](crate::reference::external_product::glwe::glwe_external_product_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_external_product_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_external_product<R, A>(
        module: &Module<Self>,
        res: &mut R,
        lhs: &A,
        rhs: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_external_product_assign_reference`](crate::reference::external_product::glwe::glwe_external_product_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_external_product_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_external_product_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;
}

/// Backend hook for batched GGLWE external products.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGLWEExternalProductImpl: Backend + GLWEExternalProductImpl + crate::oep::GLWEZeroImpl {
    /// Default body: `gglwe_external_product_tmp_bytes_derived`,
    /// [`GLWEExternalProductImpl::glwe_external_product_tmp_bytes`](crate::oep::GLWEExternalProductImpl::glwe_external_product_tmp_bytes).
    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        crate::oep::derived::external_product::gglwe_external_product_tmp_bytes_derived::<Self, _, _, _, _>(
            module, res_infos, a_infos, b_infos,
        )
    }

    /// Default body: `gglwe_external_product_derived`,
    /// [`GLWEExternalProductImpl::glwe_external_product`](crate::oep::GLWEExternalProductImpl::glwe_external_product)
    /// on each entry of the first `min(res.dnum(), a.dnum())` rows, then
    /// [`GLWEZeroImpl::glwe_zero`](crate::oep::GLWEZeroImpl::glwe_zero) on the remaining rows, within
    /// [`Self::gglwe_external_product_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn gglwe_external_product<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos,
    {
        crate::oep::derived::external_product::gglwe_external_product_derived::<Self, _, _, _>(module, res, a, b, scratch)
    }

    /// Default body: `gglwe_external_product_assign_derived`,
    /// [`GLWEExternalProductImpl::glwe_external_product_assign`](crate::oep::GLWEExternalProductImpl::glwe_external_product_assign)
    /// on each entry of `res`, within [`Self::gglwe_external_product_tmp_bytes`], under any
    /// [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
    fn gglwe_external_product_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
    {
        crate::oep::derived::external_product::gglwe_external_product_assign_derived::<Self, _, _>(module, res, a, scratch)
    }
}

/// Backend hook for GGSW external products.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGSWExternalProductImpl: Backend + GLWEExternalProductImpl + crate::oep::GLWEZeroImpl {
    /// Default body: `ggsw_external_product_tmp_bytes_derived`,
    /// [`GLWEExternalProductImpl::glwe_external_product_tmp_bytes`](crate::oep::GLWEExternalProductImpl::glwe_external_product_tmp_bytes).
    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        crate::oep::derived::external_product::ggsw_external_product_tmp_bytes_derived::<Self, _, _, _, _>(
            module, res_infos, a_infos, b_infos,
        )
    }

    /// Default body: `ggsw_external_product_derived`,
    /// [`GLWEExternalProductImpl::glwe_external_product`](crate::oep::GLWEExternalProductImpl::glwe_external_product)
    /// on each entry of the first `min(res.dnum(), a.dnum())` rows, then
    /// [`GLWEZeroImpl::glwe_zero`](crate::oep::GLWEZeroImpl::glwe_zero) on the remaining rows, within
    /// [`Self::ggsw_external_product_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn ggsw_external_product<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWAtViewRef<Self> + GGSWInfos,
    {
        crate::oep::derived::external_product::ggsw_external_product_derived::<Self, _, _, _>(module, res, a, b, scratch)
    }

    /// Default body: `ggsw_external_product_assign_derived`,
    /// [`GLWEExternalProductImpl::glwe_external_product_assign`](crate::oep::GLWEExternalProductImpl::glwe_external_product_assign)
    /// on each entry of `res`, within [`Self::ggsw_external_product_tmp_bytes`], under any
    /// [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
    fn ggsw_external_product_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
    {
        crate::oep::derived::external_product::ggsw_external_product_assign_derived::<Self, _, _>(module, res, a, scratch)
    }
}

/// Selects the portable HAL algorithms for this backend operation family.
#[macro_export]
macro_rules! impl_glwe_external_product_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEExternalProductImpl for $be {
            fn glwe_external_product_internal_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res_infos: &R,
                a_infos: &A,
                b_infos: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GGSWInfos,
            {
                $crate::reference::external_product::glwe::GLWEExternalProductDftReference::glwe_external_product_internal_tmp_bytes_reference(
                    module, res_infos, a_infos, b_infos,
                )
            }

            fn glwe_external_product_dft<'r, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res_dft: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'r, Self>,
                a: &A,
                ggsw: &$crate::layouts::GGSWPreparedBackendRef<'_, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) where
                A: $crate::layouts::GLWEToBackendRef<Self>,
            {
                $crate::reference::external_product::glwe::GLWEExternalProductDftReference::glwe_external_product_dft_reference(
                    module, res_dft, a, ggsw, scratch,
                )
            }

            fn glwe_external_product_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
                b_infos: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GGSWInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_tmp_bytes_reference::<$be, _, _, _, _>(
                    module, res_infos, a_infos, b_infos,
                )
            }

            fn glwe_external_product<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                lhs: &A,
                rhs: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_reference::<$be, _, _, _>(
                    module, res, lhs, rhs, scratch,
                )
            }

            fn glwe_external_product_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_assign_reference::<$be, _, _>(
                    module, res, a, scratch,
                )
            }
        }
    };
}

/// Selects the same-layer row composition for this backend.
#[macro_export]
macro_rules! impl_gglwe_external_product_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGLWEExternalProductImpl for $be {}
    };
}

/// Selects the same-layer row composition for this backend.
#[macro_export]
macro_rules! impl_ggsw_external_product_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGSWExternalProductImpl for $be {}
    };
}
