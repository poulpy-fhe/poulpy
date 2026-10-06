use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use crate::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};

/// Backend-provided maps between conjugate-invariant GLWEs of degree `N` and
/// standard GLWEs of degree `2N`, executed by a standard backend of degree at least `2N`.
///
/// # Safety
/// Outputs follow the [noise metadata rule](crate::oep#noise-metadata). Delegates only forward calls.
/// Implementations must only read and write the regions described by the provided layouts, respect
/// scratch-space requirements, and produce results equivalent to the reference maps.
pub unsafe trait GLWECIConversionImpl: Backend<Ring = Standard> {
    fn glwe_ci_embed<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_ci_trace_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_ci_trace<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;
}

/// Selects the portable HAL algorithms for the conjugate-invariant maps.
#[macro_export]
macro_rules! impl_glwe_ci_conversion_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWECIConversionImpl for $be {
            fn glwe_ci_embed<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::ci_conversion::glwe_ci_embed_reference::<$be, _, _, _>(module, res, a)
            }

            fn glwe_ci_trace_tmp_bytes<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::ci_conversion::glwe_ci_trace_tmp_bytes_reference::<$be, _, _, _>(module, res_infos, a_infos)
            }

            fn glwe_ci_trace<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::ci_conversion::glwe_ci_trace_reference::<$be, _, _, _>(module, res, a, scratch)
            }
        }
    };
}
