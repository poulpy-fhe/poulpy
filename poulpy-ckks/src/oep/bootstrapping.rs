//! Backend seam for the secret-switching encapsulation around CKKS ModUp.
//!
//! ModUp's known-zero low limbs are a CKKS pipeline property, not a general
//! Core key-switch operation.

use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGLWEPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos};

/// Backend implementation of
/// `dense-to-sparse key switch -> ModUp -> sparse-to-dense key switch`.
/// A backend may exploit the known-zero limbs produced by ModUp.
///
/// # Safety
///
/// Implementations must preserve the exact CKKS metadata and ciphertext
/// semantics of the reference composition, honor all key layouts, and stay
/// within the supplied scratch arena.
pub unsafe trait CKKSEncapsulatedModUpImpl<BE: Backend>: Backend {
    fn ckks_encapsulated_mod_up_tmp_bytes<Dst, Src, D2S, S2D>(
        module: &Module<BE>,
        dst_infos: &Dst,
        src_infos: &Src,
        dense_to_sparse_infos: &D2S,
        sparse_to_dense_infos: &S2D,
    ) -> usize
    where
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        D2S: GGLWEInfos,
        S2D: GGLWEInfos;

    fn ckks_encapsulated_mod_up<Dst, Src, D2S, S2D>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &mut Src,
        dense_to_sparse: &D2S,
        sparse_to_dense: &S2D,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        D2S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        S2D: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Opts a backend into the CKKS reference encapsulated-ModUp pipeline.
#[macro_export]
macro_rules! impl_ckks_encapsulated_mod_up_default {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSEncapsulatedModUpImpl<$be> for $be {
            fn ckks_encapsulated_mod_up_tmp_bytes<Dst, Src, D2S, S2D>(
                module: &::poulpy_hal::layouts::Module<$be>,
                dst_infos: &Dst,
                src_infos: &Src,
                dense_to_sparse_infos: &D2S,
                sparse_to_dense_infos: &S2D,
            ) -> usize
            where
                Dst: $crate::CKKSCtBounds,
                Src: $crate::CKKSCtBounds,
                D2S: ::poulpy_core::layouts::GGLWEInfos,
                S2D: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::default::bootstrapping::ckks_encapsulated_mod_up_tmp_bytes_default(
                    module,
                    dst_infos,
                    src_infos,
                    dense_to_sparse_infos,
                    sparse_to_dense_infos,
                )
            }

            fn ckks_encapsulated_mod_up<Dst, Src, D2S, S2D>(
                module: &::poulpy_hal::layouts::Module<$be>,
                dst: &mut Dst,
                src: &mut Src,
                dense_to_sparse: &D2S,
                sparse_to_dense: &S2D,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWEToBackendRef<$be>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWEToBackendRef<$be>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                D2S: ::poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
                S2D: ::poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::default::bootstrapping::ckks_encapsulated_mod_up_default(
                    module,
                    dst,
                    src,
                    dense_to_sparse,
                    sparse_to_dense,
                    scratch,
                )
            }
        }
    };
}

pub use crate::impl_ckks_encapsulated_mod_up_default;
