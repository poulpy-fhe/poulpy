//! Backend seam for the secret-switching encapsulation around CKKS ModUp.
//!
//! ModUp's known-zero low limbs are a CKKS pipeline property, not a general
//! Core key-switch operation.

use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGLWEPreparedBackendRef};
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

    /// `scale_up` is applied to the raised ciphertext between ModUp and the
    /// sparse-to-dense switch, so the message is already at its final scale when
    /// that key-switch's noise is added.
    fn ckks_encapsulated_mod_up<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
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

            fn ckks_encapsulated_mod_up<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                dst: &mut Dst,
                src: &mut Src,
                scale_up: usize,
                dense_to_sparse: &::poulpy_core::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                sparse_to_dense: &::poulpy_core::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
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
            {
                $crate::default::bootstrapping::ckks_encapsulated_mod_up_default(
                    module,
                    dst,
                    src,
                    scale_up,
                    dense_to_sparse,
                    sparse_to_dense,
                    scratch,
                )
            }
        }
    };
}

pub use crate::impl_ckks_encapsulated_mod_up_default;
