//! Backend seams for CKKS bootstrapping: the pipeline and the
//! secret-switching encapsulation around ModUp.
//!
//! ModUp's known-zero low limbs are a CKKS pipeline property, not a general
//! Core key-switch operation.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, prepared::GGLWEPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos,
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned, CKKSPlaintextOwned, EncodedLut,
        EvalModPlan,
    },
};

/// Backend override hook for [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps).
/// [`impl_ckks_bootstrapping_reference`] wires every method to the reference pipeline.
///
/// # Safety
/// Outputs follow the [noise metadata rule](poulpy_core::oep#noise-metadata). Delegates only forward calls.
///
/// Implementations must preserve the exact CKKS metadata and ciphertext
/// semantics of the reference pipeline, honor all key layouts, and stay within
/// the scratch they report.
pub unsafe trait CKKSBootstrappingImpl: Backend<Ring = Standard> {
    fn ckks_mod_up_tmp_bytes_impl(module: &Module<Self>, res_size: usize) -> usize;

    fn ckks_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<Self>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<Self, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    fn ckks_functional_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<Self>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<Self, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    fn ckks_mod_up_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    fn ckks_bootstrap_mod_up_impl<Dst, Src, K>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        K: BootstrappingKeys<Self>;

    fn ckks_bootstrap_impl<F, K>(
        module: &Module<Self>,
        ct_out: &mut CKKSCiphertextOwned<Self>,
        ct_in: &CKKSCiphertextOwned<Self>,
        ctx: &BootstrappingContext<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        F: Sync,
        K: BootstrappingKeys<Self, TensorKey = GLWETensorKeyPrepared<Self::OwnedBuf, Self>> + Sync;

    fn ckks_functional_bootstrap_impl<F, K>(
        module: &Module<Self>,
        ct_outs: &mut [CKKSCiphertextOwned<Self>],
        ct_in: &CKKSCiphertextOwned<Self>,
        ctx: &BootstrappingContext<Self, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        K: BootstrappingKeys<Self, TensorKey = GLWETensorKeyPrepared<Self::OwnedBuf, Self>>;
}

/// Backend implementation of
/// `dense-to-sparse key switch -> ModUp -> sparse-to-dense key switch`.
/// A backend may exploit the known-zero limbs produced by ModUp.
///
/// # Safety
/// Outputs follow the [noise metadata rule](poulpy_core::oep#noise-metadata). Delegates only forward calls.
///
/// Implementations must preserve the exact CKKS metadata and ciphertext
/// semantics of the reference composition, honor all key layouts, and stay
/// within the supplied scratch arena.
pub unsafe trait CKKSEncapsulatedModUpImpl: Backend<Ring = Standard> {
    fn ckks_encapsulated_mod_up_tmp_bytes<Dst, Src, D2S, S2D>(
        module: &Module<Self>,
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
        module: &Module<Self>,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, Self>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos;
}

/// Opts a standard backend into the CKKS reference bootstrapping pipeline.
#[macro_export]
macro_rules! impl_ckks_bootstrapping_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSBootstrappingImpl for $be {
            fn ckks_mod_up_tmp_bytes_impl(module: &::poulpy_hal::layouts::Module<$be>, res_size: usize) -> usize {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_mod_up_tmp_bytes_reference(module, res_size)
            }

            fn ckks_bootstrap_tmp_bytes_impl<C1, C2, F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_out: &C1,
                ct_in: &C2,
                ctx: &$crate::layouts::BootstrappingContext<$be, F>,
                keys_layout: &$crate::layouts::BootstrappingKeysLayout,
            ) -> usize
            where
                C1: $crate::CKKSCtBounds,
                C2: $crate::CKKSCtBounds,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_tmp_bytes_reference(
                    module,
                    ct_out,
                    ct_in,
                    ctx,
                    keys_layout,
                )
            }

            fn ckks_functional_bootstrap_tmp_bytes_impl<C1, C2, F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_out: &C1,
                ct_in: &C2,
                ctx: &$crate::layouts::BootstrappingContext<$be, F>,
                luts: &[$crate::layouts::EncodedLut<$crate::layouts::CKKSPlaintextOwned<$be>>],
                keys_layout: &$crate::layouts::BootstrappingKeysLayout,
            ) -> usize
            where
                C1: $crate::CKKSCtBounds,
                C2: $crate::CKKSCtBounds,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_functional_bootstrap_tmp_bytes_reference(
                    module,
                    ct_out,
                    ct_in,
                    ctx,
                    luts,
                    keys_layout,
                )
            }

            fn ckks_mod_up_into_impl<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                dst: &mut Dst,
                src: &Src,
                eval_mod: &$crate::layouts::EvalModPlan,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_mod_up_into_reference(
                    module, dst, src, eval_mod, scratch,
                )
            }

            fn ckks_bootstrap_mod_up_impl<Dst, Src, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                dst: &mut Dst,
                src: &Src,
                eval_mod: &$crate::layouts::EvalModPlan,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWEToBackendRef<$be>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                K: $crate::layouts::BootstrappingKeys<$be>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_mod_up_reference(
                    module, dst, src, eval_mod, keys, scratch,
                )
            }

            fn ckks_bootstrap_impl<F, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_out: &mut $crate::layouts::CKKSCiphertextOwned<$be>,
                ct_in: &$crate::layouts::CKKSCiphertextOwned<$be>,
                ctx: &$crate::layouts::BootstrappingContext<$be, F>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                F: Sync,
                K: $crate::layouts::BootstrappingKeys<
                        $be,
                        TensorKey = ::poulpy_core::layouts::GLWETensorKeyPrepared<
                            <$be as ::poulpy_hal::layouts::Backend>::OwnedBuf,
                            $be,
                        >,
                    > + Sync,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_reference(
                    module, ct_out, ct_in, ctx, keys, scratch,
                )
            }

            fn ckks_functional_bootstrap_impl<F, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_outs: &mut [$crate::layouts::CKKSCiphertextOwned<$be>],
                ct_in: &$crate::layouts::CKKSCiphertextOwned<$be>,
                ctx: &$crate::layouts::BootstrappingContext<$be, F>,
                luts: &[$crate::layouts::EncodedLut<$crate::layouts::CKKSPlaintextOwned<$be>>],
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                K: $crate::layouts::BootstrappingKeys<
                        $be,
                        TensorKey = ::poulpy_core::layouts::GLWETensorKeyPrepared<
                            <$be as ::poulpy_hal::layouts::Backend>::OwnedBuf,
                            $be,
                        >,
                    >,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_functional_bootstrap_reference(
                    module, ct_outs, ct_in, ctx, luts, keys, scratch,
                )
            }
        }
    };
}

/// Opts a backend into the CKKS reference encapsulated-ModUp pipeline.
#[macro_export]
macro_rules! impl_ckks_encapsulated_mod_up_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSEncapsulatedModUpImpl for $be {
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
                $crate::reference::bootstrapping::ckks_encapsulated_mod_up_tmp_bytes_reference(
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
                $crate::reference::bootstrapping::ckks_encapsulated_mod_up_reference(
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

pub use crate::impl_ckks_encapsulated_mod_up_reference;

pub use crate::impl_ckks_bootstrapping_reference;
