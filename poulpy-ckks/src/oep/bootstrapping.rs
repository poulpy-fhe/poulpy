//! Backend seam for the secret-switching encapsulation around CKKS ModUp.
//!
//! ModUp's known-zero low limbs are a CKKS pipeline property, not a general
//! Core key-switch operation.

use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
    prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_core::{GLWEBytesOf, GLWECopy, GLWEKeyswitch, GLWEShift};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSDFTOps, CKKSEvalModOps, CKKSImagOps, CKKSPow2Ops,
        CKKSSubOps,
    },
    default::bootstrapping::CKKSBootstrapDefault,
    layouts::{BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertextOwned},
};

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
    fn ckks_encapsulated_mod_up<Dst, Src, D2S, S2D>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
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
                scale_up: usize,
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

/// Backend override hook for the whole CKKS bootstrap.
///
/// The blanket impl below forwards to the reference composition
/// ([`CKKSBootstrapDefault`]): ModUp → CoeffsToSlots → paired EvalMod →
/// SlotsToCoeffs. A backend that implements this trait directly owns the whole
/// operation instead: the bootstrap scratch lifetime, temporary ciphertext
/// allocation and reuse, any backend-private intermediate representation, the
/// transitions between the stages, and its own scheduling and synchronization.
/// The public ciphertext representation is unchanged: a private representation
/// is converted at the bootstrap boundaries.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSBootstrapImpl<BE: Backend>: Backend {
    /// See [`CKKSBootstrappingOps::ckks_bootstrap_tmp_bytes`](crate::api::CKKSBootstrappingOps::ckks_bootstrap_tmp_bytes).
    fn ckks_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// See [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap).
    fn ckks_bootstrap_impl<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;
}

unsafe impl<BE: Backend> CKKSBootstrapImpl<BE> for BE
where
    BE: CKKSEncapsulatedModUpImpl<BE>,
    Module<BE>: CKKSBootstrapDefault<BE>
        + GLWEBytesOf<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    fn ckks_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        module.ckks_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_bootstrap_impl<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        module.ckks_bootstrap_default(ct_out, ct_in, ctx, keys, scratch)
    }
}

/// Wires the reference whole-bootstrap composition into `$be`.
///
/// Emits the marker impl the [`CKKSBootstrapImpl`] blanket is keyed on. A
/// backend that owns the bootstrap omits this and implements
/// [`CKKSBootstrapImpl`] directly.
#[macro_export]
macro_rules! impl_ckks_bootstrap_defaults {
    ($be:ty) => {
        impl $crate::default::bootstrapping::CKKSBootstrapDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_bootstrap_defaults;
