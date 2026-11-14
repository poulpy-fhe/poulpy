use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    GetAutomorphismKey, GetTensorKey,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{
        BootstrappingKeys, ReplicatedBootstrapWorkspace, ReplicatedBootstrapping, ReplicatedConversion,
        ReplicatedTransformWorkspace,
    },
    oep::derived::replicated::{
        ckks_from_replicated_derived, ckks_replicated_conversion_tmp_bytes_derived, ckks_to_replicated_derived,
    },
};

/// Backend contract of [`CKKSReplicatedOps`](crate::api::CKKSReplicatedOps), the conversions between
/// CKKS and a replicated slot layout. Every method defaults to its derived composition of the CKKS
/// operations of its supertraits on the ring `Rg`; [`impl_ckks_replicated_derived_full`] selects them all.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the derived body does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSReplicatedImpl<Rg: Ring = Standard>:
    Backend
    + super::CKKSLinearTransformationImpl<Rg>
    + super::CKKSRotateImpl<Rg>
    + super::CKKSAddImpl<Rg>
    + super::CKKSCopyImpl<Rg>
    + super::CKKSPow2Impl<Rg>
    + super::CKKSRoundImpl<Rg>
{
    /// Scratch bound of [`Self::ckks_to_replicated`] and [`Self::ckks_from_replicated`]; may report less than the
    /// derived body, never less than the body takes.
    fn ckks_replicated_conversion_tmp_bytes<C, K>(
        module: &Module<Self>,
        ct_infos: &C,
        key_infos: &K,
        conv: &ReplicatedConversion<Self, Rg>,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_replicated_conversion_tmp_bytes_derived(module, ct_infos, key_infos, conv)
    }

    /// Reproduces the derived conversion onto `conv`'s replicated layout bit for bit.
    fn ckks_to_replicated<Dst, Src, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<Self, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<Self, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg>,
    {
        ckks_to_replicated_derived(module, dst, src, conv, workspace, keys, scratch)
    }

    /// Reproduces the derived conversion from `conv`'s replicated layout bit for bit.
    fn ckks_from_replicated<Dst, Src, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<Self, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<Self, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg>,
    {
        ckks_from_replicated_derived(module, dst, src, conv, workspace, keys, scratch)
    }
}

/// Selects the derived defaults of [`CKKSReplicatedImpl`] for `$be`, on both rings from `($be)` and on `$ring` from
/// `($be, $ring)`.
#[macro_export]
macro_rules! impl_ckks_replicated_derived_full {
    ($be:ty) => {
        $crate::impl_ckks_replicated_derived_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_replicated_derived_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSReplicatedImpl<$ring> for $be {}
    };
}

pub use crate::impl_ckks_replicated_derived_full;

/// Backend contract of [`CKKSReplicatedBootstrapOps`](crate::api::CKKSReplicatedBootstrapOps), the
/// bootstrapping of a replicated slot layout. [`impl_ckks_replicated_bootstrap_reference_full`] wires every method to
/// [`CKKSReplicatedBootstrapReference`](crate::reference::replicated::CKKSReplicatedBootstrapReference).
///
/// # Safety
/// Implementations reproduce the reference bodies bit for bit (values, metadata, canonical flag,
/// refusals and their order), report for each `_tmp_bytes` query a bound that covers the body, and
/// choose only their kernels and scratch within it. The memory, layout, synchrony and aliasing rules
/// of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSReplicatedBootstrapImpl: Backend {
    /// Scratch bound of [`Self::ckks_replicated_bootstrap`]; may report less than the reference, never less than the
    /// body takes.
    fn ckks_replicated_bootstrap_tmp_bytes<C, I, F, A, T, S>(
        module: &Module<Self>,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<Self, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos;

    /// Reproduces
    /// [`ckks_replicated_bootstrap_reference`](crate::reference::replicated::CKKSReplicatedBootstrapReference::ckks_replicated_bootstrap_reference)
    /// bit for bit.
    fn ckks_replicated_bootstrap<Dst, Src, F, K>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<Self, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        K: BootstrappingKeys<Self>;
}

/// Backend contract of [`CKKSReplicatedCIBootstrapOps`](crate::api::CKKSReplicatedCIBootstrapOps), the
/// bootstrapping of a replicated layout on the conjugate-invariant ring of degree `N` through the standard ring of
/// degree `2N` of the same module. [`impl_ckks_replicated_ci_bootstrap_reference_full`] wires it to
/// [`CKKSReplicatedCIBootstrapReference`](crate::reference::replicated::CKKSReplicatedCIBootstrapReference).
///
/// # Safety
/// Implementations reproduce the reference bodies bit for bit (values, metadata, canonical flag,
/// refusals and their order), report for each `_tmp_bytes` query a bound that covers the body,
/// and choose only their kernels and scratch within it. The memory, layout, synchrony and aliasing
/// rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSReplicatedCIBootstrapImpl: Backend {
    /// Scratch bound of [`Self::ckks_replicated_ci_bootstrap`]; may report less than the reference, never less than
    /// the body takes.
    #[expect(
        clippy::too_many_arguments,
        reason = "the scratch covers the bootstrapping keys and the conversion keys"
    )]
    fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, A, T, S, K>(
        module: &Module<Self>,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<Self, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`ckks_replicated_ci_bootstrap_reference`](crate::reference::replicated::CKKSReplicatedCIBootstrapReference::ckks_replicated_ci_bootstrap_reference)
    /// bit for bit.
    #[expect(
        clippy::too_many_arguments,
        reason = "the bootstrap takes its context, workspace and both key sets"
    )]
    fn ckks_replicated_ci_bootstrap<Dst, Src, F, K, E, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<Self, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<Self>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, Self>, &GLWECITraceKeyPrepared<T, Self>),
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, ConjugateInvariant> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self, ConjugateInvariant> + CKKSCtBounds,
        K: BootstrappingKeys<Self>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, Self>: GGLWEPreparedToBackendRef<Self>,
        GLWECITraceKeyPrepared<T, Self>: GGLWEPreparedToBackendRef<Self>;
}

/// Implements [`CKKSReplicatedBootstrapImpl`] for `$be` by forwarding every method to
/// [`CKKSReplicatedBootstrapReference`](crate::reference::replicated::CKKSReplicatedBootstrapReference).
#[macro_export]
macro_rules! impl_ckks_replicated_bootstrap_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSReplicatedBootstrapImpl for $be {
            fn ckks_replicated_bootstrap_tmp_bytes<C, I, F, A, T, S>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_out: &C,
                ct_in: &I,
                ctx: &$crate::layouts::ReplicatedBootstrapping<Self, F>,
                atk_infos: &A,
                tsk_infos: &T,
                encapsulation_infos: Option<(&S, &S)>,
            ) -> usize
            where
                C: $crate::CKKSCtBounds,
                I: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GGLWEInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
                S: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::replicated::CKKSReplicatedBootstrapReference::ckks_replicated_bootstrap_tmp_bytes_reference(
                    module,
                    ct_out,
                    ct_in,
                    ctx,
                    atk_infos,
                    tsk_infos,
                    encapsulation_infos,
                )
            }

            fn ckks_replicated_bootstrap<Dst, Src, F, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                ctx: &$crate::layouts::ReplicatedBootstrapping<Self, F>,
                workspace: &mut $crate::layouts::ReplicatedBootstrapWorkspace<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                K: $crate::layouts::BootstrappingKeys<Self>,
            {
                $crate::reference::replicated::CKKSReplicatedBootstrapReference::ckks_replicated_bootstrap_reference(
                    module, dst, src, ctx, workspace, keys, scratch,
                )
            }
        }
    };
}

pub use crate::impl_ckks_replicated_bootstrap_reference_full;

/// Implements [`CKKSReplicatedCIBootstrapImpl`] for `$be` by forwarding every method to
/// [`CKKSReplicatedCIBootstrapReference`](crate::reference::replicated::CKKSReplicatedCIBootstrapReference).
#[macro_export]
macro_rules! impl_ckks_replicated_ci_bootstrap_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSReplicatedCIBootstrapImpl for $be {
            fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, A, T, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_out: &C,
                ct_in: &I,
                ctx: &$crate::layouts::ReplicatedBootstrapping<Self, F>,
                atk_infos: &A,
                tsk_infos: &T,
                encapsulation_infos: Option<(&S, &S)>,
                conversion_infos: &K,
            ) -> usize
            where
                C: $crate::CKKSCtBounds,
                I: $crate::CKKSCtBounds,
                A: ::poulpy_core::layouts::GGLWEInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
                S: ::poulpy_core::layouts::GGLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::replicated::CKKSReplicatedCIBootstrapReference::ckks_replicated_ci_bootstrap_tmp_bytes_reference(
                    module,
                    ct_out,
                    ct_in,
                    ctx,
                    atk_infos,
                    tsk_infos,
                    encapsulation_infos,
                    conversion_infos,
                )
            }

            fn ckks_replicated_ci_bootstrap<Dst, Src, F, K, E, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                ctx: &$crate::layouts::ReplicatedBootstrapping<Self, F>,
                workspace: &mut $crate::layouts::ReplicatedBootstrapWorkspace<Self>,
                keys: &K,
                conversion_keys: (
                    &::poulpy_core::layouts::GLWECIEmbedKeyPrepared<E, Self>,
                    &::poulpy_core::layouts::GLWECITraceKeyPrepared<T, Self>,
                ),
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, ::poulpy_hal::layouts::ConjugateInvariant>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self, ::poulpy_hal::layouts::ConjugateInvariant>
                    + $crate::CKKSCtBounds,
                K: $crate::layouts::BootstrappingKeys<Self>,
                E: ::poulpy_hal::layouts::Data,
                T: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWECIEmbedKeyPrepared<E, Self>: ::poulpy_core::layouts::GGLWEPreparedToBackendRef<Self>,
                ::poulpy_core::layouts::GLWECITraceKeyPrepared<T, Self>: ::poulpy_core::layouts::GGLWEPreparedToBackendRef<Self>,
            {
                $crate::reference::replicated::CKKSReplicatedCIBootstrapReference::ckks_replicated_ci_bootstrap_reference(
                    module,
                    dst,
                    src,
                    ctx,
                    workspace,
                    keys,
                    conversion_keys,
                    scratch,
                )
            }
        }
    };
}

pub use crate::impl_ckks_replicated_ci_bootstrap_reference_full;
