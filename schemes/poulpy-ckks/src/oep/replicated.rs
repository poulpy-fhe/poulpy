use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    GetAutomorphismKey, GetTensorKey,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

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
/// operations of its supertraits; [`impl_ckks_replicated_derived_full`] selects them all.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the derived body does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSReplicatedImpl:
    Backend
    + super::CKKSLinearTransformationImpl
    + super::CKKSRotateImpl
    + super::CKKSAddImpl
    + super::CKKSCopyImpl
    + super::CKKSPow2Impl
    + super::CKKSRoundImpl
{
    /// Scratch bound of [`Self::ckks_to_replicated`] and [`Self::ckks_from_replicated`]; may report less than the
    /// derived body, never less than the body takes.
    fn ckks_replicated_conversion_tmp_bytes<C, K>(
        module: &Module<Self>,
        ct_infos: &C,
        key_infos: &K,
        conv: &ReplicatedConversion<Self>,
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
        conv: &ReplicatedConversion<Self>,
        workspace: &mut ReplicatedTransformWorkspace<Self>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self> + GetTensorKey<Self>,
    {
        ckks_to_replicated_derived(module, dst, src, conv, workspace, keys, scratch)
    }

    /// Reproduces the derived conversion from `conv`'s replicated layout bit for bit.
    fn ckks_from_replicated<Dst, Src, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<Self>,
        workspace: &mut ReplicatedTransformWorkspace<Self>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self> + GetTensorKey<Self>,
    {
        ckks_from_replicated_derived(module, dst, src, conv, workspace, keys, scratch)
    }
}

/// Selects the derived defaults of [`CKKSReplicatedImpl`] for `$be`.
#[macro_export]
macro_rules! impl_ckks_replicated_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSReplicatedImpl for $be {}
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
pub unsafe trait CKKSReplicatedBootstrapImpl: Backend<Ring = Standard> {
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
/// bootstrapping of a replicated layout on the conjugate-invariant ring of [`Self::CI`] through the standard
/// ring of `Self`. [`impl_ckks_replicated_ci_bootstrap_reference_full`] wires it to
/// [`CKKSReplicatedCIBootstrapReference`](crate::reference::replicated::CKKSReplicatedCIBootstrapReference).
///
/// # Safety
/// Implementations reproduce the reference bodies bit for bit (values, metadata, canonical flag,
/// refusals and their order), report for each `_tmp_bytes` query a bound that covers the body in
/// each arena, and choose only their kernels and scratch within it. The memory, layout, synchrony
/// and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSReplicatedCIBootstrapImpl: Backend<Ring = Standard> {
    /// The conjugate-invariant twin that holds the input and output, sharing the buffers of `Self`.
    type CI: Backend<Ring = ConjugateInvariant, OwnedBuf = Self::OwnedBuf, ZnxWord = Self::ZnxWord>
        + for<'a> Backend<BufRef<'a> = Self::BufRef<'a>, BufMut<'a> = Self::BufMut<'a>>
        + 'static;

    /// Scratch bounds of [`Self::ckks_replicated_ci_bootstrap`], `(scratch, scratch_ci)`; may report less than the
    /// reference, never less than the body takes.
    #[expect(
        clippy::too_many_arguments,
        reason = "the scratch covers the bootstrapping keys and the conversion keys"
    )]
    fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, A, T, S, K>(
        module: &Module<Self>,
        ci: &Module<Self::CI>,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<Self, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> (usize, usize)
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
        reason = "the bootstrap takes a module, keys and a scratch arena per ring"
    )]
    fn ckks_replicated_ci_bootstrap<Dst, Src, F, K, E, T>(
        module: &Module<Self>,
        ci: &Module<Self::CI>,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<Self, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<Self>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, Self>, &GLWECITraceKeyPrepared<T, Self>),
        scratch: &mut ScratchArena<'_, Self>,
        scratch_ci: &mut ScratchArena<'_, Self::CI>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self::CI> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self::CI> + CKKSCtBounds,
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

/// Implements [`CKKSReplicatedCIBootstrapImpl`] for `$be` with the twin `$ci` by forwarding every method to
/// [`CKKSReplicatedCIBootstrapReference`](crate::reference::replicated::CKKSReplicatedCIBootstrapReference).
#[macro_export]
macro_rules! impl_ckks_replicated_ci_bootstrap_reference_full {
    ($be:ty, $ci:ty) => {
        unsafe impl $crate::oep::CKKSReplicatedCIBootstrapImpl for $be {
            type CI = $ci;

            fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, A, T, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ci: &::poulpy_hal::layouts::Module<$ci>,
                ct_out: &C,
                ct_in: &I,
                ctx: &$crate::layouts::ReplicatedBootstrapping<Self, F>,
                atk_infos: &A,
                tsk_infos: &T,
                encapsulation_infos: Option<(&S, &S)>,
                conversion_infos: &K,
            ) -> (usize, usize)
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
                    ci,
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
                ci: &::poulpy_hal::layouts::Module<$ci>,
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
                scratch_ci: &mut ::poulpy_hal::layouts::ScratchArena<'_, $ci>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$ci> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$ci> + $crate::CKKSCtBounds,
                K: $crate::layouts::BootstrappingKeys<Self>,
                E: ::poulpy_hal::layouts::Data,
                T: ::poulpy_hal::layouts::Data,
                ::poulpy_core::layouts::GLWECIEmbedKeyPrepared<E, Self>: ::poulpy_core::layouts::GGLWEPreparedToBackendRef<Self>,
                ::poulpy_core::layouts::GLWECITraceKeyPrepared<T, Self>: ::poulpy_core::layouts::GGLWEPreparedToBackendRef<Self>,
            {
                $crate::reference::replicated::CKKSReplicatedCIBootstrapReference::ckks_replicated_ci_bootstrap_reference(
                    module,
                    ci,
                    dst,
                    src,
                    ctx,
                    workspace,
                    keys,
                    conversion_keys,
                    scratch,
                    scratch_ci,
                )
            }
        }
    };
}

pub use crate::impl_ckks_replicated_ci_bootstrap_reference_full;
