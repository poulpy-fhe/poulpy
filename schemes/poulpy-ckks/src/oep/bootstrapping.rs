//! Backend seams for CKKS bootstrapping: the pipeline and the
//! secret-switching encapsulation around ModUp.
//!
//! ModUp's known-zero low limbs are a CKKS pipeline property, not a general
//! Core key-switch operation.

use poulpy_core::layouts::{
    GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LinearTransformation, SetBSGSMeta, prepared::GGLWEPreparedBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, CKKSResult, SetCKKSInfos,
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSCITwin,
        CKKSCiphertextOwned, CKKSEvalModInfos, CKKSModRaiseInfos, CKKSPlaintextOwned, CKKSRightOperandPrepared, EncodedLut,
    },
};

/// Backend override hook for [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps).
/// [`impl_ckks_bootstrapping_reference_full`] wires every method to the reference pipeline.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSBootstrappingImpl: CKKSCITwin {
    /// Scratch bound of [`Self::ckks_bootstrap`]; may report less than the reference, never less than the body takes.
    fn ckks_bootstrap_tmp_bytes<C1, C2, F>(
        module: &Module<Self>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<Self, F>,
        ci: Option<&Module<Self::CI>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch bound of [`Self::ckks_functional_bootstrap`]; may report less than the reference, never less than the
    /// body takes. It does not grow with the number of LUTs.
    fn ckks_functional_bootstrap_tmp_bytes<C1, C2, F>(
        module: &Module<Self>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<Self, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch bound of [`Self::ckks_mod_raise`]; may report less than the reference, never less than the body takes.
    fn ckks_mod_raise_tmp_bytes<R, A, P>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<Self>;

    /// Reproduces
    /// [`ckks_mod_raise_reference`](crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_mod_raise_reference)
    /// bit for bit, within [`Self::ckks_mod_raise_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_mod_raise<R, A, P, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &mut A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<Self>,
        K: BootstrappingLinearKeys<Self>;

    /// Scratch bound of [`Self::ckks_bootstrap_eval_mod`]; may report less than the reference, never less than the body
    /// takes.
    fn ckks_bootstrap_eval_mod_tmp_bytes<R, A, P, F>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        plan: &P,
        ci: Option<&Module<Self::CI>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>;

    /// Reproduces
    /// [`ckks_bootstrap_eval_mod_reference`](crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_eval_mod_reference)
    /// bit for bit, within [`Self::ckks_bootstrap_eval_mod_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_bootstrap_eval_mod<R, A, P, F, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        plan: &P,
        ci: Option<&Module<Self::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        F: Sync,
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>,
        K: BootstrappingKeys<Self> + Sync,
        K::TensorKey: Sync;

    /// Scratch bound of [`Self::ckks_eval_lut`]; may report less than the reference, never less than the body takes.
    fn ckks_eval_lut_tmp_bytes<R, A, P, F>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>;

    /// Reproduces
    /// [`ckks_eval_lut_reference`](crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_eval_lut_reference)
    /// bit for bit, within [`Self::ckks_eval_lut_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_eval_lut<A, P, F, K>(
        module: &Module<Self>,
        res: &mut [CKKSCiphertextOwned<Self>],
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: CKKSEvalModInfos<Self, F>,
        K: BootstrappingKeys<Self>;

    /// Reproduces
    /// [`ckks_bootstrap_reference`](crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_reference)
    /// bit for bit, within [`Self::ckks_bootstrap_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_bootstrap<F, K>(
        module: &Module<Self>,
        ct_out: &mut CKKSCiphertextOwned<Self>,
        ct_in: &CKKSCiphertextOwned<Self>,
        ctx: &BootstrappingContextPrepared<Self, F>,
        ci: Option<&Module<Self::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        F: Sync,
        K: BootstrappingKeys<Self> + Sync,
        K::TensorKey: Sync;

    /// Reproduces
    /// [`ckks_functional_bootstrap_reference`](crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_functional_bootstrap_reference)
    /// bit for bit, within [`Self::ckks_functional_bootstrap_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_functional_bootstrap<F, K>(
        module: &Module<Self>,
        ct_outs: &mut [CKKSCiphertextOwned<Self>],
        ct_in: &CKKSCiphertextOwned<Self>,
        ctx: &BootstrappingContextPrepared<Self, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<Self>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        K: BootstrappingKeys<Self>;
}

/// Backend override hook for [`CKKSBootstrappingCIOps`](crate::api::CKKSBootstrappingCIOps),
/// the bootstrap with EvalMod on the conjugate-invariant ring of [`CKKSCITwin::CI`].
/// [`impl_ckks_bootstrapping_ci_reference_full`] wires it to the reference pipeline.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSBootstrappingCIImpl: CKKSBootstrappingImpl {
    /// Scratch bound of [`Self::ckks_bootstrap_ci_into`], one size per arena as `(scratch, scratch_ci)`; may report
    /// less than the reference, never less than the body uses.
    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        module: &Module<Self>,
        ci: &Module<Self::CI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<Self, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> (usize, usize)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Reproduces
    /// [`ckks_bootstrap_ci_reference`](crate::reference::bootstrapping::CKKSBootstrappingCIReference::ckks_bootstrap_ci_reference)
    /// bit for bit: values, metadata, canonical flag, refusals and their order; free in kernels. It runs within what
    /// [`Self::ckks_bootstrap_ci_tmp_bytes`] reports for each arena.
    #[expect(
        clippy::too_many_arguments,
        reason = "the bootstrap takes a module and a scratch arena per ring"
    )]
    fn ckks_bootstrap_ci_into<Dst, Src, F, K>(
        module: &Module<Self>,
        ci: &Module<Self::CI>,
        dst: &mut Dst,
        src: &Src,
        ctx: &BootstrappingContextPrepared<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
        scratch_ci: &mut ScratchArena<'_, Self::CI>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        K: BootstrappingKeys<Self>;
}

/// Backend implementation of the modulus raise that applies the first
/// `CoeffsToSlots` factor on the raised ciphertext while it is small
/// ([`DFTPlan::with_first_factor_on_raise`](crate::layouts::DFTPlan::with_first_factor_on_raise)).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSModUpFirstFactorImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::ckks_mod_up_first_factor`]; may report less than the reference, never less than the
    /// body takes. `d2s_infos` is the dense-to-sparse key under encapsulation, `key_infos` the layout of the
    /// first-factor keys.
    fn ckks_mod_up_first_factor_tmp_bytes<Dst, Src, D2S, K>(
        module: &Module<Self>,
        dst_infos: &Dst,
        src_infos: &Src,
        factor: &LinearTransformation<CKKSRightOperandPrepared<Self>>,
        d2s_infos: Option<&D2S>,
        key_infos: &K,
    ) -> usize
    where
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        D2S: GGLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`ckks_mod_up_first_factor_reference`](crate::reference::first_factor::CKKSModUpFirstFactorReference::ckks_mod_up_first_factor_reference)
    /// bit for bit, within [`Self::ckks_mod_up_first_factor_tmp_bytes`]; free in kernels and scratch use. It honors the
    /// key layouts of the reference composition.
    fn ckks_mod_up_first_factor<Dst, Src, K>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        factor: &LinearTransformation<CKKSRightOperandPrepared<Self>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        K: BootstrappingLinearKeys<Self>;
}

/// Backend implementation of
/// `dense-to-sparse key switch -> ModUp -> sparse-to-dense key switch`.
/// A backend may exploit the known-zero limbs produced by ModUp. A
/// sparse-to-dense key with a row offset makes ModUp and its switch one key
/// switch, which reads the raised ciphertext at its own width.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSEncapsulatedModUpImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::ckks_encapsulated_mod_up`]; may report less than the reference, never less than the
    /// body takes.
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

    /// Reproduces
    /// [`ckks_encapsulated_mod_up_reference`](crate::reference::bootstrapping::CKKSEncapsulatedModUpReference::ckks_encapsulated_mod_up_reference)
    /// bit for bit, within [`Self::ckks_encapsulated_mod_up_tmp_bytes`]; free in kernels and scratch use. `scale_up` is
    /// applied to the raised ciphertext between ModUp and the sparse-to-dense switch, so the message is already at its
    /// final scale when that key-switch's noise is added. It honors the key layouts of the reference composition.
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

/// Opts a standard backend into the CKKS reference bootstrapping pipeline. The
/// backend also implements [`CKKSEncapsulatedModUpImpl`] and
/// [`CKKSModUpFirstFactorImpl`], the two raises the pipeline runs.
#[macro_export]
macro_rules! impl_ckks_bootstrapping_reference_full {
    ($be:ty, $ci:ty) => {
        impl $crate::layouts::CKKSCITwin for $be {
            type CI = $ci;
        }

        unsafe impl $crate::oep::CKKSBootstrappingImpl for $be {
            fn ckks_bootstrap_tmp_bytes<C1, C2, F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_out: &C1,
                ct_in: &C2,
                ctx: &$crate::layouts::BootstrappingContextPrepared<$be, F>,
                ci: ::core::option::Option<&::poulpy_hal::layouts::Module<$ci>>,
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
                    ci,
                    keys_layout,
                )
            }

            fn ckks_functional_bootstrap_tmp_bytes<C1, C2, F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_out: &C1,
                ct_in: &C2,
                ctx: &$crate::layouts::BootstrappingContextPrepared<$be, F>,
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

            fn ckks_mod_raise_tmp_bytes<R, A, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                plan: &P,
                keys_layout: &$crate::layouts::BootstrappingKeysLayout,
            ) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
                P: $crate::layouts::CKKSModRaiseInfos<$be>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_mod_raise_tmp_bytes_reference(
                    module,
                    res,
                    a,
                    plan,
                    keys_layout,
                )
            }

            fn ckks_mod_raise<R, A, P, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &mut A,
                plan: &P,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWEToBackendRef<$be>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWEToBackendRef<$be>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                P: $crate::layouts::CKKSModRaiseInfos<$be>,
                K: $crate::layouts::BootstrappingLinearKeys<$be>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_mod_raise_reference(
                    module, res, a, plan, keys, scratch,
                )
            }

            fn ckks_bootstrap_eval_mod_tmp_bytes<R, A, P, F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                plan: &P,
                ci: ::core::option::Option<&::poulpy_hal::layouts::Module<$ci>>,
                keys_layout: &$crate::layouts::BootstrappingKeysLayout,
            ) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
                P: $crate::layouts::CKKSEvalModInfos<$be, F>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_eval_mod_tmp_bytes_reference(
                    module,
                    res,
                    a,
                    plan,
                    ci,
                    keys_layout,
                )
            }

            fn ckks_bootstrap_eval_mod<R, A, P, F, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                plan: &P,
                ci: ::core::option::Option<&::poulpy_hal::layouts::Module<$ci>>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                F: Sync,
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWEToBackendRef<$be>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::SetBSGSMeta
                    + Send,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                P: $crate::layouts::CKKSEvalModInfos<$be, F>,
                K: $crate::layouts::BootstrappingKeys<$be> + Sync,
                K::TensorKey: Sync,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_eval_mod_reference(
                    module, res, a, plan, ci, keys, scratch,
                )
            }

            fn ckks_eval_lut_tmp_bytes<R, A, P, F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                plan: &P,
                luts: &[$crate::layouts::EncodedLut<$crate::layouts::CKKSPlaintextOwned<$be>>],
                keys_layout: &$crate::layouts::BootstrappingKeysLayout,
            ) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
                P: $crate::layouts::CKKSEvalModInfos<$be, F>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_eval_lut_tmp_bytes_reference(
                    module,
                    res,
                    a,
                    plan,
                    luts,
                    keys_layout,
                )
            }

            fn ckks_eval_lut<A, P, F, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut [$crate::layouts::CKKSCiphertextOwned<$be>],
                a: &A,
                plan: &P,
                luts: &[$crate::layouts::EncodedLut<$crate::layouts::CKKSPlaintextOwned<$be>>],
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                P: $crate::layouts::CKKSEvalModInfos<$be, F>,
                K: $crate::layouts::BootstrappingKeys<$be>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_eval_lut_reference(
                    module, res, a, plan, luts, keys, scratch,
                )
            }

            fn ckks_bootstrap<F, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_out: &mut $crate::layouts::CKKSCiphertextOwned<$be>,
                ct_in: &$crate::layouts::CKKSCiphertextOwned<$be>,
                ctx: &$crate::layouts::BootstrappingContextPrepared<$be, F>,
                ci: ::core::option::Option<&::poulpy_hal::layouts::Module<$ci>>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                F: Sync,
                K: $crate::layouts::BootstrappingKeys<$be> + Sync,
                K::TensorKey: Sync,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_bootstrap_reference(
                    module, ct_out, ct_in, ctx, ci, keys, scratch,
                )
            }

            fn ckks_functional_bootstrap<F, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_outs: &mut [$crate::layouts::CKKSCiphertextOwned<$be>],
                ct_in: &$crate::layouts::CKKSCiphertextOwned<$be>,
                ctx: &$crate::layouts::BootstrappingContextPrepared<$be, F>,
                luts: &[$crate::layouts::EncodedLut<$crate::layouts::CKKSPlaintextOwned<$be>>],
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                K: $crate::layouts::BootstrappingKeys<$be>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingReference::ckks_functional_bootstrap_reference(
                    module, ct_outs, ct_in, ctx, luts, keys, scratch,
                )
            }
        }
    };
}

/// Opts a backend into the reference bootstrap with EvalMod on the
/// conjugate-invariant ring ([`CKKSBootstrappingCIImpl`]), run by its twin `$ci`.
#[macro_export]
macro_rules! impl_ckks_bootstrapping_ci_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSBootstrappingCIImpl for $be {
            fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ci: &::poulpy_hal::layouts::Module<<$be as $crate::layouts::CKKSCITwin>::CI>,
                ct_out: &C1,
                ct_in: &C2,
                ctx: &$crate::layouts::BootstrappingContextPrepared<$be, F>,
                keys_layout: &$crate::layouts::BootstrappingKeysLayout,
            ) -> (usize, usize)
            where
                C1: $crate::CKKSCtBounds,
                C2: $crate::CKKSCtBounds,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingCIReference::ckks_bootstrap_ci_tmp_bytes_reference(
                    module,
                    ci,
                    ct_out,
                    ct_in,
                    ctx,
                    keys_layout,
                )
            }

            fn ckks_bootstrap_ci_into<Dst, Src, F, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ci: &::poulpy_hal::layouts::Module<<$be as $crate::layouts::CKKSCITwin>::CI>,
                dst: &mut Dst,
                src: &Src,
                ctx: &$crate::layouts::BootstrappingContextPrepared<$be, F>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
                scratch_ci: &mut ::poulpy_hal::layouts::ScratchArena<'_, <$be as $crate::layouts::CKKSCITwin>::CI>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                K: $crate::layouts::BootstrappingKeys<$be>,
            {
                $crate::reference::bootstrapping::CKKSBootstrappingCIReference::ckks_bootstrap_ci_reference(
                    module, ci, dst, src, ctx, keys, scratch, scratch_ci,
                )
            }
        }
    };
}

/// Opts a backend into the CKKS reference encapsulated-ModUp pipeline.
#[macro_export]
macro_rules! impl_ckks_encapsulated_mod_up_reference_full {
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
                $crate::reference::bootstrapping::CKKSEncapsulatedModUpReference::ckks_encapsulated_mod_up_tmp_bytes_reference(
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
                $crate::reference::bootstrapping::CKKSEncapsulatedModUpReference::ckks_encapsulated_mod_up_reference(
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

/// Opts a standard backend into the reference raise with the first factor.
#[macro_export]
macro_rules! impl_ckks_mod_up_first_factor_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSModUpFirstFactorImpl for $be {
            fn ckks_mod_up_first_factor_tmp_bytes<Dst, Src, D2S, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                dst_infos: &Dst,
                src_infos: &Src,
                factor: &::poulpy_core::layouts::LinearTransformation<$crate::layouts::CKKSRightOperandPrepared<$be>>,
                d2s_infos: Option<&D2S>,
                key_infos: &K,
            ) -> usize
            where
                Dst: $crate::CKKSCtBounds,
                Src: $crate::CKKSCtBounds,
                D2S: ::poulpy_core::layouts::GGLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::first_factor::CKKSModUpFirstFactorReference::ckks_mod_up_first_factor_tmp_bytes_reference(
                    module, dst_infos, src_infos, factor, d2s_infos, key_infos,
                )
            }

            fn ckks_mod_up_first_factor<Dst, Src, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                dst: &mut Dst,
                src: &mut Src,
                scale_up: usize,
                factor: &::poulpy_core::layouts::LinearTransformation<$crate::layouts::CKKSRightOperandPrepared<$be>>,
                keys: &K,
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
                K: $crate::layouts::BootstrappingLinearKeys<$be>,
            {
                $crate::reference::first_factor::CKKSModUpFirstFactorReference::ckks_mod_up_first_factor_reference(
                    module, dst, src, scale_up, factor, keys, scratch,
                )
            }
        }
    };
}

pub use crate::impl_ckks_mod_up_first_factor_reference_full;

pub use crate::impl_ckks_encapsulated_mod_up_reference_full;

pub use crate::impl_ckks_bootstrapping_reference_full;

pub use crate::impl_ckks_bootstrapping_ci_reference_full;
