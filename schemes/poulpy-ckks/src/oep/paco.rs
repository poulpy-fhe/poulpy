use crate::CKKSResult as Result;
use crate::layouts::CKKSPlaintextOwned;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, Module, ScratchArena, ScratchOwned, Standard},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::CKKSPaCoScalar,
    layouts::{CKKSCiphertextOwned, PaCoContext, PaCoKeys, PaCoWorker},
    oep::CKKSEncodingImpl,
};

/// Backend extension point for the PaCo coefficient encoding: the ciphertext
/// → four-β-plaintexts transformation consumed by PaCo's blind rotation.
///
/// Implementing this trait is the entry requirement for the PaCo
/// bootstrapping operations
/// ([`CKKSPaCoOps`](crate::api::CKKSPaCoOps)): a backend without a ready
/// coefficient-encoding implementation cannot bootstrap. The trait imposes no
/// FFT engine, encoder, host codec, or transfer capability: the encoding
/// method sees only the ciphertext, the compiled context (its validated plan
/// and radix), and the backend's own precomputed material, so a backend may
/// implement the whole step as one fused native kernel. This crate carries no implementation; the
/// reference implementation is [`crate::reference::encoding::ckks_paco_coeff_encodings_host`], and the complete
/// scheme definition of the step (host ciphertext → four host plaintexts) is
/// exposed as
/// [`ckks_paco_coeff_encodings_host`](crate::encoding::ckks_paco_coeff_encodings_host)
/// so host reference implementations derive from the same math.
///
/// # Safety
/// Implementations return plaintexts byte-identical with the host definition
/// [`ckks_paco_coeff_encodings_host`](crate::reference::encoding::ckks_paco_coeff_encodings_host) on every backend and
/// platform, report for the `_tmp_bytes` query a bound that covers the body, and refuse as documented. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSPaCoCoeffEncodingImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::ckks_paco_coeff_encodings`] for an input of the layout `input`, the scalar buffers of
    /// native FFT kernels included. That method refuses a smaller arena with `CKKSError::Internal` before any work.
    fn ckks_paco_coeff_encodings_tmp_bytes<F, A>(
        module: &Module<Self>,
        context: &PaCoContext<Self, F>,
        input: &A,
    ) -> Result<usize>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        A: GLWEInfos;

    /// No reference body is wired: this crate carries no implementation. Encodes the exhausted ciphertext's public
    /// coefficients as the four dense beta plaintexts consumed by PaCo blind rotation, byte-identical with the host
    /// definition [`ckks_paco_coeff_encodings_host`](crate::reference::encoding::ckks_paco_coeff_encodings_host) on every
    /// backend and platform, within [`Self::ckks_paco_coeff_encodings_tmp_bytes`]. Preserves the plan's degree,
    /// radix, scale, budget, full-slot ordering and backend ownership, and rejects malformed ciphertext layouts before
    /// reading their active limbs. The beta packing agrees numerically (within CKKS precision) with the `2C`-block
    /// coefficient-to-evaluation packing used by the sigma vectors of
    /// [`PaCoSecretSpec`](crate::layouts::PaCoSecretSpec) at key generation.
    fn ckks_paco_coeff_encodings<F, Src>(
        module: &Module<Self>,
        ciphertext: &Src,
        context: &PaCoContext<Self, F>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<[CKKSPlaintextOwned<Self>; 4]>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Backend contract of the PaCo bootstrapping operations
/// ([`CKKSPaCoOps`](crate::api::CKKSPaCoOps)) beyond the coefficient encoding, on the standard ring. The scalar `F`
/// is a method parameter bounded by `Self: CKKSEncodingImpl<F>`.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSPaCoImpl: Backend<Ring = Standard> {
    /// Returns
    /// [`ckks_paco_bootstrap_direct_tmp_bytes_reference`](crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_direct_tmp_bytes_reference)
    /// or a smaller bound covering the body, refusing as it does. Scratch bound of
    /// [`Self::ckks_paco_bootstrap_direct_into`]; every parallel worker arena must provide at least the branch part.
    fn ckks_paco_bootstrap_direct_tmp_bytes<F, K, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
    ) -> Result<usize>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Returns
    /// [`ckks_paco_bootstrap_tmp_bytes_reference`](crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_tmp_bytes_reference)
    /// or a smaller bound covering the body, refusing as it does. Scratch bound of [`Self::ckks_paco_bootstrap_into`];
    /// parallel worker arenas need only the direct bound.
    fn ckks_paco_bootstrap_tmp_bytes<F, K, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
    ) -> Result<usize>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_paco_bootstrap_direct_into_reference`](crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_direct_into_reference)
    /// bit for bit, within [`Self::ckks_paco_bootstrap_direct_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_paco_bootstrap_direct_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_paco_bootstrap_into_reference`](crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_into_reference)
    /// bit for bit, within [`Self::ckks_paco_bootstrap_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_paco_bootstrap_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_paco_bootstrap_parallel_direct_into_reference`](crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_parallel_direct_into_reference)
    /// bit for bit, within [`Self::ckks_paco_bootstrap_direct_tmp_bytes`] on the caller and its branch part on each
    /// worker; free in kernels and scratch use.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_direct_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        workers: &mut [PaCoWorker<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self> + Sync,
        ScratchOwned<Self>: ScratchOwnedBorrow<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds + Sync;

    /// Reproduces
    /// [`ckks_paco_bootstrap_parallel_into_reference`](crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_parallel_into_reference)
    /// bit for bit, within [`Self::ckks_paco_bootstrap_tmp_bytes`] on the caller and the direct branch part on each
    /// worker; free in kernels and scratch use.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        workers: &mut [PaCoWorker<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: CKKSPaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self> + Sync,
        ScratchOwned<Self>: ScratchOwnedBorrow<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Scratch bound of [`Self::ckks_slot_trace_assign`] under keys laid out like `key_infos`; may report less than the
    /// reference, never less than the body takes.
    fn ckks_slot_trace_assign_tmp_bytes<C, K>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Hook without an API twin. Reproduces
    /// [`ckks_slot_trace_assign_reference`](crate::reference::paco::CKKSPaCoReference::ckks_slot_trace_assign_reference)
    /// bit for bit, within [`Self::ckks_slot_trace_assign_tmp_bytes`] at every key layout of the fold; free in kernels
    /// and scratch use.
    fn ckks_slot_trace_assign<Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>;

    /// Scratch bound of [`Self::ckks_slot_product_assign`] under rotation keys laid out like `key_infos` and a tensor
    /// key like `tsk_infos`; may report less than the reference, never less than the body takes.
    fn ckks_slot_product_assign_tmp_bytes<C, K, T>(module: &Module<Self>, ct_infos: &C, key_infos: &K, tsk_infos: &T) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos;

    /// Hook without an API twin. Reproduces
    /// [`ckks_slot_product_assign_reference`](crate::reference::paco::CKKSPaCoReference::ckks_slot_product_assign_reference)
    /// bit for bit, within [`Self::ckks_slot_product_assign_tmp_bytes`] at every key layout of the fold; free in kernels
    /// and scratch use.
    fn ckks_slot_product_assign<Dst, H, TH>(
        module: &Module<Self>,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        tsk: &TH,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
        TH: GetTensorKey<Self>;
}

/// Selects the reference implementation of `CKKSPaCoImpl` for the standard backend `$be`.
#[macro_export]
macro_rules! impl_ckks_paco_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSPaCoImpl for $be {
            fn ckks_paco_bootstrap_direct_tmp_bytes<F, K, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &$crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                context: &$crate::layouts::PaCoContext<Self, F>,
                keys: &K,
            ) -> $crate::CKKSResult<usize>
            where
                F: $crate::api::CKKSPaCoScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                K: $crate::layouts::PaCoKeys<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_direct_tmp_bytes_reference(
                    module, output, input, context, keys,
                )
            }

            fn ckks_paco_bootstrap_tmp_bytes<F, K, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &$crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                context: &$crate::layouts::PaCoContext<Self, F>,
                keys: &K,
            ) -> $crate::CKKSResult<usize>
            where
                F: $crate::api::CKKSPaCoScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                K: $crate::layouts::PaCoKeys<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_tmp_bytes_reference(
                    module, output, input, context, keys,
                )
            }

            fn ckks_paco_bootstrap_direct_into<F, K, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                context: &$crate::layouts::PaCoContext<Self, F>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                F: $crate::api::CKKSPaCoScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                K: $crate::layouts::PaCoKeys<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_direct_into_reference(
                    module, output, input, context, keys, scratch,
                )
            }

            fn ckks_paco_bootstrap_into<F, K, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                context: &$crate::layouts::PaCoContext<Self, F>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                F: $crate::api::CKKSPaCoScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                K: $crate::layouts::PaCoKeys<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_into_reference(
                    module, output, input, context, keys, scratch,
                )
            }

            fn ckks_paco_bootstrap_parallel_direct_into<F, K, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                context: &$crate::layouts::PaCoContext<Self, F>,
                keys: &K,
                workers: &mut [$crate::layouts::PaCoWorker<Self>],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                F: $crate::api::CKKSPaCoScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                K: $crate::layouts::PaCoKeys<Self> + Sync,
                ::poulpy_hal::layouts::ScratchOwned<Self>: ::poulpy_hal::api::ScratchOwnedBorrow<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + Sync,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_parallel_direct_into_reference(
                    module, output, input, context, keys, workers, scratch,
                )
            }

            fn ckks_paco_bootstrap_parallel_into<F, K, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                context: &$crate::layouts::PaCoContext<Self, F>,
                keys: &K,
                workers: &mut [$crate::layouts::PaCoWorker<Self>],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                F: $crate::api::CKKSPaCoScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                K: $crate::layouts::PaCoKeys<Self> + Sync,
                ::poulpy_hal::layouts::ScratchOwned<Self>: ::poulpy_hal::api::ScratchOwnedBorrow<Self>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_paco_bootstrap_parallel_into_reference(
                    module, output, input, context, keys, workers, scratch,
                )
            }

            fn ckks_slot_trace_assign_tmp_bytes<C, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_infos: &C,
                key_infos: &K,
            ) -> usize
            where
                C: $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_slot_trace_assign_tmp_bytes_reference(
                    module, ct_infos, key_infos,
                )
            }

            fn ckks_slot_trace_assign<Dst, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &mut Dst,
                a: usize,
                b: usize,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_slot_trace_assign_reference(module, ct, a, b, keys, scratch)
            }

            fn ckks_slot_product_assign_tmp_bytes<C, K, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct_infos: &C,
                key_infos: &K,
                tsk_infos: &T,
            ) -> usize
            where
                C: $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_slot_product_assign_tmp_bytes_reference(
                    module, ct_infos, key_infos, tsk_infos,
                )
            }

            fn ckks_slot_product_assign<Dst, H, TH>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &mut Dst,
                a: usize,
                b: usize,
                keys: &H,
                tsk: &TH,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
                TH: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::paco::CKKSPaCoReference::ckks_slot_product_assign_reference(
                    module, ct, a, b, keys, tsk, scratch,
                )
            }
        }
    };
}

pub use crate::impl_ckks_paco_reference_full;
