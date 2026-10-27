use std::collections::HashMap;

use crate::CKKSResult as Result;
use poulpy_core::layouts::{Base2K, GGLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::{
    layouts::{Backend, CnvPVecL, Module, ScratchArena, Standard},
    oep::HalVecZnxDftImpl,
};

use crate::{
    CKKSCtBounds,
    api::ShipScalar,
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, HMuxRotKeyPrepared, ShipCoeffEncodings, ShipKeysPrepared, ShipPlan},
    oep::CKKSEncodingImpl,
};

/// Backend extension point for the SHIP coefficient encoding: the bottom
/// ciphertext -> hoisted-plaintext-material transformation consumed by the
/// SHIP pipeline.
///
/// Implementing this trait is the entry requirement for the SHIP operations
/// ([`CKKSShipOps`](crate::api::CKKSShipOps)). The trait imposes no FFT
/// engine, encoder, host codec, or transfer capability; a backend may
/// implement the whole step as one fused native kernel. The complete scheme
/// definition (host ciphertext -> plaintext material) is exposed as
/// [`ship_coeff_encodings_host`](crate::encoding::ship_coeff_encodings_host)
/// so host reference implementations derive from the same math.
///
/// # Safety
/// Implementations return material byte-identical with the host definition
/// [`ship_coeff_encodings_host`](crate::reference::encoding::ship_coeff_encodings_host) on every backend and
/// platform, report for the `_tmp_bytes` query a bound that covers the body, and refuse as documented. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSShipCoeffEncodingImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::ckks_ship_coeff_encodings`], in backend-native arena bytes. That method refuses a
    /// smaller arena with `CKKSError::Internal` before any work.
    fn ckks_ship_coeff_encodings_tmp_bytes<F>(
        module: &Module<Self>,
        base2k: Base2K,
        complex: bool,
        plan: &ShipPlan,
    ) -> Result<usize>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>;

    /// No reference body is wired: this crate carries no implementation. Builds the input-dependent SHIP plaintext
    /// material from the bottom ciphertext's public coefficients, byte-identical with the host definition
    /// [`ship_coeff_encodings_host`](crate::reference::encoding::ship_coeff_encodings_host) on every backend and
    /// platform, within [`Self::ckks_ship_coeff_encodings_tmp_bytes`]. Preserves the plan's degree, radix, scales and
    /// backend ownership, and rejects malformed ciphertext layouts before reading their limbs.
    fn ckks_ship_coeff_encodings<F, Src>(
        module: &Module<Self>,
        base2k: Base2K,
        complex: bool,
        ciphertext: &Src,
        plan: &ShipPlan,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<ShipCoeffEncodings<Self::OwnedBuf, Self::ZnxWord>>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Backend contract of the SHIP bootstrapping operations
/// ([`CKKSShipOps`](crate::api::CKKSShipOps)) beyond the coefficient encoding, on the standard ring. The scalar `F`
/// is a method parameter bounded by `Self: CKKSEncodingImpl<F>`.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSShipImpl: Backend<Ring = Standard> {
    /// Returns
    /// [`ckks_ship_bootstrap_tmp_bytes_reference`](crate::reference::ship::CKKSShipReference::ckks_ship_bootstrap_tmp_bytes_reference)
    /// or a smaller bound covering the body, refusing as it does. Scratch bound of [`Self::ckks_ship_bootstrap_into`]
    /// and [`Self::ckks_ship_bootstrap_complex_into`].
    fn ckks_ship_bootstrap_tmp_bytes<F, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        keys: &ShipKeysPrepared<Self::OwnedBuf, Self>,
    ) -> Result<usize>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_ship_bootstrap_into_reference`](crate::reference::ship::CKKSShipReference::ckks_ship_bootstrap_into_reference)
    /// bit for bit, within [`Self::ckks_ship_bootstrap_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_ship_bootstrap_into<F, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        keys: &ShipKeysPrepared<Self::OwnedBuf, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_ship_bootstrap_complex_into_reference`](crate::reference::ship::CKKSShipReference::ckks_ship_bootstrap_complex_into_reference)
    /// bit for bit, within [`Self::ckks_ship_bootstrap_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_ship_bootstrap_complex_into<F, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        keys: &ShipKeysPrepared<Self::OwnedBuf, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;

    /// Scratch bound of [`Self::ckks_ship_masking_accumulate`] for `plan` at `base2k`; may report less than the
    /// reference, never less than the body takes.
    fn ckks_ship_masking_accumulate_tmp_bytes(module: &Module<Self>, plan: &ShipPlan, base2k: usize) -> usize;

    /// Hook without an API twin. Reproduces
    /// [`ckks_ship_masking_accumulate_reference`](crate::reference::ship::CKKSShipReference::ckks_ship_masking_accumulate_reference)
    /// bit for bit, within [`Self::ckks_ship_masking_accumulate_tmp_bytes`]; free in kernels and scratch use. It
    /// computes `acc = Σ_i masks[i] · pis[i]` at the plan's working scale. Required: the masks are key-generation
    /// prepared convolution operands with no CKKS metadata, which no CKKS product or dot product takes. The caller, the
    /// leaf construction of the bootstrap, allocates `acc` and records its metadata.
    fn ckks_ship_masking_accumulate(
        module: &Module<Self>,
        acc: &mut CKKSCiphertextOwned<Self>,
        plan: &ShipPlan,
        masks: &[CnvPVecL<Self::OwnedBuf, Self::DftWord, Self>],
        pis: &[CKKSPlaintextOwned<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>;

    /// Scratch bound of [`Self::ckks_ship_mux_rotate`] on `ct` for `term_count` keys like `key`; may report less than
    /// the reference, never less than the body takes.
    fn ckks_ship_mux_rotate_tmp_bytes<C, K>(module: &Module<Self>, ct: &C, key: &K, term_count: usize) -> usize
    where
        C: LWEInfos,
        K: GGLWEInfos;

    /// Hook without an API twin. Reproduces
    /// [`ckks_ship_mux_rotate_reference`](crate::reference::ship::CKKSShipReference::ckks_ship_mux_rotate_reference)
    /// bit for bit, within [`Self::ckks_ship_mux_rotate_tmp_bytes`]; free in kernels and scratch use. It computes the
    /// hoisted mux rotation `ct ← Σ_d β_d · Rot_{rot_d}(ct)` over one digit position's `keys`, the output automorphisms
    /// read from `plans`, whose type the HAL DFT family fixes. Required: one gadget decomposition serves every key of
    /// the group and each product is twisted in the DFT domain before the single normalization, which no core key
    /// switch or automorphism offers. The caller, the leaf construction of the bootstrap, records the metadata of `ct`.
    fn ckks_ship_mux_rotate(
        module: &Module<Self>,
        ct: &mut CKKSCiphertextOwned<Self>,
        keys: &[HMuxRotKeyPrepared<Self::OwnedBuf, Self>],
        plans: &HashMap<i64, <Self as HalVecZnxDftImpl>::AutomorphismPlan>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Self: HalVecZnxDftImpl;
}

/// Selects the reference implementation of `CKKSShipImpl` for the standard backend `$be`.
#[macro_export]
macro_rules! impl_ckks_ship_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSShipImpl for $be {
            fn ckks_ship_bootstrap_tmp_bytes<F, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &$crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                keys: &$crate::layouts::ShipKeysPrepared<<Self as ::poulpy_hal::layouts::Backend>::OwnedBuf, Self>,
            ) -> $crate::CKKSResult<usize>
            where
                F: $crate::api::ShipScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::ship::CKKSShipReference::ckks_ship_bootstrap_tmp_bytes_reference::<F, Src>(
                    module, output, input, keys,
                )
            }

            fn ckks_ship_bootstrap_into<F, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                keys: &$crate::layouts::ShipKeysPrepared<<Self as ::poulpy_hal::layouts::Backend>::OwnedBuf, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                F: $crate::api::ShipScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::ship::CKKSShipReference::ckks_ship_bootstrap_into_reference::<F, Src>(
                    module, output, input, keys, scratch,
                )
            }

            fn ckks_ship_bootstrap_complex_into<F, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                output: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                input: &Src,
                keys: &$crate::layouts::ShipKeysPrepared<<Self as ::poulpy_hal::layouts::Backend>::OwnedBuf, Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                F: $crate::api::ShipScalar,
                Self: $crate::oep::CKKSEncodingImpl<F>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::ship::CKKSShipReference::ckks_ship_bootstrap_complex_into_reference::<F, Src>(
                    module, output, input, keys, scratch,
                )
            }

            fn ckks_ship_masking_accumulate_tmp_bytes(
                module: &::poulpy_hal::layouts::Module<Self>,
                plan: &$crate::layouts::ShipPlan,
                base2k: usize,
            ) -> usize {
                $crate::reference::ship::CKKSShipReference::ckks_ship_masking_accumulate_tmp_bytes_reference(module, plan, base2k)
            }

            fn ckks_ship_masking_accumulate(
                module: &::poulpy_hal::layouts::Module<Self>,
                acc: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                plan: &$crate::layouts::ShipPlan,
                masks: &[::poulpy_hal::layouts::CnvPVecL<
                    <Self as ::poulpy_hal::layouts::Backend>::OwnedBuf,
                    <Self as ::poulpy_hal::layouts::Backend>::DftWord,
                    Self,
                >],
                pis: &[$crate::layouts::CKKSPlaintextOwned<Self>],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()> {
                $crate::reference::ship::CKKSShipReference::ckks_ship_masking_accumulate_reference(
                    module, acc, plan, masks, pis, scratch,
                )
            }

            fn ckks_ship_mux_rotate_tmp_bytes<C, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &C,
                key: &K,
                term_count: usize,
            ) -> usize
            where
                C: ::poulpy_core::layouts::LWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::ship::CKKSShipReference::ckks_ship_mux_rotate_tmp_bytes_reference(module, ct, key, term_count)
            }

            fn ckks_ship_mux_rotate(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &mut $crate::layouts::CKKSCiphertextOwned<Self>,
                keys: &[$crate::layouts::HMuxRotKeyPrepared<<Self as ::poulpy_hal::layouts::Backend>::OwnedBuf, Self>],
                plans: &::std::collections::HashMap<i64, <Self as ::poulpy_hal::oep::HalVecZnxDftImpl>::AutomorphismPlan>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Self: ::poulpy_hal::oep::HalVecZnxDftImpl,
            {
                $crate::reference::ship::CKKSShipReference::ckks_ship_mux_rotate_reference(module, ct, keys, plans, scratch)
            }
        }
    };
}

pub use crate::impl_ckks_ship_reference_full;
