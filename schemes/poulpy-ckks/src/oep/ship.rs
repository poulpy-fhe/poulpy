use std::collections::HashMap;

use crate::CKKSResult as Result;
use poulpy_core::{
    layouts::{Base2K, GGLWEInfos, GLWEToBackendRef, LWEInfos},
    oep::{GLWEKeyswitchImpl, GLWEZeroImpl},
};
use poulpy_hal::{
    layouts::{Backend, CnvPVecL, Module, ScratchArena, Standard},
    oep::HalVecZnxDftImpl,
};

use crate::{
    CKKSCtBounds,
    api::ShipScalar,
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, HMuxRotKeyPrepared, ShipCoeffEncodings, ShipKeysPrepared, ShipPlan},
    oep::{
        CKKSAddImpl, CKKSConjugateImpl, CKKSEncodingImpl, CKKSImagImpl, CKKSMulImpl, CKKSSubImpl,
        derived::ship::{
            ckks_ship_bootstrap_complex_into_derived, ckks_ship_bootstrap_into_derived, ckks_ship_bootstrap_tmp_bytes_derived,
        },
    },
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
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by
/// the associated method signatures.
pub unsafe trait CKKSShipCoeffEncodingImpl: Backend<Ring = Standard> {
    /// Backend-native arena bytes required by the coefficient encoding.
    /// [`Self::ckks_ship_coeff_encodings`] refuses a smaller arena with
    /// `CKKSError::Internal` before any work.
    fn ckks_ship_coeff_encodings_tmp_bytes<F>(
        module: &Module<Self>,
        plan: &ShipPlan,
        base2k: Base2K,
        complex: bool,
    ) -> Result<usize>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>;

    /// Builds the input-dependent SHIP plaintext material from the bottom
    /// ciphertext's public coefficients. Implementations must preserve the
    /// plan's degree, radix, scales, and backend ownership, and must reject
    /// malformed ciphertext layouts before reading their limbs.
    fn ckks_ship_coeff_encodings<F, Src>(
        module: &Module<Self>,
        ciphertext: &Src,
        plan: &ShipPlan,
        base2k: Base2K,
        complex: bool,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<ShipCoeffEncodings<Self::OwnedBuf, Self::ZnxWord>>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Backend contract of the SHIP bootstrapping operations
/// ([`CKKSShipOps`](crate::api::CKKSShipOps)) beyond the coefficient encoding, on the standard ring.
///
/// The bootstrap and its query default to the SHIP drivers over the selected CKKS
/// operations, the core key switch and zero families, the HAL DFT automorphism plans,
/// [`CKKSShipCoeffEncodingImpl`] and the two required hooks of this trait; an override
/// reproduces them bit for bit within its own query. The scalar `F` is a method
/// parameter bounded by `Self: CKKSEncodingImpl<F>`.
///
/// # Safety
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand, key and scratch region they touch.
pub unsafe trait CKKSShipImpl:
    Backend<Ring = Standard>
    + CKKSShipCoeffEncodingImpl
    + CKKSMulImpl
    + CKKSAddImpl
    + CKKSSubImpl
    + CKKSImagImpl
    + CKKSConjugateImpl
    + GLWEKeyswitchImpl
    + GLWEZeroImpl
    + HalVecZnxDftImpl
{
    /// Scratch of both bootstraps after their runtime validation: the largest stage, the complex
    /// variant included when the keys carry its masks.
    fn ckks_ship_bootstrap_tmp_bytes<F, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        keys: &ShipKeysPrepared<Self::OwnedBuf, Self>,
    ) -> Result<usize>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_ship_bootstrap_tmp_bytes_derived::<Self, F, Src>(module, output, input, keys)
    }

    /// Validates, switches `input` to the sparse secret, builds the leaves (coefficient encoding,
    /// [`Self::ckks_ship_masking_accumulate`], [`Self::ckks_ship_mux_rotate`]), multiplies them in a
    /// binary tree, then `output = root + conj(root)`.
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
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_ship_bootstrap_into_derived::<Self, F, Src>(module, output, input, keys, scratch)
    }

    /// As [`Self::ckks_ship_bootstrap_into`] over both coefficient halves, then
    /// `output = (v1 + i·v2) + conj(v1 − i·v2)`.
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
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_ship_bootstrap_complex_into_derived::<Self, F, Src>(module, output, input, keys, scratch)
    }

    /// Scratch of [`Self::ckks_ship_masking_accumulate`] for `plan` at `base2k`.
    fn ckks_ship_masking_accumulate_tmp_bytes(module: &Module<Self>, plan: &ShipPlan, base2k: usize) -> usize;

    /// Hook without an API twin: `acc = Σ_i masks[i] · pis[i]` at the plan's working scale,
    /// reproducing [`CKKSShipReference::ckks_ship_masking_accumulate_reference`](crate::reference::ship::CKKSShipReference::ckks_ship_masking_accumulate_reference)
    /// bit for bit within [`Self::ckks_ship_masking_accumulate_tmp_bytes`]. Required: the masks are
    /// key-generation prepared convolution operands with no CKKS metadata, which no CKKS
    /// product or dot product takes.
    fn ckks_ship_masking_accumulate(
        module: &Module<Self>,
        acc: &mut CKKSCiphertextOwned<Self>,
        plan: &ShipPlan,
        masks: &[CnvPVecL<Self::OwnedBuf, Self::DftWord, Self>],
        pis: &[CKKSPlaintextOwned<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>;

    /// Scratch of [`Self::ckks_ship_mux_rotate`] on `ct` for `term_count` keys like `key`.
    fn ckks_ship_mux_rotate_tmp_bytes<C, K>(module: &Module<Self>, ct: &C, key: &K, term_count: usize) -> usize
    where
        C: LWEInfos,
        K: GGLWEInfos;

    /// Hook without an API twin: the hoisted mux rotation `ct ← Σ_d β_d · Rot_{rot_d}(ct)` over
    /// one digit position's `keys`, the output automorphisms read from `plans`, reproducing
    /// [`CKKSShipReference::ckks_ship_mux_rotate_reference`](crate::reference::ship::CKKSShipReference::ckks_ship_mux_rotate_reference)
    /// bit for bit within [`Self::ckks_ship_mux_rotate_tmp_bytes`]. Required: one gadget
    /// decomposition serves every key of the group and each product is twisted in the DFT
    /// domain before the single normalization, which no core key switch or automorphism offers.
    fn ckks_ship_mux_rotate(
        module: &Module<Self>,
        ct: &mut CKKSCiphertextOwned<Self>,
        keys: &[HMuxRotKeyPrepared<Self::OwnedBuf, Self>],
        plans: &HashMap<i64, Self::AutomorphismPlan>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>;
}

/// Selects the reference SHIP hooks of `CKKSShipImpl` for the standard backend `$be`.
#[macro_export]
macro_rules! impl_ckks_ship_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSShipImpl for $be {
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
            ) -> $crate::CKKSResult<()> {
                $crate::reference::ship::CKKSShipReference::ckks_ship_mux_rotate_reference(module, ct, keys, plans, scratch)
            }
        }
    };
}

pub use crate::impl_ckks_ship_reference_full;
