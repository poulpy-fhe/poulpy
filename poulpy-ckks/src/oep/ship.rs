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
/// Implementations meet the contract their methods document, a numeric one with no bit-exact body, report for the
/// `_tmp_bytes` query a bound that covers the body, and refuse as documented. The memory, layout, synchrony and
/// aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
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

    /// No reference body is wired: this crate carries no implementation, and the host scheme definition
    /// [`ship_coeff_encodings_host`](crate::reference::encoding::ship_coeff_encodings_host) is the math. Builds the
    /// input-dependent SHIP plaintext material from the bottom ciphertext's public coefficients, within
    /// [`Self::ckks_ship_coeff_encodings_tmp_bytes`]. Preserves the plan's degree, radix, scales and backend ownership,
    /// and rejects malformed ciphertext layouts before reading their limbs.
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
/// ([`CKKSShipOps`](crate::api::CKKSShipOps)) beyond the coefficient encoding, on the standard ring.
///
/// The bootstrap and its query default to the SHIP drivers over the selected CKKS
/// operations, the core key switch and zero families, the HAL DFT automorphism plans,
/// [`CKKSShipCoeffEncodingImpl`] and the two required hooks of this trait; an override
/// reproduces them bit for bit within its own query. The scalar `F` is a method
/// parameter bounded by `Self: CKKSEncodingImpl<F>`.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
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
    /// Default body: `ckks_ship_bootstrap_tmp_bytes_derived`, the runtime validation (refusing as the drivers do), then
    /// the largest of the encapsulation (the larger of the selected `glwe_keyswitch_tmp_bytes` and the
    /// coefficient-encoding query, plus the bottom ciphertext it carves), the product tree (the add-plaintext,
    /// [`Self::ckks_ship_masking_accumulate_tmp_bytes`], [`Self::ckks_ship_mux_rotate_tmp_bytes`] of each group at its
    /// key count, and `ckks_mul_tmp_bytes` queries) and the final combination (its temporaries plus the conjugate, add,
    /// sub and multiply-by-`i` queries), the complex combination counted whenever the keys carry its masks. Scratch
    /// bound of [`Self::ckks_ship_bootstrap_into`] and [`Self::ckks_ship_bootstrap_complex_into`]; may report less than
    /// the reference, never less than the body takes.
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

    /// Default body: `ckks_ship_bootstrap_into_derived`. The runtime validation of the output, input and key layouts
    /// refuses before any write. Then the selected `glwe_keyswitch` of `input` into a carved ciphertext under the
    /// dense-to-sparse key (carrying the input's metadata), [`CKKSShipCoeffEncodingImpl::ckks_ship_coeff_encodings`], a
    /// leaf 0 (`glwe_zero` then `ckks_add_pt_vec_assign` of the constant encoding), one leaf per index key (a
    /// [`Self::ckks_ship_masking_accumulate`] then a [`Self::ckks_ship_mux_rotate`] per mux group), and a binary
    /// product tree of `ckks_mul_into` under the tensor key, an odd leftover carrying to the next level, a tree that
    /// exhausts the budget being a refusal, and `output = root + conj(root)` with the selected `ckks_conjugate_into`
    /// and `ckks_add_into`, the slot kind set to real.
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

    /// Default body: `ckks_ship_bootstrap_complex_into_derived`. The runtime validation of the output, input and key
    /// layouts refuses before any write. Then the encapsulation and coefficient encodings of the real bootstrap once,
    /// then for each of the two coefficient halves its own leaves (the second half with the second mask set) and
    /// product tree as there, and with `v1` and `v2` the two roots, `ckks_mul_i_into` of `v2`, `ckks_add_into` and
    /// `ckks_sub_into` of it with `v1`, `ckks_conjugate_into` of the difference, and `output = (v1 + i·v2) + conj(v1 −
    /// i·v2)` by `ckks_add_into`.
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
    /// read from `plans`. Required: one gadget decomposition serves every key of the group and each product is twisted
    /// in the DFT domain before the single normalization, which no core key switch or automorphism offers. The caller,
    /// the leaf construction of the bootstrap, records the metadata of `ct`.
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
