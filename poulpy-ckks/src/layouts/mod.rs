//! CKKS-level data structures.
//!
//! Each layout wraps the corresponding `poulpy-core` GLWE primitive and adds
//! the CKKS-specific metadata needed for leveled arithmetic.
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | `CKKSCiphertext<D>` | Encrypted CKKS value: CKKS wrapper over the core GLWE ciphertext |
//! | `CKKSPlaintext<D>` | Quantized CKKS plaintext in the torus / ZNX domain |

/// Implements the full CKKS scratch-view trait bundle for a nominal backend
/// view wrapper: [`CKKSInfos`](crate::CKKSInfos), [`SetCKKSInfos`](crate::SetCKKSInfos), `SetK`, `SetSize`, `Compact` (deliberate no-op — a scratch view's borrowed limb count is fixed for the lifetime of the arena allocation), `BSGSMeta`, and `SetBSGSMeta`.
///
/// Two arms, selected by where the CKKS metadata lives:
///
/// - `inner_meta`: the wrapped `inner` itself carries the metadata (e.g. [`CKKSPlaintextViewMut`] wrapping a `CKKSPlaintext<BufMut>` via [`poulpy_core::view_wrapper!`], which also supplies `LWEInfos`). Everything forwards to `self.inner`.
/// - `self_meta`: the view stores `meta: CKKSMeta` beside a GLWE-level `inner` (e.g. [`CKKSCiphertextViewMut`]). This arm additionally implements `LWEInfos`/`GLWEInfos` by forwarding to `inner`, and derives `log_budget` from `inner.k() − meta.log_delta`.
///
/// New scratch-backed CKKS containers should be one `view_wrapper!`-style
/// struct plus one invocation of this macro (plus the backend to-ref/to-mut
/// conversions, which stay hand-written — they depend on the wrapped core
/// type's reborrow plumbing).
#[macro_export]
macro_rules! impl_ckks_infos {
    (@ckks_bundle $name:ident) => {
        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::poulpy_core::layouts::SetK for $name<'a, BE> {
            fn set_k(&mut self, k: ::poulpy_core::layouts::TorusPrecision) {
                ::poulpy_core::layouts::SetK::set_k(&mut self.inner, k);
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::poulpy_core::layouts::SetSize for $name<'a, BE> {
            fn set_size(&mut self, size: usize) {
                ::poulpy_core::layouts::SetSize::set_size(&mut self.inner, size);
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::poulpy_core::layouts::Compact for $name<'a, BE> {
            // Scratch-backed views intentionally skip compaction; the borrowed
            // limb count is fixed for the lifetime of the arena allocation.
            fn compact(&mut self) {}
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::poulpy_core::layouts::BSGSMeta for $name<'a, BE> {
            fn bsgs_log_budget(&self) -> usize {
                $crate::CKKSInfos::log_budget(self)
            }

            fn bsgs_log_delta(&self) -> usize {
                $crate::CKKSInfos::log_delta(self)
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::poulpy_core::layouts::SetBSGSMeta for $name<'a, BE> {
            fn set_bsgs_log_budget(&mut self, log_budget: usize) {
                $crate::SetCKKSInfos::set_log_budget(self, log_budget);
            }

            fn set_bsgs_log_delta(&mut self, log_delta: usize) {
                $crate::SetCKKSInfos::set_log_delta(self, log_delta);
            }
        }
    };
    (inner_meta $name:ident) => {
        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> $crate::CKKSInfos for $name<'a, BE> {
            fn meta(&self) -> $crate::CKKSMeta {
                $crate::CKKSInfos::meta(&self.inner)
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> $crate::SetCKKSInfos for $name<'a, BE> {
            fn set_meta(&mut self, meta: $crate::CKKSMeta) {
                $crate::SetCKKSInfos::set_meta(&mut self.inner, meta);
            }

            fn set_k(&mut self, k: ::poulpy_core::layouts::TorusPrecision) {
                $crate::SetCKKSInfos::set_k(&mut self.inner, k);
            }
        }

        $crate::impl_ckks_infos!(@ckks_bundle $name);
    };
    (self_meta $name:ident) => {
        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::poulpy_core::layouts::LWEInfos for $name<'a, BE> {
            fn base2k(&self) -> ::poulpy_core::layouts::Base2K {
                ::poulpy_core::layouts::LWEInfos::base2k(&self.inner)
            }

            fn n(&self) -> ::poulpy_core::layouts::Degree {
                ::poulpy_core::layouts::LWEInfos::n(&self.inner)
            }

            fn max_size(&self) -> usize {
                ::poulpy_core::layouts::LWEInfos::max_size(&self.inner)
            }

            fn k(&self) -> ::poulpy_core::layouts::TorusPrecision {
                ::poulpy_core::layouts::LWEInfos::k(&self.inner)
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> ::poulpy_core::layouts::GLWEInfos for $name<'a, BE> {
            fn rank(&self) -> ::poulpy_core::layouts::Rank {
                ::poulpy_core::layouts::GLWEInfos::rank(&self.inner)
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> $crate::CKKSInfos for $name<'a, BE> {
            fn meta(&self) -> $crate::CKKSMeta {
                self.meta
            }
        }

        impl<'a, BE: ::poulpy_hal::layouts::Backend + 'a> $crate::SetCKKSInfos for $name<'a, BE> {
            fn set_meta(&mut self, meta: $crate::CKKSMeta) {
                self.meta = meta;
            }

            fn set_k(&mut self, k: ::poulpy_core::layouts::TorusPrecision) {
                ::poulpy_core::layouts::SetK::set_k(&mut self.inner, k);
            }
        }

        $crate::impl_ckks_infos!(@ckks_bundle $name);
    };
}

mod alloc;
pub mod bootstrapping;
pub mod bootstrapping_keys;
pub mod ciphertext;
pub mod complex_diagonals;
pub mod dft;
mod encoding_buffer;
pub mod eval_mod;
pub mod mul;
pub mod paco;
pub mod plaintext;
pub mod ship;
pub(crate) mod validation;

pub use alloc::CKKSModuleAlloc;
pub use bootstrapping::{BootstrappingContext, BootstrappingPlan};
pub use bootstrapping_keys::{
    BootstrappingKeySet, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingKeysPrepared, EncapsulationKeysLayout,
};
pub use ciphertext::{
    CKKSCiphertext, CKKSCiphertextViewMut, CKKSNormalizationState, Normalized, ScratchArenaTakeCKKS, Unnormalized,
    UnnormalizedCKKSCiphertext,
};
pub use complex_diagonals::ComplexDiagonals;
pub use dft::{
    DFTMatrix, DFTMatrixFactors, DFTMatrixPrepared, DFTOutputFormat, DFTPlan, DFTType, Decode, DftDirection, DftFormat, Encode,
    Repack, Split, Standard,
};
pub use encoding_buffer::{
    CKKSEncodingBuffer, CKKSEncodingBufferBackendMut, CKKSEncodingBufferBackendRef, CKKSEncodingBufferInfos,
    CKKSEncodingBufferToBackendMut, CKKSEncodingBufferToBackendRef, CKKSEncodingBufferViewMut,
};
pub(crate) use encoding_buffer::{
    copy_encoding_buffer_into_host, copy_encoding_buffer_into_reim_host, copy_host_into_encoding_buffer,
    copy_reim_host_into_encoding_buffer,
};
pub use eval_mod::{EvalMod, EvalModBsgs, EvalModPlan, EvalModPoly, EvalModType, compile_eval_mod};
pub use mul::CKKSPreparedRight;
pub use paco::{
    PaCoContext, PaCoDFTPlan, PaCoKeyParameters, PaCoKeySet, PaCoKeySetParts, PaCoKeys, PaCoKeysPrepared, PaCoKeysPreparedParts,
    PaCoPlan, PaCoSecretSpec, PaCoSlotOrder, PaCoWorker,
};
pub use plaintext::{CKKSPlaintext, CKKSPlaintextViewMut};
pub use ship::{
    HMuxRotKey, HMuxRotKeyPrepared, ShipCoeffEncodings, ShipIndexKeys, ShipIndexKeysPrepared, ShipKeyParameters, ShipKeySet,
    ShipKeysLayout, ShipKeysPrepared, ShipPlan, ShipSecretSpec,
};

use std::fmt::Debug;

use num_traits::{Float, FromPrimitive, ToPrimitive};
pub trait CKKSScalar: Float + FromPrimitive + ToPrimitive + Debug {}

impl<T> CKKSScalar for T where T: Float + FromPrimitive + ToPrimitive + Debug {}

pub use plaintext::CKKSPlaintextVecHostCodec;
