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

mod alloc;
pub mod bootstrapping;
pub mod bootstrapping_keys;
pub mod ciphertext;
pub mod complex_diagonals;
pub mod dft;
pub mod eval_mod;
pub mod mul;
pub mod plaintext;

pub use alloc::CKKSModuleAlloc;
pub use bootstrapping::{BootstrappingContext, BootstrappingPipeline, BootstrappingPlan};
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
pub use eval_mod::{EvalMod, EvalModBsgs, EvalModPlan, EvalModPoly, EvalModType};
pub use mul::CKKSPreparedRight;
pub use plaintext::CKKSPlaintext;

use std::fmt::Debug;

use rand_distr::num_traits::{Float, FromPrimitive, ToPrimitive};
pub trait CKKSScalar: Float + FromPrimitive + ToPrimitive + Debug {}

impl<T> CKKSScalar for T where T: Float + FromPrimitive + ToPrimitive + Debug {}

pub use plaintext::CKKSPlaintextVecHostCodec;
