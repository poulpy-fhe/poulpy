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
pub mod ciphertext;
pub mod plaintext;

pub use alloc::CKKSModuleAlloc;
pub use ciphertext::{CKKSCiphertext, CKKSCiphertextViewMut, CKKSMaintainOps, ScratchArenaTakeCKKS, UnnormalizedCKKSCiphertext};
pub use plaintext::CKKSPlaintext;

use std::fmt::Debug;

use rand_distr::num_traits::{Float, FromPrimitive, ToPrimitive};
pub trait CKKSScalar: Float + FromPrimitive + ToPrimitive + Debug {}

impl<T> CKKSScalar for T where T: Float + FromPrimitive + ToPrimitive + Debug {}

pub use plaintext::CKKSPlaintextVecHostCodec;
