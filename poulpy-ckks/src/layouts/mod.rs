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
//! | `CKKSPlaintextVecZnx<D>` | Quantized vector CKKS plaintext in the torus / ZNX domain |
//! | `CKKSPlaintextVecRnx<F>` | Floating-point vector CKKS plaintext in the RNX domain |
//! | `CKKSPlaintextCstZnx` | Quantized constant CKKS plaintext in the torus / ZNX domain |
//! | `CKKSPlaintextCstRnx<F>` | Floating-point constant CKKS plaintext in the RNX domain |

pub mod ciphertext;
pub mod plaintext;

pub use ciphertext::{CKKSCiphertext, CKKSMaintainOps};
pub use plaintext::{
    CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
    CKKSPlaintextVecZnx,
};
pub use plaintext::{CKKSPlaintextRnx, CKKSPlaintextZnx};
