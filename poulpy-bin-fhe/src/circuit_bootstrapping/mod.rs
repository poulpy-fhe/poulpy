//! Circuit bootstrapping: lifting a GLWE ciphertext to a GGSW.
//!
//! Circuit bootstrapping transforms a GLWE ciphertext that encrypts a small
//! plaintext value (typically a single bit) into a GGSW ciphertext.  The
//! output GGSW can be used as a CMux selector, enabling the evaluation of
//! arbitrary Boolean circuits on ciphertexts without additional noise growth
//! per gate.
//!
//! ## Algorithm
//!
//! The circuit bootstrapping procedure composes three operations:
//!
//! 1. **Blind rotation** ([`super::blind_rotation`]): A programmable LUT is
//!    evaluated on an LWE ciphertext, producing a fresh GLWE ciphertext whose
//!    coefficients encode GGSW row values at the desired plaintext precision.
//! 2. **Trace / packing**: The Galois-automorphism key (`atk`) is applied to
//!    project a single coefficient out of the GLWE via a partial trace (with an
//!    optional re-packing step when `log_gap_out != log_gap_in`).
//! 3. **GGLWE-to-GGSW key-switch**: The tensor-switching key (`tsk`) converts
//!    the GGLWE intermediate into the final GGSW form.
//!
//! ## Output Modes
//!
//! - **`execute_to_constant`**: The GGSW encrypts the plaintext value as the
//!   constant term of a polynomial, which is the standard form.
//! - **`execute_to_exponent`**: The GGSW encrypts the value in the exponent
//!   of the polynomial variable; useful for ring-homomorphic products.
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |------|------|
//! | [`CircuitBootstrappingKey`] | Raw key bundle (BRK + ATK + TSK) |
//! | [`CircuitBootstrappingKeyPrepared`] | DFT-prepared form for on-line evaluation |
//! | [`CircuitBootstrappingKeyLayout`] | Dimension descriptor |
//! | [`CircuitBootstrappingKeyInfos`] | Accessor trait for key dimensions |
mod circuit;
mod key;
mod key_compressed;
mod key_prepared;

#[cfg(all(test, feature = "enable-bin-fhe"))]
pub mod tests;

pub(crate) fn trace_galois_elements(log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
    (0..log_n)
        .map(|i| {
            if i == 0 {
                -1
            } else {
                poulpy_hal::layouts::galois_element(1 << (i - 1), cyclotomic_order)
            }
        })
        .collect()
}

pub use circuit::*;
pub use key::*;
// pub use key_compressed::*;
pub use key_prepared::*;
