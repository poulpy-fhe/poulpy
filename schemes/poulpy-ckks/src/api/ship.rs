//! Public SHIP half-bootstrapping operations.
//!
//! SHIP (Cheon, Hanrot, Kim, Stehlé) refreshes a one-limb bottom ciphertext
//! under a dense secret into a slots-domain ciphertext at the raised
//! precision, without EvalMod: the bottom ciphertext is switched to a
//! regularly-spaced sparse secret, each support slot contributes one omega
//! factor through hoisted masking and base-B mux blind rotations, and a
//! binary product tree assembles the factors. The API follows the crate's
//! caller-allocated convention.

use crate::CKKSResult as Result;
use poulpy_core::layouts::{Base2K, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::CKKSEncodingScalar,
    layouts::{CKKSCiphertextOwned, ShipCoeffEncodings, ShipKeysPrepared, ShipPlan},
};

mod sealed {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for crate::Quad {}
}

/// Scalar precision contract for SHIP's unit-circle phase encoding.
///
/// SHIP converts one-limb residues to floating-point phases, so the scalar
/// must represent every residue exactly: operations reject radixes reaching
/// [`MANTISSA_BITS`](Self::MANTISSA_BITS). Sealed to `f64` and the crate's
/// binary128 [`Quad`](crate::Quad), whose precision contracts are known here.
pub trait CKKSShipScalar: sealed::Sealed + CKKSEncodingScalar {
    /// Number of significant binary digits in the scalar representation.
    const MANTISSA_BITS: u32;
}

impl CKKSShipScalar for f64 {
    const MANTISSA_BITS: u32 = f64::MANTISSA_DIGITS;
}

impl CKKSShipScalar for crate::Quad {
    const MANTISSA_BITS: u32 = 113;
}

/// Caller-allocated CKKS SHIP half-bootstrapping.
///
/// All methods validate degree, rank, radix, metadata, and key layouts before
/// evaluating. The input is a one-limb bottom ciphertext under the dense
/// secret; it is switched to the sparse secret internally through the key
/// bundle's encapsulation key. The output is a slots-domain ciphertext at the
/// plan's raised precision under the dense secret.
pub trait CKKSShipOps<BE: Backend, F: CKKSShipScalar> {
    /// Scratch bytes required by [`Self::ckks_ship_bootstrap_into`] and
    /// [`Self::ckks_ship_bootstrap_complex_into`]. The ciphertext and key
    /// layouts are validated while computing the bound.
    ///
    /// # Errors
    ///
    /// `InvalidOperand` for the layouts [`Self::ckks_ship_bootstrap_into`] refuses, under the op
    /// `ckks_ship_bootstrap_tmp_bytes`, the complex ones checked when `keys` carry the complex masks; then the errors
    /// of [`Self::ckks_ship_coeff_encodings_tmp_bytes`], and an untyped error when the bound overflows `usize`.
    fn ckks_ship_bootstrap_tmp_bytes<Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    ) -> Result<usize>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Standard-arena bytes required to build the input-dependent SHIP
    /// plaintext material, including backend-native scalar/FFT workspace.
    ///
    /// # Errors
    ///
    /// The backend's refusal of `base2k`, `complex` or `plan`; the CPU backends have none.
    fn ckks_ship_coeff_encodings_tmp_bytes(&self, base2k: Base2K, complex: bool, plan: &ShipPlan) -> Result<usize>;

    /// Builds the input-dependent SHIP plaintext material from the bottom
    /// ciphertext's public coefficients. This is the only SHIP primitive
    /// intended for native backend specialization; the bootstrap itself
    /// composes existing CKKS/core operations.
    ///
    /// # Errors
    ///
    /// An untyped error, before anything is built, when the module or `ciphertext` is not of the plan's degree,
    /// `ciphertext` is not of rank 1 or not one limb of `base2k`, or `base2k` is outside `1..63` or reaches
    /// [`CKKSShipScalar::MANTISSA_BITS`]; then an untyped error for a phase the scalar cannot represent.
    fn ckks_ship_coeff_encodings<Src>(
        &self,
        base2k: Base2K,
        complex: bool,
        ciphertext: &Src,
        plan: &ShipPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<ShipCoeffEncodings<BE::OwnedBuf, BE::ZnxWord>>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Bootstraps the bottom ciphertext `input` (one limb, real cleartexts
    /// with gap `gamma` in its coefficients) into `output`, a slots-domain
    /// ciphertext at the raised precision encoding the cleartexts.
    ///
    /// # Errors
    ///
    /// `InvalidOperand` when the module or `input` is not of the plan's degree, `input` is not of rank 1 or not one
    /// limb of the key radix, `output` is not of the plan's degree, of rank 1 and of the key radix or holds less than
    /// the raised precision, or a mux key group of `keys` has another radix than the key radix, a key whose
    /// `max_input_k` is below the width it rotates, digits off a limb boundary of more than one width and count, keys of
    /// two sizes or a row offset; then
    /// an untyped error for an arena below [`Self::ckks_ship_bootstrap_tmp_bytes`], all before any write. Then the
    /// errors of [`Self::ckks_ship_coeff_encodings`], of the product tree (`MultiplicationPrecisionUnderflow` when it
    /// exhausts the budget, and the errors of `ckks_mul_into`) and of the final conjugation and addition. `output` is
    /// untouched on `Err`.
    fn ckks_ship_bootstrap_into<Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Complex variant: `input` carries `Re(mu)` in its first `N/2`
    /// coefficients and `Im(mu)` in the last `N/2`; `output` encodes `mu`.
    /// Requires keys generated with `complex`.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_ship_bootstrap_into`], under the op `ckks_ship_bootstrap_complex_into`, and `InvalidOperand`
    /// for keys generated without `complex`. `output` is untouched on `Err`.
    fn ckks_ship_bootstrap_complex_into<Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}
