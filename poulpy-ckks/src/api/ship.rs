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
    layouts::{CKKSCiphertext, ShipCoeffEncodings, ShipKeysPrepared, ShipPlan},
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
pub trait ShipScalar: sealed::Sealed + CKKSEncodingScalar {
    /// Number of significant binary digits in the scalar representation.
    const MANTISSA_BITS: u32;
}

impl ShipScalar for f64 {
    const MANTISSA_BITS: u32 = f64::MANTISSA_DIGITS;
}

impl ShipScalar for crate::Quad {
    const MANTISSA_BITS: u32 = 113;
}

/// Caller-allocated CKKS SHIP half-bootstrapping.
///
/// All methods validate degree, rank, radix, metadata, and key layouts before
/// evaluating. The input is a one-limb bottom ciphertext under the dense
/// secret; it is switched to the sparse secret internally through the key
/// bundle's encapsulation key. The output is a slots-domain ciphertext at the
/// plan's raised precision under the dense secret.
pub trait CKKSShipOps<BE: Backend, F: ShipScalar> {
    /// Scratch bytes required by [`Self::ckks_ship_bootstrap_into`] and
    /// [`Self::ckks_ship_bootstrap_complex_into`]. The ciphertext and key
    /// layouts are validated while computing the bound.
    fn ckks_ship_bootstrap_tmp_bytes<Src>(
        &self,
        output: &CKKSCiphertext<BE::OwnedBuf>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    ) -> Result<usize>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Standard-arena bytes required to build the input-dependent SHIP
    /// plaintext material, including backend-native scalar/FFT workspace.
    fn ckks_ship_coeff_encodings_tmp_bytes(&self, plan: &ShipPlan, base2k: Base2K, complex: bool) -> Result<usize>;

    /// Builds the input-dependent SHIP plaintext material from the bottom
    /// ciphertext's public coefficients. This is the only SHIP primitive
    /// intended for native backend specialization; the bootstrap itself
    /// composes existing CKKS/core operations.
    fn ckks_ship_coeff_encodings<Src>(
        &self,
        ciphertext: &Src,
        plan: &ShipPlan,
        base2k: Base2K,
        complex: bool,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<ShipCoeffEncodings<BE::OwnedBuf>>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Bootstraps the bottom ciphertext `input` (one limb, real cleartexts
    /// with gap `gamma` in its coefficients) into `output`, a slots-domain
    /// ciphertext at the raised precision encoding the cleartexts.
    fn ckks_ship_bootstrap_into<Src>(
        &self,
        output: &mut CKKSCiphertext<BE::OwnedBuf>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Complex variant: `input` carries `Re(mu)` in its first `N/2`
    /// coefficients and `Im(mu)` in the last `N/2`; `output` encodes `mu`.
    /// Requires keys generated with `complex`.
    fn ckks_ship_bootstrap_complex_into<Src>(
        &self,
        output: &mut CKKSCiphertext<BE::OwnedBuf>,
        input: &Src,
        keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}
