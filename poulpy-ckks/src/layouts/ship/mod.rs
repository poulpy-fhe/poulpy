//! SHIP half-bootstrapping layouts.
//!
//! - [`plan`]: validated instance dimensions and derived widths.
//! - [`secret`]: the regularly-spaced sparse support.
//! - [`keyset`]: key generation, storage, and backend preparation.

pub mod keyset;
pub mod plan;
pub mod secret;

pub use keyset::{
    HMuxRotKey, HMuxRotKeyPrepared, ShipIndexKeys, ShipIndexKeysPrepared, ShipKeyParameters, ShipKeySet, ShipKeysLayout,
    ShipKeysPrepared,
};
pub use plan::ShipPlan;
pub use secret::ShipSecretSpec;

use poulpy_hal::layouts::Data;

use crate::layouts::CKKSPlaintext;

/// Input-dependent SHIP plaintext material: `pt0` (and `pt0_2` for the
/// complex bootstrap) at the raised width, plus per support slot the
/// `4*theta` rotated `pi` vectors, candidate-major, at the working width.
pub struct ShipCoeffEncodings<D: Data> {
    /// `Ecd((gamma/(4*i*pi)) * w^{b_i})` over the first coefficient half.
    pub pt0: CKKSPlaintext<D>,
    /// Second-half `pt0`, present when built for the complex bootstrap.
    pub pt0_2: Option<CKKSPlaintext<D>>,
    /// Per support slot, the `4*theta` `Rot_{p+c}(Ecd(pi_k(a)))` plaintexts.
    pub pi: Vec<Vec<CKKSPlaintext<D>>>,
}
