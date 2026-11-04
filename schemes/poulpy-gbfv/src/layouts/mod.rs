//! GBFV layouts.

mod batch_encoder;
mod ntt;
mod plaintext_modulus;

pub use batch_encoder::GBFVBatchEncoder;
#[cfg(test)]
pub(crate) use ntt::bit_reverse;
pub use ntt::{
    GBFVNttBuffer, GBFVNttBufferBackendMut, GBFVNttBufferBackendRef, GBFVNttBufferToBackendMut, GBFVNttBufferToBackendRef,
    GBFVNttInfos, GBFVNttPlan, GBFVNttPlanBackendRef, GBFVNttPlanToBackendRef,
};
pub(crate) use ntt::{read_residue, write_residue};
pub use plaintext_modulus::GBFVPlaintextModulus;
