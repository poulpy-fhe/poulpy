//! Prepared right operand for hoisted CKKS `ct×ct` multiplication.

use poulpy_core::layouts::GLWELayout;
use poulpy_hal::layouts::{Backend, CnvPVecR};

/// A CKKS ciphertext prepared as the right operand of a `ct×ct` multiply.
///
/// Produced by
/// [`CKKSMulOps::ckks_prepare_right`](crate::api::CKKSMulOps::ckks_prepare_right)
/// and consumed by
/// [`CKKSMulOps::ckks_mul_prepared_assign`](crate::api::CKKSMulOps::ckks_mul_prepared_assign).
/// It bundles the backend-resident convolution operand with the scale metadata of
/// the ciphertext it was prepared from, so the multiply can derive the result
/// precision without re-reading the source. Preparing once and multiplying many
/// times (e.g. the same `X^{gsp}` across a BSGS giant-step level) hoists the
/// forward transform out of the per-multiply path.
pub struct CKKSPreparedRight<BE: Backend> {
    /// Backend-resident prepared convolution operand.
    pub(crate) prep: CnvPVecR<BE::OwnedBuf, BE>,
    /// Limb count consumed at prepare time: `ceil(k / base2k)`.
    pub(crate) size: usize,
    /// `log_delta` of the source ciphertext.
    pub(crate) log_delta: usize,
    /// Torus width `k` of the source ciphertext.
    pub(crate) k: usize,
    /// `log_sparsity` of the source ciphertext.
    pub(crate) log_sparsity: usize,
    /// Ring/radix/rank identity captured at prepare time; prepared operands are
    /// long-lived cached objects, so `ckks_mul_prepared_assign` validates this
    /// against the destination before use.
    pub(crate) layout: GLWELayout,
}
