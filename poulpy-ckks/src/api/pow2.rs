use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Multiplication and division of a ciphertext by a power of two.
///
/// These operations shift the torus polynomial by `bits` positions.  Their
/// metadata effects are **asymmetric**: `mul_pow2` preserves `log_delta` and
/// only loses `log_budget` if the destination is undersized, while `div_pow2`
/// transfers `bits` from `log_budget` to `log_delta` (a precision-aware
/// rescale) and again loses extra bits if the destination is undersized.
///
/// # Metadata
///
/// ## Multiply by `2^bits` (`ckks_mul_pow2_into` / `ckks_mul_pow2_assign`)
///
/// The polynomial is left-shifted by `bits` positions.  The encrypted value
/// becomes `message * 2^bits`, encoded at the same precision `log_delta`.
///
/// For `_into` variants:
///
/// ```text
/// offset         = max(0, src.effective_k() − dst.max_k())
///
/// log_delta_out  = src.log_delta
/// log_budget_out = src.log_budget − offset
/// ```
///
/// For `_assign` variants `offset = 0` and metadata is unchanged.
///
/// **Note**: the metadata does not account for the increased magnitude of the
/// encrypted value.  It is the caller's responsibility to ensure that
/// `message * 2^bits` still fits within the nominal `log_budget` headroom.
///
/// ## Divide by `2^bits` (`ckks_div_pow2_into` / `ckks_div_pow2_assign`)
///
/// The polynomial is right-shifted by `bits` positions, increasing precision
/// at the cost of capacity.
///
/// For `_into` variants:
///
/// ```text
/// offset         = max(0, src.effective_k() − dst.max_k())
///
/// log_delta_out  = src.log_delta + bits
/// log_budget_out = src.log_budget − bits − offset
/// ```
///
/// For `_assign` variants `offset = 0`:
///
/// ```text
/// log_delta_out  = dst.log_delta + bits
/// log_budget_out = dst.log_budget − bits
/// ```
///
/// **Capacity consumed**: `bits` bits (plus `offset` for undersized destinations).
/// Errors with `InsufficientHomomorphicCapacity` if `bits (+ offset) > src.log_budget`.
pub trait CKKSPow2Ops<BE: Backend> {
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize;

    /// Computes `dst = src * 2^bits`.
    ///
    /// Left-shifts the torus polynomial by `bits` positions.
    /// See the trait-level documentation for the metadata rule.
    fn ckks_mul_pow2_into<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst *= 2^bits` in-place.  Metadata is unchanged.
    fn ckks_mul_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_div_pow2_tmp_bytes(&self) -> usize;

    /// Computes `dst = src / 2^bits`.
    ///
    /// Right-shifts the torus polynomial by `bits` positions.
    /// See the trait-level documentation for the metadata rule.
    fn ckks_div_pow2_into<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst /= 2^bits` in-place.
    ///
    /// ```text
    /// log_delta_out  = dst.log_delta + bits
    /// log_budget_out = dst.log_budget − bits
    /// ```
    ///
    /// Errors with `InsufficientHomomorphicCapacity` if `bits > dst.log_budget`.
    fn ckks_div_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
