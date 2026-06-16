use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// CKKS rescaling and level-alignment.
///
/// Rescaling discards the least-significant `k` bits of the torus
/// representation by right-shifting the polynomial.  This lowers
/// `log_budget` without changing `log_delta`, and is the standard way to
/// recover homomorphic capacity consumed by multiplications.
///
/// Unlike most other `_into` operations, `ckks_rescale_into` does **not**
/// apply a destination-capacity offset: after right-shifting by `k` bits the
/// result is strictly smaller than the source, so a correctly sized `dst`
/// (at least `src.effective_k() − k` bits of capacity) will always hold the
/// result without truncation.
///
/// # Metadata
///
/// ## Rescale by `k` bits (`ckks_rescale_assign` / `ckks_rescale_into`)
///
/// ```text
/// log_delta_out  = src.log_delta   (unchanged)
/// log_budget_out = src.log_budget − k
/// ```
///
/// Errors with `InsufficientHomomorphicCapacity` if `k > src.log_budget`.
///
/// ## Align two ciphertexts (`ckks_align_pair`)
///
/// Brings two ciphertexts to the same `log_budget` by rescaling whichever
/// has more remaining capacity:
///
/// ```text
/// // let higher = whichever of a, b has the larger log_budget
/// higher.log_budget -= higher.log_budget − lower.log_budget
/// ```
///
/// After the call both operands share the same `log_budget`.  If they already
/// have equal budgets neither is modified.
///
/// ## Re-scale by `bits` (`ckks_scale_down_assign` / `ckks_scale_up_assign`)
///
/// These move the boundary between precision and headroom **without changing
/// the encrypted value**, by shifting the torus polynomial to compensate.
/// They keep `effective_k = log_delta + log_budget` constant, so storage is
/// never reallocated:
///
/// ```text
/// // ckks_scale_down_assign: trade precision for headroom
/// log_delta_out  = ct.log_delta  − bits
/// log_budget_out = ct.log_budget + bits
///
/// // ckks_scale_up_assign: trade headroom for precision
/// log_delta_out  = ct.log_delta  + bits
/// log_budget_out = ct.log_budget − bits
/// ```
///
/// `ckks_scale_down_assign` errors with `InsufficientScalePrecision` if
/// `bits > ct.log_delta`; `ckks_scale_up_assign` errors with
/// `InsufficientHomomorphicCapacity` if `bits > ct.log_budget`.
pub trait CKKSRescaleOps<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_rescale_into`].
    fn ckks_rescale_tmp_bytes(&self) -> usize;

    /// Rescales `ct` in-place by `k` bits.
    ///
    /// Errors with `InsufficientHomomorphicCapacity` if `k > ct.log_budget`.
    fn ckks_rescale_assign<Dst>(&self, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Writes a `k`-bit rescaled copy of `src` into `dst`.
    ///
    /// Errors with `InsufficientHomomorphicCapacity` if `k > src.log_budget`.
    /// `dst` must have at least `src.effective_k() − k` bits of capacity.
    fn ckks_rescale_into<Dst, Src>(&self, dst: &mut Dst, k: usize, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Lowers `ct`'s working scale by `bits`, preserving the encrypted value.
    ///
    /// Right-shifts the polynomial by `bits`, discarding that many
    /// least-significant precision bits and turning the freed space into
    /// budget.  A subsequent multiplication then consumes only the reduced
    /// scale.
    ///
    /// Errors with `InsufficientScalePrecision` if `bits > ct.log_delta`.
    /// Uses the scratch reported by [`Self::ckks_rescale_tmp_bytes`].
    fn ckks_scale_down_assign<Dst>(&self, ct: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Inverse of [`Self::ckks_scale_down_assign`]: raises `ct`'s working scale
    /// by `bits`, preserving the encrypted value.
    ///
    /// Left-shifts the polynomial into the budget headroom.
    ///
    /// Errors with `InsufficientHomomorphicCapacity` if `bits > ct.log_budget`.
    /// Uses the scratch reported by [`Self::ckks_rescale_tmp_bytes`].
    fn ckks_scale_up_assign<Dst>(&self, ct: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Returns scratch bytes required by [`Self::ckks_align_pair`].
    fn ckks_align_tmp_bytes(&self) -> usize;

    /// Equalizes the `log_budget` of `a` and `b` by rescaling whichever has
    /// more remaining capacity down to the other's level.
    ///
    /// Either `a` or `b` may be modified; the caller cannot assume which one
    /// is left unchanged.  If both already share the same `log_budget`,
    /// neither is touched.  Errors propagate from the underlying rescale.
    fn ckks_align_pair<A, B>(&self, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
