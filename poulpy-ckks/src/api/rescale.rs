use anyhow::Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Crate-private scale management: move the precision/headroom boundary by
/// shifting the torus, keeping `k = log_delta + log_budget` (and the modulus)
/// constant, so storage is never reallocated:
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
/// (`log_budget` is derived as `k − log_delta`, so leaving the torus width `k`
/// untouched and relabelling `log_delta` is what shifts the boundary.)
///
/// This is **not** exposed publicly because `scale_down` is not value-preserving
/// over the full modulus: it does not drop the modulus, so right-shifting the
/// mask `a` leaves a residual `a·s` wraparound in the **top `bits`** of the
/// modulus. Only the message region is preserved; the result survives a *single*
/// multiply and must be flushed by `ckks_scale_up_assign` before any full-range
/// read. The overflow sits at the top of the live range and each further multiply
/// rescales it down toward the message, so callers must honour that discipline —
/// which is why this trait is crate-internal.
pub(crate) trait CKKSScaleManage<BE: Backend> {
    /// Lowers `ct`'s working scale by `bits`, preserving only the message region.
    ///
    /// Right-shifts the polynomial by `bits`, discarding that many
    /// least-significant precision bits and turning the freed space into budget.
    /// A subsequent multiplication then consumes only the reduced scale.
    ///
    /// The modulus is **not** dropped, so the right-shifted mask leaves an `a·s`
    /// overflow in the **top `bits`** of the modulus. The message survives a single
    /// following multiply (the overflow stays high) and is flushed by
    /// [`Self::ckks_scale_up_assign`], but until then the value is dirty above the
    /// message: never decode, keyswitch, or relinearize the result over its full
    /// range, and never assume the freed budget bits are zero.
    ///
    /// Errors with `InsufficientScalePrecision` if `bits > ct.log_delta`.
    fn ckks_scale_down_assign<Dst>(&self, ct: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Inverse of [`Self::ckks_scale_down_assign`]: raises `ct`'s working scale
    /// by `bits`, flushing the `scale_down` overflow and restoring the value.
    ///
    /// Left-shifts the polynomial into the budget headroom, pushing any top-`bits`
    /// overflow to bit `≥ max_k` (`≡ 0 mod q`).
    ///
    /// Errors with `InsufficientHomomorphicCapacity` if `bits > ct.log_budget`.
    fn ckks_scale_up_assign<Dst>(&self, ct: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
