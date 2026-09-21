use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Multiplication and division of a ciphertext by the imaginary unit `i`.
///
/// Multiplication by `i` uses the monomial `X^(N/2)`; division by `i`
/// uses `X^(-N/2)` in the ring modulo `X^N + 1`. The reference implementations
/// apply one core monomial rotation, after any required destination narrowing.
///
/// These operations consume homomorphic capacity only when narrowing.
///
/// # Metadata
///
/// For `_into` variants:
///
/// ```text
/// offset         = max(0, src.k() − dst.k())
///
/// log_delta_out  = src.log_delta
/// log_budget_out = src.log_budget − offset
/// ```
///
/// For `_assign` variants `offset = 0`. Both variants preserve sparsity and
/// set the slot kind to `SlotsKind::Complex`; scale and budget follow the rules
/// above.
pub trait CKKSImagOps<BE: Backend> {
    fn ckks_mul_i_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = i · src` (multiply every slot by the imaginary unit).
    fn ckks_mul_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = i · dst` in-place, setting the slot kind to complex.
    fn ckks_mul_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_div_i_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = src / i = −i · src` (multiply every slot by `−i`).
    fn ckks_div_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = dst / i` in-place, setting the slot kind to complex.
    fn ckks_div_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
