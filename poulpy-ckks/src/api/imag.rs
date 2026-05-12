use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Multiplication and division of a ciphertext by the imaginary unit `i`.
///
/// In the CKKS slot layout each pair of conjugate slots `(z_j, z̄_j)` is
/// mapped to real and imaginary interleaved coefficients of the underlying
/// polynomial.  Multiplying by `i` rotates every complex slot value by 90
/// degrees: `z_j ↦ i · z_j`.
///
/// These operations do not consume homomorphic capacity.
///
/// # Metadata
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
pub trait CKKSImagOps<BE: Backend> {
    fn ckks_mul_i_tmp_bytes(&self) -> usize;

    /// Computes `dst = i · src` (multiply every slot by the imaginary unit).
    fn ckks_mul_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = i · dst` in-place.  Metadata is unchanged.
    fn ckks_mul_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_div_i_tmp_bytes(&self) -> usize;

    /// Computes `dst = src / i = −i · src` (multiply every slot by `−i`).
    fn ckks_div_i_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = dst / i` in-place.  Metadata is unchanged.
    fn ckks_div_i_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
