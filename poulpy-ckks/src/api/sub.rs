use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Ciphertext and plaintext subtraction.
///
/// Subtraction is the additive inverse of addition. Normalization and metadata
/// rules are identical to those of [`CKKSAddOps`](crate::api::CKKSAddOps).
///
/// # Metadata
///
/// ## Ciphertext–ciphertext subtraction (`ckks_sub_into` / `ckks_sub_assign`)
///
/// ```text
/// offset         = max(0, min(a.k(), b.k()) − dst.k())
///
/// log_delta_out  = min(a.log_delta,  b.log_delta)
/// log_budget_out = min(a.log_budget, b.log_budget) − offset
/// ```
///
/// For `_assign` variants `offset = 0`.
///
/// ## Ciphertext–plaintext-vector subtraction (`ckks_sub_pt_vec_*`)
///
/// ```text
/// offset         = max(0, a.k() − dst.k())
///
/// log_delta_out  = a.log_delta
/// log_budget_out = a.log_budget − offset
/// ```
///
/// **Precondition**: `a.log_budget + pt.log_delta >= pt.k()`.
/// Returns `PlaintextAlignmentImpossible` otherwise.
///
/// ## Ciphertext–plaintext-constant subtraction (`ckks_sub_pt_const_*`)
///
/// Metadata follows the same rule as the `pt_vec` variants above.
pub trait CKKSSubOps<BE: Backend> {
    fn ckks_sub_tmp_bytes(&self, res_size: usize) -> usize;
    fn ckks_sub_pt_vec_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = a - b`.
    ///
    /// Operands with differing `log_budget` are aligned automatically.
    fn ckks_sub_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst -= a` in-place.
    fn ckks_sub_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Scratch bytes for [`Self::ckks_sub_one_assign`].
    fn ckks_sub_one_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst -= 1` in-place.
    ///
    /// The exact integer constant is subtracted from coefficient slot `0`.
    /// Metadata is preserved. Size scratch with [`Self::ckks_sub_one_tmp_bytes`].
    fn ckks_sub_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Computes `dst = a - pt` where `pt` is a full plaintext polynomial.
    fn ckks_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos;

    /// Computes `dst -= pt` in-place.
    fn ckks_sub_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos;

    fn ckks_sub_pt_const_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = a - pt[pt_coeff]`, subtracting one quantized constant
    /// from a single coefficient slot of the ciphertext.
    ///
    /// - `dst_coeff`: target ZNX coefficient of `dst`.  Use `0` for the
    ///   real-slot constant term and `n/2` for the imaginary-slot constant term.
    /// - `pt_coeff`: source coefficient index in `pt`. It indexes the
    ///   coefficients of the plaintext as stored, whatever its degree.
    ///
    /// See [`CKKSAddOps::ckks_add_pt_const_into`](crate::api::CKKSAddOps::ckks_add_pt_const_into)
    /// for further semantics.
    fn ckks_sub_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos;

    /// Computes `dst -= pt[pt_coeff]` in-place.
    ///
    /// See [`Self::ckks_sub_pt_const_into`] for the semantics of
    /// `dst_coeff` and `pt_coeff`. `pt_coeff` indexes the coefficients of the
    /// plaintext as stored, whatever its degree.
    fn ckks_sub_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos;
}
