use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos, layouts::UnnormalizedCKKSCiphertext};

/// Normalized ciphertext and plaintext subtraction.
///
/// Subtraction is the additive inverse of addition.  Metadata rules are
/// identical to those of [`CKKSAddOps`](crate::api::CKKSAddOps).
///
/// # Metadata
///
/// ## Ciphertext–ciphertext subtraction (`ckks_sub_into` / `ckks_sub_assign`)
///
/// ```text
/// offset         = max(0, min(a.effective_k(), b.effective_k()) − dst.max_k())
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
/// offset         = max(0, a.effective_k() − dst.max_k())
///
/// log_delta_out  = a.log_delta
/// log_budget_out = a.log_budget − offset
/// ```
///
/// **Precondition**: `a.log_budget + pt.log_delta >= pt.effective_k()`.
/// Returns `PlaintextAlignmentImpossible` otherwise.
///
/// ## Ciphertext–plaintext-constant subtraction (`ckks_sub_pt_const_*`)
///
/// Metadata follows the same rule as the `pt_vec` variants above.
pub trait CKKSSubOps<BE: Backend> {
    fn ckks_sub_tmp_bytes(&self) -> usize;
    fn ckks_sub_pt_vec_tmp_bytes(&self) -> usize;

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

    /// Computes `dst -= 1` in-place.
    ///
    /// The exact integer constant is subtracted from coefficient slot `0`.
    /// Metadata is preserved.
    fn ckks_sub_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Computes `dst = a - pt` where `pt` is a full plaintext polynomial.
    fn ckks_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst -= pt` in-place.
    fn ckks_sub_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize;

    /// Computes `dst = a - pt[pt_coeff]`, subtracting one quantized constant
    /// from a single coefficient slot of the ciphertext.
    ///
    /// - `dst_coeff`: target ZNX coefficient of `dst`.  Use `0` for the
    ///   real-slot constant term and `n/2` for the imaginary-slot constant term.
    /// - `pt_coeff`: source coefficient index in `pt`.
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst -= pt[pt_coeff]` in-place.
    ///
    /// See [`Self::ckks_sub_pt_const_into`] for the semantics of
    /// `dst_coeff` and `pt_coeff`.
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
}

/// Unnormalized subtraction variants for explicit fusion loops.
///
/// See [`CKKSAddOpsUnnormalized`](crate::api::CKKSAddOpsUnnormalized) for
/// the full contract, motivation, metadata rules, and digit-growth analysis
/// (worst-case linear, typical-case Irwin–Hall O(√n)).  Everything stated
/// there applies here with `+` replaced by `−`.
pub trait CKKSSubOpsUnnormalized<BE: Backend> {
    fn ckks_sub_into_unnormalized<Dst, A, B>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_sub_assign_unnormalized<Dst, A>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_vec_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_sub_pt_vec_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_sub_pt_const_into_unnormalized<Dst, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    fn ckks_sub_pt_const_assign_unnormalized<Dst, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        dst_coeff: usize,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        UnnormalizedCKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
}
