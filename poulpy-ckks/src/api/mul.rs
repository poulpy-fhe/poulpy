use anyhow::Result;
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, CKKSInfos, SetCKKSInfos};

/// Ciphertext–ciphertext and ciphertext–plaintext multiplication.
///
/// Multiplication is the **primary consumer** of homomorphic capacity.  Every
/// multiplication reduces `log_budget` by an amount proportional to the
/// precision of the operands, plus an additional reduction if the destination
/// buffer cannot hold the full natural result.
///
/// # Metadata
///
/// ## Ciphertext–ciphertext multiplication (`ckks_mul_*`, `ckks_square_*`)
///
/// Let:
/// ```text
/// natural_budget = min(a.log_budget, b.log_budget) − max(a.log_delta, b.log_delta)
/// log_delta_out  = min(a.log_delta, b.log_delta)
/// natural_eff_k  = natural_budget + log_delta_out
/// offset         = max(0, natural_eff_k − dst.max_k())
///
/// log_budget_out = natural_budget − offset
/// ```
///
/// **Capacity consumed by the multiplication itself**: `max(a.log_delta, b.log_delta)` bits.
/// **Additional reduction from small `dst`**: `offset` bits.
///
/// For the common case of equal-precision operands (`a.log_delta == b.log_delta == Δ`):
///
/// ```text
/// natural_eff_k  = a.log_budget   (= b.log_budget when budgets are also equal)
/// log_delta_out  = Δ
/// offset         = max(0, a.log_budget − dst.max_k())
/// log_budget_out = a.log_budget − Δ − offset
/// ```
///
/// Errors with `MultiplicationPrecisionUnderflow` if `natural_budget < 0`
/// (i.e. `min(log_budget) < max(log_delta)`).
///
/// ## Ciphertext–plaintext-vector multiplication (`ckks_mul_pt_vec_*`)
///
/// ```text
/// natural_budget = a.log_budget − pt.log_delta
/// log_delta_out  = a.log_delta
/// natural_eff_k  = natural_budget + a.log_delta
///                = a.effective_k() − pt.log_delta
/// offset         = max(0, natural_eff_k − dst.max_k())
///
/// log_budget_out = natural_budget − offset
/// ```
///
/// **Capacity consumed**: `pt.log_delta` bits (precision of the plaintext
/// multiplier), plus `offset`.
///
/// ## Ciphertext–plaintext-constant multiplication (`ckks_mul_pt_const_*`)
///
/// Identical metadata rule to the `pt_vec` variant above, using
/// `pt.log_delta` as the plaintext precision.
///
/// # Rescaling after multiplication
///
/// After a ciphertext–ciphertext multiplication the result has a lower
/// `log_budget` but the same `log_delta`.  To release the physical limbs
/// no longer needed, call
/// [`CKKSMaintainOps::ckks_compact_limbs`](crate::layouts::CKKSMaintainOps::ckks_compact_limbs).
pub trait CKKSMulOps<BE: Backend> {
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Computes `dst = a * b` using tensor-product keyswitching via `tsk`.
    ///
    /// See the trait-level documentation for the exact metadata rule including
    /// the capacity offset.
    fn ckks_mul_into<Dst, A, B, T>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst *= a` in-place using tensor-product keyswitching via `tsk`.
    fn ckks_mul_assign<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst = a * a` (squaring) using tensor-product keyswitching.
    ///
    /// Equivalent to `ckks_mul_into(dst, a, a, tsk)` with the same metadata rule.
    fn ckks_square_into<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst = dst * dst` (squaring in-place) using tensor-product keyswitching.
    fn ckks_square_assign<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Computes `dst = a * pt` where `pt` is a full plaintext polynomial.
    ///
    /// See the trait-level documentation for the exact metadata rule including
    /// the capacity offset.
    fn ckks_mul_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst *= pt` in-place.
    fn ckks_mul_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = a * pt[pt_coeff]`, multiplying by a single
    /// quantized constant from coefficient `pt_coeff` of `pt`.
    ///
    /// See the trait-level documentation for the exact metadata rule including
    /// the capacity offset.
    fn ckks_mul_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst *= pt[pt_coeff]` in-place.
    fn ckks_mul_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds;
}
