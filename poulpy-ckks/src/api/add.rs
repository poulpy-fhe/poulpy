use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Ciphertext and plaintext addition.
///
/// The sum is not normalized: the destination's canonical flag is cleared and
/// the next operation that reads it through a DFT normalizes it first. An
/// operand aligned by a shift is normalized by that shift.
///
/// # Metadata
///
/// ## Ciphertext–ciphertext addition (`ckks_add_into` / `ckks_add_assign`)
///
/// Both operands are automatically shifted to the same torus level before
/// addition, so different `log_budget` values are accepted without manual
/// alignment.
///
/// For `_into` variants the destination capacity can reduce the result:
///
/// ```text
/// offset         = max(0, min(a.k(), b.k()) − dst.k())
///
/// log_delta_out  = min(a.log_delta,  b.log_delta)
/// log_budget_out = min(a.log_budget, b.log_budget) − offset
/// ```
///
/// For `_assign` variants `dst` is already the buffer being operated on so
/// `offset = 0` and the formula reduces to:
///
/// ```text
/// log_delta_out  = min(dst.log_delta, a.log_delta)
/// log_budget_out = min(dst.log_budget, a.log_budget)
/// ```
///
/// Addition does not consume homomorphic capacity beyond the offset.
///
/// ## Ciphertext–plaintext-vector addition (`ckks_add_pt_vec_*`)
///
/// The full plaintext polynomial is added coefficient-wise in the ZNX domain.
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
/// ## Ciphertext–plaintext-constant addition (`ckks_add_pt_const_*`)
///
/// Adds a single quantized constant (one ZNX coefficient of a plaintext) to
/// one coefficient slot of the ciphertext.  Metadata follows the same rule as
/// the `pt_vec` variants above.
pub trait CKKSAddOps<BE: Backend> {
    fn ckks_add_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = a + b`.
    ///
    /// Operands with differing `log_budget` are aligned automatically.
    /// See the trait-level documentation for the full metadata rule.
    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst += a` in-place.
    ///
    /// `dst` and `a` are aligned automatically if their `log_budget` differs.
    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Scratch bytes for [`Self::ckks_add_one_assign`].
    fn ckks_add_one_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst += 1` in-place.
    ///
    /// The exact integer constant is added to coefficient slot `0`.
    /// Metadata is preserved. Size scratch with [`Self::ckks_add_one_tmp_bytes`].
    fn ckks_add_one_assign<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_add_pt_vec_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = a + pt` where `pt` is a full plaintext polynomial.
    ///
    /// `pt` is added coefficient-wise after being aligned to the ciphertext's
    /// torus level.  Metadata is inherited from `a` with the capacity offset
    /// applied (see trait-level doc).
    fn ckks_add_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos;

    /// Computes `dst += pt` in-place, where `pt` is a full plaintext polynomial.
    fn ckks_add_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos;

    fn ckks_add_pt_const_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = a + pt[pt_coeff]`, adding one quantized constant to
    /// a single coefficient slot of the ciphertext.
    ///
    /// - `dst_coeff`: target ZNX coefficient of `dst`.  Use `0` for the
    ///   real-slot constant term and `n/2` for the imaginary-slot constant
    ///   term (standard CKKS real/imaginary packing split).
    /// - `pt_coeff`: source coefficient index in `pt`. It indexes the
    ///   coefficients of the plaintext as stored, whatever its degree.
    ///
    /// Metadata is inherited from `a` with the capacity offset applied.
    fn ckks_add_pt_const_into<Dst, A, P>(
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

    /// Computes `dst += pt[pt_coeff]` in-place.
    ///
    /// See [`Self::ckks_add_pt_const_into`] for the semantics of `dst_coeff`
    /// and `pt_coeff`. `pt_coeff` indexes the coefficients of the plaintext as
    /// stored, whatever its degree.
    fn ckks_add_pt_const_assign<Dst, P>(
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
