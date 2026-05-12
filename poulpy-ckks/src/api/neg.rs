use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Homomorphic negation.
///
/// Negation does not consume homomorphic capacity.
///
/// # Metadata
///
/// ```text
/// offset         = max(0, src.effective_k() − dst.max_k())
///
/// log_delta_out  = src.log_delta
/// log_budget_out = src.log_budget − offset
/// ```
///
/// For `_assign` the buffer is the same as the source so `offset = 0` and
/// metadata is unchanged.
pub trait CKKSNegOps<BE: Backend> {
    fn ckks_neg_tmp_bytes(&self) -> usize;

    /// Computes `dst = -src`.
    fn ckks_neg_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes `dst = -dst` in-place.  Metadata is unchanged.
    fn ckks_neg_assign<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos;
}
