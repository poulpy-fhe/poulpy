use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Homomorphic negation.
///
/// Negation does not consume homomorphic capacity.
///
/// # Metadata
///
/// ```text
/// offset         = max(0, src.k() − dst.k())
///
/// log_delta_out  = src.log_delta
/// log_budget_out = src.log_budget − offset
/// ```
///
/// For `_assign` the buffer is the same as the source so `offset = 0` and
/// CKKS layout and scale metadata is unchanged. Component noise metadata is cleared.
pub trait CKKSNegOps<BE: Backend, Rg: Ring = Standard> {
    fn ckks_neg_tmp_bytes(&self, res_size: usize) -> usize;

    /// Computes `dst = -src`.
    fn ckks_neg_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, Rg> + CKKSCtBounds;

    /// Computes `dst = -dst` in-place, preserving CKKS layout and scale metadata.
    /// Component noise metadata is cleared.
    fn ckks_neg_assign<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos;
}
