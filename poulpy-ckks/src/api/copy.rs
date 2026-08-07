use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Level-aware ciphertext copy.
///
/// Copies the polynomial data from `src` into `dst`, adjusting for any
/// mismatch between the source's precision and the destination's allocated
/// capacity.
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
/// When `dst.k() >= src.k()` the copy is lossless and
/// `offset = 0`.  When `dst.k() < src.k()`, the
/// most-significant bits that do not fit are silently dropped and
/// `log_budget` is reduced by the deficit.
pub trait CKKSCopyOps<BE: Backend> {
    fn ckks_copy_tmp_bytes(&self) -> usize;

    /// Copies `src` into `dst`.
    fn ckks_copy<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}
