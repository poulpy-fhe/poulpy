use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Precision reduction at a fixed torus value.
///
/// Rounds `src` to `dst.k()`, dropping `drop = src.k() − dst.k()` bits of
/// precision under the scale. A [`SlotLayout::Full`](crate::SlotLayout::Full)
/// source rounds to nearest. A
/// [`SlotLayout::Replicated`](crate::SlotLayout::Replicated) source rounds with
/// error `−g·h·2^-(dst.k() + λ)`, `||h||_∞ <= 2^(λ−1)`, plus an unshaped error of
/// at most `2^-(dst.k() + λ + 1)`, where `g` is the layout's shaping polynomial
/// and `λ = min(drop, max_pass_bits)` the shaping depth
/// ([`ShapingProfile`](crate::shaping::ShapingProfile)).
///
/// # Metadata
///
/// ```text
/// log_delta_out  = src.log_delta − drop
/// log_budget_out = src.log_budget
/// ```
///
/// Other metadata is `src`'s. Component noise metadata is cleared. Fails when
/// `dst.k() > src.k()` or `drop > src.log_delta`, and for a replicated layout
/// with `2R > N`. Source and destination must have equal degrees and ranks;
/// their limb radices may differ.
pub trait CKKSRoundOps<BE: Backend> {
    fn ckks_round_tmp_bytes<Dst: CKKSCtBounds, Src: CKKSCtBounds>(&self, dst: &Dst, src: &Src) -> usize;

    /// Rounds `src` into `dst` at `dst.k()`.
    ///
    /// # Errors
    /// Returns [`crate::CKKSCompositionError::InvalidOperand`] for mismatched
    /// degrees or ranks, a wider destination, a drop exceeding the source
    /// scale, or a shaping period exceeding the degree. Refusal leaves both
    /// operands unchanged.
    ///
    /// # Panics
    /// Panics before writing when the arena is smaller than the queried budget.
    fn ckks_round_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}
