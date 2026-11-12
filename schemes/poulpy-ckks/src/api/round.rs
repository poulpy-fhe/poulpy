use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Precision reduction at a fixed torus value.
///
/// Rounds `a` to `res.k()`, dropping `drop = a.k() - res.k()` bits of
/// precision under the scale. A [`SlotLayout::Full`](crate::SlotLayout::Full)
/// source rounds to nearest. A
/// [`SlotLayout::Replicated`](crate::SlotLayout::Replicated) source rounds with
/// error `-g·h·2^-(res.k() + λ)`, `||h||_∞ <= 2^(λ-1)`, plus an unshaped error of
/// at most `2^-(res.k() + λ + 1)`, where `g` is the layout's shaping polynomial
/// and `λ = min(drop, max_pass_bits)` the shaping depth
/// ([`ShapingProfile`](crate::shaping::ShapingProfile)).
///
/// Shaped rounding is how a replicated layout leaves its extra bits. Replicated
/// encryption rounds through it after encrypting `λ` bits wider. A caller rounds
/// the finer of two replicated operands to the other's scale before a sum, which
/// refuses replicated operands at two scales rather than round one plainly. The
/// products, rotations and key switches refuse a replicated operand, so there is
/// no shaped rescale or key switch yet.
///
/// # Metadata
///
/// ```text
/// log_delta_out  = a.log_delta - drop
/// log_budget_out = a.log_budget
/// ```
///
/// Other metadata is `a`'s. Component noise metadata is cleared. `res` comes
/// out normalized at its width and flagged canonical, whatever the flag of `a`.
/// Fails when `res.k() > a.k()` or `drop > a.log_delta`, and for a replicated
/// layout whose period exceeds the degree, `2R > N` on the standard ring and
/// `R > N` on the conjugate-invariant ring, or, on the conjugate-invariant ring,
/// a shaping polynomial without a conjugate-invariant shift. Source and destination must
/// have equal degrees and ranks. Their limb radices may differ.
pub trait CKKSRoundOps<BE: Backend, Rg: Ring = Standard> {
    fn ckks_round_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds;

    /// Rounds `a` into `res` at `res.k()`.
    ///
    /// # Errors
    /// Returns [`crate::CKKSCompositionError::InvalidOperand`] for mismatched
    /// degrees or ranks, a wider destination, a drop exceeding the source
    /// scale, a shaping period exceeding the degree, or a shaping polynomial the
    /// operand's ring cannot use. Refusal leaves both
    /// operands unchanged. Returns [`crate::CKKSError::Internal`] before writing
    /// when the arena is smaller than the queried budget.
    fn ckks_round<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds;
}
