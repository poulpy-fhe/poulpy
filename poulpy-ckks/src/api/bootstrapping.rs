use anyhow::Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSEvalModOps, DFTOps},
};

/// CKKS bootstrapping primitives.
///
/// Bootstrapping is the pipeline `ModUp → CoeffsToSlots → EvalMod →
/// SlotsToCoeffs`. Of those, only **ModUp** (the modulus raise) is a new
/// primitive, defined directly on this trait. The other three stages are reused
/// verbatim from their own op traits, which this trait **re-exports as
/// supertraits**:
///
/// - CoeffsToSlots / SlotsToCoeffs via [`DFTOps`](crate::api::DFTOps)
///   ([`ckks_coeffs_to_slots`](DFTOps::ckks_coeffs_to_slots),
///   [`ckks_slots_to_coeffs`](DFTOps::ckks_slots_to_coeffs), and their
///   `_split` / `_repack` variants);
/// - EvalMod via [`CKKSEvalModOps`](crate::api::CKKSEvalModOps)
///   ([`ckks_eval_mod`](CKKSEvalModOps::ckks_eval_mod)).
///
/// So a value bounded by `CKKSBootstrappingOps` can drive the entire pipeline
/// through one bound. This crate intentionally provides **no orchestrator** —
/// these stay composable building blocks; assemble them (with the keys for the
/// rotation/relinearization steps) at a higher level.
/// [`BootstrappingPlan`](crate::layouts::BootstrappingPlan) carries the
/// per-stage parameterization.
///
/// # ModUp (modulus raise) in the base-`2^base2k` torus model
///
/// Unlike RNS schemes there is no prime-basis extension: raising the modulus is
/// a digit shift. A CKKS ciphertext stores its torus digits most-significant
/// first, normalized by its modulus `2^k`. Reinterpreting the *same* integer
/// coefficients under a wider modulus `2^{k'}` (`k' ≥ k`) divides the normalized
/// value by `2^{k'−k}`, i.e. right-shifts the digits.
///
/// Concretely, [`Self::ckks_mod_up_into`] copies the input (MSB-aligned) into a
/// caller-allocated wider destination and right-shifts by `dst.max_k() −
/// src.k()`. After decryption the secret-dependent term that used to
/// wrap modulo the input modulus `q = 2^{src.k()}` is no longer
/// reduced, so the cleartext is `I(X)·q + Δ·m`: exactly the input
/// [`CKKSEvalModOps`](crate::api::CKKSEvalModOps) expects, with message ratio
/// `q/Δ = 2^{src.log_budget()}`.
///
/// ## Metadata
///
/// ```text
/// log_delta_out  = src.log_delta                 (unchanged)
/// log_budget_out = dst.max_k() − src.log_delta   (the full raised headroom)
/// // k_out = dst.max_k()
/// ```
///
/// Errors if `dst.max_k() < src.k()` (ModUp must widen the modulus).
pub trait CKKSBootstrappingOps<BE: Backend>: DFTOps<BE> + CKKSEvalModOps<BE> {
    /// Returns scratch bytes required by [`Self::ckks_mod_up_into`].
    fn ckks_mod_up_tmp_bytes(&self) -> usize;

    /// Raises the modulus of `src` into the wider `dst`.
    ///
    /// `dst` must be allocated at the target (raised) modulus: `dst.max_k() ≥
    /// src.k()`. See the trait docs for the exact semantics and
    /// metadata effect.
    fn ckks_mod_up_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}
