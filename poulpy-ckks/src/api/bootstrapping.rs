use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDFTOps, CKKSEvalModOps},
    layouts::{BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertext},
};

/// CKKS bootstrapping.
///
/// Bootstrapping is the pipeline `ModUp → CoeffsToSlots → EvalMod →
/// SlotsToCoeffs`. Of those, only **ModUp** (the modulus raise) is a new
/// primitive, defined directly on this trait. The other three stages are reused
/// verbatim from their own op traits, which this trait **re-exports as
/// supertraits**:
///
/// - CoeffsToSlots / SlotsToCoeffs via [`CKKSDFTOps`](crate::api::CKKSDFTOps)
///   ([`ckks_coeffs_to_slots`](CKKSDFTOps::ckks_coeffs_to_slots),
///   [`ckks_slots_to_coeffs`](CKKSDFTOps::ckks_slots_to_coeffs), and their
///   `_split` / `_repack` variants);
/// - EvalMod via [`CKKSEvalModOps`](crate::api::CKKSEvalModOps)
///   ([`ckks_eval_mod`](CKKSEvalModOps::ckks_eval_mod)).
///
/// The composable stages stay public so callers can assemble custom pipelines,
/// but a ready-made orchestrator is provided: [`ckks_bootstrap`](Self::ckks_bootstrap).
/// It consumes a compiled [`BootstrappingContext`] and a prepared
/// [`BootstrappingKeys`], and selects the pipeline from the context — the classic
/// refresh when [`coeffs_to_slots_bypass`](BootstrappingContext::coeffs_to_slots_bypass)
/// is absent, the EvalRound+ variant (<https://eprint.iacr.org/2024/1379>) when it
/// is present. Sparse-secret encapsulation of ModUp is applied automatically when
/// the keys carry [encapsulation keys](BootstrappingKeys::encapsulation_keys).
pub trait CKKSBootstrappingOps<BE: Backend>: CKKSDFTOps<BE> + CKKSEvalModOps<BE> {
    /// Returns scratch bytes required by [`Self::ckks_mod_up_into`].
    fn ckks_mod_up_tmp_bytes(&self) -> usize;

    /// Scratch upper bound for a full [`Self::ckks_bootstrap`] call: the
    /// pipeline working ciphertexts it carves from scratch plus the largest
    /// nested stage. `ct_out`/`ct_in` provide the bootstrap and input widths,
    /// `ctx` selects the pipeline variant (standard vs EvalRound+) and the
    /// EvalMod parameters, and `keys_layout` sizes the key-dependent stages
    /// (rotations, tensor key, optional encapsulation switches).
    fn ckks_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Raises the modulus of `src` into the wider `dst`.
    ///
    /// `dst` must be allocated at the target (raised) modulus: `dst.max_k() ≥
    /// src.k()`. See the trait docs for the exact semantics and
    /// metadata effect.
    fn ckks_mod_up_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// One-shot CKKS bootstrap, driven by the compiled context.
    ///
    /// `ct_in` is at the input ("level 0") modulus; `ct_out` must be allocated at
    /// the bootstrap modulus (its `k()` sets the working width). When `keys` carries
    /// [encapsulation keys](BootstrappingKeys::encapsulation_keys), the sparse-secret
    /// trick wraps ModUp (`denseToSparse → ModUp → sparseToDense`). The `1/K`
    /// amplitude bridge must be folded into the CoeffsToSlots plan's `scaling` by
    /// the caller **before** compiling the context (see
    /// [`BootstrappingContext::coeffs_to_slots`]); EvalMod applies its own scale
    /// round-trip, so no further scaling is needed at call time.
    ///
    /// The pipeline is selected from the context:
    ///
    /// - **standard** (`ModUp → CoeffsToSlots → EvalMod → SlotsToCoeffs`) when
    ///   [`coeffs_to_slots_bypass`](BootstrappingContext::coeffs_to_slots_bypass) is
    ///   `None`;
    /// - **EvalRound+** (<https://eprint.iacr.org/2024/1379>) when it is `Some`: EvalMod
    ///   runs on the low-precision CoeffsToSlots whose DFT error `e` cancels in the
    ///   round `r0_hp − K·r0_lp + EvalMod(r0_lp)`, recovering the message at the
    ///   high-precision bypass transform's precision (`K = f_mod_interval`, read from
    ///   the compiled EvalMod, must be a power of two).
    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;
}
