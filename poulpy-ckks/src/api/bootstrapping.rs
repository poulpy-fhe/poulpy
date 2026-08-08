use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDFTOps, CKKSEvalModOps},
    layouts::{BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertext, CKKSPlaintext, EncodedLut},
};

/// CKKS bootstrapping.
///
/// Only **ModUp** is a bootstrapping-specific primitive. CoeffsToSlots,
/// EvalMod, and SlotsToCoeffs are reused from their own op traits, which this
/// trait re-exports as supertraits:
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
/// [`BootstrappingKeys`], and selects C2S-first or S2C-first from the context.
/// EvalRound+ is selected separately by either recipe's optional bypass.
/// Sparse-secret encapsulation is also selected by the compiled recipe; the
/// supplied keys must carry the matching
/// [encapsulation keys](BootstrappingKeys::encapsulation_keys).
pub trait CKKSBootstrappingOps<BE: Backend>: CKKSDFTOps<BE> + CKKSEvalModOps<BE> {
    /// Returns scratch bytes required by [`Self::ckks_mod_up_into`].
    fn ckks_mod_up_tmp_bytes(&self) -> usize;

    /// Scratch upper bound for a full [`Self::ckks_bootstrap`] call: the
    /// pipeline working ciphertexts it carves from scratch plus the largest
    /// nested stage. `ct_out`/`ct_in` provide the bootstrap and input widths,
    /// `ctx` selects the pipeline, optional EvalRound+ variant, and EvalMod
    /// parameters; `keys_layout` sizes the key-dependent stages
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

    /// Scratch upper bound for functional bootstrap with `lut`.
    fn ckks_functional_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Raises the modulus of `src` into the wider `dst`.
    ///
    /// `dst` must carry the target (raised) modulus: `dst.k() ≥
    /// src.k()`. See the trait docs for the exact semantics and
    /// metadata effect.
    fn ckks_mod_up_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// One-shot CKKS bootstrap, driven by the compiled context.
    ///
    /// `ct_in` is at the input ("level 0") modulus; `ct_out` must be allocated at
    /// the bootstrap modulus (its `k()` sets the working width). When the compiled
    /// recipe enables sparse-secret encapsulation, `keys` must carry the matching
    /// [encapsulation keys](BootstrappingKeys::encapsulation_keys), which wrap ModUp
    /// (`denseToSparse → ModUp → sparseToDense`). The `1/K`
    /// amplitude bridge must be folded into the CoeffsToSlots plan's `scaling` by
    /// the caller **before** compiling the context (see
    /// [`BootstrappingContext::coeffs_to_slots`]); EvalMod applies its own scale
    /// round-trip, so no further scaling is needed at call time.
    ///
    /// The pipeline is selected from [`BootstrappingContext::pipeline`]:
    ///
    /// - [`C2SFirst`](crate::layouts::BootstrappingPipeline::C2SFirst): `ModUp → CoeffsToSlots →
    ///   EvalMod → SlotsToCoeffs`. The final transform restores the message ratio.
    /// - [`S2CFirst`](crate::layouts::BootstrappingPipeline::S2CFirst): `SlotsToCoeffs → ModUp →
    ///   CoeffsToSlots → EvalMod`. The first transform uses scaling `1/2`; the
    ///   output is relabeled at `ct_in.log_delta`.
    ///
    /// Use [`BootstrappingPlan::input_k`](crate::layouts::BootstrappingPlan::input_k)
    /// and [`BootstrappingPlan::bootstrap_k`](crate::layouts::BootstrappingPlan::bootstrap_k)
    /// to place the pre- and post-ModUp costs correctly.
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

    /// Standard S2C-first bootstrap for ciphertexts known to contain real slots
    /// only. EvalRound+ contexts are not supported by this specialized path.
    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap_real<F, K>(
        &self,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;

    /// Refreshes `ct_in` through an S2C-first context and applies `lut` to each
    /// real and imaginary slot half before recombining them in `ct_out`. The LUT
    /// derives the required message ratio from its table length.
    #[allow(clippy::too_many_arguments)]
    fn ckks_functional_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;

    /// Functional bootstrap for ciphertexts known to contain real slots only.
    #[allow(clippy::too_many_arguments)]
    fn ckks_functional_bootstrap_real<F, K>(
        &self,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;

    /// Applies several equal-arity general LUTs through one shared functional bootstrap.
    #[allow(clippy::too_many_arguments)]
    fn ckks_functional_bootstrap_multi<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertext<BE::OwnedBuf>],
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintext<BE::OwnedBuf>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;
}
