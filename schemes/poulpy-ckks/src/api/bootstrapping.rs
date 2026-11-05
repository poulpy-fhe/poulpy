use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSCITwin,
        CKKSCiphertextOwned, CKKSEvalModInfos, CKKSModRaiseInfos, CKKSPlaintextOwned, EncodedLut,
    },
};

/// CKKS bootstrapping.
///
/// Each step of a bootstrap is a public operation, and the plan decides how each one runs. Three are specific to
/// bootstrapping: the modulus raise ([`ckks_mod_raise`](Self::ckks_mod_raise)), the EvalMod of a bootstrap
/// ([`ckks_bootstrap_eval_mod`](Self::ckks_bootstrap_eval_mod)) and the LUT step ([`ckks_eval_lut`](Self::ckks_eval_lut)).
/// CoeffsToSlots, EvalMod, and SlotsToCoeffs are reused from their own op traits:
///
/// - CoeffsToSlots / SlotsToCoeffs via [`CKKSDFTOps`](crate::api::CKKSDFTOps)
///   ([`ckks_coeffs_to_slots_assign`](crate::api::CKKSDFTOps::ckks_coeffs_to_slots_assign),
///   [`ckks_slots_to_coeffs_assign`](crate::api::CKKSDFTOps::ckks_slots_to_coeffs_assign), and their
///   `_split` / `_repack` variants);
/// - EvalMod via [`CKKSEvalModOps`](crate::api::CKKSEvalModOps)
///   ([`ckks_eval_mod`](crate::api::CKKSEvalModOps::ckks_eval_mod)).
///
/// The composable stages stay public so callers can assemble custom pipelines,
/// but a ready-made orchestrator is provided: [`ckks_bootstrap`](Self::ckks_bootstrap).
/// It consumes a compiled [`BootstrappingContextPrepared`] and a prepared
/// [`BootstrappingKeys`], and selects C2S-first or S2C-first from the context.
/// EvalRound+ is selected separately by either recipe's optional bypass, and the ring of EvalMod
/// by the plan, with no other operation.
/// Sparse-secret encapsulation is also selected by the compiled recipe; the
/// supplied keys must carry the matching
/// [encapsulation keys](crate::layouts::BootstrappingLinearKeys::encapsulation_keys).
pub trait CKKSBootstrappingOps<BE: CKKSCITwin> {
    /// Scratch upper bound for a full [`Self::ckks_bootstrap`] call: the
    /// pipeline working ciphertexts it carves from scratch plus the largest
    /// nested stage. `ct_out`/`ct_in` provide the bootstrap and input widths,
    /// `ctx` selects the pipeline, optional EvalRound+ variant, and EvalMod
    /// parameters; `keys_layout` sizes the key-dependent stages
    /// (rotations, tensor key, optional encapsulation switches). `ci` is the module of
    /// [`Self::ckks_bootstrap`]; one arena serves both rings.
    fn ckks_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        ci: Option<&Module<BE::CI>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch upper bound for [`Self::ckks_functional_bootstrap`] with `luts`.
    /// `ct_out` provides the shared output layout every result must have; the
    /// bound does not grow with the number of LUTs.
    fn ckks_functional_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch bytes for [`Self::ckks_mod_raise`] into a ciphertext like `res` from one like `a`, with the plan and the
    /// key layout of the raise.
    fn ckks_mod_raise_tmp_bytes<R, A, P>(&self, res: &R, a: &A, plan: &P, keys_layout: &BootstrappingKeysLayout) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<BE>;

    /// Whole raise step of the bootstrap, as one call: on C2S-first, lift `Δ·m` to the plan's message ratio; switch
    /// to the sparse secret when `keys` carry the encapsulation switches, ModUp into `res` (with the second lift, or
    /// the C2S guard bits, fused into its shift), apply the first CoeffsToSlots factor when the plan leaves it to the
    /// raise, switch back, and relabel so `res` comes back fully labelled: C2S-first by the message ratio, so that
    /// `I(X)·q` is the integer part and the message the residue.
    ///
    /// `res` must be allocated at the bootstrap modulus. `a` is consumed: its content and metadata are unspecified on
    /// a successful return. Scratch is bounded by [`Self::ckks_mod_raise_tmp_bytes`]. `plan` is a compiled context, a
    /// [`run`](BootstrappingContextPrepared::run) of it, its [`mod_raise`](BootstrappingContextPrepared::mod_raise), or,
    /// on a plan that leaves no CoeffsToSlots factor to the raise, the [`ModRaisePlan`](crate::layouts::ModRaisePlan) itself.
    ///
    /// # Errors
    ///
    /// Before any write, leaving `res` and `a` unchanged: [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand)
    /// for a rank other than 1, ciphertexts of two degrees or base2k or of a degree the module does not serve, an
    /// encapsulation that does not match the plan, a plan that moves the first CoeffsToSlots factor onto the raise
    /// without that factor, a `res` too narrow for the raise of `a` and its lifts, encapsulation keys absent or too narrow
    /// for what they switch, or first-factor keys absent, of two layouts or too narrow for the raise.
    /// `CKKSError::Internal` for an input without the bits of the plan's lifts, or an arena under
    /// [`Self::ckks_mod_raise_tmp_bytes`], when the keys report their layout.
    ///
    /// # Metadata
    ///
    /// `res` keeps its `k`, so `log_budget_out = res.k() - log_delta_out`, and takes the slot kind and sparsity of `a`.
    /// S2C-first has `log_delta_out = a.k() + c2s_guard_bits`. C2S-first has `log_delta_out = a.log_delta() + lifts +
    /// log_msg_ratio`, `lifts` being the two lifts of [`EvalModPlan`](crate::layouts::EvalModPlan), so that the message
    /// is the residue.
    fn ckks_mod_raise<R, A, P, K>(
        &self,
        res: &mut R,
        a: &mut A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<BE>,
        K: BootstrappingLinearKeys<BE>;

    /// Scratch bytes for [`Self::ckks_bootstrap_eval_mod`] into a ciphertext like `res` from one like `a`, with the
    /// plan, the module and the key layout of the step. It holds on any slot kind.
    fn ckks_bootstrap_eval_mod_tmp_bytes<R, A, P, F>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        ci: Option<&Module<BE::CI>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>;

    /// EvalMod of a bootstrap, on `a`, the output of CoeffsToSlots, as the plan and the slots of the message say:
    ///
    /// - real slots of an S2C-first plan without a bypass, which [`CKKSEvalModInfos::one_eval_mod_on_real_slots`] names:
    ///   the real part and one EvalMod;
    /// - any other input of a plan on the standard ring: the split, one EvalMod per half, on a parallel executor at once,
    ///   and the recombination;
    /// - a plan on the conjugate-invariant ring, with `ci` of degree `N` and this module of degree `2N`: the conversion
    ///   to that ring, one EvalMod for all coefficients on `ci`, and the conversion back.
    ///
    /// `plan` is a [`BootstrappingRun`](crate::layouts::BootstrappingRun), which holds the labels of the message, or a
    /// plain [`EvalMod`](crate::layouts::EvalMod), which reads the slots of `a` and stamps nothing.
    ///
    /// `a` is consumed: its content and metadata are unspecified on a successful return, and a refusal leaves it unchanged.
    ///
    /// The output is the message's: the guard bits leave `res` here and, on an S2C-first plan, `res` takes the message's
    /// metadata; a C2S-first plan ends in `ckks_slots_to_coeffs_assign`, which labels it. Under EvalRound+,
    /// [`CKKSEvalModInfos::eval_round_plus`], `res` is `EvalMod(x) - K·x`, with its working label and the guard bits: the
    /// composer adds [`BootstrappingRun::bypass`](crate::layouts::BootstrappingRun::bypass) back, then calls
    /// [`CKKSEvalModInfos::stamp_eval_mod_output`].
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) before any write to `res`, for `ci` on a plan of
    /// the standard ring, no `ci` on a plan of the conjugate-invariant ring, a plan with no EvalMod, a rank other than 1,
    /// ciphertexts of two degrees or base2k or of a degree the module does not serve, an input of fewer scale bits than
    /// the plan's guard bits, or keys without the tensor key of the plan's ring.
    /// On the conjugate-invariant ring also, before its query, for modules of other degrees than `2N` and `N`, `a` wider
    /// than `res`, or conversion keys absent or of another shape, and after the product by `u` for conversion keys
    /// narrower than the widths they switch. `CKKSError::Internal` for an arena under
    /// [`Self::ckks_bootstrap_eval_mod_tmp_bytes`], when the keys report their layout.
    ///
    /// [`MissingAutomorphismKey`](crate::CKKSCompositionError::MissingAutomorphismKey), before any write to `res`, for a
    /// conjugation key that does not cover `a`, or a rotation key of a folded `u` on the conjugate-invariant ring.
    /// [`InsufficientHomomorphicCapacity`](crate::CKKSCompositionError::InsufficientHomomorphicCapacity), before any write
    /// to `res` or `a`, for an input without the budget of EvalMod.
    /// [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey), before any write to `res`
    /// or `a`, for a tensor key narrower than a product of EvalMod.
    ///
    /// # Metadata
    ///
    /// EvalMod spends `consumed = plan.eval_mod_plan().consumed_bits()` of the budget of `a`, and the conversions of the
    /// conjugate-invariant ring the scales of their products:
    ///
    /// - a [`BootstrappingRun`](crate::layouts::BootstrappingRun) of an S2C-first plan: `res.k() = a.k() - c2s_guard_bits -
    ///   consumed` at the scale, slot kind and sparsity of the message;
    /// - a run of a C2S-first plan: `log_delta_out = a.log_delta()`, `log_budget_out = a.log_budget() - consumed`, until
    ///   `ckks_slots_to_coeffs_assign` labels it;
    /// - under EvalRound+ and on a plain [`EvalMod`](crate::layouts::EvalMod): the same, with the guard bits still on and
    ///   the working label; on a plain `EvalMod`, slot kind and sparsity are those of the route.
    ///
    /// The split route leaves the canonical flag clear: its recombination is a lazy sum.
    fn ckks_bootstrap_eval_mod<R, A, P, F, K>(
        &self,
        res: &mut R,
        a: &mut A,
        plan: &P,
        ci: Option<&Module<BE::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync;

    /// Scratch bytes for [`Self::ckks_eval_lut`] of `luts` into outputs like `res` from `a`, with the plan and the key
    /// layout of the step. The bytes do not grow with the number of LUTs.
    fn ckks_eval_lut_tmp_bytes<R, A, P, F>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>;

    /// Evaluates each LUT of `luts` on `a`, the output of CoeffsToSlots, into the matching ciphertext of `res`.
    ///
    /// The guard bits leave the input here, before the exponential. Real slots of a plan for which
    /// [`CKKSEvalModInfos::real_part_on_real_slots`] holds take the real part of `a` and skip the imaginary half, with or
    /// without a bypass in the plan, which this step does not run. Any other input is split, each half is evaluated and
    /// the halves are recombined. General LUTs of a batch share the power basis of each half.
    ///
    /// `a` is consumed: its content and metadata are unspecified on a successful return, and a refusal leaves it unchanged.
    ///
    /// `plan` is a [`BootstrappingRun`](crate::layouts::BootstrappingRun), whose message gives the slots and each output
    /// its [`eval_lut_output`](CKKSEvalModInfos::eval_lut_output), or a plain [`EvalMod`](crate::layouts::EvalMod), which
    /// reads the slots of `a` and leaves the outputs the scale and widths of the evaluation.
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) before any write to `res`, for no LUT, a number of
    /// outputs other than the number of LUTs, LUTs of different message moduli, a plan that is not the unit-circle
    /// complex exponential on the standard ring when a LUT needs EvalMod, outputs that do not share one rank-1 layout,
    /// an input of another degree, base2k or rank, an input of fewer scale bits than the plan's guard bits, keys
    /// without the tensor key of the plan's ring, a C2S-first run, an input without the bits an output of a run takes,
    /// outputs narrower than the width a run gives them, or outputs narrower than `a`: the evaluation runs at the width of
    /// `res`, so every output must be at least as wide as `a`.
    /// [`MissingAutomorphismKey`](crate::CKKSCompositionError::MissingAutomorphismKey), before any write to `res`, for a
    /// conjugation key that does not cover `a`.
    /// `CKKSError::Internal` for an arena under [`Self::ckks_eval_lut_tmp_bytes`], when the keys report their layout.
    ///
    /// [`InsufficientHomomorphicCapacity`](crate::CKKSCompositionError::InsufficientHomomorphicCapacity) for an input
    /// without the budget of a LUT and
    /// [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey) for a tensor key narrower than
    /// its widest product, each before any write to `res` or `a`.
    ///
    /// # Metadata
    ///
    /// Each LUT spends `bits`, EvalMod's when it needs it and its own series', of the budget of `a`, the guard bits
    /// leaving first. On a [`BootstrappingRun`](crate::layouts::BootstrappingRun), an output has the width and label of
    /// [`eval_lut_output`](CKKSEvalModInfos::eval_lut_output): `k_out = a.k() - c2s_guard_bits - bits`,
    /// `log_delta_out = log_delta + log_msg_ratio` from the `log_delta` of the message, and the slot kind and sparsity of
    /// the message. On a plain [`EvalMod`](crate::layouts::EvalMod) it keeps what the evaluation leaves it. The split
    /// route leaves the canonical flag clear: its recombination is a lazy sum.
    fn ckks_eval_lut<R, A, P, F, K>(
        &self,
        res: &mut [R],
        a: &mut A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE>;

    /// One-shot CKKS bootstrap, driven by the compiled context.
    ///
    /// `ct_in` is at the input ("level 0") modulus; `ct_out` must be allocated at
    /// the bootstrap modulus (its `k()` sets the working width). `ci` is `None` on a
    /// plan with EvalMod on the standard ring. On a plan with EvalMod on the
    /// conjugate-invariant ring ([`EvalModRing::ConjugateInvariant`](crate::layouts::EvalModRing::ConjugateInvariant)),
    /// it is the CI module of degree `N`, on the twin [`CKKSCITwin::CI`], and this module serves degree
    /// `2N` for the conversions and `N` for the other stages. When the compiled
    /// recipe enables sparse-secret encapsulation, `keys` must carry the matching
    /// [encapsulation keys](crate::layouts::BootstrappingLinearKeys::encapsulation_keys), which wrap ModUp
    /// (`denseToSparse → ModUp → sparseToDense`). [`BootstrappingPlan::new`](crate::layouts::BootstrappingPlan::new)
    /// folds EvalMod's required input scaling into CoeffsToSlots, so no further
    /// scaling is needed at call time.
    ///
    /// The pipeline is selected from [`BootstrappingContextPrepared::pipeline`]:
    ///
    /// - [`C2SFirst`](crate::layouts::BootstrappingPipeline::C2SFirst): `ModUp → CoeffsToSlots →
    ///   EvalMod → SlotsToCoeffs`. The final transform restores the message ratio,
    ///   then the output scale and modulus width are reduced to return to the input scale.
    /// - [`S2CFirst`](crate::layouts::BootstrappingPipeline::S2CFirst): `SlotsToCoeffs → ModUp →
    ///   CoeffsToSlots → EvalMod`. The first transform uses scaling `1/2`; the
    ///   output is relabeled at `ct_in.log_delta`.
    ///
    /// Use [`BootstrappingPlan::input_k`](crate::layouts::BootstrappingPlan::input_k)
    /// and [`BootstrappingPlan::bootstrap_k`](crate::layouts::BootstrappingPlan::bootstrap_k)
    /// to place the pre- and post-ModUp costs correctly. Pass the input `log_delta`
    /// to `bootstrap_k` so it accounts for C2S-first's output scale restoration.
    ///
    /// Before any work, every rotation key and the tensor key must cover the
    /// widest ciphertext the bootstrap switches with it at this working width
    /// ([`BootstrappingPlan::key_widths`](crate::layouts::BootstrappingPlan::key_widths)):
    /// otherwise it fails with
    /// [`MissingAutomorphismKey`](crate::CKKSCompositionError::MissingAutomorphismKey)
    /// or [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey),
    /// whose `source` is the key source's refusal. Keys laid out by a preset
    /// cover its `bootstrap_k`. A working width or an input narrower than the bits
    /// the stages running on it consume fails with
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) before any work.
    ///
    /// # Errors
    ///
    /// Before any work, and leaving `ct_out` unchanged, besides the key refusals above:
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for a functional context, `ci` on a plan of the
    /// standard ring, no `ci` on a plan of the conjugate-invariant ring, a rank other than 1, ciphertexts of two degrees
    /// or base2k or of a degree the module does not serve, an encapsulation that does not match the recipe, or keys
    /// without the tensor key of the plan's ring. On the conjugate-invariant ring also, for modules of other degrees than `2N` and `N`, or
    /// conversion keys absent, of another degree, radix or rank, or narrower than the widths they switch.
    /// `CKKSError::Internal` for an arena under [`Self::ckks_bootstrap_tmp_bytes`], when the keys report their layout.
    ///
    /// A step refusing on its own working ciphertexts, such as an input without the budget of SlotsToCoeffs, leaves
    /// `ct_out` unchanged as well.
    ///
    /// # Metadata
    ///
    /// `ct_out` ends at the output width of the recipe below its working width, at the scale, slot kind and sparsity
    /// of `ct_in`.
    fn ckks_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        ci: Option<&Module<BE::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync;

    /// Refreshes `ct_in` through an S2C-first context and applies each LUT of
    /// `luts` to it, writing the results into `ct_outs`.
    ///
    /// One LUT or many: the batch shares the SlotsToCoeffs, ModUp and
    /// CoeffsToSlots stages, and general LUTs additionally share the power basis
    /// of each transformed half, so `n` LUTs cost one bootstrap
    /// plus `n` polynomial evaluations. `ct_outs` and `luts` must have the same
    /// length, every LUT must have the same message modulus, and every output
    /// must share one rank-1 layout.
    ///
    /// Inputs contain integer messages. Configure the plan with
    /// [`BootstrappingPlan::with_functional_bootstrap`](crate::layouts::BootstrappingPlan::with_functional_bootstrap).
    /// The slot kind of `ct_in` selects the pipeline:
    /// [`SlotsKind::Real`] slots skip the imaginary branch entirely.
    ///
    /// # Errors
    ///
    /// Before any work, and leaving `ct_outs` unchanged:
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for a C2S-first context, the refusals of
    /// [`Self::ckks_eval_lut`] on the batch, an encapsulation that does not match the recipe, a message ratio or modulus
    /// the context or the LUTs do not take, or an output width below the bits the stages after the raise consume;
    /// [`MissingAutomorphismKey`](crate::CKKSCompositionError::MissingAutomorphismKey) or
    /// [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey) for a key narrower than
    /// [`BootstrappingPlan::key_widths`](crate::layouts::BootstrappingPlan::key_widths).
    /// `CKKSError::Internal` for an arena under [`Self::ckks_functional_bootstrap_tmp_bytes`], when the keys report their
    /// layout.
    ///
    /// [`SlotsKind::Real`]: crate::SlotsKind::Real
    fn ckks_functional_bootstrap<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>;
}
