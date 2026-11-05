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
/// Only the **modulus raise** ([`ckks_mod_raise`](Self::ckks_mod_raise)) is a bootstrapping-specific primitive. CoeffsToSlots,
/// EvalMod, and SlotsToCoeffs are reused from their own op traits:
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
/// EvalRound+ is selected separately by either recipe's optional bypass.
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
    /// [`Self::ckks_bootstrap`].
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
    /// return. Scratch is bounded by [`Self::ckks_mod_raise_tmp_bytes`].
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

    /// EvalMod of a bootstrap, on `a`, the output of CoeffsToSlots, as the plan and the slots of `a` say:
    ///
    /// - real slots of an S2C-first plan without a bypass, which [`CKKSEvalModInfos::one_eval_mod_on_real_slots`] names:
    ///   the real part and one EvalMod. CoeffsToSlots labels its output complex, so the caller stamps the slot kind of the
    ///   message on `a`;
    /// - any other input of a plan on the standard ring: the split, one EvalMod per half, on a parallel executor at once,
    ///   and the recombination;
    /// - a plan on the conjugate-invariant ring, with `ci` of degree `N` and this module of degree `2N`: the conversion
    ///   to that ring, one EvalMod for all coefficients on `ci`, and the conversion back.
    ///
    /// The guard bits leave `res` here. Under EvalRound+, [`CKKSEvalModInfos::eval_round_plus`], `res` is
    /// `EvalMod(x) - K·x` and keeps them: the caller adds the bypass back, then removes them with `set_log_delta`. A
    /// step does not know the message scale of a raised ciphertext, so the caller stamps the scale, slot kind and
    /// sparsity of the input at the end.
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) before any write to `res`, for `ci` on a plan of
    /// the standard ring, no `ci` on a plan of the conjugate-invariant ring, a rank other than 1, modules or ciphertexts
    /// of other degrees, or keys without the tensor key of the plan's ring or its conversion keys, of another shape or
    /// narrower than the widths they switch. `CKKSError::Internal` for an arena under
    /// [`Self::ckks_bootstrap_eval_mod_tmp_bytes`], when the keys report their layout.
    fn ckks_bootstrap_eval_mod<R, A, P, F, K>(
        &self,
        res: &mut R,
        a: &A,
        plan: &P,
        ci: Option<&Module<BE::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync;

    /// One-shot CKKS bootstrap, driven by the compiled context.
    ///
    /// `ct_in` is at the input ("level 0") modulus; `ct_out` must be allocated at
    /// the bootstrap modulus (its `k()` sets the working width). `ci` is `None` on a
    /// plan with EvalMod on the standard ring. When the compiled
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

/// [`CKKSBootstrappingOps::ckks_bootstrap`] with EvalMod on the
/// conjugate-invariant ring ([`EvalModRing::ConjugateInvariant`](crate::layouts::EvalModRing::ConjugateInvariant)),
/// for a context compiled on that ring by [`BootstrappingContextPrepared::compile`]: this module
/// serves degree `2N` for the conversions and `N` for the other stages, and `ci`
/// is the CI module of degree `N`, on the twin [`CKKSCITwin::CI`].
/// The keys, the conversion keys included, are checked at entry.
pub trait CKKSBootstrappingCIOps<BE: CKKSCITwin> {
    /// Scratch of [`Self::ckks_bootstrap_ci_into`]: `(scratch, scratch_ci)`.
    fn ckks_bootstrap_ci_tmp_bytes<C1, C2, F>(
        &self,
        ci: &Module<BE::CI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> (usize, usize)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Bootstraps `src` into `dst`, at the width of `dst`, with `scratch`
    /// and `scratch_ci` of [`Self::ckks_bootstrap_ci_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// Before any work, and leaving `dst` unchanged:
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for a context not compiled
    /// on that ring or keys without its tensor and conversion keys, modules of other degrees than
    /// `2N` and `N`, ciphertexts of another degree or rank, an encapsulation that does not match the
    /// recipe, a working width or an input narrower than the bits the stages running on it
    /// consume, or conversion keys of another degree, radix or rank, or narrower than the widths
    /// they switch; [`MissingAutomorphismKey`](crate::CKKSCompositionError::MissingAutomorphismKey)
    /// or [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey)
    /// for a rotation or tensor key narrower than its widest use, as
    /// [`CKKSBootstrappingOps::ckks_bootstrap`]. An error past these checks leaves `dst`
    /// unspecified.
    ///
    /// # Metadata
    ///
    /// `dst` ends at the output width of the recipe below its working width, at the scale,
    /// slot kind and sparsity of `src`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the bootstrap takes a module and a scratch arena per ring"
    )]
    fn ckks_bootstrap_ci_into<Dst, Src, F, K>(
        &self,
        ci: &Module<BE::CI>,
        dst: &mut Dst,
        src: &Src,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BE::CI>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: BootstrappingKeys<BE>;
}
