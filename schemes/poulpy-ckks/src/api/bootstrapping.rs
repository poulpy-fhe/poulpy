use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSCiphertextOwned,
        CKKSEvalModInfos, CKKSModRaiseInfos, CKKSPlaintextOwned, EncodedLut,
    },
};

/// CKKS bootstrapping: public steps that run according to a [`BootstrappingPlan`](crate::layouts::BootstrappingPlan).
///
/// This trait holds the raise ([`ckks_mod_raise`](Self::ckks_mod_raise)), the EvalMod of a bootstrap
/// ([`ckks_bootstrap_eval_mod`](Self::ckks_bootstrap_eval_mod)) and the LUT step ([`ckks_eval_lut`](Self::ckks_eval_lut)).
/// CoeffsToSlots and SlotsToCoeffs come from [`CKKSDFTOps`](crate::api::CKKSDFTOps), and
/// [`ckks_bootstrap`](Self::ckks_bootstrap) and [`ckks_functional_bootstrap`](Self::ckks_functional_bootstrap) compose
/// them.
///
/// # Refusals
///
/// A step refuses before it writes `res`, and [`ckks_mod_raise`](Self::ckks_mod_raise) before it writes `a`. A bootstrap
/// refuses before any work, a step refusing on its working ciphertexts leaving the output unchanged as well:
///
/// - [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand): a context, module, operand, key or LUT that does
///   not match the plan or each other, a ciphertext of another degree than the one the context bootstraps (`2·g` times
///   the slots of its transforms, `g` the gap of the plaintexts), a bootstrap input off the subring of that gap, a
///   module of another degree than `2N` with EvalMod on the conjugate-invariant ring, a width too narrow for a stage,
///   an input holding fewer bits than its scale plus the plan's message ratio, or a plan whose EvalMod scale is below
///   that input scale plus the message ratio (the lifts of a C2S-first raise), a LUT on a plan with the EvalRound+
///   bypass, a complex message under an [`ExpCmplx`](crate::layouts::eval_mod::EvalModType::ExpCmplx) EvalMod, and an
///   input of a [replicated](crate::SlotLayout::Replicated) slot layout.
/// - [`MissingAutomorphismKey`](crate::CKKSCompositionError::MissingAutomorphismKey) and
///   [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey): a key narrower than its
///   stage ([`key_widths`](crate::layouts::BootstrappingPlan::key_widths)), `source` being the key source's refusal.
/// - [`InsufficientHomomorphicCapacity`](crate::CKKSCompositionError::InsufficientHomomorphicCapacity): an input
///   without the budget of EvalMod or a LUT.
/// - `CKKSError::Internal`: an arena under the query of the call, when the keys report their layout.
pub trait CKKSBootstrappingOps<BE: Backend> {
    /// Scratch upper bound for [`Self::ckks_bootstrap`], whose arguments it takes, `keys_layout` sizing the
    /// key-dependent stages. One arena serves both rings.
    fn ckks_bootstrap_tmp_bytes<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch upper bound for [`Self::ckks_functional_bootstrap`], which does not grow with the number of LUTs.
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

    /// Raises `a` into `res`, allocated at the bootstrap modulus, according to the plan: see
    /// [`ModRaisePlan`](crate::layouts::ModRaisePlan).
    ///
    /// `plan` is a compiled context, a [`run`](BootstrappingContextPrepared::run) of it, its
    /// [`mod_raise`](BootstrappingContextPrepared::mod_raise), or, on a plan that leaves no CoeffsToSlots factor to the
    /// raise, the [`ModRaisePlan`](crate::layouts::ModRaisePlan) itself.
    ///
    /// `a` is clobbered: the plan's lift and switch run on it in place, and its content and metadata are unspecified on a
    /// successful return. It is borrowed and not moved, so a refusal leaves it unchanged for a retry and the caller keeps
    /// its buffer.
    ///
    /// # Errors
    ///
    /// `InvalidOperand` or `CKKSError::Internal`: see [refusals](CKKSBootstrappingOps#refusals).
    ///
    /// # Metadata
    ///
    /// `res` comes back fully labelled, as [`ModRaisePlan`](crate::layouts::ModRaisePlan) says.
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
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>;

    /// Evaluates EvalMod on `a`, the output of CoeffsToSlots, into `res`, according to the bootstrapping plan: see
    /// [its EvalMod section](crate::layouts::BootstrappingPlan#evalmod).
    ///
    /// `plan` is a [`BootstrappingRun`](crate::layouts::BootstrappingRun) or a plain
    /// [`EvalMod`](crate::layouts::EvalMod). Under
    /// [`EvalModRing::ConjugateInvariant`](crate::layouts::EvalModRing::ConjugateInvariant), the module, of degree `2N`,
    /// also runs EvalMod on the conjugate-invariant ring of degree `N`.
    ///
    /// # Errors
    ///
    /// See [refusals](CKKSBootstrappingOps#refusals).
    ///
    /// # Metadata
    ///
    /// `res` is labelled as [the plan](crate::layouts::BootstrappingPlan#evalmod) says.
    fn ckks_bootstrap_eval_mod<R, A, P, F, K>(
        &self,
        res: &mut R,
        a: &A,
        plan: &P,
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

    /// Evaluates each LUT of `luts` on `a`, the output of CoeffsToSlots, into the matching ciphertext of `res`,
    /// according to the bootstrapping plan: see [its LUT section](crate::layouts::BootstrappingPlan#luts).
    ///
    /// `plan` is a [`BootstrappingRun`](crate::layouts::BootstrappingRun) or a plain
    /// [`EvalMod`](crate::layouts::EvalMod).
    ///
    /// # Errors
    ///
    /// See [refusals](CKKSBootstrappingOps#refusals). A batch holds one rank-1 output per LUT, of one layout no
    /// narrower than `a`, and one message modulus.
    ///
    /// # Metadata
    ///
    /// Each output is labelled as [the plan](crate::layouts::BootstrappingPlan#luts) says.
    fn ckks_eval_lut<R, A, P, F, K>(
        &self,
        res: &mut [R],
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE>;

    /// Refreshes `ct_in` into `ct_out`, allocated at the bootstrap modulus, by running the steps according to the
    /// compiled plan: see [`BootstrappingPlan`](crate::layouts::BootstrappingPlan).
    ///
    /// Under [`EvalModRing::ConjugateInvariant`](crate::layouts::EvalModRing::ConjugateInvariant), the module, of degree
    /// `2N`, also runs EvalMod on the conjugate-invariant ring of degree `N`.
    /// [`BootstrappingPlan::input_k`](crate::layouts::BootstrappingPlan::input_k) and
    /// [`BootstrappingPlan::bootstrap_k`](crate::layouts::BootstrappingPlan::bootstrap_k) place the widths.
    ///
    /// # Errors
    ///
    /// See [refusals](CKKSBootstrappingOps#refusals).
    ///
    /// # Metadata
    ///
    /// `ct_out` ends at the output width of the plan below its working width, at the scale, slot kind and sparsity of
    /// `ct_in`.
    fn ckks_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync;

    /// Refreshes `ct_in`, which holds integer messages, and applies each LUT of `luts`, writing into `ct_outs`, by
    /// running the steps according to a plan configured with
    /// [`BootstrappingPlan::with_functional_bootstrap`](crate::layouts::BootstrappingPlan::with_functional_bootstrap).
    ///
    /// The batch shares the stages up to the LUT step, so `n` LUTs cost one bootstrap plus `n` polynomial evaluations.
    ///
    /// # Errors
    ///
    /// See [refusals](CKKSBootstrappingOps#refusals), and the batch of [`Self::ckks_eval_lut`].
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
