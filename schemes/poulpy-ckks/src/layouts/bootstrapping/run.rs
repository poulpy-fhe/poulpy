//! A compiled bootstrap bound to its message, the operand of the steps.

use poulpy_core::layouts::{LinearTransformation, LinearTransformationPlan};
use poulpy_hal::layouts::Backend;

use crate::{
    CKKSResult, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind, ckks_ensure_operand, ckks_log_delta,
    layouts::{
        CKKSCITwin, CKKSDFTFactors, CKKSEvalModInfos, CKKSModRaiseInfos, CKKSPlaintextOwned, CKKSRightOperandPrepared,
        DFTMatrixPrepared, DFTPlan, Decode, Encode, EncodedLut, EvalMod, EvalModPlan, ModRaisePlan, Split, Standard,
        validation::{functional_output_contract, remove_guard_bits},
    },
};

use super::{BootstrappingContextPrepared, BootstrappingPipeline, EvalModCIPrepared};

/// A compiled bootstrap bound to the labels of the message it runs on. The steps read those labels here, so a
/// composition stamps nothing.
pub struct BootstrappingRun<'a, BE: Backend, F> {
    context: &'a BootstrappingContextPrepared<BE, F>,
    message: GLWEPlaintextMeta,
}

impl<BE: Backend, F> BootstrappingRun<'_, BE, F> {
    /// The EvalRound+ bypass of the plan, `None` without one. A composer runs it on a copy of the raised ciphertext,
    /// `ckks_coeffs_to_slots_assign(&mut copy, &bypass, ..)`, which leaves half a `Standard` encode, then doubles the
    /// copy and adds the output of `ckks_bootstrap_eval_mod`.
    pub fn bypass(&self) -> Option<&DFTMatrixPrepared<BE, Encode, Split>> {
        self.context.coeffs_to_slots_bypass()
    }
}

impl<BE: Backend, F> CKKSDFTFactors<BE, Encode, Standard> for BootstrappingRun<'_, BE, F> {
    type Diagonal = <DFTMatrixPrepared<BE, Encode, Split> as CKKSDFTFactors<BE, Encode, Split>>::Diagonal;

    fn dft_plan(&self) -> &DFTPlan {
        CKKSDFTFactors::<BE, Encode, Standard>::dft_plan(self.context)
    }

    fn factors<'a>(&'a self) -> impl Iterator<Item = &'a LinearTransformation<Self::Diagonal>>
    where
        BE: 'a,
        Encode: 'a,
        Standard: 'a,
    {
        CKKSDFTFactors::<BE, Encode, Standard>::factors(self.context)
    }

    fn factor_plans(&self) -> &[LinearTransformationPlan] {
        CKKSDFTFactors::<BE, Encode, Standard>::factor_plans(self.context)
    }
}

impl<BE: Backend, F> CKKSDFTFactors<BE, Decode, Standard> for BootstrappingRun<'_, BE, F> {
    type Diagonal = <DFTMatrixPrepared<BE, Decode, Split> as CKKSDFTFactors<BE, Decode, Split>>::Diagonal;

    fn dft_plan(&self) -> &DFTPlan {
        self.context.slots_to_coeffs().dft_plan()
    }

    fn factors<'a>(&'a self) -> impl Iterator<Item = &'a LinearTransformation<Self::Diagonal>>
    where
        BE: 'a,
        Decode: 'a,
        Standard: 'a,
    {
        self.context.slots_to_coeffs().factors()
    }

    fn factor_plans(&self) -> &[LinearTransformationPlan] {
        self.context.slots_to_coeffs().factor_plans()
    }

    fn stamp_transform_output<C: SetCKKSInfos>(&self, ct: &mut C) {
        // `set_log_delta` lowers `k`, so it waits for the end of SlotsToCoeffs.
        if self.context.pipeline() == BootstrappingPipeline::C2SFirst {
            ct.set_log_delta(ckks_log_delta(&self.message));
            ct.set_slots(self.message.slots);
        }
    }
}

impl<BE: Backend, F> CKKSModRaiseInfos<BE> for BootstrappingRun<'_, BE, F> {
    fn mod_raise_plan(&self) -> &ModRaisePlan {
        self.context.mod_raise_plan()
    }

    fn first_factor(&self) -> Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>> {
        CKKSModRaiseInfos::first_factor(self.context)
    }
}

impl<BE: CKKSCITwin, F> CKKSEvalModInfos<BE, F> for BootstrappingRun<'_, BE, F> {
    fn eval_mod_plan(&self) -> &EvalModPlan {
        self.context.eval_mod_plan()
    }

    fn eval_mod_standard(&self) -> Option<&EvalMod<F, CKKSPlaintextOwned<BE>>> {
        self.context.eval_mod()
    }

    fn eval_mod_ci(&self) -> Option<&EvalModCIPrepared<BE, F>> {
        self.context.eval_mod_ci()
    }

    fn c2s_guard_bits(&self) -> usize {
        self.context.c2s_guard_bits()
    }

    fn real_part_on_real_slots(&self) -> bool {
        self.context.pipeline() == BootstrappingPipeline::S2CFirst
    }

    fn eval_round_plus(&self) -> bool {
        self.context.coeffs_to_slots_bypass().is_some()
    }

    fn message_slots(&self) -> Option<SlotsKind> {
        Some(self.message.slots)
    }

    fn stamp_eval_mod_output<C: SetCKKSInfos>(&self, res: &mut C) -> CKKSResult<()> {
        let log_delta = remove_guard_bits("ckks_bootstrap_eval_mod", res.log_delta(), self.context.c2s_guard_bits())?;
        res.set_log_delta(log_delta);
        // C2S-first ends in SlotsToCoeffs, which stamps the rest.
        if self.context.pipeline() == BootstrappingPipeline::S2CFirst {
            res.set_meta(self.message);
        }
        Ok(())
    }

    fn eval_lut_output(
        &self,
        op: &'static str,
        lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
        a_k: usize,
    ) -> CKKSResult<Option<(GLWEPlaintextMeta, usize)>> {
        ckks_ensure_operand!(
            op,
            self.context.pipeline() == BootstrappingPipeline::S2CFirst,
            "EvalLut requires an S2C-first run"
        );
        let bootstrap_k = a_k + self.context.coeffs_to_slots().consumed_bits();
        functional_output_contract(op, self.message, self.context, lut, bootstrap_k).map(Some)
    }
}

impl<BE: Backend, F> BootstrappingContextPrepared<BE, F> {
    /// Binds this bootstrap to the labels of the message it runs on: the steps read them from the returned run. It takes
    /// the metadata, not a ciphertext: the raise clobbers its input, and a composer may state a message kind the label
    /// does not carry.
    pub fn run(&self, message: GLWEPlaintextMeta) -> BootstrappingRun<'_, BE, F> {
        BootstrappingRun { context: self, message }
    }
}
