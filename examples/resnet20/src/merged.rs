use anyhow::{Result, ensure};
use poulpy_ckks::{
    api::{CKKSLinearTransformationOps, ckks_plaintext_diagonal_layout},
    layouts::MergedS2CFactorPrepared,
};
use poulpy_core::{
    GLWELinearTransformationBabyStepsBytesOf, ScratchArenaTakeGLWELinearTransformationBabySteps,
    layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey, LinearTransformationPlan},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{Context, Ct, ServerBackend, kernel::kernel_polynomials, layout::CoeffLayout, model::Window};

/// A convolution held in the last SlotsToCoeffs factor: evaluated on a ciphertext that SlotsToCoeffs has run on short
/// of that factor, it leaves the convolution in the coefficients, for the one level of that factor. Coefficients
/// outside the output samples hold other terms.
///
/// The factor is prepared once, here, and stays resident.
pub(crate) struct MergedConv {
    factor: MergedS2CFactorPrepared<ServerBackend>,
    /// The rotations the factor reads, those of the last SlotsToCoeffs factor whatever the kernels.
    plan: LinearTransformationPlan,
}

impl MergedConv {
    /// Scratch of [`Self::new`] at scale `2^log_delta`.
    pub(crate) fn new_tmp_bytes(module: &Module<ServerBackend>, context: &Context, log_delta: usize) -> Result<usize> {
        context.merged_s2c_factor_tmp_bytes(module, log_delta)
    }

    /// The convolution of window `window` from `input` to `output` at scale `2^log_delta`: it consumes `log_delta`
    /// bits. `weights` is `[out][in][kernel][kernel]`, row-major.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        module: &Module<ServerBackend>,
        context: &Context,
        input: &CoeffLayout,
        output: &CoeffLayout,
        weights: &[f64],
        window: Window,
        log_delta: usize,
        scratch: &mut ScratchArena<'_, ServerBackend>,
    ) -> Result<Self> {
        ensure!(input.n() == module.n(), "the layouts must span the module's degree");
        let polys = kernel_polynomials(input, output, weights, window, context.merged_s2c_cosets(), context.log_gap())?;
        let factor = context.merged_s2c_factor(module, &polys, log_delta, scratch)?;
        let plan = factor.factor().index();
        Ok(Self { factor, plan })
    }

    /// Scratch of [`Self::apply`] on `blocks`, with `shortcut` when it runs beside this convolution and rotation keys of
    /// layout `key`.
    pub(crate) fn apply_tmp_bytes<K, T>(
        &self,
        module: &Module<ServerBackend>,
        shortcut: Option<&Self>,
        blocks: &Ct,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        let diagonal = ckks_plaintext_diagonal_layout(blocks);
        let eval = |conv: &Self| {
            <Module<ServerBackend> as CKKSLinearTransformationOps<ServerBackend>>::ckks_linear_transformation_eval_tmp_bytes(
                module, blocks, &diagonal, &conv.plan, tsk, key,
            )
        };
        let prepare =
            <Module<ServerBackend> as CKKSLinearTransformationOps<ServerBackend>>::ckks_linear_transformation_baby_steps_prepare_tmp_bytes(
                module, blocks, &self.plan, key,
            );
        module.glwe_linear_transformation_baby_steps_bytes_of(&self.plan, blocks)
            + prepare.max(eval(self)).max(shortcut.map_or(0, eval))
    }

    /// `res = conv(blocks)` with `blocks` short of the last SlotsToCoeffs factor and at least `log_delta` wider than
    /// `res`. A `shortcut`, another convolution of `blocks` and its destination, is evaluated beside it on the
    /// rotations of `blocks` this convolution prepares.
    ///
    /// # Errors
    ///
    /// A `shortcut` that reads a rotation of `blocks` this convolution does not.
    pub(crate) fn apply<H>(
        &self,
        module: &Module<ServerBackend>,
        res: &mut Ct,
        shortcut: Option<(&Self, &mut Ct)>,
        blocks: &Ct,
        keys: &H,
        scratch: &mut ScratchArena<'_, ServerBackend>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<ServerBackend> + GetTensorKey<ServerBackend>,
    {
        let (mut babies, mut scratch) = scratch
            .borrow()
            .take_glwe_linear_transformation_baby_steps_scratch(module, &self.plan, blocks);
        module.ckks_linear_transformation_baby_steps_prepare(&mut babies, blocks, keys, &mut scratch)?;
        module.ckks_linear_transformation_eval_into(res, blocks, &babies, self.factor.factor(), keys, &mut scratch)?;
        if let Some((conv, res)) = shortcut {
            module.ckks_linear_transformation_eval_into(res, blocks, &babies, conv.factor.factor(), keys, &mut scratch)?;
        }
        Ok(())
    }
}
