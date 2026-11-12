use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDFTOps, CKKSLinearTransformationOps, ckks_plaintext_diagonal_layout},
    layouts::{BootstrappingContextPrepared, MergedS2CFactorPrepared},
};
use poulpy_core::{
    GLWELinearTransformationBabyStepsBytesOf, ScratchArenaTakeGLWELinearTransformationBabySteps,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{kernel::kernel_polynomials, layout::CoeffLayout, network::ServerBackend};

type Context = BootstrappingContextPrepared<ServerBackend, f64>;
type Arena<'a, 'b> = &'a mut ScratchArena<'b, ServerBackend>;

/// SlotsToCoeffs of `ct` in place, short of its last factor, which a [`MergedConv`] takes.
pub fn slots_to_blocks<C, H>(
    module: &Module<ServerBackend>,
    ct: &mut C,
    context: &Context,
    keys: &H,
    scratch: Arena<'_, '_>,
) -> Result<()>
where
    C: GLWEToBackendMut<ServerBackend> + GLWEToBackendRef<ServerBackend> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<ServerBackend> + GetTensorKey<ServerBackend>,
{
    Ok(module.ckks_slots_to_coeffs_assign(ct, &context.slots_to_coeffs().head(), keys, scratch)?)
}

/// A convolution held in the last SlotsToCoeffs factor: evaluated on a ciphertext [`slots_to_blocks`] has run on, it
/// leaves the convolution in the coefficients, for the one level of that factor. Coefficients outside the output
/// samples hold other terms.
///
/// The factor is prepared once, here, and stays resident.
pub struct MergedConv {
    output: CoeffLayout,
    factor: MergedS2CFactorPrepared<ServerBackend>,
}

impl MergedConv {
    /// Scratch of [`Self::new`] on a context of radix `base2k`: a factor holds on one limb.
    pub fn new_tmp_bytes(module: &Module<ServerBackend>, context: &Context, base2k: usize) -> Result<usize> {
        context.merged_s2c_factor_tmp_bytes(module, base2k)
    }

    /// The `kernel x kernel` convolution of stride `stride` from `input` to `output` at scale `2^log_delta`: it consumes
    /// `log_delta` bits. `weights` is `[out][in][kernel][kernel]`, row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        module: &Module<ServerBackend>,
        context: &Context,
        input: &CoeffLayout,
        output: &CoeffLayout,
        weights: &[f64],
        kernel: usize,
        stride: usize,
        log_delta: usize,
        scratch: Arena<'_, '_>,
    ) -> Result<Self> {
        ensure!(input.n() == module.n(), "the layouts must span the module's degree");
        let polys = kernel_polynomials(
            input,
            output,
            weights,
            kernel,
            stride,
            context.merged_s2c_cosets(),
            context.log_gap(),
        )?;
        Ok(Self {
            output: output.clone(),
            factor: context.merged_s2c_factor(module, &polys, log_delta, scratch)?,
        })
    }

    pub fn output(&self) -> &CoeffLayout {
        &self.output
    }

    /// Scratch of [`Self::apply`] on `blocks`, with rotation keys of layout `key`.
    pub fn apply_tmp_bytes<A, K, T>(&self, module: &Module<ServerBackend>, blocks: &A, tsk: &T, key: &K) -> usize
    where
        A: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        let plan = self.factor.factor().index();
        let diagonal = ckks_plaintext_diagonal_layout(blocks);
        module.glwe_linear_transformation_baby_steps_bytes_of(&plan, blocks)
            + <Module<ServerBackend> as CKKSLinearTransformationOps<ServerBackend>>::ckks_linear_transformation_baby_steps_prepare_tmp_bytes(module, blocks, &plan, key)
                .max(<Module<ServerBackend> as CKKSLinearTransformationOps<ServerBackend>>::ckks_linear_transformation_eval_tmp_bytes(module, blocks, &diagonal, &plan, tsk, key))
    }

    /// `res = conv(a)` with `a` the state [`slots_to_blocks`] leaves, at least `log_delta` wider than `res`.
    pub fn apply<R, A, H>(
        &self,
        module: &Module<ServerBackend>,
        res: &mut R,
        blocks: &A,
        keys: &H,
        scratch: Arena<'_, '_>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<ServerBackend> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<ServerBackend> + CKKSCtBounds,
        H: GetAutomorphismKey<ServerBackend> + GetTensorKey<ServerBackend>,
    {
        let factor = self.factor.factor();
        let plan = factor.index();
        let (mut babies, mut scratch) = scratch
            .borrow()
            .take_glwe_linear_transformation_baby_steps_scratch(module, &plan, blocks);
        module.ckks_linear_transformation_baby_steps_prepare(&mut babies, blocks, keys, &mut scratch)?;
        module.ckks_linear_transformation_eval_into(res, blocks, &babies, factor, keys, &mut scratch)?;
        Ok(())
    }
}
