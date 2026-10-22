use crate::{CKKSCompositionError, CKKSResult as Result, ckks_ensure, ckks_log_delta, layouts::eval_mod::EvalModPlan};
use poulpy_core::layouts::{GLWEPlaintextInfos, SetGLWEPlaintextInfos};
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
        GetAutomorphismKey, GetTensorKey, LWEInfos, LinearTransformation, Rank, SetBSGSMeta, SetK, TorusPrecision,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    execution::{TaskExecutor, worker_scratch_bytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps, CKKSCopyOps,
        CKKSDFTOps, CKKSEvalModOps, CKKSImagOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    eval_lut::{ckks_eval_lut, ckks_eval_lut_binary, ckks_eval_lut_from_basis, ckks_lut_power_basis},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, BootstrappingPipeline,
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, EncodedLut, EncodedLutKind, EvalModType, ScratchArenaTakeCKKS,
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl},
};
use poulpy_core::{GLWEBytesOf, GLWENormalize, ScratchArenaTakeCore};

/// [`CKKSBootstrappingReference::ckks_bootstrap_tmp_bytes_reference`]: the terms
/// of the rotation keys are maximized over their layouts, those of EvalMod come
/// from the tensor key alone.
fn bootstrap_tmp_bytes_for<BE, C1, C2, F>(
    module: &Module<BE>,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContext<BE, F>,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWEBytesOf<BE>,
    Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
    CKKSCiphertextOwned<BE>: CKKSCtBounds,
{
    let (boot_layout, in_layout) = bootstrap_layouts(ct_out, ct_in);
    let base2k = ct_in.base2k();
    let boot_ct_bytes = module.glwe_bytes_of_from_infos(&boot_layout);
    // The coefficient-plaintext proxy for the plaintext ops: EvalMod's
    // encoded polynomial coefficients, the widest plaintext the pipeline
    // multiplies by.
    let coeffs_meta = ctx.eval_mod().plan.coeffs_meta;
    let coeffs_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_out.n(),
            base2k,
            k: coeffs_meta.k,
            rank: Rank(1),
        },
        meta: coeffs_meta.meta,
    };

    let in_ct_bytes = module.glwe_bytes_of_from_infos(&in_layout);

    let post_mod_up = if ctx.coeffs_to_slots_bypass().is_some() {
        5 * boot_ct_bytes
    } else {
        3 * boot_ct_bytes
    };
    let mut carved = match ctx.pipeline() {
        BootstrappingPipeline::C2SFirst => post_mod_up,
        BootstrappingPipeline::S2CFirst => post_mod_up.max(boot_ct_bytes + in_ct_bytes),
    };

    // EvalMod writes into scratch-carved boot-width ciphertexts and, on the
    // real-slot path, into `ct_out` itself, which may be allocated wider
    // than its `k`.
    let tsk = &keys_layout.tensor_key;
    let eval_mod_tmp = module
        .ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, ctx.eval_mod(), tsk)
        .max(module.ckks_eval_mod_tmp_bytes(ct_out, &boot_layout, ctx.eval_mod(), tsk));
    let eval_mod_tmp = if <BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
        2 * worker_scratch_bytes::<BE>(eval_mod_tmp)
    } else {
        eval_mod_tmp
    };
    let mut nested = eval_mod_tmp;

    // CoeffsToSlots always reads the raised ciphertext. SlotsToCoeffs runs
    // in place on the input before ModUp in the S2C-first pipeline.
    // Otherwise it reads the EvalMod outputs and writes `ct_out`, whose
    // allocation can be wider than its `k`.
    let out_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            k: ct_out.max_k(),
            ..boot_layout.glwe_layout
        },
        meta: boot_layout.meta,
    };
    let (s2c_dst, s2c_src) = match ctx.pipeline() {
        BootstrappingPipeline::C2SFirst => (&out_layout, &boot_layout),
        BootstrappingPipeline::S2CFirst => (&in_layout, &in_layout),
    };
    // Every rotation key shape against the widest ciphertext: a narrow key has a
    // wider digit, which can need more scratch than the wide key.
    for atk in &keys_layout.rotation_key_layouts() {
        nested = nested
            .max(module.ckks_all_ops_with_atk_tmp_bytes(&boot_layout, tsk, atk, &coeffs_layout))
            .max(module.ckks_all_ops_with_atk_tmp_bytes(&in_layout, tsk, atk, &coeffs_layout))
            .max(module.ckks_dft_tmp_bytes(&boot_layout, &boot_layout, ctx.coeffs_to_slots(), atk))
            .max(module.ckks_dft_tmp_bytes(s2c_dst, s2c_src, ctx.slots_to_coeffs(), atk));
        if let Some(bypass) = ctx.coeffs_to_slots_bypass() {
            nested = nested.max(module.ckks_dft_tmp_bytes(&boot_layout, &boot_layout, bypass, atk));
        }
    }

    if ctx.pipeline() == BootstrappingPipeline::C2SFirst {
        carved += in_ct_bytes;
    }

    if let Some((encaps, sparse_to_dense)) = keys_layout
        .encapsulation
        .as_ref()
        .and_then(|encaps| encaps.sparse_to_dense.as_ref().map(|s2d| (encaps, s2d)))
    {
        nested = nested.max(BE::ckks_encapsulated_mod_up_tmp_bytes(
            module,
            &boot_layout,
            &in_layout,
            &encaps.dense_to_sparse,
            sparse_to_dense,
        ));
    }

    // A raise that applies the first CoeffsToSlots factor works on a few small
    // ciphertexts, as wide as its keys switch.
    if let (Some(factor), Some(key)) = (ctx.coeffs_to_slots_first_factor(), &keys_layout.first_factor_key) {
        nested = nested.max(BE::ckks_mod_up_first_factor_tmp_bytes(
            module,
            &boot_layout,
            &in_layout,
            factor,
            keys_layout.encapsulation.as_ref().map(|encaps| &encaps.dense_to_sparse),
            key,
        ));
    }

    carved + nested
}

/// Fails before any work unless every key covers the widest ciphertext the
/// bootstrap switches with it.
///
/// The bootstrap runs at `ct_out.k()`, which need not be the width the keys
/// were laid out for. Each stage's key lookup also refuses a key too narrow for
/// its width; this check only moves the failure ahead of the computation. An
/// EvalRound+ recipe runs its bypass at widths of its own and is left to those
/// checks. Keys with a row offset (sparse-to-dense, first factor) are laid out
/// for one raise: they are checked against the raise this bootstrap runs.
fn check_key_widths<BE, F, K>(
    ct_out: &CKKSCiphertextOwned<BE>,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    keys: &K,
) -> Result<()>
where
    BE: Backend,
    K: BootstrappingKeys<BE>,
    CKKSCiphertextOwned<BE>: CKKSCtBounds,
{
    let widths = ctx.key_widths();
    let (bootstrap_k, input_k) = (ct_out.k().as_usize(), ct_in.k().as_usize());
    if ctx.coeffs_to_slots_bypass().is_none() {
        let op = "ckks_bootstrap";
        let log_n = ct_out.n().as_usize().trailing_zeros() as usize;
        for (p, k) in widths.rotation_key_inputs(log_n, bootstrap_k, input_k) {
            let k = TorusPrecision(k as u32);
            keys.rotation_keys()
                .get_automorphism_key(p, k)
                .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                    op,
                    rotation: p,
                    k: k.into(),
                    source,
                })?;
        }
        let c2s = ctx.coeffs_to_slots().consumed_bits();
        ckks_ensure!(
            bootstrap_k >= c2s,
            "{op}: a working width of {bootstrap_k} bits is below the {c2s} that CoeffsToSlots consumes"
        );
        let eval_mod_k = bootstrap_k - c2s;
        let k = TorusPrecision(widths.eval_mod_relinearization_k(eval_mod_k, ct_in.log_delta()) as u32);
        keys.tensor_key()
            .get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into(), source })?;
    }
    check_raise_keys(ct_out, ct_in, ctx, keys)
}

/// Keys with a row offset are laid out for the raise: fails before any work
/// unless the first-factor keys, or a sparse-to-dense key with an offset, fit
/// the raise of `ct_in` into `ct_out.k()` bits.
pub(crate) fn check_raise_keys<BE, F, K, C1, C2, BEM>(
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContext<BE, F, BEM>,
    keys: &K,
) -> Result<()>
where
    BE: Backend,
    BEM: Backend,
    K: BootstrappingLinearKeys<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let bootstrap_k = ct_out.k().as_usize();
    let (k_small, raised) = ctx.key_widths().raise(ct_in.k().as_usize());
    if let Some(factor) = ctx.coeffs_to_slots_first_factor() {
        crate::reference::first_factor::check_first_factor_raise(
            ct_out.base2k().as_usize(),
            ct_out.n().as_usize().ilog2() as usize,
            bootstrap_k,
            raised,
            factor,
            keys,
        )?;
    } else if let Some((_, Some(sparse_to_dense))) = keys.encapsulation_keys()
        && sparse_to_dense.k_offset().as_usize() > 0
    {
        check_offset_raise(bootstrap_k, k_small, raised - k_small, sparse_to_dense)?;
    }
    Ok(())
}

/// Scratch of [`ckks_bootstrap_s2c_mod_up`] into a ciphertext like `ct_raised`
/// followed by CoeffsToSlots on it, with every rotation key shape against the
/// widest ciphertext: the linear stages of an S2C-first bootstrap.
pub(crate) fn s2c_first_linear_tmp_bytes<BE, C1, C2, F, BEM>(
    module: &Module<BE>,
    ct_raised: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContext<BE, F, BEM>,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    BEM: Backend,
    Module<BE>: GLWEBytesOf<BE> + CKKSDFTOps<BE> + CKKSPow2Ops<BE> + GLWEShift<BE> + GLWEKeyswitch<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let (boot_layout, in_layout) = bootstrap_layouts(ct_raised, ct_in);
    // The raise: a shift, under encapsulation between two switches, or with the
    // first factor, on a few small ciphertexts as wide as its keys switch.
    let mut raise = module.glwe_shift_tmp_bytes(boot_layout.size());
    if let Some((encaps, sparse_to_dense)) = keys_layout
        .encapsulation
        .as_ref()
        .and_then(|encaps| encaps.sparse_to_dense.as_ref().map(|s2d| (encaps, s2d)))
    {
        raise = raise.max(BE::ckks_encapsulated_mod_up_tmp_bytes(
            module,
            &boot_layout,
            &in_layout,
            &encaps.dense_to_sparse,
            sparse_to_dense,
        ));
    }
    if let (Some(factor), Some(key)) = (ctx.coeffs_to_slots_first_factor(), &keys_layout.first_factor_key) {
        raise = raise.max(BE::ckks_mod_up_first_factor_tmp_bytes(
            module,
            &boot_layout,
            &in_layout,
            factor,
            keys_layout.encapsulation.as_ref().map(|encaps| &encaps.dense_to_sparse),
            key,
        ));
    }
    // SlotsToCoeffs and the raise work on a copy of the input, CoeffsToSlots on
    // the raised ciphertext.
    let in_ct_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&in_layout));
    let mut bytes = 0;
    for atk in &keys_layout.rotation_key_layouts() {
        let s2c_mod_up = module
            .ckks_dft_tmp_bytes(&in_layout, &in_layout, ctx.slots_to_coeffs(), atk)
            .max(module.ckks_mul_pow2_tmp_bytes(in_layout.size()))
            .max(raise);
        let c2s = module.ckks_dft_tmp_bytes(&boot_layout, &boot_layout, ctx.coeffs_to_slots(), atk);
        bytes = bytes.max(in_ct_bytes + s2c_mod_up).max(c2s);
    }
    bytes
}

fn bootstrap_layouts<C1, C2>(ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
where
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let base2k = ct_in.base2k();
    let boot_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_out.n(),
            base2k,
            k: ct_out.k(),
            rank: Rank(1),
        },
        // `log_delta = 0` maximizes `log_budget`, which upper-bounds the
        // EvalMod working width.
        meta: GLWEPlaintextMeta::default(),
    };
    let in_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_in.n(),
            base2k,
            k: ct_in.k(),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta::default(),
    };
    (boot_layout, in_layout)
}

/// Backend-generic reference of the CKKS bootstrapping pipeline, selected by
/// [`impl_ckks_bootstrapping_reference`](crate::oep::impl_ckks_bootstrapping_reference).
pub trait CKKSBootstrappingReference<BE: Backend> {
    fn ckks_mod_up_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    /// Scratch upper bound for [`Self::ckks_bootstrap_reference`].
    fn ckks_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds;

    /// Scratch upper bound for [`Self::ckks_functional_bootstrap_reference`], for one
    /// LUT or a batch: the batch folds each imaginary half as it is produced,
    /// so the carved working set does not grow with `luts.len()`.
    fn ckks_functional_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds;

    /// ModUp with the lift of `eval_mod` fused into its shift, relabeled by the
    /// message ratio.
    fn ckks_mod_up_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSInfos;

    /// Whole raise step: lift to the plan's message ratio, (encapsulate) ModUp
    /// with the second lift fused into its shift, then relabel by the message
    /// ratio so `I(X)·q` is the integer part and the message the residue.
    fn ckks_bootstrap_mod_up_reference<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE> + CKKSCopyOps<BE>,
        K: BootstrappingLinearKeys<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Refreshes one standard ciphertext. The pipeline is selected from the context.
    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap_reference<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>
            + GLWENormalize<BE>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    /// Backend-generic reference for
    /// [`CKKSBootstrappingOps::ckks_functional_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_functional_bootstrap).
    ///
    /// One LUT or many: the batch shares the S2C, ModUp and CoeffsToSlots stages,
    /// and general LUTs additionally share the power basis of each transformed
    /// half. The slot kind of `ct_in` selects the pipeline: real slots skip the
    /// imaginary branch entirely.
    fn ckks_functional_bootstrap_reference<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + CKKSModuleAlloc<BE>
            + GLWEKeyswitch<BE>
            + CKKSDFTOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSImagOps<BE>
            + CKKSEvalModOps<BE>
            + CKKSPolynomialEvaluationOps<BE>
            + CKKSComplexPolynomialEvaluationOps<BE>
            + CKKSCopyOps<BE>
            + CKKSMulOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAffineOps<BE>
            + GLWENormalize<BE>,
        K: BootstrappingKeys<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

impl<BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl> CKKSBootstrappingReference<BE> for Module<BE> {
    fn ckks_mod_up_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Module<BE>: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes(res_size)
    }

    fn ckks_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        bootstrap_tmp_bytes_for(self, ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_functional_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        let base = self.ckks_bootstrap_tmp_bytes_reference(ct_out, ct_in, ctx, keys_layout);
        let (boot_layout, in_layout) = bootstrap_layouts(ct_out, ct_in);
        let boot_bytes = self.glwe_bytes_of_from_infos(&boot_layout);
        let in_bytes = self.glwe_bytes_of_from_infos(&in_layout);
        // The complex path holds `ct_raised`, `r0` and `i0` while carving one
        // LUT-evaluation temporary. Direct S2C instead holds one boot-width
        // ciphertext beside its input-width buffer.
        let carved = (4 * boot_bytes).max(boot_bytes + in_bytes);
        let tsk = &keys_layout.tensor_key;
        let mut nested = 0usize;
        if luts.iter().any(EncodedLut::requires_eval_mod) {
            nested = nested
                .max(self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, ctx.eval_mod(), tsk))
                .max(self.ckks_eval_mod_tmp_bytes(ct_out, &boot_layout, ctx.eval_mod(), tsk));
        }
        for atk in &keys_layout.rotation_key_layouts() {
            for lut in luts {
                nested = nested
                    .max(self.ckks_all_ops_with_atk_tmp_bytes(&boot_layout, tsk, atk, lut.coeffs()))
                    .max(self.ckks_all_ops_with_atk_tmp_bytes(&in_layout, tsk, atk, lut.coeffs()));
            }
        }
        base.max(carved + nested)
    }

    fn ckks_mod_up_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSInfos,
    {
        let scale_up = eval_mod.raised_scale_up(src.log_delta())?;
        ckks_mod_up_scaled_reference(self, dst, src, scale_up, scratch)?;
        let mut meta = dst.meta();
        meta.scale = Scale::Log(ckks_log_delta(&meta) + eval_mod.log_msg_ratio);
        dst.set_meta(meta);
        Ok(())
    }

    fn ckks_bootstrap_mod_up_reference<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        eval_mod: &EvalModPlan,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE> + CKKSCopyOps<BE>,
        K: BootstrappingLinearKeys<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        bootstrap_mod_up(self, dst, src, eval_mod, None, keys, scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap_reference<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>
            + GLWENormalize<BE>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ckks_ensure!(
            ctx.functional_message_modulus().is_none(),
            "ckks_bootstrap requires an identity bootstrapping context"
        );
        // All pipeline intermediates are rank-1 working ciphertexts carved from
        // scratch (accounted for by `ckks_bootstrap_tmp_bytes`); reject
        // higher-rank inputs up front.
        ckks_ensure!(
            ct_in.rank().as_usize() == 1 && ct_out.rank().as_usize() == 1,
            "ckks_bootstrap supports rank-1 ciphertexts only, got rank {} -> {}",
            ct_in.rank().as_usize(),
            ct_out.rank().as_usize()
        );

        let encapsulation_keys = keys.encapsulation_keys();
        ckks_ensure!(
            encapsulation_keys.is_some() == ctx.sparse_secret_hamming_weight().is_some(),
            "bootstrapping key encapsulation does not match the compiled recipe (expected {}, got {})",
            ctx.sparse_secret_hamming_weight().is_some(),
            encapsulation_keys.is_some()
        );
        check_key_widths(ct_out, ct_in, ctx, keys)?;

        if ctx.pipeline() == BootstrappingPipeline::S2CFirst {
            // Known-real slots skip the imaginary nonlinear branch: one EvalMod
            // instead of two. EvalRound+ has no real-slot form, so a bypass
            // recipe falls back to the general pipeline rather than erroring.
            if ct_in.slots().is_real() && ctx.coeffs_to_slots_bypass().is_none() {
                return bootstrap_real_inner(self, ct_out, ct_in, ctx, keys, scratch);
            }
            return ckks_bootstrap_s2c_first(self, ct_out, ct_in, ctx, keys, scratch);
        }

        let base2k = ct_in.base2k();
        let k_boot = ct_out.k();
        let boot_layout = GLWELayout {
            n: ct_out.n(),
            base2k,
            k: k_boot,
            rank: Rank(1),
        };

        scratch.scope(|scratch_local| {
            // (encapsulate) denseToSparse → ModUp → sparseToDense. With encapsulation the
            // integer wrap-around `I·q` exposed by ModUp is bounded by the *sparse* secret's
            // Hamming weight (https://eprint.iacr.org/2022/024).
            let (mut ct, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            bootstrap_mod_up(
                self,
                &mut ct,
                ct_in,
                &ctx.eval_mod().plan,
                ctx.coeffs_to_slots_first_factor(),
                keys,
                &mut scratch_local,
            )?;

            // CoeffsToSlots (split): coefficients → (real, imag) slots. In the standard
            // pipeline this feeds EvalMod directly; in EvalRound+ it is the low-precision
            // transform feeding the round.
            let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
            ckks_bootstrap_coeffs_to_slots(self, &ct, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;
            match ctx.coeffs_to_slots_bypass() {
                // Standard: EvalMod's clean residue goes straight to SlotsToCoeffs.
                None => {
                    self.ckks_eval_mod(&mut ct, &r0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    r0.set_k(k_boot);
                    self.ckks_eval_mod(&mut r0, &i0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    self.ckks_slots_to_coeffs_split(
                        ct_out,
                        &ct,
                        &r0,
                        ctx.slots_to_coeffs(),
                        keys.rotation_keys(),
                        &mut scratch_local,
                    )?;
                }
                // EvalRound+: r1 = r0_hp − K·r0_lp + EvalMod(r0_lp) = IDFT(Δ·m). The
                // integer part and the low-precision error `e` both cancel, leaving the
                // message at the high-precision (bypass) transform's precision.
                Some(bypass) => {
                    let (mut r0_hp, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
                    let (mut i0_hp, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct.meta());
                    self.ckks_coeffs_to_slots_split(
                        &mut r0_hp,
                        &mut i0_hp,
                        &ct,
                        bypass,
                        keys.rotation_keys(),
                        &mut scratch_local,
                    )?;

                    // `BootstrappingContext::compile` rejects this, but re-check
                    // before the power-of-two shift.
                    ckks_ensure!(
                        ctx.eval_mod().plan.f_mod_interval.is_power_of_two(),
                        "EvalRound+ requires a power-of-two f_mod_interval, got {}",
                        ctx.eval_mod().plan.f_mod_interval
                    );
                    let log2_k = ctx.eval_mod().plan.f_mod_interval.trailing_zeros() as usize;

                    self.ckks_eval_mod(&mut ct, &r0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    self.ckks_mul_pow2_assign(&mut r0, log2_k, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut r0_hp, &r0, &mut scratch_local)?;
                    self.ckks_add_assign(&mut r0_hp, &ct, &mut scratch_local)?;

                    r0.set_k(k_boot);
                    self.ckks_eval_mod(&mut r0, &i0, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
                    self.ckks_mul_pow2_assign(&mut i0, log2_k, &mut scratch_local)?;
                    self.ckks_sub_assign(&mut i0_hp, &i0, &mut scratch_local)?;
                    self.ckks_add_assign(&mut i0_hp, &r0, &mut scratch_local)?;

                    self.ckks_slots_to_coeffs_split(
                        ct_out,
                        &r0_hp,
                        &i0_hp,
                        ctx.slots_to_coeffs(),
                        keys.rotation_keys(),
                        &mut scratch_local,
                    )?;
                }
            }
            Result::Ok(())
        })?;
        ct_out.set_log_delta(ct_in.log_delta());
        ct_out.set_slots(ct_in.slots());
        Ok(())
    }

    fn ckks_functional_bootstrap_reference<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + CKKSModuleAlloc<BE>
            + GLWEKeyswitch<BE>
            + CKKSDFTOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSImagOps<BE>
            + CKKSEvalModOps<BE>
            + CKKSPolynomialEvaluationOps<BE>
            + CKKSComplexPolynomialEvaluationOps<BE>
            + CKKSCopyOps<BE>
            + CKKSMulOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAffineOps<BE>
            + GLWENormalize<BE>,
        K: BootstrappingKeys<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ckks_ensure!(!luts.is_empty(), "functional bootstrap requires at least one LUT");
        ckks_ensure!(
            ct_outs.len() == luts.len(),
            "functional bootstrap ct_outs/luts length mismatch ({} vs {})",
            ct_outs.len(),
            luts.len()
        );
        ckks_ensure!(
            ctx.pipeline() == BootstrappingPipeline::S2CFirst,
            "functional bootstrapping requires an S2C-first context"
        );
        let log_msg_ratio = luts[0].log_msg_ratio();
        ckks_ensure!(
            luts.iter().all(|lut| lut.message_modulus() == luts[0].message_modulus()),
            "functional bootstrap LUTs must have the same message modulus"
        );
        if luts.iter().any(EncodedLut::requires_eval_mod) {
            ensure_unit_circle_exp_context(ctx)?;
        }
        let head = &ct_outs[0];
        let (out_n, out_k) = (head.n(), head.k());
        ckks_ensure!(
            ct_in.rank().as_usize() == 1
                && ct_outs.iter().all(|ct| {
                    ct.rank().as_usize() == 1 && ct.n() == head.n() && ct.base2k() == head.base2k() && ct.k() == head.k()
                }),
            "functional bootstrapping outputs must share one rank-1 layout"
        );
        ckks_ensure!(
            keys.encapsulation_keys().is_some() == ctx.sparse_secret_hamming_weight().is_some(),
            "bootstrapping key encapsulation does not match the compiled recipe"
        );
        ensure_functional_message_ratio(ct_in, ctx.slots_to_coeffs().consumed_bits(), log_msg_ratio)?;
        // Keys laid out for the raise fail here rather than after SlotsToCoeffs.
        check_raise_keys(head, ct_in, ctx, keys)?;
        ckks_ensure!(
            ctx.functional_message_modulus().unwrap_or(1usize << log_msg_ratio) == luts[0].message_modulus(),
            "functional bootstrap context must be configured for message modulus {}",
            luts[0].message_modulus()
        );
        let output_contracts = ct_outs
            .iter()
            .zip(luts)
            .map(|(ct_out, lut)| functional_output_contract(ct_in, ctx, lut, ct_out.k().as_usize()))
            .collect::<Result<Vec<_>>>()?;

        let shared = luts.len() > 1 && luts.iter().all(|lut| lut.general_series().is_some());
        let boot_layout = GLWELayout {
            n: out_n,
            base2k: ct_in.base2k(),
            k: out_k,
            rank: Rank(1),
        };
        scratch.scope(|scratch_local| {
            let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            ckks_bootstrap_s2c_mod_up(self, &mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;

            if ct_in.slots().is_real() {
                let (mut r0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
                ckks_bootstrap_coeffs_to_slots_real(self, &mut ct_raised, &mut r0, ctx, keys, &mut scratch_local)?;
                ct_raised.set_log_delta(ct_raised.log_delta() - ctx.c2s_guard_bits());
                return eval_lut_batch(self, ct_outs, &ct_raised, ctx, luts, keys, shared, &mut scratch_local);
            }

            let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            ckks_bootstrap_coeffs_to_slots(self, &ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;
            r0.set_log_delta(r0.log_delta() - ctx.c2s_guard_bits());
            i0.set_log_delta(i0.log_delta() - ctx.c2s_guard_bits());

            eval_lut_batch(self, ct_outs, &r0, ctx, luts, keys, shared, &mut scratch_local)?;

            // `ct_raised` is dead after CoeffsToSlots, so it serves as the single
            // imaginary accumulator: each LUT's imaginary half is folded into its
            // output before the next one is evaluated, which keeps the carved
            // working set independent of the batch size. Multiplication produces
            // its result at the destination's requested `k`, so the accumulator is
            // restored to the full boot width before each reuse; otherwise the
            // first LUT's consumption would clamp every later one.
            let im_meta = ct_raised.meta();
            if shared {
                let basis = ckks_lut_power_basis(
                    self,
                    &i0,
                    &boot_layout,
                    ctx.eval_mod(),
                    luts,
                    keys.tensor_key(),
                    &mut scratch_local,
                )?;
                for (ct_out, lut) in ct_outs.iter_mut().zip(luts) {
                    ct_raised.set_meta(im_meta);
                    ct_raised.set_k(boot_layout.k);
                    ckks_eval_lut_from_basis(
                        self,
                        &mut ct_raised,
                        lut,
                        &basis,
                        keys.rotation_keys(),
                        keys.tensor_key(),
                        &mut scratch_local,
                    )?;
                    recombine_halves(self, ct_out, &mut ct_raised, &mut scratch_local)?;
                }
            } else {
                for (ct_out, lut) in ct_outs.iter_mut().zip(luts) {
                    ct_raised.set_meta(im_meta);
                    ct_raised.set_k(boot_layout.k);
                    ckks_eval_encoded_lut(self, &mut ct_raised, &i0, ctx, lut, keys, &mut scratch_local)?;
                    recombine_halves(self, ct_out, &mut ct_raised, &mut scratch_local)?;
                }
            }
            Result::Ok(())
        })?;
        for (ct_out, (meta, expected_k)) in ct_outs.iter_mut().zip(output_contracts) {
            ct_out.set_meta(meta);
            ckks_ensure!(
                ct_out.k().as_usize() >= expected_k,
                "functional bootstrap produced k={}, below required {expected_k}",
                ct_out.k().as_usize()
            );
            ct_out.set_k(expected_k.into());
        }
        Ok(())
    }
}

/// `scale_up` lifts the message that many bits above its natural magnitude,
/// fused into the widening shift rather than costing a second pass.
pub(crate) fn ckks_mod_up_scaled_reference<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    scale_up: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSInfos,
{
    // `dst` is the (freshly-allocated) widened target: ModUp raises the modulus
    // to the destination's requested modulus, so the widening width is `dst.k()`.
    // `dst.k()` is meta-derived and is `0` on a fresh ciphertext, which would
    // spuriously fail the "must widen" check below.
    let k_large: usize = dst.k().as_usize();
    let k_small: usize = src.k().as_usize();

    ckks_ensure!(
        k_large >= k_small + scale_up,
        "ckks_mod_up: dst.k ({k_large}) < src.k ({k_small}) + scale_up ({scale_up}); ModUp must widen, not shrink, the modulus"
    );

    // MSB-align `src` into the wider `dst`: the value occupies the top
    // `k_small` bits and the freshly-introduced low-order limbs are zero.
    module.glwe_copy(dst, src, scratch);

    // Shift the digits down to their natural integer magnitude. This is the
    // modulus raise: the raised-from modulus `q = 2^k_small` becomes an
    // explicit, un-reduced multiple `I(X)·q` living in the `[0, 2^k_large)`
    // window, which EvalMod subsequently removes. (A zero shift, when the
    // input already fills the storage, is a no-op.) Stopping `scale_up` bits
    // short of the natural magnitude is the fused lift: the value lands
    // multiplied by `2^scale_up`, absorbed by the matching `log_delta`.
    module.glwe_rsh(k_large - k_small - scale_up, dst, scratch);

    // `log_delta` picks up the fused lift, so the encoded value is unchanged;
    // the remaining headroom now spans the full raised modulus, so the torus
    // width `k` becomes `k_large` and `log_budget = k_large - log_delta`.
    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
    });
    dst.set_k(k_large.into());

    Ok(())
}

/// Whole raise step of the C2S-first pipeline: lift to the plan's message
/// ratio, (encapsulated) ModUp, and the first CoeffsToSlots factor when the
/// plan leaves it to the raise.
fn bootstrap_mod_up<BE, Dst, Src, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    eval_mod: &EvalModPlan,
    first_factor: Option<&LinearTransformation<CKKSPlaintextOwned<BE>>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE> + CKKSCopyOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    // The pipeline carves rank-1 intermediates, so reject higher-rank inputs
    // rather than silently copying into a rank-1 scratch ciphertext.
    ckks_ensure!(
        src.rank().as_usize() == 1 && dst.rank().as_usize() == 1,
        "ckks_bootstrap_mod_up supports rank-1 ciphertexts only, got rank {} -> {}",
        src.rank().as_usize(),
        dst.rank().as_usize()
    );

    // The input-width copy is scoped so its scratch is released right after
    // ModUp widens it into `dst`.
    scratch.scope(|scratch_inner| {
        let (mut ct0, mut scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(src, src.meta());
        module.ckks_copy(&mut ct0, src, &mut scratch_inner)?;
        ckks_bootstrap_mod_up_from_mut(
            module,
            dst,
            &mut ct0,
            Some(eval_mod),
            0,
            first_factor,
            keys,
            &mut scratch_inner,
        )
    })?;

    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(dst.log_delta() + eval_mod.log_msg_ratio),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
    });
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the raise takes its lift, its scale and the first factor as separate operands"
)]
fn ckks_bootstrap_mod_up_from_mut<BE, Dst, Src, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    lift: Option<&EvalModPlan>,
    mut scale_up: usize,
    first_factor: Option<&LinearTransformation<CKKSPlaintextOwned<BE>>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
    K: BootstrappingLinearKeys<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    // C2S-first derives both lifts from EvalMod; S2C-first supplies only
    // the internal guard bits, fused into ModUp's shift.
    if let Some(plan) = lift {
        let k: usize = src.k().as_usize();
        let log_delta: usize = src.log_delta();
        let scale_up_in = plan.input_scale_up(k, log_delta)?;
        if scale_up_in != 0 {
            module.ckks_mul_pow2_assign(src, scale_up_in, scratch)?;
            // Not `set_log_delta`: it preserves `log_budget` and grows `k`, which
            // cancels the shift. The lift spends budget at a fixed modulus.
            let mut meta = src.meta();
            meta.scale = Scale::Log(log_delta + scale_up_in);
            src.set_meta(meta);
        }
        scale_up = plan.raised_scale_up(src.log_delta())?;
    }

    // A first CoeffsToSlots factor left to the raise is applied here, on the
    // raised ciphertext while it is small; its switches replace sparse-to-dense.
    if let Some(factor) = first_factor {
        return BE::ckks_mod_up_first_factor(module, dst, src, scale_up, factor, keys, scratch);
    }

    match keys.encapsulation_keys() {
        Some((dense_to_sparse, sparse_to_dense)) => {
            ckks_ensure!(
                sparse_to_dense.is_some(),
                "the encapsulated ModUp needs a sparse-to-dense key, which a key set for the first factor on the raise has not"
            );
            let sparse_to_dense = sparse_to_dense.unwrap();
            BE::ckks_encapsulated_mod_up(
                module,
                dst,
                src,
                scale_up,
                &dense_to_sparse.to_backend_ref(),
                &sparse_to_dense.to_backend_ref(),
                scratch,
            )?;
        }
        None => ckks_mod_up_scaled_reference(module, dst, src, scale_up, scratch)?,
    }
    Ok(())
}

fn recombine_halves<BE, R1, R2>(
    module: &Module<BE>,
    res_re: &mut R1,
    res_im: &mut R2,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: CKKSImagOps<BE> + CKKSAddOps<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_mul_i_assign(res_im, scratch)?;
    module.ckks_add_assign(res_re, &*res_im, scratch)?;
    Ok(())
}

fn ckks_bootstrap_coeffs_to_slots<BE, F, K, C, R>(
    module: &Module<BE>,
    ct: &C,
    r0: &mut R,
    i0: &mut R,
    ctx: &BootstrappingContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: CKKSDFTOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_coeffs_to_slots_split(r0, i0, ct, ctx.coeffs_to_slots(), keys.rotation_keys(), scratch)
}

fn ckks_bootstrap_coeffs_to_slots_real<BE, F, K, R1, R2>(
    module: &Module<BE>,
    ct: &mut R1,
    conjugate: &mut R2,
    ctx: &BootstrappingContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: CKKSDFTOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_dft_evaluate_assign(ct, ctx.coeffs_to_slots(), keys.rotation_keys(), scratch)?;
    module.ckks_conjugate_into(conjugate, &*ct, keys.rotation_keys(), scratch)?;
    module.ckks_add_assign(ct, &*conjugate, scratch)?;
    // `z + conj(z) = 2·Re(z)` holds the input polynomial's coefficients.
    ct.set_slots(SlotsKind::Real);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ckks_bootstrap_eval_mod_halves<BE, F, K, C, R1, R2>(
    module: &Module<BE>,
    r0: &C,
    i0: &C,
    res_real: &mut R1,
    res_imag: &mut R2,
    ctx: &BootstrappingContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: CKKSEvalModOps<BE>,
    F: Sync,
    K: BootstrappingKeys<BE> + Sync,
    K::TensorKey: Sync,
    C: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
{
    let tensor_keys = keys.tensor_key();
    if !<BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
        module.ckks_eval_mod(res_real, r0, ctx.eval_mod(), tensor_keys, scratch)?;
        module.ckks_eval_mod(res_imag, i0, ctx.eval_mod(), tensor_keys, scratch)?;
        return Ok(());
    }

    // The two halves share what the layout-sized query reserved for them
    // (`ckks_bootstrap_tmp_bytes`): two worker arenas of EvalMod's scratch.
    let task_bytes = scratch.available() / 2 / BE::SCRATCH_ALIGN * BE::SCRATCH_ALIGN;
    let (arenas, _) = scratch.borrow().split(2, task_bytes);
    let mut arenas = arenas.into_iter();
    let mut scratch_real = arenas.next().unwrap();
    let mut scratch_imag = arenas.next().unwrap();
    let (real, imag) = <BE::TaskExecutor as TaskExecutor>::join(
        || module.ckks_eval_mod(res_real, r0, ctx.eval_mod(), tensor_keys, &mut scratch_real),
        || module.ckks_eval_mod(res_imag, i0, ctx.eval_mod(), tensor_keys, &mut scratch_imag),
    );
    real?;
    imag?;
    Ok(())
}

fn ckks_bootstrap_s2c_first<BE, F, K>(
    module: &Module<BE>,
    ct_out: &mut CKKSCiphertextOwned<BE>,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    F: Sync,
    Module<BE>: GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + GLWENormalize<BE>,
    K: BootstrappingKeys<BE> + Sync,
    K::TensorKey: Sync,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let boot_layout = GLWELayout {
        n: ct_out.n(),
        base2k: ct_in.base2k(),
        k: ct_out.k(),
        rank: Rank(1),
    };

    scratch.scope(|scratch_local| {
        let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
        ckks_bootstrap_s2c_mod_up(module, &mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;

        let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
        ckks_bootstrap_coeffs_to_slots(module, &ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;
        match ctx.coeffs_to_slots_bypass() {
            None => {
                {
                    let r0_ref = r0.to_backend_view_ref();
                    let i0_ref = i0.to_backend_view_ref();
                    ckks_bootstrap_eval_mod_halves(
                        module,
                        &r0_ref,
                        &i0_ref,
                        ct_out,
                        &mut ct_raised,
                        ctx,
                        keys,
                        &mut scratch_local,
                    )?;
                }
                recombine_halves(module, ct_out, &mut ct_raised, &mut scratch_local)?;
            }
            Some(bypass) => {
                let (mut r0_hp, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
                let (mut i0_hp, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
                module.ckks_coeffs_to_slots_split(
                    &mut r0_hp,
                    &mut i0_hp,
                    &ct_raised,
                    bypass,
                    keys.rotation_keys(),
                    &mut scratch_local,
                )?;
                {
                    let r0_ref = r0.to_backend_view_ref();
                    let i0_ref = i0.to_backend_view_ref();
                    ckks_bootstrap_eval_mod_halves(
                        module,
                        &r0_ref,
                        &i0_ref,
                        ct_out,
                        &mut ct_raised,
                        ctx,
                        keys,
                        &mut scratch_local,
                    )?;
                }

                ckks_ensure!(
                    ctx.eval_mod().plan.f_mod_interval.is_power_of_two(),
                    "EvalRound+ requires a power-of-two f_mod_interval, got {}",
                    ctx.eval_mod().plan.f_mod_interval
                );
                let log2_k = ctx.eval_mod().plan.f_mod_interval.trailing_zeros() as usize;
                module.ckks_mul_pow2_assign(&mut r0, log2_k, &mut scratch_local)?;
                module.ckks_mul_pow2_assign(&mut i0, log2_k, &mut scratch_local)?;
                module.ckks_sub_assign(&mut r0_hp, &r0, &mut scratch_local)?;
                module.ckks_sub_assign(&mut i0_hp, &i0, &mut scratch_local)?;
                module.ckks_add_assign(&mut r0_hp, ct_out, &mut scratch_local)?;
                module.ckks_add_assign(&mut i0_hp, &ct_raised, &mut scratch_local)?;
                recombine_halves(module, &mut r0_hp, &mut i0_hp, &mut scratch_local)?;
                module.ckks_copy(ct_out, &r0_hp, &mut scratch_local)?;
            }
        }
        Result::Ok(())
    })?;
    // Keep the C2S guard bits through EvalMod, then relabel at the output width.
    ct_out.set_log_delta(ct_out.log_delta() - ctx.c2s_guard_bits());
    ct_out.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(ct_in.log_delta()),
        slots: ct_in.slots(),
        log_sparsity: ct_in.log_sparsity(),
    });
    Ok(())
}

pub(crate) fn ckks_bootstrap_s2c_mod_up<BE, F, K, R, BEM>(
    module: &Module<BE>,
    ct_raised: &mut R,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F, BEM>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    BEM: Backend,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + CKKSDFTOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
{
    let input_layout = GLWELayout {
        n: ct_in.n(),
        base2k: ct_in.base2k(),
        k: ct_in.k(),
        rank: Rank(1),
    };

    scratch.scope(|scratch_inner| {
        let (mut ct_coeffs, mut scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(&input_layout, ct_in.meta());
        // A `Split` decode matrix is numerically identical to the standard
        // matrix after the two halves are recombined. Preserve the split
        // path's `2 * ct_in`, then intentionally use the format-agnostic
        // evaluator directly.
        module.ckks_double_into(&mut ct_coeffs, ct_in, &mut scratch_inner)?;
        module.ckks_dft_evaluate_assign(
            &mut ct_coeffs,
            ctx.slots_to_coeffs(),
            keys.rotation_keys(),
            &mut scratch_inner,
        )?;

        let log_modulus_in = ct_coeffs.k().as_usize();
        ckks_bootstrap_mod_up_from_mut(
            module,
            ct_raised,
            &mut ct_coeffs,
            None,
            ctx.c2s_guard_bits(),
            ctx.coeffs_to_slots_first_factor(),
            keys,
            &mut scratch_inner,
        )?;
        ct_raised.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(log_modulus_in + ctx.c2s_guard_bits()),
            slots: ct_in.slots(),
            log_sparsity: ct_in.log_sparsity(),
        });
        Result::Ok(())
    })
}

/// Real-slot S2C-first pipeline: one EvalMod instead of two. Selected by
/// [`Self::ckks_bootstrap_reference`] from `ct_in.slots()`; the recipe
/// preconditions are part of that selection, not checked again here.
fn bootstrap_real_inner<BE, F, K>(
    module: &Module<BE>,
    ct_out: &mut CKKSCiphertextOwned<BE>,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAddOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + GLWENormalize<BE>,
    K: BootstrappingKeys<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let boot_layout = GLWELayout {
        n: ct_out.n(),
        base2k: ct_in.base2k(),
        k: ct_out.k(),
        rank: Rank(1),
    };
    scratch.scope(|scratch_local| {
        let (mut ct_raised, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
        let (mut r0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
        ckks_bootstrap_s2c_mod_up(module, &mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;
        ckks_bootstrap_coeffs_to_slots_real(module, &mut ct_raised, &mut r0, ctx, keys, &mut scratch_local)?;
        module.ckks_eval_mod(ct_out, &ct_raised, ctx.eval_mod(), keys.tensor_key(), &mut scratch_local)?;
        ct_out.set_log_delta(ct_out.log_delta() - ctx.c2s_guard_bits());
        ct_out.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(ct_in.log_delta()),
            slots: ct_in.slots(),
            log_sparsity: ct_in.log_sparsity(),
        });
        Result::Ok(())
    })
}

/// Reference SSE pipeline: switch to the sparse secret before ModUp to bound
/// wrap-around, then restore the dense secret afterward.
#[doc(hidden)]
pub fn ckks_encapsulated_mod_up_reference<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    scale_up: usize,
    dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
    sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    ensure_encapsulation_covers(dst, src, dense_to_sparse, sparse_to_dense)?;
    module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
    if sparse_to_dense.k_offset().as_usize() > 0 {
        return ckks_mod_up_offset_reference(module, dst, src, scale_up, sparse_to_dense, scratch);
    }
    // The lift is fused into ModUp, so the message is already at its final scale
    // when sparse-to-dense adds its noise.
    ckks_mod_up_scaled_reference(module, dst, src, scale_up, scratch)?;
    module.glwe_keyswitch_assign(dst, &sparse_to_dense.to_backend_ref(), scratch);
    Ok(())
}

/// Bits of input a sparse-to-dense key with a row offset takes for a raise
/// into `dst_k` bits: those below its offset, which its rows must reach.
fn offset_key_window<K: GGLWEInfos>(dst_k: usize, key: &K) -> usize {
    dst_k.saturating_sub(key.k_offset().as_usize())
}

/// Checks that a sparse-to-dense key with a row offset serves the raise of a
/// `k_small`-bit ciphertext lifted by `scale_up` into `k_large` bits, and
/// returns the raise's shift and the bits the key's window then holds.
pub(crate) fn check_offset_raise<K: GGLWEInfos>(
    k_large: usize,
    k_small: usize,
    scale_up: usize,
    key: &K,
) -> Result<(usize, usize)> {
    ckks_ensure!(
        k_large >= k_small + scale_up,
        "ckks_mod_up: dst.k ({k_large}) < src.k ({k_small}) + scale_up ({scale_up}); ModUp must widen, not shrink, the modulus"
    );
    let shift = k_large - k_small - scale_up;
    let k_offset = key.k_offset().as_usize();
    ckks_ensure!(
        k_offset <= shift,
        "ckks_mod_up: the sparse-to-dense key lowers its input by {k_offset} bits, more than the raise's {shift}"
    );
    let window = offset_key_window(k_large, key) - scale_up;
    let max_input_k = key.gglwe_layout().max_input_k().as_usize();
    let reach = k_offset + key.gglwe_layout().gadget_k().as_usize();
    ckks_ensure!(
        window <= max_input_k && k_large <= reach,
        "ckks_mod_up: the sparse-to-dense key takes {max_input_k} bits of input and reaches bit {reach}; the raise needs {window} bits up to bit {k_large}"
    );
    Ok((shift, window))
}

/// ModUp and the sparse-to-dense switch as one key switch, with a key whose
/// rows sit `k_offset` bits lower: the switch reads the digits of `src` as
/// balanced integers and lowers them, which is the raise, integer multiple of
/// the modulus included. The raise lowers `src` by `dst.k - src.k - scale_up`
/// bits; a key of a smaller offset takes the rest as a shift of `src` inside
/// its window first, exact since the window holds every bit of it.
fn ckks_mod_up_offset_reference<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    scale_up: usize,
    sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    Src: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let k_large: usize = dst.k().as_usize();
    let k_small: usize = src.k().as_usize();
    let (shift, window) = check_offset_raise(k_large, k_small, scale_up, sparse_to_dense)?;
    let k_offset = sparse_to_dense.k_offset().as_usize();
    let key = sparse_to_dense.to_backend_ref();
    if shift == k_offset {
        module.glwe_keyswitch(dst, src, &key, scratch);
    } else {
        let layout = GLWELayout {
            n: src.n(),
            base2k: src.base2k(),
            k: window.into(),
            rank: src.rank(),
        };
        scratch.scope(|scratch_inner| {
            let (mut lowered, mut scratch_inner) = scratch_inner.take_glwe_scratch(&layout);
            module.glwe_copy(&mut lowered, src, &mut scratch_inner);
            module.glwe_rsh(shift - k_offset, &mut lowered, &mut scratch_inner);
            module.glwe_keyswitch(dst, &lowered, &key, &mut scratch_inner);
        });
    }
    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
    });
    Ok(())
}

/// Refuses encapsulation keys too narrow for their inputs before any write: `src`
/// for the dense-to-sparse switch, `dst` for an offset-free sparse-to-dense one. A
/// sparse-to-dense key with a row offset is checked against its raise, by
/// [`check_offset_raise`].
fn ensure_encapsulation_covers<Dst: CKKSCtBounds, Src: CKKSCtBounds, D2S: GGLWEInfos, S2D: GGLWEInfos>(
    dst: &Dst,
    src: &Src,
    dense_to_sparse: &D2S,
    sparse_to_dense: &S2D,
) -> Result<()> {
    let sparse_to_dense_covers =
        sparse_to_dense.k_offset().as_usize() > 0 || dst.k() <= sparse_to_dense.gglwe_layout().max_input_k();
    if src.k() > dense_to_sparse.gglwe_layout().max_input_k() || !sparse_to_dense_covers {
        return Err(CKKSCompositionError::InvalidOperand {
            op: "ckks_encapsulated_mod_up",
            reason: "the input is wider than an encapsulation key's max_input_k".into(),
        }
        .into());
    }
    Ok(())
}

/// Scratch bound for [`ckks_encapsulated_mod_up_reference`].
#[doc(hidden)]
pub fn ckks_encapsulated_mod_up_tmp_bytes_reference<BE, Dst, Src, D2S, S2D>(
    module: &Module<BE>,
    dst_infos: &Dst,
    src_infos: &Src,
    dense_to_sparse_infos: &D2S,
    sparse_to_dense_infos: &S2D,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEShift<BE> + GLWEKeyswitch<BE>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    D2S: GGLWEInfos,
    S2D: GGLWEInfos,
{
    let d2s = module.glwe_keyswitch_tmp_bytes(src_infos, src_infos, dense_to_sparse_infos);
    if sparse_to_dense_infos.k_offset().as_usize() > 0 {
        // The input of the switch, `src` or its shift inside the key's window.
        let window = GLWELayout {
            n: src_infos.n(),
            base2k: src_infos.base2k(),
            k: offset_key_window(dst_infos.max_k().as_usize(), sparse_to_dense_infos)
                .max(src_infos.k().as_usize())
                .into(),
            rank: src_infos.rank(),
        };
        let switch = module.glwe_keyswitch_tmp_bytes(dst_infos, &window, sparse_to_dense_infos);
        let shifted = module.glwe_bytes_of_from_infos(&window) + module.glwe_shift_tmp_bytes(window.size()).max(switch);
        return d2s.max(switch).max(shifted);
    }
    d2s.max(module.glwe_shift_tmp_bytes(dst_infos.size().max(src_infos.size())))
        .max(module.glwe_keyswitch_tmp_bytes(dst_infos, dst_infos, sparse_to_dense_infos))
}

fn ensure_unit_circle_exp_context<BE: Backend, F>(ctx: &BootstrappingContext<BE, F>) -> Result<()> {
    ckks_ensure!(
        ctx.eval_mod().plan.eval_mod_type == EvalModType::ExpCmplx,
        "general functional bootstrapping requires an ExpCmplx EvalMod context"
    );
    ckks_ensure!(
        ctx.eval_mod().plan.scaling == Some(std::f64::consts::TAU),
        "general functional bootstrapping requires a unit-circle ExpCmplx context (scaling = 2π)"
    );
    Ok(())
}

fn functional_output_contract<BE: Backend, F>(
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    bootstrap_k: usize,
) -> Result<(GLWEPlaintextMeta, usize)> {
    let log_delta = ct_in
        .log_delta()
        .checked_add(lut.log_msg_ratio())
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap output scale overflow")))?;
    let eval_mod = if lut.requires_eval_mod() {
        ctx.eval_mod().plan.consumed_bits()
    } else {
        0
    };
    let consumed = ctx
        .coeffs_to_slots()
        .consumed_bits()
        .checked_add(ctx.c2s_guard_bits())
        .and_then(|bits| bits.checked_add(eval_mod))
        .and_then(|bits| bits.checked_add(lut.consumed_bits(log_delta)))
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap budget overflow")))?;
    let output_k = bootstrap_k.checked_sub(consumed).ok_or_else(|| {
        crate::CKKSError::Internal(anyhow::anyhow!(
            "functional bootstrap output width {bootstrap_k} is smaller than its {consumed}-bit post-ModUp cost"
        ))
    })?;
    ckks_ensure!(
        output_k >= log_delta,
        "functional bootstrap leaves output k={output_k} below log_delta={log_delta}"
    );
    Ok((
        GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: ct_in.slots(),
            log_sparsity: ct_in.log_sparsity(),
        },
        output_k,
    ))
}

fn ckks_eval_encoded_lut<BE, F, K, C, R>(
    module: &Module<BE>,
    ct_out: &mut R,
    ct_in: &C,
    ctx: &BootstrappingContext<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSComplexPolynomialEvaluationOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>
        + GLWENormalize<BE>,
    K: BootstrappingKeys<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    match lut.kind() {
        EncodedLutKind::General(series) => ckks_eval_lut(
            module,
            ct_out,
            ct_in,
            ctx.eval_mod(),
            series,
            keys.rotation_keys(),
            keys.tensor_key(),
            scratch,
        )?,
        EncodedLutKind::Binary {
            cos,
            affine,
            log_interval_reduction,
        } => ckks_eval_lut_binary(
            module,
            ct_out,
            ct_in,
            cos,
            *log_interval_reduction,
            affine,
            keys.tensor_key(),
            scratch,
        )?,
    }
    Ok(())
}

/// Evaluates every LUT of a batch against one transformed half, writing into
/// `outs`. Equal-arity general LUTs share one power basis; a batch containing a
/// binary LUT falls back to evaluating each LUT against the shared input.
#[allow(clippy::too_many_arguments)]
fn eval_lut_batch<BE, F, K, C>(
    module: &Module<BE>,
    outs: &mut [CKKSCiphertextOwned<BE>],
    ct: &C,
    ctx: &BootstrappingContext<BE, F>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    keys: &K,
    shared: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSComplexPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAffineOps<BE>
        + GLWENormalize<BE>,
    K: BootstrappingKeys<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    if shared {
        let layout = GLWELayout {
            n: outs[0].n(),
            base2k: outs[0].base2k(),
            k: outs[0].k(),
            rank: outs[0].rank(),
        };
        let basis = ckks_lut_power_basis(module, ct, &layout, ctx.eval_mod(), luts, keys.tensor_key(), scratch)?;
        for (out, lut) in outs.iter_mut().zip(luts) {
            ckks_eval_lut_from_basis(module, out, lut, &basis, keys.rotation_keys(), keys.tensor_key(), scratch)?;
        }
        return Ok(());
    }
    for (out, lut) in outs.iter_mut().zip(luts) {
        ckks_eval_encoded_lut(module, out, ct, ctx, lut, keys, scratch)?;
    }
    Ok(())
}

fn ensure_functional_message_ratio<C>(ct_in: &C, s2c_consumed_bits: usize, expected: usize) -> Result<()>
where
    C: CKKSInfos,
{
    let log_budget = ct_in.log_budget();
    ckks_ensure!(
        log_budget >= s2c_consumed_bits,
        "functional bootstrap needs log_budget >= {s2c_consumed_bits}, got {log_budget}"
    );
    let actual = log_budget - s2c_consumed_bits;
    ckks_ensure!(
        actual == expected,
        "functional bootstrap LUT requires log_msg_ratio {expected}, got {actual}"
    );
    Ok(())
}
