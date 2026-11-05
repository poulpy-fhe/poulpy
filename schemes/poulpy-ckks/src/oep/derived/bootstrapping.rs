//! The bootstrap and the functional bootstrap, composed from the public steps.
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, SetBSGSMeta},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSBootstrappingOps, CKKSCopyOps, CKKSDFTOps, CKKSPow2Ops},
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingRun,
        CKKSCiphertextOwned, CKKSDFTFactors, CKKSEvalModInfos, CKKSPlaintextOwned, EncodedLut, ScratchArenaTakeCKKSCiphertext,
        bootstrapping::bootstrap_layouts,
        validation::{check_bootstrap_entry, check_functional_entry},
    },
    oep::CKKSBootstrappingImpl,
};

/// The largest query of `dft` over the rotation key layouts: a narrow key has a wider digit, which can need more
/// scratch than the wide key.
fn dft_tmp_bytes<BE, Dir, Fmt, D>(
    module: &Module<BE>,
    dst: &CKKSLayout,
    src: &CKKSLayout,
    dft: &D,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: CKKSBootstrappingImpl,
    D: CKKSDFTFactors<BE, Dir, Fmt>,
{
    keys_layout
        .rotation_key_layouts()
        .iter()
        .map(|atk| module.ckks_dft_tmp_bytes(dst, src, dft, atk))
        .max()
        .unwrap_or(0)
}

/// `ckks_bootstrap_derived`: the raised ciphertext beside every stage, the copy of the input beside the stages up to
/// the raise, the EvalRound+ bypass beside CoeffsToSlots and EvalMod.
pub(crate) fn ckks_bootstrap_tmp_bytes_derived<BE, C1, C2, F>(
    module: &Module<BE>,
    res: &C1,
    a: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    ci: Option<&Module<BE::CI>>,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: CKKSBootstrappingImpl,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let (boot_layout, in_layout) = bootstrap_layouts(res, a);
    let boot_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&boot_layout));
    let in_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&in_layout));

    let mut after_raise = dft_tmp_bytes(module, &boot_layout, &boot_layout, ctx.coeffs_to_slots(), keys_layout)
        .max(module.ckks_bootstrap_eval_mod_tmp_bytes(res, &boot_layout, &ctx.run(a.meta()), ci, keys_layout));
    if let Some(bypass) = ctx.coeffs_to_slots_bypass() {
        let size = boot_layout.size();
        after_raise = boot_bytes
            + after_raise
                .max(dft_tmp_bytes(module, &boot_layout, &boot_layout, bypass, keys_layout))
                .max(module.ckks_copy_tmp_bytes(&boot_layout, &boot_layout))
                .max(module.ckks_copy_tmp_bytes(res, &boot_layout))
                .max(module.ckks_mul_pow2_tmp_bytes(size))
                .max(module.ckks_add_tmp_bytes(size));
    }

    let raise = module.ckks_mod_raise_tmp_bytes(&boot_layout, &in_layout, ctx, keys_layout);
    let stages = match ctx.pipeline() {
        BootstrappingPipeline::S2CFirst => {
            let before_raise = dft_tmp_bytes(module, &in_layout, &in_layout, ctx.slots_to_coeffs(), keys_layout)
                .max(module.ckks_double_tmp_bytes(in_layout.size()))
                .max(raise);
            after_raise.max(in_bytes + before_raise)
        }
        BootstrappingPipeline::C2SFirst => {
            // SlotsToCoeffs writes `res`, whose allocation can be wider than its `k`.
            let out_layout = CKKSLayout {
                glwe_layout: GLWELayout {
                    k: res.max_k(),
                    ..boot_layout.glwe_layout
                },
                meta: boot_layout.meta,
            };
            let before_raise = module.ckks_copy_tmp_bytes(&in_layout, &in_layout).max(raise);
            after_raise
                .max(dft_tmp_bytes(
                    module,
                    &out_layout,
                    &boot_layout,
                    ctx.slots_to_coeffs(),
                    keys_layout,
                ))
                .max(in_bytes + before_raise)
        }
    };
    boot_bytes + stages
}

/// `ckks_functional_bootstrap_derived`: the raised ciphertext beside every stage, the copy of the input beside the
/// stages up to the raise. The bytes of the LUT step do not grow with the batch.
pub(crate) fn ckks_functional_bootstrap_tmp_bytes_derived<BE, C1, C2, F>(
    module: &Module<BE>,
    res: &C1,
    a: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: CKKSBootstrappingImpl,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let (boot_layout, in_layout) = bootstrap_layouts(res, a);
    let boot_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&boot_layout));
    let in_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&in_layout));
    let before_raise = dft_tmp_bytes(module, &in_layout, &in_layout, ctx.slots_to_coeffs(), keys_layout)
        .max(module.ckks_double_tmp_bytes(in_layout.size()))
        .max(module.ckks_mod_raise_tmp_bytes(&boot_layout, &in_layout, ctx, keys_layout));
    boot_bytes
        + dft_tmp_bytes(module, &boot_layout, &boot_layout, ctx.coeffs_to_slots(), keys_layout)
            .max(module.ckks_eval_lut_tmp_bytes(res, &boot_layout, &ctx.run(a.meta()), luts, keys_layout))
            .max(in_bytes + before_raise)
}

/// Doubling, SlotsToCoeffs and the raise of `a` into `z`, on a copy released once `z` holds the raised ciphertext.
fn slots_to_coeffs_and_raise<BE, F, K, Z>(
    module: &Module<BE>,
    z: &mut Z,
    a: &CKKSCiphertextOwned<BE>,
    run: &BootstrappingRun<'_, BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSBootstrappingImpl,
    K: BootstrappingKeys<BE>,
    Z: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    scratch.scope(|scratch| {
        let (mut coeffs, mut scratch) = scratch.take_ckks_ciphertext_scratch(a, a.meta());
        // The split path's `2 * a`: a `Split` decode matrix equals the standard one once its halves recombine.
        module.ckks_double_into(&mut coeffs, a, &mut scratch)?;
        module.ckks_slots_to_coeffs_assign(&mut coeffs, run, keys.rotation_keys(), &mut scratch)?;
        module.ckks_mod_raise(z, &mut coeffs, run, keys, &mut scratch)
    })
}

/// Default body of `ckks_bootstrap`, by the pipeline of `ctx`: doubling, SlotsToCoeffs, the raise, CoeffsToSlots and
/// the EvalMod of a bootstrap, or the raise, CoeffsToSlots, EvalMod and SlotsToCoeffs, with the EvalRound+ bypass when
/// the plan has one. A [`BootstrappingRun`] of the input's metadata gives the steps the slots and the labels of `res`.
pub(crate) fn ckks_bootstrap_derived<BE, F, K>(
    module: &Module<BE>,
    res: &mut CKKSCiphertextOwned<BE>,
    a: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContextPrepared<BE, F>,
    ci: Option<&Module<BE::CI>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSBootstrappingImpl,
    F: Sync,
    K: BootstrappingKeys<BE> + Sync,
    K::TensorKey: Sync,
{
    check_bootstrap_entry(module, res, a, ctx, ci, keys, scratch.available())?;
    let run = ctx.run(a.meta());
    let boot_layout = GLWELayout {
        n: res.n(),
        base2k: a.base2k(),
        k: res.k(),
        rank: Rank(1),
    };
    scratch.scope(|scratch| {
        let (mut z, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, a.meta());
        match ctx.pipeline() {
            BootstrappingPipeline::S2CFirst => {
                slots_to_coeffs_and_raise(module, &mut z, a, &run, keys, &mut scratch)?;
                coeffs_to_slots_and_eval_mod(module, res, &mut z, &run, ci, keys, &mut scratch)
            }
            BootstrappingPipeline::C2SFirst => {
                // The raise consumes its input: it works on a copy, released once `z` holds the raised ciphertext.
                scratch.scope(|scratch| {
                    let (mut copy, mut scratch) = scratch.take_ckks_ciphertext_scratch(a, a.meta());
                    module.ckks_copy(&mut copy, a, &mut scratch)?;
                    module.ckks_mod_raise(&mut z, &mut copy, &run, keys, &mut scratch)
                })?;
                coeffs_to_slots_and_eval_mod(module, res, &mut z, &run, ci, keys, &mut scratch)?;
                module.ckks_slots_to_coeffs_assign(res, &run, keys.rotation_keys(), &mut scratch)
            }
        }
    })
}

/// CoeffsToSlots on `z` and EvalMod into `res`, with the EvalRound+ bypass when the plan has one.
fn coeffs_to_slots_and_eval_mod<BE, F, K, R, Z>(
    module: &Module<BE>,
    res: &mut R,
    z: &mut Z,
    run: &BootstrappingRun<'_, BE, F>,
    ci: Option<&Module<BE::CI>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSBootstrappingImpl,
    F: Sync,
    K: BootstrappingKeys<BE> + Sync,
    K::TensorKey: Sync,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    Z: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let Some(bypass) = run.context().coeffs_to_slots_bypass() else {
        module.ckks_coeffs_to_slots_assign(z, run, keys.rotation_keys(), scratch)?;
        return module.ckks_bootstrap_eval_mod(res, &*z, run, ci, keys, scratch);
    };
    let (mut z_hp, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&*z, z.meta());
    module.ckks_copy(&mut z_hp, &*z, &mut scratch)?;
    module.ckks_dft_matrix_eval_assign(&mut z_hp, bypass, keys.rotation_keys(), &mut scratch)?;
    module.ckks_coeffs_to_slots_assign(z, run, keys.rotation_keys(), &mut scratch)?;
    module.ckks_bootstrap_eval_mod(res, &*z, run, ci, keys, &mut scratch)?;
    // `2·z_hp`, the bypass on the packed halves, which the split doubles as `2·Re` and `2·Im`.
    module.ckks_mul_pow2_assign(1, &mut z_hp, &mut scratch)?;
    module.ckks_add_assign(&mut z_hp, &*res, &mut scratch)?;
    module.ckks_copy(res, &z_hp, &mut scratch)?;
    run.stamp_eval_mod_output(res);
    Ok(())
}

/// Default body of `ckks_functional_bootstrap`: doubling, SlotsToCoeffs, the raise, CoeffsToSlots and the LUT step,
/// which shares them across the batch and gives each output its contract from a [`BootstrappingRun`] of the input's
/// metadata.
pub(crate) fn ckks_functional_bootstrap_derived<BE, F, K>(
    module: &Module<BE>,
    res: &mut [CKKSCiphertextOwned<BE>],
    a: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContextPrepared<BE, F>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSBootstrappingImpl,
    K: BootstrappingKeys<BE>,
{
    check_functional_entry(module, res, a, ctx, luts, keys, scratch.available())?;
    let run = ctx.run(a.meta());
    let boot_layout = GLWELayout {
        n: res[0].n(),
        base2k: a.base2k(),
        k: res[0].k(),
        rank: Rank(1),
    };
    scratch.scope(|scratch| {
        let (mut z, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, a.meta());
        slots_to_coeffs_and_raise(module, &mut z, a, &run, keys, &mut scratch)?;
        module.ckks_coeffs_to_slots_assign(&mut z, &run, keys.rotation_keys(), &mut scratch)?;
        module.ckks_eval_lut(res, &z, &run, luts, keys, &mut scratch)
    })
}
