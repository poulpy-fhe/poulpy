use crate::{
    CKKSInfos, CKKSLayout, CKKSMeta, CKKSModuleInfos, CKKSResult as Result, CKKSRingKind, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSBootstrappingOps, CKKSImagOps},
    layouts::{
        BootstrappingKeys, CIBootstrappingContext, CIBootstrappingKeys, CIBootstrappingKeysLayout, CKKSCiphertextOwned,
        CKKSModuleAlloc,
    },
};
use poulpy_core::{
    GLWEKeyswitch, GLWENormalize,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
        prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScratchArena, ZnxView, ZnxViewMut},
};

pub(crate) fn ckks_ci_bootstrap_tmp_bytes_reference<BE, CI, F>(
    standard_module: &Module<BE>,
    ci_module: &Module<CI>,
    ct_out: &CKKSCiphertextOwned<CI>,
    ct_in: &CKKSCiphertextOwned<CI>,
    ctx: &CIBootstrappingContext<BE, F>,
    keys_layout: &CIBootstrappingKeysLayout,
) -> usize
where
    BE: Backend,
    CI: Backend<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
    Module<CI>: GLWENormalize<CI>,
    Module<BE>: ModuleN + GLWEKeyswitch<BE> + GLWENormalize<BE> + CKKSAddOps<BE> + CKKSImagOps<BE> + CKKSBootstrappingOps<BE>,
{
    let ctx = &ctx.standard;
    assert_eq!(standard_module.n(), 2 * ci_module.n());
    let standard_base2k = keys_layout.ci_to_standard.base2k;
    let standard_in = CKKSLayout {
        ring_kind: CKKSRingKind::Standard,
        glwe_layout: GLWELayout {
            n: standard_module.n().into(),
            base2k: standard_base2k,
            k: ct_in.k(),
            rank: Rank(1),
        },
        meta: ct_in.meta(),
    };
    let standard_out = CKKSLayout {
        ring_kind: CKKSRingKind::Standard,
        glwe_layout: GLWELayout {
            n: standard_module.n().into(),
            base2k: standard_base2k,
            k: ct_out.k(),
            rank: Rank(1),
        },
        meta: ct_out.meta(),
    };
    let mut returned = standard_out;
    returned.glwe_layout.k = ct_out
        .k()
        .as_usize()
        .saturating_sub(ctx.output_consumed_bits(ct_in.log_delta()))
        .into();
    let max_size = standard_in.size().max(standard_out.size());
    standard_module
        .glwe_keyswitch_tmp_bytes(&standard_in, &standard_in, &keys_layout.ci_to_standard)
        .max(standard_module.ckks_add_tmp_bytes(max_size))
        .max(standard_module.ckks_mul_i_tmp_bytes(max_size))
        .max(standard_module.ckks_div_i_tmp_bytes(max_size))
        .max(standard_module.ckks_bootstrap_tmp_bytes(&standard_out, &standard_in, ctx, &keys_layout.bootstrap_keys))
        .max(standard_module.glwe_keyswitch_tmp_bytes(&returned, &returned, &keys_layout.standard_to_ci))
        .max(ci_module.glwe_normalize_tmp_bytes())
        .max(standard_module.glwe_normalize_tmp_bytes())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_ci_bootstrap_reference<BE, CI, F, K, S>(
    standard_module: &Module<BE>,
    ci_module: &Module<CI>,
    left_out: &mut CKKSCiphertextOwned<CI>,
    right_out: Option<&mut CKKSCiphertextOwned<CI>>,
    left_in: &CKKSCiphertextOwned<CI>,
    right_in: Option<&CKKSCiphertextOwned<CI>>,
    ctx: &CIBootstrappingContext<BE, F>,
    keys: &CIBootstrappingKeys<K, S>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend<ZnxWord = i64>,
    CI: Backend<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    for<'a> CI::BufRef<'a>: HostDataRef,
    for<'a> CI::BufMut<'a>: HostDataMut,
    Module<CI>: GLWENormalize<CI> + CKKSModuleAlloc<CI>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: ModuleN
        + GLWEKeyswitch<BE>
        + GLWENormalize<BE>
        + CKKSAddOps<BE>
        + CKKSImagOps<BE>
        + CKKSBootstrappingOps<BE>
        + CKKSModuleAlloc<BE>,
    K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>> + Sync,
    S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    F: Sync,
{
    validate_ci_bootstrap(ci_module, left_out, right_out.as_deref(), left_in, right_in)?;

    let ctx = &ctx.standard;
    crate::ckks_ensure!(
        ctx.pipeline() != crate::layouts::BootstrappingPipeline::C2SFirst
            || left_in.log_delta() <= ctx.eval_mod().plan.f_mod_log_delta,
        "CI bootstrap input scale exceeds the C2S-first working scale"
    );
    let ring = standard_module.ckks_ring();
    ring.check("CI bootstrap input key", keys.ci_to_standard.key_ring())?;
    ring.check("CI bootstrap output key", keys.standard_to_ci.key_ring())?;
    let standard_base2k = keys.ci_to_standard.base2k();
    crate::ckks_ensure!(
        (1..=BE::MAX_BASE2K).contains(&standard_base2k.as_usize()),
        "invalid CI switching-key radix"
    );
    crate::ckks_ensure!(
        standard_base2k == keys.standard_to_ci.base2k(),
        "CI bootstrap switching-key radices differ"
    );
    crate::layouts::validation::validate_gadget_backend_view(
        "CI-to-standard key",
        keys.ci_to_standard.as_core(),
        &keys.ci_to_standard.to_backend_ref(),
        standard_module.n(),
        standard_base2k,
        left_in.k().as_usize().div_ceil(standard_base2k.as_usize()),
    )?;
    crate::ckks_ensure!(
        keys.ci_to_standard.gglwe_layout().gadget_k() >= left_in.k(),
        "CI-to-standard key does not cover the input width"
    );
    // The return switch runs after the standard pipeline has consumed its budget.
    let return_k = left_out
        .k()
        .as_usize()
        .checked_sub(ctx.output_consumed_bits(left_in.log_delta()))
        .ok_or_else(|| crate::CKKSError::from(anyhow::anyhow!("insufficient CI bootstrap output width")))?;
    let output_k = return_k
        .checked_sub(ctx.output_scale_drop(left_in.log_delta()) + 1)
        .ok_or_else(|| crate::CKKSError::from(anyhow::anyhow!("insufficient CI bootstrap normalization width")))?;
    crate::ckks_ensure!(output_k > left_in.log_delta(), "CI bootstrap output has no message budget");
    crate::layouts::validation::validate_gadget_backend_view(
        "standard-to-CI key",
        keys.standard_to_ci.as_core(),
        &keys.standard_to_ci.to_backend_ref(),
        standard_module.n(),
        standard_base2k,
        return_k.div_ceil(standard_base2k.as_usize()),
    )?;
    crate::ckks_ensure!(
        keys.standard_to_ci.gglwe_layout().gadget_k().as_usize() >= return_k,
        "standard-to-CI key does not cover the return width"
    );
    let standard_in_layout = GLWELayout {
        n: standard_module.n().into(),
        base2k: standard_base2k,
        k: left_in.k(),
        rank: Rank(1),
    };
    let standard_out_layout = GLWELayout {
        n: standard_module.n().into(),
        base2k: standard_base2k,
        k: left_out.k(),
        rank: Rank(1),
    };
    let bootstrap_k = left_out.k();

    let mut packed = standard_module.ckks_ciphertext_alloc_from_glwe_infos(&standard_in_layout);
    packed.set_meta(left_in.meta());

    embed_real_ciphertext(standard_module, &mut packed, left_in, scratch)?;
    if let Some(right_in) = right_in {
        let mut right_packed = standard_module.ckks_ciphertext_alloc_from_glwe_infos(&standard_in_layout);
        embed_real_ciphertext(standard_module, &mut right_packed, right_in, scratch)?;
        standard_module.ckks_mul_i_assign(&mut right_packed, scratch)?;
        standard_module.ckks_add_assign(&mut packed, &right_packed, scratch)?;
        packed.set_slots(SlotsKind::Complex);
    }

    standard_module.glwe_keyswitch_assign(&mut packed, &keys.ci_to_standard.as_core().to_backend_ref(), scratch);

    let mut refreshed = standard_module.ckks_ciphertext_alloc_from_glwe_infos(&standard_out_layout);
    refreshed.set_meta(left_in.meta());
    refreshed.set_k(bootstrap_k);
    standard_module.ckks_bootstrap(&mut refreshed, &packed, ctx, &keys.bootstrap_keys, scratch)?;
    standard_module.glwe_keyswitch_assign(&mut refreshed, &keys.standard_to_ci.as_core().to_backend_ref(), scratch);
    {
        let mut ci_scratch = scratch.borrow().into_backend::<CI>();
        fold_complex_to_real(ci_module, left_out, &refreshed, &mut ci_scratch);
        crate::ckks_set_log_delta_normalized(ci_module, left_out, left_in.log_delta(), &mut ci_scratch);
    }
    if let Some(right_out) = right_out {
        standard_module.ckks_div_i_assign(&mut refreshed, scratch)?;
        let mut ci_scratch = scratch.borrow().into_backend::<CI>();
        fold_complex_to_real(ci_module, right_out, &refreshed, &mut ci_scratch);
        crate::ckks_set_log_delta_normalized(ci_module, right_out, left_in.log_delta(), &mut ci_scratch);
    }
    Ok(())
}

fn embed_real_ciphertext<BE>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE>,
    src: &CKKSCiphertextOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend<ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWENormalize<BE> + CKKSModuleAlloc<BE>,
{
    crate::ckks_ensure!(dst.n().as_usize() == 2 * src.n().as_usize(), "invalid CI-to-standard degrees");
    crate::ckks_ensure!(dst.rank() == src.rank(), "invalid CI-to-standard layout");
    dst.set_meta(CKKSMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_k(src.k());
    if dst.base2k() == src.base2k() {
        unfold_ciphertext::<BE>(dst, src);
        module.glwe_normalize_assign(dst, scratch);
    } else {
        let layout = GLWELayout {
            n: dst.n(),
            base2k: src.base2k(),
            k: src.k(),
            rank: src.rank(),
        };
        let mut unfolded = module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
        unfolded.set_meta(dst.meta());
        unfold_ciphertext::<BE>(&mut unfolded, src);
        module.glwe_normalize(dst, &unfolded, scratch);
    }
    Ok(())
}

fn validate_ci_bootstrap<CI: Backend>(
    ci_module: &Module<CI>,
    left_out: &CKKSCiphertextOwned<CI>,
    right_out: Option<&CKKSCiphertextOwned<CI>>,
    left_in: &CKKSCiphertextOwned<CI>,
    right_in: Option<&CKKSCiphertextOwned<CI>>,
) -> Result<()> {
    let ci_ring = ci_module.ckks_ring();
    for ct in [Some(left_in), Some(left_out), right_in, right_out].into_iter().flatten() {
        ci_ring.check_ciphertext("CI bootstrap", ct)?;
        crate::layouts::validation::validate_storage_capacity("CI bootstrap ciphertext", ct)?;
        crate::ckks_ensure!(
            ct.base2k().as_usize() <= CI::MAX_BASE2K,
            "CI ciphertext radix exceeds the backend limit"
        );
    }
    crate::ckks_ensure!(
        left_in.n().as_usize() == ci_module.n() && left_out.n().as_usize() == ci_module.n(),
        "CI ciphertext degree does not match the CI module"
    );
    crate::ckks_ensure!(
        left_in.rank().as_usize() == 1 && left_out.rank().as_usize() == 1,
        "CI bootstrapping supports rank-1 ciphertexts only"
    );
    crate::ckks_ensure!(
        left_in.log_delta() <= left_in.k().as_usize(),
        "CI input scale exceeds its width"
    );
    crate::ckks_ensure!(
        left_in.log_sparsity() < ci_module.n().ilog2() as usize + 1,
        "invalid CI input sparsity"
    );
    crate::ckks_ensure!(
        right_in.is_some() == right_out.is_some(),
        "CI pair bootstrapping requires both a right input and output"
    );
    if let (Some(right_in), Some(right_out)) = (right_in, right_out) {
        crate::ckks_ensure!(right_in.slots().is_real(), "CI bootstrapping inputs must carry real slots");
        crate::ckks_ensure!(
            right_in.n() == left_in.n()
                && right_out.n() == left_out.n()
                && right_in.rank() == left_in.rank()
                && right_out.rank() == left_out.rank()
                && right_in.base2k() == left_in.base2k()
                && right_out.base2k() == left_out.base2k()
                && right_in.k() == left_in.k()
                && right_out.k() == left_out.k()
                && right_in.meta() == left_in.meta(),
            "CI pair inputs and outputs must have matching layouts and input metadata"
        );
    }
    Ok(())
}

fn unfold_ciphertext<BE: Backend<ZnxWord = i64>>(dst: &mut CKKSCiphertextOwned<BE>, src: &CKKSCiphertextOwned<BE>)
where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
    let mut dst_mut = GLWEToBackendMut::<BE>::to_backend_mut(dst);
    let n = src.n().as_usize();
    for col in 0..=src.rank().as_usize() {
        for limb in 0..dst_mut.data().size() {
            let out = dst_mut.data_mut().at_mut(col, limb);
            out.fill(0);
            if limb >= src_ref.data().size() {
                continue;
            }
            let input = src_ref.data().at(col, limb);
            out[..n].copy_from_slice(input);
            for k in 1..n {
                out[2 * n - k] = input[k].wrapping_neg();
            }
        }
    }
}

fn fold_complex_to_real<BE: Backend<ZnxWord = i64>>(
    ci_module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE>,
    src: &CKKSCiphertextOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWENormalize<BE> + CKKSModuleAlloc<BE>,
{
    dst.set_meta(CKKSMeta {
        slots: SlotsKind::Real,
        log_delta: src.log_delta() + 1,
        ..src.meta()
    });
    dst.set_k(src.k());
    let fold_into = |dst: &mut CKKSCiphertextOwned<BE>| {
        let n = dst.n().as_usize();
        let rank = dst.rank().as_usize();
        let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
        let mut dst_mut = GLWEToBackendMut::<BE>::to_backend_mut(dst);
        for col in 0..=rank {
            for limb in 0..dst_mut.data().size() {
                let out = dst_mut.data_mut().at_mut(col, limb);
                if limb >= src_ref.data().size() {
                    out.fill(0);
                    continue;
                }
                let input = src_ref.data().at(col, limb);
                out[0] = input[0].wrapping_mul(2);
                for k in 1..n {
                    out[k] = input[k].wrapping_sub(input[2 * n - k]);
                }
            }
        }
    };
    if dst.base2k() == src.base2k() {
        fold_into(dst);
        ci_module.glwe_normalize_assign(dst, scratch);
    } else {
        let layout = GLWELayout {
            n: dst.n(),
            base2k: src.base2k(),
            k: src.k(),
            rank: src.rank(),
        };
        let mut folded = ci_module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
        folded.set_meta(dst.meta());
        fold_into(&mut folded);
        ci_module.glwe_normalize(dst, &folded, scratch);
    }
}
