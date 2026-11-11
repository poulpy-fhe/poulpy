//! The modulus raise, with the sparse-secret encapsulation around it.

use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWEKeyswitch, GLWEShift, ScratchArenaTakeGLWE,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        LinearTransformation, TorusPrecision,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef},
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult as Result, GLWEPlaintextMeta, Scale, SetCKKSInfos,
    api::CKKSPow2Ops,
    ckks_ensure, ckks_ensure_operand,
    layouts::{
        BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSModRaiseInfos, CKKSRightOperandPrepared,
        bootstrapping::bootstrap_layouts,
        validation::{
            check_first_factor_raise, check_offset_raise, ensure_encapsulation, ensure_mod_up_widens, ensure_one_base2k,
            ensure_one_degree, ensure_rank_one, offset_key_window,
        },
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl},
};

/// Scratch of the raise of `in_layout` to `boot_layout`: the lift and the shift, then under encapsulation, between
/// two switches, or with the first CoeffsToSlots factor, on a few small ciphertexts as wide as its keys switch.
fn raise_tmp_bytes<BE, P>(
    module: &Module<BE>,
    boot_layout: &CKKSLayout,
    in_layout: &CKKSLayout,
    plan: &P,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWEShift<BE> + CKKSPow2Ops<BE>,
    P: CKKSModRaiseInfos<BE>,
{
    let mut bytes = module.glwe_shift_tmp_bytes(boot_layout.size());
    if plan.mod_raise_plan().lift.is_some() {
        bytes = bytes.max(module.ckks_mul_pow2_tmp_bytes(in_layout.size()));
    }
    if let Some((encaps, sparse_to_dense)) = keys_layout
        .encapsulation
        .as_ref()
        .and_then(|encaps| encaps.sparse_to_dense.as_ref().map(|s2d| (encaps, s2d)))
    {
        bytes = bytes.max(BE::ckks_encapsulated_mod_up_tmp_bytes(
            module,
            boot_layout,
            in_layout,
            &encaps.dense_to_sparse,
            sparse_to_dense,
        ));
    }
    if let (Some(factor), Some(key)) = (plan.first_factor(), &keys_layout.first_factor_key) {
        bytes = bytes.max(BE::ckks_mod_up_first_factor_tmp_bytes(
            module,
            boot_layout,
            in_layout,
            factor,
            keys_layout.encapsulation.as_ref().map(|encaps| &encaps.dense_to_sparse),
            key,
        ));
    }
    bytes
}

/// Backend-generic reference of the modulus raise, selected by
/// [`impl_ckks_bootstrapping_reference_full`](crate::oep::impl_ckks_bootstrapping_reference_full).
pub trait CKKSModRaiseReference<BE: Backend>: std::borrow::Borrow<Module<BE>> {
    /// Scratch upper bound for [`Self::ckks_mod_raise_reference`].
    fn ckks_mod_raise_tmp_bytes_reference<R, A, P>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Module<BE>: GLWEShift<BE> + CKKSPow2Ops<BE>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<BE>,
    {
        let module: &Module<BE> = self.borrow();
        let (boot_layout, in_layout) = bootstrap_layouts(res, a);
        raise_tmp_bytes(module, &boot_layout, &in_layout, plan, keys_layout)
    }

    /// Reference body of [`ckks_mod_raise`](crate::api::CKKSBootstrappingOps::ckks_mod_raise).
    fn ckks_mod_raise_reference<R, A, P, K>(
        &self,
        res: &mut R,
        a: &mut A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<BE>,
        K: BootstrappingLinearKeys<BE>,
    {
        let module: &Module<BE> = self.borrow();
        const OP: &str = "ckks_mod_raise";
        crate::layouts::validation::ensure_full_layout(OP, crate::GLWEPlaintextInfos::slot_layout(a))?;
        let raise = plan.mod_raise_plan();
        ensure_rank_one(OP, a, res)?;
        ensure_one_degree(OP, module.n(), a, res)?;
        ensure_one_base2k(OP, a, res)?;
        ensure_encapsulation(OP, keys, raise.sparse_secret_hamming_weight)?;
        ckks_ensure_operand!(
            OP,
            !raise.first_factor_on_raise || plan.first_factor().is_some(),
            "the plan moves the first CoeffsToSlots factor onto the raise, and no prepared factor comes with it"
        );
        // C2S-first derives both lifts from EvalMod, and the first one writes `a`: the checks run on what it will leave.
        let (lift_bits, scale_up) = match &raise.lift {
            Some(lift) => {
                let lift_bits = lift.input_scale_up(a.k().as_usize(), a.log_delta())?;
                (lift_bits, lift.raised_scale_up(a.log_delta() + lift_bits)?)
            }
            None => (0, raise.c2s_guard_bits),
        };
        check_raise_widths(res, a, scale_up, plan.first_factor(), keys)?;
        if let Some(keys_layout) = keys.keys_layout() {
            let bytes = module.ckks_mod_raise_tmp_bytes_reference(res, a, plan, keys_layout);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        let (slots, log_sparsity) = (a.slots(), a.log_sparsity());
        let log_modulus_in = a.k().as_usize();
        mod_raise_body(module, res, a, lift_bits, scale_up, plan.first_factor(), keys, scratch)?;
        let scale = match &raise.lift {
            Some(lift) => res.log_delta() + lift.log_msg_ratio,
            None => log_modulus_in + raise.c2s_guard_bits,
        };
        res.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(scale),
            slots,
            log_sparsity,
            slot_layout: crate::SlotLayout::Full,
        });
        Ok(())
    }
}

impl<BE: Backend> CKKSModRaiseReference<BE> for Module<BE> {}

/// `scale_up` lifts the message that many bits above its natural magnitude,
/// fused into the widening shift rather than costing a second pass.
fn ckks_mod_up_scaled_reference<BE, Dst, Src>(
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
    let k_large: usize = dst.k().as_usize();
    let k_small: usize = src.k().as_usize();

    ensure_mod_up_widens("ckks_mod_raise", k_large, k_small, scale_up)?;

    // The shift to the natural magnitude is the raise: `q = 2^k_small` becomes an explicit multiple `I(X)·q`. Stopping
    // `scale_up` bits short of it is the fused lift.
    module.glwe_copy(dst, src, scratch);
    module.glwe_rsh_assign(k_large - k_small - scale_up, dst, scratch);

    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
        slot_layout: src.slot_layout(),
    });
    dst.set_k(k_large.into());

    Ok(())
}

/// What [`CKKSModRaiseReference::ckks_mod_raise_reference`] refuses on widths: `src`, lifted by `scale_up` fused
/// into ModUp, into `dst`, under the encapsulation keys and the first-factor keys of `keys`.
fn check_raise_widths<BE, Dst, Src, K>(
    dst: &Dst,
    src: &Src,
    scale_up: usize,
    first_factor: Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>,
    keys: &K,
) -> Result<()>
where
    BE: Backend,
    K: BootstrappingLinearKeys<BE>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
{
    const OP: &str = "ckks_mod_raise";
    let (k_large, k_small) = (dst.k().as_usize(), src.k().as_usize());
    if let Some(factor) = first_factor {
        if let Some((dense_to_sparse, _)) = keys.encapsulation_keys() {
            ensure_encapsulation_key_covers(src.k(), dense_to_sparse)?;
        }
        ensure_mod_up_widens(OP, k_large, k_small, scale_up)?;
        check_first_factor_raise(
            OP,
            dst.base2k().as_usize(),
            dst.n().as_usize().ilog2() as usize,
            k_large,
            k_small + scale_up,
            factor,
            keys,
        )?;
        return Ok(());
    }
    match keys.encapsulation_keys() {
        Some((dense_to_sparse, Some(sparse_to_dense))) => {
            check_encapsulated_raise(dst, src, scale_up, dense_to_sparse, sparse_to_dense)
        }
        Some((_, None)) => Err(CKKSCompositionError::InvalidOperand {
            op: OP,
            reason:
                "the encapsulated ModUp needs a sparse-to-dense key, which a key set for the first factor on the raise has not"
                    .into(),
        }
        .into()),
        None => ensure_mod_up_widens(OP, k_large, k_small, scale_up),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the raise takes its lift, its scale and the first factor as separate operands"
)]
fn mod_raise_body<BE, Dst, Src, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    lift_bits: usize,
    scale_up: usize,
    first_factor: Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>,
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
    if lift_bits != 0 {
        let log_delta: usize = src.log_delta();
        module.ckks_mul_pow2_assign(lift_bits, src, scratch)?;
        // Not `set_log_delta`: it preserves `log_budget` and grows `k`, which
        // cancels the shift. The lift spends budget at a fixed modulus.
        let mut meta = src.meta();
        meta.scale = Scale::Log(log_delta + lift_bits);
        src.set_meta(meta);
    }

    // A first CoeffsToSlots factor left to the raise is applied here, on the
    // raised ciphertext while it is small; its switches replace sparse-to-dense.
    if let Some(factor) = first_factor {
        return BE::ckks_mod_up_first_factor(module, dst, src, scale_up, factor, keys, scratch);
    }

    match keys.encapsulation_keys() {
        Some((dense_to_sparse, Some(sparse_to_dense))) => BE::ckks_encapsulated_mod_up(
            module,
            dst,
            src,
            scale_up,
            &dense_to_sparse.to_backend_ref(),
            &sparse_to_dense.to_backend_ref(),
            scratch,
        ),
        _ => ckks_mod_up_scaled_reference(module, dst, src, scale_up, scratch),
    }
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
    // The switch keeps the width of `src`: the raise's checks run before it.
    check_encapsulated_raise(dst, src, scale_up, dense_to_sparse, sparse_to_dense)?;
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
    let (shift, window) = check_offset_raise("ckks_mod_raise", k_large, k_small, scale_up, sparse_to_dense)?;
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
            module.glwe_rsh_assign(shift - k_offset, &mut lowered, &mut scratch_inner);
            module.glwe_keyswitch(dst, &lowered, &key, &mut scratch_inner);
        });
    }
    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
        slot_layout: src.slot_layout(),
    });
    Ok(())
}

/// `InvalidOperand` unless the encapsulation key `key` takes an input of `k` bits.
fn ensure_encapsulation_key_covers<K: GGLWEInfos>(k: TorusPrecision, key: &K) -> Result<()> {
    ckks_ensure_operand!(
        "ckks_mod_raise",
        k <= key.gglwe_layout().max_input_k(),
        "the input is wider than an encapsulation key's max_input_k"
    );
    Ok(())
}

/// Refuses encapsulation keys too narrow for their inputs before any write: `src`
/// for the dense-to-sparse switch, `dst` for an offset-free sparse-to-dense one, and
/// the raise of `src` by `scale_up` into `dst`, a sparse-to-dense key with a row
/// offset being checked against that raise, by [`check_offset_raise`].
fn check_encapsulated_raise<Dst: CKKSCtBounds, Src: CKKSCtBounds, D2S: GGLWEInfos, S2D: GGLWEInfos>(
    dst: &Dst,
    src: &Src,
    scale_up: usize,
    dense_to_sparse: &D2S,
    sparse_to_dense: &S2D,
) -> Result<()> {
    const OP: &str = "ckks_mod_raise";
    ensure_encapsulation_key_covers(src.k(), dense_to_sparse)?;
    let (k_large, k_small) = (dst.k().as_usize(), src.k().as_usize());
    if sparse_to_dense.k_offset().as_usize() > 0 {
        check_offset_raise(OP, k_large, k_small, scale_up, sparse_to_dense)?;
    } else {
        ensure_encapsulation_key_covers(dst.k(), sparse_to_dense)?;
        ensure_mod_up_widens(OP, k_large, k_small, scale_up)?;
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

/// Reference bodies of the required [`CKKSEncapsulatedModUpImpl`] methods, which
/// [`impl_ckks_encapsulated_mod_up_reference_full!`](crate::impl_ckks_encapsulated_mod_up_reference_full) forwards to.
pub trait CKKSEncapsulatedModUpReference<BE: Backend> {
    fn ckks_encapsulated_mod_up_tmp_bytes_reference<Dst, Src, D2S, S2D>(
        &self,
        dst_infos: &Dst,
        src_infos: &Src,
        dense_to_sparse_infos: &D2S,
        sparse_to_dense_infos: &S2D,
    ) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWEKeyswitch<BE>,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        D2S: GGLWEInfos,
        S2D: GGLWEInfos;

    fn ckks_encapsulated_mod_up_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

impl<BE: Backend + CKKSEncapsulatedModUpImpl> CKKSEncapsulatedModUpReference<BE> for Module<BE> {
    fn ckks_encapsulated_mod_up_tmp_bytes_reference<Dst, Src, D2S, S2D>(
        &self,
        dst_infos: &Dst,
        src_infos: &Src,
        dense_to_sparse_infos: &D2S,
        sparse_to_dense_infos: &S2D,
    ) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWEKeyswitch<BE>,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        D2S: GGLWEInfos,
        S2D: GGLWEInfos,
    {
        ckks_encapsulated_mod_up_tmp_bytes_reference(self, dst_infos, src_infos, dense_to_sparse_infos, sparse_to_dense_infos)
    }

    fn ckks_encapsulated_mod_up_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        ckks_encapsulated_mod_up_reference(self, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch)
    }
}
