use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSModuleInfos, ReplicatedLayout, SetCKKSInfos,
    SetGLWEPlaintextInfos, SlotLayout,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSPow2Ops, CKKSReplicatedOps, CKKSRotateOps, CKKSRoundOps},
    layouts::{CKKSCiphertextViewMut, CKKSPlaintextOwned, ReplicatedConversion, ScratchArenaTakeCKKS},
};

fn invalid(op: &'static str, reason: &str) -> crate::CKKSError {
    CKKSCompositionError::InvalidOperand {
        op,
        reason: reason.into(),
    }
    .into()
}

/// The working layout of a stage on `ct`: `extra_bits` under its scale and
/// room for a trace's division by `2^trace_steps`.
fn working_layout<C: GLWEInfos>(ct: &C, extra_bits: usize, trace_steps: usize) -> GLWELayout {
    GLWELayout {
        k: (ct.k().as_usize() + extra_bits + trace_steps).into(),
        ..ct.glwe_layout()
    }
}

/// Wide stage onto a replicated layout: `dst = Σ_o D_o ⊙ rot_o(src)`
/// computed `extra_bits` under `src`'s scale, then rounded with `layout`'s
/// shaping to `log_delta`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wide_onto_replicated<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    diagonals: &[(i64, CKKSPlaintextOwned<BE>)],
    layout: ReplicatedLayout,
    log_delta: usize,
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSRoundOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    let working = working_layout(src, extra_bits, 0);
    let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut acc, scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut rotated, scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut product, mut scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());

    module.ckks_copy(&mut x, src, &mut scratch)?;
    x.set_log_delta(src.log_delta() + extra_bits);
    x.set_slot_layout(SlotLayout::Full);
    apply_diagonals(
        module,
        &mut acc,
        &x,
        diagonals,
        keys,
        &mut rotated,
        &mut product,
        &mut scratch,
    )?;
    acc.set_slot_layout(SlotLayout::Replicated(layout));
    dst.set_k((acc.log_budget() + log_delta).into());
    module.ckks_round_into(dst, &acc, &mut scratch)
}

/// Wide stage gathering onto period `k`: `dst = T(Σ_o D_o ⊙ rot_o(src))`
/// computed `extra_bits` under `src`'s scale, where the normalized trace `T`
/// sums the rotations by `trace` after dividing by `2^trace.len()`, which lifts
/// the modulus so that its multiples vanish. Rounds plainly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wide_gather_trace<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    diagonals: &[(i64, CKKSPlaintextOwned<BE>)],
    trace: &[i64],
    extra_bits: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    let working = working_layout(src, extra_bits, trace.len());
    let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut acc, scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut rotated, scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());
    let (mut product, mut scratch) = scratch.take_ckks_ciphertext_scratch(&working, src.meta());

    module.ckks_copy(&mut x, src, &mut scratch)?;
    x.set_log_delta(src.log_delta() + extra_bits);
    x.set_slot_layout(SlotLayout::Full);
    apply_diagonals(
        module,
        &mut acc,
        &x,
        diagonals,
        keys,
        &mut rotated,
        &mut product,
        &mut scratch,
    )?;
    normalized_trace(module, &mut acc, trace, keys, &mut rotated, &mut scratch)?;
    module.ckks_copy(dst, &acc, &mut scratch)
}

/// `ct = (1/2^L) Σ_(i < 2^L) rot_(i·step)(ct)` by `L` rotations by `step, 2·step, ...`.
/// The division comes first and lifts the modulus: the trace of the modulus'
/// multiples lies in `2^L·R`, so they vanish on the torus. A trace closing the
/// slot cycle, `step·2^L = N/2`, leaves a polynomial in `X^(2^L)`.
pub(crate) fn normalized_trace<BE, H>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertextViewMut<'_, BE>,
    shifts: &[i64],
    keys: &H,
    rotated: &mut CKKSCiphertextViewMut<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSPow2Ops<BE>,
    H: GetAutomorphismKey<BE>,
{
    module.ckks_div_pow2_assign(ct, shifts.len())?;
    for &shift in shifts {
        module.ckks_rotate_into(rotated, &*ct, shift, keys, scratch)?;
        module.ckks_add_assign(ct, &*rotated, scratch)?;
    }
    if shifts
        .first()
        .is_some_and(|&step| (step as usize) << shifts.len() == module.ckks_max_slots())
    {
        ct.set_log_sparsity(shifts.len());
    }
    Ok(())
}

/// `acc = Σ_o D_o ⊙ rot_o(x)` over `diagonals`.
#[allow(clippy::too_many_arguments)]
fn apply_diagonals<BE, H>(
    module: &Module<BE>,
    acc: &mut CKKSCiphertextViewMut<'_, BE>,
    x: &CKKSCiphertextViewMut<'_, BE>,
    diagonals: &[(i64, CKKSPlaintextOwned<BE>)],
    keys: &H,
    rotated: &mut CKKSCiphertextViewMut<'_, BE>,
    product: &mut CKKSCiphertextViewMut<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE>,
    H: GetAutomorphismKey<BE>,
{
    for (index, (offset, diagonal)) in diagonals.iter().enumerate() {
        if *offset == 0 {
            accumulate(module, acc, product, x, diagonal, index == 0, scratch)?;
        } else {
            module.ckks_rotate_into(rotated, x, *offset, keys, scratch)?;
            accumulate(module, acc, product, &*rotated, diagonal, index == 0, scratch)?;
        }
    }
    Ok(())
}

/// `acc = read ⊙ diagonal` for the first diagonal, `acc += read ⊙ diagonal` after.
fn accumulate<BE, R>(
    module: &Module<BE>,
    acc: &mut CKKSCiphertextViewMut<'_, BE>,
    product: &mut CKKSCiphertextViewMut<'_, BE>,
    read: &R,
    diagonal: &CKKSPlaintextOwned<BE>,
    first: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSMulOps<BE> + CKKSAddOps<BE>,
    R: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    if first {
        return module.ckks_mul_pt_vec_into(acc, read, diagonal, scratch);
    }
    module.ckks_mul_pt_vec_into(product, read, diagonal, scratch)?;
    module.ckks_add_assign(acc, product, scratch)
}

impl<BE: Backend> CKKSReplicatedOps<BE> for Module<BE>
where
    Module<BE>: CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + CKKSRoundOps<BE>,
{
    fn ckks_replicated_conversion_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K, conv: &ReplicatedConversion<BE>) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        let layout = working_layout(ct_infos, conv.extra_bits, conv.trace.len());
        let wide = CKKSLayout {
            glwe_layout: layout,
            meta: ct_infos.meta(),
        };
        let ct_bytes = BE::scratch_aligned(BE::bytes_of_vec_znx(
            layout.n.as_usize(),
            layout.rank.as_usize() + 1,
            layout.size(),
        ));
        let product = conv
            .to
            .iter()
            .chain(&conv.from)
            .map(|(_, pt)| self.ckks_mul_pt_vec_tmp_bytes(&wide, &wide, pt))
            .max()
            .unwrap_or(0);
        let ops = product
            .max(self.ckks_rotate_tmp_bytes(&wide, key_infos))
            .max(self.ckks_add_tmp_bytes(layout.size()))
            .max(self.ckks_copy_tmp_bytes(&wide, ct_infos))
            .max(self.ckks_round_tmp_bytes(ct_infos, &wide))
            .max(self.ckks_div_pow2_tmp_bytes(layout.size()));
        4 * ct_bytes + ops
    }

    fn ckks_to_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        const OP: &str = "ckks_to_replicated_into";
        if src.slot_layout() != SlotLayout::Full {
            return Err(invalid(OP, "the source must have the full slot layout"));
        }
        if conv.log_delta > src.log_delta() + conv.extra_bits {
            return Err(invalid(OP, "the replicated scale exceeds the source's working scale"));
        }
        wide_onto_replicated(
            self,
            dst,
            src,
            &conv.to,
            conv.layout,
            conv.log_delta,
            conv.extra_bits,
            keys,
            scratch,
        )
    }

    fn ckks_from_replicated_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        const OP: &str = "ckks_from_replicated_into";
        if src.slot_layout() != SlotLayout::Replicated(conv.layout) {
            return Err(invalid(OP, "the source must have the conversion's replicated layout"));
        }
        if conv.from.is_empty() {
            return Err(invalid(OP, "the gather needs the period to divide the batch size"));
        }
        wide_gather_trace(self, dst, src, &conv.from, &conv.trace, conv.extra_bits, keys, scratch)
    }
}
