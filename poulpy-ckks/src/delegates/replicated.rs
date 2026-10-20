use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout,
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

/// The working layout of a conversion of `ct`: `extra_bits` under its scale and
/// room for the trace's division by `N/(2k)`.
fn working_layout<BE: Backend, C: GLWEInfos>(ct: &C, conv: &ReplicatedConversion<BE>) -> GLWELayout {
    GLWELayout {
        k: (ct.k().as_usize() + conv.extra_bits + conv.trace.len()).into(),
        ..ct.glwe_layout()
    }
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
        let layout = working_layout(ct_infos, conv);
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
        let layout = working_layout(src, conv);
        let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&layout, src.meta());
        let (mut acc, scratch) = scratch.take_ckks_ciphertext_scratch(&layout, src.meta());
        let (mut rotated, scratch) = scratch.take_ckks_ciphertext_scratch(&layout, src.meta());
        let (mut product, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout, src.meta());

        self.ckks_copy(&mut x, src, &mut scratch)?;
        x.set_log_delta(src.log_delta() + conv.extra_bits);
        apply_diagonals(self, &mut acc, &x, &conv.to, keys, &mut rotated, &mut product, &mut scratch)?;
        acc.set_slot_layout(SlotLayout::Replicated(conv.layout));
        dst.set_k((acc.log_budget() + conv.log_delta).into());
        self.ckks_round_into(dst, &acc, &mut scratch)
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
        let layout = working_layout(src, conv);
        let (mut x, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&layout, src.meta());
        let (mut acc, scratch) = scratch.take_ckks_ciphertext_scratch(&layout, src.meta());
        let (mut rotated, scratch) = scratch.take_ckks_ciphertext_scratch(&layout, src.meta());
        let (mut product, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout, src.meta());

        self.ckks_copy(&mut x, src, &mut scratch)?;
        x.set_log_delta(src.log_delta() + conv.extra_bits);
        x.set_slot_layout(SlotLayout::Full);
        apply_diagonals(self, &mut acc, &x, &conv.from, keys, &mut rotated, &mut product, &mut scratch)?;
        // Dividing by `h` lifts the modulus: the trace of the lifted multiples
        // of the modulus lies in `h·R`, so they vanish on the torus.
        self.ckks_div_pow2_assign(&mut acc, conv.trace.len())?;
        for &shift in &conv.trace {
            self.ckks_rotate_into(&mut rotated, &acc, shift, keys, &mut scratch)?;
            self.ckks_add_assign(&mut acc, &rotated, &mut scratch)?;
        }
        // The trace onto period `k` leaves a polynomial in `X^(N/(2k))`.
        acc.set_log_sparsity(conv.trace.len());
        self.ckks_copy(dst, &acc, &mut scratch)
    }
}
