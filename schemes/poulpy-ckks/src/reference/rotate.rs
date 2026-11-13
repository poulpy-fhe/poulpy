use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision,
        prepared::GLWEAutomorphismKeyPreparedBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, GLWEPlaintextInfos, SetCKKSInfos,
    api::CKKSRoundOps,
    checked_log_budget_sub, ckks_offset_unary,
    layouts::validation::ensure_scratch,
    reference::mul::{is_replicated, take_wide, with_shaped_bytes},
    shaping::product_depth,
};

pub trait CKKSRotateReference<BE: Backend, Rg: Ring = Standard> {
    fn ckks_rotate_with_key_tmp_bytes_reference<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE, Rg> + GLWEShift<BE, Rg> + CKKSRoundOps<BE, Rg>,
    {
        automorphism_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_rotate_with_key_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE, Rg> + GLWEShift<BE, Rg> + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, Rg> + GLWEInfos + CKKSInfos,
    {
        automorphism_into("ckks_rotate_into", self, dst, src, key, scratch)
    }

    fn ckks_rotate_with_key_assign_reference<Dst>(
        &self,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE, Rg> + GLWEShift<BE, Rg> + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    {
        automorphism_assign("ckks_rotate_assign", self, dst, key, scratch)
    }
}

impl<BE: Backend, Rg: Ring> CKKSRotateReference<BE, Rg> for poulpy_hal::layouts::Module<BE> {}

/// Scratch of [`automorphism_into`] and [`automorphism_assign`].
pub(crate) fn automorphism_tmp_bytes<BE, Rg, M, C, K>(module: &M, ct_infos: &C, key_infos: &K) -> usize
where
    BE: Backend,
    Rg: Ring,
    M: GLWEAutomorphism<BE, Rg> + GLWEShift<BE, Rg> + CKKSRoundOps<BE, Rg> + ?Sized,
    C: CKKSCtBounds,
    K: GGLWEInfos,
{
    let plain = |res: &GLWELayout, _: TorusPrecision| {
        module
            .glwe_automorphism_tmp_bytes(res, res, key_infos)
            .max(module.glwe_shift_tmp_bytes(res.max_size()))
    };
    with_shaped_bytes(module, ct_infos, ct_infos.k(), is_replicated(ct_infos), plain)
}

/// `dst = τ(src · 2^offset)` by the automorphism of `key`, where `offset` aligns `src` on the pre-call `dst`, which
/// then reports `src`'s metadata at the budget left. A replicated layout key-switches `λ` bits under the scale, then
/// rounds into `dst` with shaping. Refusals carry the caller's `op`, before any write.
pub(crate) fn automorphism_into<BE, Rg, M, Dst, Src>(
    op: &'static str,
    module: &M,
    dst: &mut Dst,
    src: &Src,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Rg: Ring,
    M: GLWEAutomorphism<BE, Rg> + GLWEShift<BE, Rg> + CKKSRoundOps<BE, Rg> + ?Sized,
    Dst: GLWEToBackendMut<BE, Rg> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    Src: GLWEToBackendRef<BE, Rg> + GLWEInfos + CKKSInfos,
{
    let offset = ckks_offset_unary(dst, src);
    let log_budget = checked_log_budget_sub(op, src.log_budget(), offset)?;
    if is_replicated(src) {
        let stamped = CKKSLayout {
            glwe_layout: GLWELayout {
                k: (log_budget + src.log_delta()).into(),
                ..dst.glwe_layout()
            },
            meta: src.meta(),
        };
        ensure_scratch(op, scratch, automorphism_tmp_bytes(module, &stamped, key))?;
    }

    // Stamp before the key switch, not after. The key switch normalizes its
    // output at `dst.k()`, so a stale, wider `k` leaves the result carrying
    // key-switch noise below the width it is about to be labelled with, and
    // a convolution consumer rescales by `2^k`, which lifts that noise back
    // to full magnitude. `offset` is computed against the pre-stamp `dst`,
    // so moving the stamp does not change it.
    dst.set_meta(src.meta());
    dst.set_log_budget(log_budget);
    let Some(lambda) = product_depth(dst.slot_layout(), dst.log_delta()) else {
        if offset != 0 {
            module.glwe_lsh(dst, src, offset, scratch);
            module.glwe_automorphism_assign(dst, key, scratch);
        } else {
            module.glwe_automorphism(dst, src, key, scratch);
        }
        return Ok(());
    };
    let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
    wide.set_meta(dst.meta());
    wide.set_log_budget(dst.log_budget());
    wide.set_log_delta(dst.log_delta() + lambda);
    if offset != 0 {
        module.glwe_lsh(&mut wide, src, offset, &mut scratch);
        module.glwe_automorphism_assign(&mut wide, key, &mut scratch);
    } else {
        module.glwe_automorphism(&mut wide, src, key, &mut scratch);
    }
    module.ckks_round(dst, &wide, &mut scratch)
}

/// `dst = τ(dst)` by the automorphism of `key`, rounded with shaping on a replicated layout. Refusals carry the
/// caller's `op`, before any write.
pub(crate) fn automorphism_assign<BE, Rg, M, Dst>(
    op: &'static str,
    module: &M,
    dst: &mut Dst,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Rg: Ring,
    M: GLWEAutomorphism<BE, Rg> + GLWEShift<BE, Rg> + CKKSRoundOps<BE, Rg> + ?Sized,
    Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + GLWEInfos + CKKSInfos + SetCKKSInfos,
{
    let Some(lambda) = product_depth(dst.slot_layout(), dst.log_delta()) else {
        module.glwe_automorphism_assign(dst, key, scratch);
        return Ok(());
    };
    ensure_scratch(op, scratch, automorphism_tmp_bytes(module, &*dst, key))?;
    let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
    wide.set_meta(dst.meta());
    wide.set_log_budget(dst.log_budget());
    wide.set_log_delta(dst.log_delta() + lambda);
    module.glwe_automorphism(&mut wide, &*dst, key, &mut scratch);
    module.ckks_round(dst, &wide, &mut scratch)
}
