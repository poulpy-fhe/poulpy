use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision,
        prepared::GLWEAutomorphismKeyPreparedBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextInfos, SetCKKSInfos,
    api::CKKSRoundOps,
    checked_log_budget_sub, ckks_offset_unary,
    reference::mul::{is_replicated, product_depth, take_wide, with_shaped_bytes},
};

pub trait CKKSRotateReference<BE: Backend> {
    fn ckks_rotate_tmp_bytes_reference<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE>,
    {
        automorphism_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_rotate_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        let offset = ckks_offset_unary(dst, src);
        // Validate before mutating: on error `dst` must remain untouched.
        let log_budget = checked_log_budget_sub("rotate", src.log_budget(), offset)?;

        // Stamp before the key switch, not after. The key switch normalizes its
        // output at `dst.k()`, so a stale, wider `k` leaves the result carrying
        // key-switch noise below the width it is about to be labelled with, and
        // a convolution consumer rescales by `2^k`, which lifts that noise back
        // to full magnitude. `offset` is computed against the pre-stamp `dst`,
        // so moving the stamp does not change it.
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        automorphism_into(self, dst, src, key, offset, scratch)
    }

    fn ckks_rotate_assign_reference<Dst>(
        &self,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    {
        automorphism_assign(self, dst, key, scratch)
    }
}

impl<BE: Backend> CKKSRotateReference<BE> for poulpy_hal::layouts::Module<BE> {}

/// Input width consumed by the automorphism, including a shifted shaping temporary.
pub(crate) fn automorphism_input_k<Dst: CKKSInfos, Src: CKKSInfos>(dst: &Dst, src: &Src) -> TorusPrecision {
    if ckks_offset_unary(dst, src) != 0 {
        let lambda = product_depth(src.slot_layout(), src.log_delta()).unwrap_or(0);
        (dst.k().as_usize() + lambda).into()
    } else {
        src.k()
    }
}

/// Scratch of [`automorphism_into`] and [`automorphism_assign`].
pub(crate) fn automorphism_tmp_bytes<BE, M, C, K>(module: &M, ct_infos: &C, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE> + ?Sized,
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

/// `dst = τ(src · 2^offset)` by the automorphism of `key`, `dst` already
/// labelled. A replicated layout key-switches `λ` bits under the scale, then
/// rounds into `dst` with shaping.
pub(crate) fn automorphism_into<BE, M, Dst, Src>(
    module: &M,
    dst: &mut Dst,
    src: &Src,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    offset: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
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
    module.ckks_round_into(dst, &wide, &mut scratch)
}

/// `dst = τ(dst)` by the automorphism of `key`, rounded with shaping on a replicated layout.
pub(crate) fn automorphism_assign<BE, M, Dst>(
    module: &M,
    dst: &mut Dst,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: GLWEAutomorphism<BE> + CKKSRoundOps<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
{
    let Some(lambda) = product_depth(dst.slot_layout(), dst.log_delta()) else {
        module.glwe_automorphism_assign(dst, key, scratch);
        return Ok(());
    };
    let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
    wide.set_meta(dst.meta());
    wide.set_log_budget(dst.log_budget());
    wide.set_log_delta(dst.log_delta() + lambda);
    module.glwe_automorphism(&mut wide, &*dst, key, &mut scratch);
    module.ckks_round_into(dst, &wide, &mut scratch)
}
