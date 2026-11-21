//! Wide temporaries: work `lambda` bits under the scale that `ckks_round` lands.
use crate::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::CKKSCopyOps,
    layouts::{CKKSCiphertextViewMut, ScratchArenaTakeCKKSCiphertext, replicated::layout_at},
};
use poulpy_core::{GLWEBytesOf, layouts::GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena};

/// A ciphertext shaped as `dst` at width `k`, for a result rounded into `dst`.
pub(crate) fn take_wide<'a, BE: Backend + 'a, D: CKKSCtBounds, Rg: Ring>(
    scratch: &'a mut ScratchArena<'_, BE>,
    dst: &D,
    k: usize,
) -> (CKKSCiphertextViewMut<'a, BE, Rg>, ScratchArena<'a, BE>) {
    let layout = layout_at(dst, k);
    scratch.borrow().take_ckks_ciphertext_scratch(&layout, layout.meta)
}

/// The width of a copy of `a` at a budget of `log_budget` bits, `lambda` bits under its scale.
pub(crate) fn wide_k(a: &impl CKKSCtBounds, lambda: usize, log_budget: usize) -> usize {
    log_budget + a.log_delta() + lambda
}

/// Scratch of [`ckks_wide_copy`] beside `stage`, the scratch of the work on the copy, given its layout.
pub(crate) fn ckks_wide_copy_tmp_bytes<BE: Backend, Rg: Ring, M: CKKSCopyOps<BE, Rg> + GLWEBytesOf<BE> + ?Sized>(
    module: &M,
    a: &impl CKKSCtBounds,
    lambda: usize,
    log_budget: usize,
    stage: impl FnOnce(&CKKSLayout) -> usize,
) -> usize {
    let wide = layout_at(a, wide_k(a, lambda, log_budget));
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(&wide)) + module.ckks_copy_tmp_bytes(&wide, a).max(stage(&wide))
}

/// `a` copied at a budget of `log_budget` bits, relabelled `lambda` bits under its scale, beside the rest of the arena.
pub(crate) fn ckks_wide_copy<'s, BE: Backend + 's, Rg: Ring, M: CKKSCopyOps<BE, Rg> + ?Sized>(
    module: &M,
    a: &(impl GLWEToBackendRef<BE, Rg> + CKKSCtBounds),
    lambda: usize,
    log_budget: usize,
    scratch: &'s mut ScratchArena<'_, BE>,
) -> Result<(CKKSCiphertextViewMut<'s, BE, Rg>, ScratchArena<'s, BE>)> {
    let (mut x, mut rest) = take_wide(scratch, a, wide_k(a, lambda, log_budget));
    x.set_log_budget(log_budget);
    module.ckks_copy(&mut x, a, &mut rest)?;
    x.set_log_delta(a.log_delta() + lambda);
    Ok((x, rest))
}
