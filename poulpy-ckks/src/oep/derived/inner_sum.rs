//! The inner sum and the replicate: poulpy-core's summed trace over a window of levels, forward or reversed, on a copy
//! of the input, then the inner sum's optional division by `ckks_div_pow2_assign`.
use std::ops::Range;

use poulpy_core::{
    GLWETrace,
    layouts::{
        GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, TorusPrecision,
        prepared::GLWEAutomorphismKeyPreparedBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos, SlotsKind,
    api::{CKKSCopyOps, CKKSModuleInfos, CKKSPow2Ops},
    error::checked_log_budget_sub,
    oep::CKKSInnerSumImpl,
};

pub(crate) fn ckks_inner_sum_assign_tmp_bytes_derived<BE: CKKSInnerSumImpl, R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
where
    R: CKKSCtBounds,
    K: GGLWEInfos,
{
    module.glwe_trace_assign_tmp_bytes(res, key)
}

pub(crate) fn ckks_inner_sum_into_tmp_bytes_derived<BE: CKKSInnerSumImpl, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    key: &K,
) -> usize
where
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    K: GGLWEInfos,
{
    module
        .ckks_copy_tmp_bytes(res, a)
        .max(ckks_inner_sum_assign_tmp_bytes_derived(module, res, key))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the sum's operands, batch, count and normalization, keys and scratch"
)]
pub(crate) fn ckks_inner_sum_into_derived<BE: CKKSInnerSumImpl, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    log_batch: usize,
    log_count: usize,
    normalize: bool,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    let sum = Sum::inner(log_batch, log_count, normalize);
    sum_into(module, dst, src, sum, keys, scratch)
}

pub(crate) fn ckks_inner_sum_assign_derived<BE: CKKSInnerSumImpl, Dst, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    log_batch: usize,
    log_count: usize,
    normalize: bool,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    sum_assign(module, dst, Sum::inner(log_batch, log_count, normalize), keys, scratch)
}

pub(crate) fn ckks_replicate_into_derived<BE: CKKSInnerSumImpl, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    log_batch: usize,
    log_count: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    sum_into(module, dst, src, Sum::replicate(log_batch, log_count), keys, scratch)
}

pub(crate) fn ckks_replicate_assign_derived<BE: CKKSInnerSumImpl, Dst, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    log_batch: usize,
    log_count: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    sum_assign(module, dst, Sum::replicate(log_batch, log_count), keys, scratch)
}

/// An inner sum (`reverse` clear) or a replicate of `2^log_count` batches of `2^log_batch` slots.
#[derive(Clone, Copy)]
struct Sum {
    op: &'static str,
    log_batch: usize,
    log_count: usize,
    reverse: bool,
    normalize: bool,
}

impl Sum {
    fn inner(log_batch: usize, log_count: usize, normalize: bool) -> Self {
        Self {
            op: "inner_sum",
            log_batch,
            log_count,
            reverse: false,
            normalize,
        }
    }

    fn replicate(log_batch: usize, log_count: usize) -> Self {
        Self {
            op: "replicate",
            log_batch,
            log_count,
            reverse: true,
            normalize: false,
        }
    }

    /// The trace levels of the sum, once [`Self::validate`] accepted it.
    fn levels(self) -> Range<usize> {
        self.log_batch + 1..self.log_batch + 1 + self.log_count
    }

    /// The checks before any write: the window within the slots, the division of `slots` at `log_budget`, and the key
    /// of every rotation at width `k`. Returns the largest scratch `bytes` gives for those keys' layouts, if any.
    fn validate<BE: Backend, H>(
        self,
        module: &Module<BE>,
        slots: SlotsKind,
        log_budget: usize,
        k: TorusPrecision,
        keys: &H,
        bytes: impl Fn(&GLWEAutomorphismKeyPreparedBackendRef<'_, BE>) -> usize,
    ) -> Result<Option<usize>>
    where
        H: GetAutomorphismKey<BE>,
    {
        let op = self.op;
        let max_slots = module.ckks_max_slots();
        if self
            .log_batch
            .checked_add(self.log_count)
            .is_none_or(|log| log > max_slots.ilog2() as usize)
        {
            return Err(CKKSCompositionError::InvalidOperand {
                op,
                reason: format!(
                    "2^{} batches of 2^{} slots exceed the {max_slots} slots",
                    self.log_count, self.log_batch
                ),
            }
            .into());
        }
        if self.normalize && self.log_count > 0 {
            if slots == SlotsKind::Integer {
                return Err(CKKSCompositionError::InvalidOperand {
                    op,
                    reason: format!(
                        "dividing Integer slots by 2^{} need not give integers; relabel them Real first",
                        self.log_count
                    ),
                }
                .into());
            }
            checked_log_budget_sub(op, log_budget, self.log_count)?;
        }
        let mut most = None;
        for i in 0..self.log_count {
            let rotation = (1i64 << (self.log_batch + i)) * if self.reverse { -1 } else { 1 };
            let key = keys
                .get_automorphism_key(module.ckks_galois_element(rotation), k)
                .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                    op,
                    rotation,
                    k: k.into(),
                    source,
                })?;
            most = most.max(Some(bytes(&key)));
        }
        Ok(most)
    }
}

fn sum_into<BE: CKKSInnerSumImpl, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    sum: Sum,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    // The copy narrows the sum to `dst`'s width, at which every rotation then switches.
    let k = src.k().min(dst.k());
    let log_budget = checked_log_budget_sub(sum.op, src.log_budget(), src.k().as_usize() - k.as_usize())?;
    let bytes = sum.validate(module, src.slots(), log_budget, k, keys, |key| {
        ckks_inner_sum_into_tmp_bytes_derived(module, &*dst, src, key)
    })?;
    // A sum that looks up no key has no key layout to size its query with; its constituents check their own.
    if let Some(bytes) = bytes {
        assert!(
            scratch.available() >= bytes,
            "{}: scratch below its _into_tmp_bytes query",
            sum.op
        );
    }
    module.ckks_copy(dst, src, scratch)?;
    trace(module, dst, sum, keys, scratch)
}

fn sum_assign<BE: CKKSInnerSumImpl, Dst, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    sum: Sum,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    let (slots, log_budget, k) = (dst.slots(), dst.log_budget(), dst.k());
    let bytes = sum.validate(module, slots, log_budget, k, keys, |key| {
        ckks_inner_sum_assign_tmp_bytes_derived(module, &*dst, key)
    })?;
    if let Some(bytes) = bytes {
        assert!(
            scratch.available() >= bytes,
            "{}: scratch below its _assign_tmp_bytes query",
            sum.op
        );
    }
    trace(module, dst, sum, keys, scratch)
}

/// The summed trace of a validated sum on `dst`, then the inner sum's division.
fn trace<BE: CKKSInnerSumImpl, Dst, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    sum: Sum,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    module.glwe_trace_assign(dst, sum.levels(), sum.reverse, false, keys, scratch);
    if sum.normalize {
        module.ckks_div_pow2_assign(dst, sum.log_count)?;
    }
    Ok(())
}
