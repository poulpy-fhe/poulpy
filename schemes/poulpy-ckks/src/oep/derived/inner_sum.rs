//! The inner sum and the replicate: poulpy-core's summed trace over a window of levels, forward or reversed, on a copy
//! of the input, then the inner sum's optional division by `ckks_div_pow2_assign`.
use std::ops::Range;

use poulpy_core::{
    GLWETrace,
    layouts::{
        Degree, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, TorusPrecision,
        prepared::GLWEAutomorphismKeyPreparedBackendRef,
    },
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos, SlotsKind,
    api::{CKKSCopyOps, CKKSModuleInfos, CKKSPow2Ops},
    ckks_ensure,
    error::checked_log_budget_sub,
    oep::CKKSInnerSumImpl,
};

pub(crate) fn ckks_inner_sum_galois_elements_derived<BE: CKKSInnerSumImpl>(
    module: &Module<BE>,
    n: Degree,
    log_batch: usize,
    log_count: usize,
) -> Vec<i64> {
    let sum = Sum::inner(log_batch, log_count, false);
    assert!(
        sum.fits(module, n),
        "ckks_inner_sum_galois_elements: the window exceeds the slot count"
    );
    sum.galois_elements(module, n)
}

pub(crate) fn ckks_replicate_galois_elements_derived<BE: CKKSInnerSumImpl>(
    module: &Module<BE>,
    n: Degree,
    log_batch: usize,
    log_count: usize,
) -> Vec<i64> {
    let sum = Sum::replicate(log_batch, log_count);
    assert!(
        sum.fits(module, n),
        "ckks_replicate_galois_elements: the window exceeds the slot count"
    );
    sum.galois_elements(module, n)
}

pub(crate) fn ckks_inner_sum_tmp_bytes_derived<BE: CKKSInnerSumImpl, R, A, K>(
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
        .max(module.glwe_trace_tmp_bytes(res, res, key))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the sum's operands, batch, count and normalization, keys and scratch"
)]
pub(crate) fn ckks_inner_sum_derived<BE: CKKSInnerSumImpl, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    log_batch: usize,
    log_count: usize,
    normalize: bool,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    sum_into(module, res, a, Sum::inner(log_batch, log_count, normalize), keys, scratch)
}

pub(crate) fn ckks_inner_sum_assign_derived<BE: CKKSInnerSumImpl, R, H>(
    module: &Module<BE>,
    res: &mut R,
    log_batch: usize,
    log_count: usize,
    normalize: bool,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    sum_assign(module, res, Sum::inner(log_batch, log_count, normalize), keys, scratch)
}

pub(crate) fn ckks_replicate_derived<BE: CKKSInnerSumImpl, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    log_batch: usize,
    log_count: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    sum_into(module, res, a, Sum::replicate(log_batch, log_count), keys, scratch)
}

pub(crate) fn ckks_replicate_assign_derived<BE: CKKSInnerSumImpl, R, H>(
    module: &Module<BE>,
    res: &mut R,
    log_batch: usize,
    log_count: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    sum_assign(module, res, Sum::replicate(log_batch, log_count), keys, scratch)
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

    /// Whether the `2^log_count` batches of `2^log_batch` slots fit in the slots of a degree-`n` ciphertext.
    fn fits<BE: CKKSInnerSumImpl>(self, module: &Module<BE>, n: Degree) -> bool {
        self.log_batch
            .checked_add(self.log_count)
            .is_some_and(|log| log <= module.ckks_slots_of_degree(n.as_usize()).ilog2() as usize)
    }

    /// The Galois elements of the sum's keys on a degree-`n` ciphertext, in the order the trace looks them up.
    fn galois_elements<BE: CKKSInnerSumImpl>(self, module: &Module<BE>, n: Degree) -> Vec<i64> {
        module.glwe_trace_galois_elements(n, self.levels(), self.reverse)
    }

    /// The trace levels of the sum, once [`Self::fits`] holds.
    fn levels(self) -> Range<usize> {
        self.log_batch + 1..self.log_batch + 1 + self.log_count
    }

    /// The slot rotation of each level, in the order of [`Self::galois_elements`].
    fn rotations(self) -> impl Iterator<Item = i64> {
        (0..self.log_count).map(move |t| {
            if self.reverse {
                -(1i64 << (self.log_batch + self.log_count - 1 - t))
            } else {
                1i64 << (self.log_batch + t)
            }
        })
    }

    /// The checks before any write of a sum of `a` on a degree-`n` result at width `k`, in the documented order: the
    /// window, the division of `a`'s slots, the key of every rotation, then the budget of the copy from `a.k` to `k`
    /// and of the division. Returns the largest scratch `bytes` gives for those keys' layouts, if any key is looked up.
    fn validate<BE: CKKSInnerSumImpl, A, H>(
        self,
        module: &Module<BE>,
        a: &A,
        n: Degree,
        k: TorusPrecision,
        keys: &H,
        bytes: impl Fn(&GLWEAutomorphismKeyPreparedBackendRef<'_, BE>) -> usize,
    ) -> Result<Option<usize>>
    where
        A: CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        let op = self.op;
        if !self.fits(module, n) {
            return Err(CKKSCompositionError::InvalidOperand {
                op,
                reason: format!(
                    "2^{} batches of 2^{} slots exceed the {} slots",
                    self.log_count,
                    self.log_batch,
                    module.ckks_slots_of_degree(n.as_usize())
                ),
            }
            .into());
        }
        let divides = self.normalize && self.log_count > 0;
        if divides && a.slots() == SlotsKind::Integer {
            return Err(CKKSCompositionError::InvalidOperand {
                op,
                reason: format!(
                    "dividing Integer slots by 2^{} need not give integers; relabel them Real first",
                    self.log_count
                ),
            }
            .into());
        }
        let mut most = None;
        for (rotation, element) in self.rotations().zip(self.galois_elements(module, n)) {
            let key = keys
                .get_automorphism_key(element, k)
                .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                    op,
                    rotation,
                    k: k.into(),
                    source,
                })?;
            most = most.max(Some(bytes(&key)));
        }
        let log_budget = checked_log_budget_sub(op, a.log_budget(), a.k().as_usize() - k.as_usize())?;
        if divides {
            checked_log_budget_sub(op, log_budget, self.log_count)?;
        }
        Ok(most)
    }
}

fn sum_into<BE: CKKSInnerSumImpl, R, A, H>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    sum: Sum,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    // The copy narrows the sum to `res`'s width, at which every rotation then switches.
    let k = a.k().min(res.k());
    let bytes = sum.validate(module, a, res.n(), k, keys, |key| {
        ckks_inner_sum_tmp_bytes_derived(module, &*res, a, key)
    })?;
    // A sum that looks up no key has no key layout to size its query with; its constituents check their own.
    if let Some(bytes) = bytes {
        ckks_ensure!(
            scratch.available() >= bytes,
            "{}: scratch below CKKSInnerSumOps::ckks_inner_sum_tmp_bytes",
            sum.op
        );
    }
    module.ckks_copy(res, a, scratch)?;
    trace(module, res, sum, keys, scratch)
}

fn sum_assign<BE: CKKSInnerSumImpl, R, H>(
    module: &Module<BE>,
    res: &mut R,
    sum: Sum,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    let bytes = sum.validate(module, &*res, res.n(), res.k(), keys, |key| {
        ckks_inner_sum_tmp_bytes_derived(module, &*res, &*res, key)
    })?;
    if let Some(bytes) = bytes {
        ckks_ensure!(
            scratch.available() >= bytes,
            "{}: scratch below CKKSInnerSumOps::ckks_inner_sum_tmp_bytes",
            sum.op
        );
    }
    trace(module, res, sum, keys, scratch)
}

/// The summed trace of a validated sum on `res`, then the inner sum's division.
fn trace<BE: CKKSInnerSumImpl, R, H>(
    module: &Module<BE>,
    res: &mut R,
    sum: Sum,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    module.glwe_trace_assign(res, sum.levels(), sum.reverse, false, keys, scratch);
    if sum.normalize {
        module.ckks_div_pow2_assign(res, sum.log_count)?;
    }
    Ok(())
}
