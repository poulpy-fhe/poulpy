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
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos, SlotsKind,
    api::{CKKSCopyOps, CKKSModuleInfos, CKKSPow2Ops},
    ckks_ensure,
    error::checked_log_budget_sub,
};

/// Reference bodies of [`CKKSInnerSumImpl`](crate::oep::CKKSInnerSumImpl), forwarded by
/// `impl_ckks_inner_sum_reference_full!`.
pub trait CKKSInnerSumReference<BE: Backend> {
    /// poulpy-core's `glwe_trace_galois_elements` at degree `n` over the levels of the inner sum.
    ///
    /// # Panics
    ///
    /// When the window exceeds the slots of degree `n`.
    fn ckks_inner_sum_galois_elements_reference(&self, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64>
    where
        Self: CKKSModuleInfos + GLWETrace<BE>,
    {
        let sum = Sum::inner(log_batch, log_count, false);
        assert!(
            sum.fits(self, n),
            "ckks_inner_sum_galois_elements: the window exceeds the slot count"
        );
        sum.galois_elements(self, n)
    }

    /// poulpy-core's `glwe_trace_galois_elements` at degree `n` over the levels of the replicate, reversed.
    ///
    /// # Panics
    ///
    /// When the window exceeds the slots of degree `n`.
    fn ckks_replicate_galois_elements_reference(&self, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64>
    where
        Self: CKKSModuleInfos + GLWETrace<BE>,
    {
        let sum = Sum::replicate(log_batch, log_count);
        assert!(
            sum.fits(self, n),
            "ckks_replicate_galois_elements: the window exceeds the slot count"
        );
        sum.galois_elements(self, n)
    }

    /// The larger of the selected `ckks_copy_tmp_bytes` from `a` to `res` and poulpy-core's
    /// `glwe_trace_assign_tmp_bytes` of `res` under `key`.
    fn ckks_inner_sum_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        Self: CKKSCopyOps<BE> + GLWETrace<BE>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        sum_tmp_bytes(self, res, a, key)
    }

    /// Refuses, before any write and in this order, a window exceeding the slots of `res`, a mean of `Integer` slots,
    /// a missing key of a rotation at the width `min(a.k(), res.k())` (`MissingAutomorphismKey`), the budget of the
    /// copy into `res` and of the mean's division (`InsufficientHomomorphicCapacity`), then an arena below
    /// [`Self::ckks_inner_sum_tmp_bytes_reference`] at the widest key layout. Then the selected `ckks_copy` of `a` into
    /// `res`, poulpy-core's summed `glwe_trace_assign` over the levels of the window, and, for a mean, the selected
    /// `ckks_div_pow2_assign`. Refusals carry `op: "ckks_inner_sum"`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the sum's operands, batch, count and normalization, keys and scratch"
    )]
    fn ckks_inner_sum_reference<R, A, H>(
        &self,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        res: &mut R,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSModuleInfos + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + GLWETrace<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        sum_into(self, res, a, Sum::inner(log_batch, log_count, normalize), keys, scratch)
    }

    /// [`Self::ckks_inner_sum_reference`] in place, the keys looked up at `res.k()` and no copy.
    fn ckks_inner_sum_assign_reference<R, H>(
        &self,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        res: &mut R,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSModuleInfos + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + GLWETrace<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        sum_assign(self, res, Sum::inner(log_batch, log_count, normalize), keys, scratch)
    }

    /// [`Self::ckks_inner_sum_reference`] over the reversed levels and without division, refusals carrying
    /// `op: "ckks_replicate"`.
    fn ckks_replicate_reference<R, A, H>(
        &self,
        log_batch: usize,
        log_count: usize,
        res: &mut R,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSModuleInfos + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + GLWETrace<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        sum_into(self, res, a, Sum::replicate(log_batch, log_count), keys, scratch)
    }

    /// [`Self::ckks_replicate_reference`] in place, the keys looked up at `res.k()` and no copy.
    fn ckks_replicate_assign_reference<R, H>(
        &self,
        log_batch: usize,
        log_count: usize,
        res: &mut R,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSModuleInfos + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + GLWETrace<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        sum_assign(self, res, Sum::replicate(log_batch, log_count), keys, scratch)
    }
}

impl<BE: Backend> CKKSInnerSumReference<BE> for Module<BE> {}

fn sum_tmp_bytes<BE: Backend, M, R, A, K>(module: &M, res: &R, a: &A, key: &K) -> usize
where
    M: CKKSCopyOps<BE> + GLWETrace<BE> + ?Sized,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    K: GGLWEInfos,
{
    module
        .ckks_copy_tmp_bytes(res, a)
        .max(module.glwe_trace_assign_tmp_bytes(res, key))
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
            op: "ckks_inner_sum",
            log_batch,
            log_count,
            reverse: false,
            normalize,
        }
    }

    fn replicate(log_batch: usize, log_count: usize) -> Self {
        Self {
            op: "ckks_replicate",
            log_batch,
            log_count,
            reverse: true,
            normalize: false,
        }
    }

    /// Whether the `2^log_count` batches of `2^log_batch` slots fit in the slots of a degree-`n` ciphertext.
    fn fits<M: CKKSModuleInfos + ?Sized>(self, module: &M, n: Degree) -> bool {
        self.log_batch
            .checked_add(self.log_count)
            .is_some_and(|log| log <= module.ckks_slots_of_degree(n.as_usize()).ilog2() as usize)
    }

    /// The Galois elements of the sum's keys on a degree-`n` ciphertext, in the order the trace looks them up.
    fn galois_elements<BE: Backend, M: GLWETrace<BE> + ?Sized>(self, module: &M, n: Degree) -> Vec<i64> {
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
    fn validate<BE: Backend, M, A, H>(
        self,
        module: &M,
        a: &A,
        n: Degree,
        k: TorusPrecision,
        keys: &H,
        bytes: impl Fn(&GLWEAutomorphismKeyPreparedBackendRef<'_, BE>) -> usize,
    ) -> Result<Option<usize>>
    where
        M: CKKSModuleInfos + GLWETrace<BE> + ?Sized,
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

fn sum_into<BE: Backend, M, R, A, H>(
    module: &M,
    res: &mut R,
    a: &A,
    sum: Sum,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: CKKSModuleInfos + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + GLWETrace<BE> + ?Sized,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    // The copy narrows the sum to `res`'s width, at which every rotation then switches.
    let k = a.k().min(res.k());
    let bytes = sum.validate(module, a, res.n(), k, keys, |key| sum_tmp_bytes(module, &*res, a, key))?;
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

fn sum_assign<BE: Backend, M, R, H>(module: &M, res: &mut R, sum: Sum, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
where
    M: CKKSModuleInfos + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + GLWETrace<BE> + ?Sized,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    let bytes = sum.validate(module, &*res, res.n(), res.k(), keys, |key| {
        sum_tmp_bytes(module, &*res, &*res, key)
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
fn trace<BE: Backend, M, R, H>(module: &M, res: &mut R, sum: Sum, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
where
    M: CKKSModuleInfos + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + GLWETrace<BE> + ?Sized,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    module.glwe_trace_assign(res, sum.levels(), sum.reverse, false, keys, scratch);
    if sum.normalize {
        module.ckks_div_pow2_assign(sum.log_count, res)?;
    }
    Ok(())
}
