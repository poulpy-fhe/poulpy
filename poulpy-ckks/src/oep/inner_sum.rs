use crate::CKKSResult as Result;

use poulpy_core::{
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey},
    oep::GLWETraceImpl,
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    oep::derived::inner_sum::{
        ckks_inner_sum_assign_derived, ckks_inner_sum_derived, ckks_inner_sum_galois_elements_derived,
        ckks_inner_sum_tmp_bytes_derived, ckks_replicate_assign_derived, ckks_replicate_derived,
        ckks_replicate_galois_elements_derived,
    },
};

/// Extension point of [`crate::api::CKKSInnerSumOps`]. Every method is provided, composed over the selected copy,
/// poulpy-core's trace and the division by a power of two.
///
/// # Safety
///
/// An override reproduces the provided body of the method it replaces bit for bit, digits, metadata, canonical flag,
/// noise and refusals included, in their order and before any write, choosing only its kernels and its scratch use:
/// - `ckks_inner_sum_galois_elements` and `ckks_replicate_galois_elements` return poulpy-core's
///   `glwe_trace_galois_elements` over the levels of the sum, forward and reversed, and panic on the same windows;
/// - `ckks_inner_sum_tmp_bytes` covers the scratch the four sums take at its arguments, which may be less than the
///   provided query, never less than an overriding sum takes;
/// - `ckks_inner_sum`, `ckks_inner_sum_assign`, `ckks_replicate` and `ckks_replicate_assign` give the result of
///   `ckks_copy` (out of place), poulpy-core's summed `glwe_trace_assign` and, for a normalized inner sum,
///   `ckks_div_pow2_assign`, fusing each level's key switch and addition if they wish.
///
/// Memory, layout, synchrony and aliasing follow `poulpy-hal/docs/backend_safety_contract.md`.
pub unsafe trait CKKSInnerSumImpl: super::CKKSCopyImpl + super::CKKSPow2Impl + GLWETraceImpl {
    fn ckks_inner_sum_galois_elements(module: &Module<Self>, log_batch: usize, log_count: usize) -> Vec<i64> {
        ckks_inner_sum_galois_elements_derived(module, log_batch, log_count)
    }

    fn ckks_replicate_galois_elements(module: &Module<Self>, log_batch: usize, log_count: usize) -> Vec<i64> {
        ckks_replicate_galois_elements_derived(module, log_batch, log_count)
    }

    fn ckks_inner_sum_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_inner_sum_tmp_bytes_derived(module, res, a, key)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the sum's operands, batch, count and normalization, keys and scratch"
    )]
    fn ckks_inner_sum<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        ckks_inner_sum_derived(module, res, a, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_inner_sum_assign<R, H>(
        module: &Module<Self>,
        res: &mut R,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_inner_sum_assign_derived(module, res, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_replicate<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        ckks_replicate_derived(module, res, a, log_batch, log_count, keys, scratch)
    }

    fn ckks_replicate_assign<R, H>(
        module: &Module<Self>,
        res: &mut R,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_replicate_assign_derived(module, res, log_batch, log_count, keys, scratch)
    }
}

/// Selects every provided method of [`CKKSInnerSumImpl`] for `$be`.
#[macro_export]
macro_rules! impl_ckks_inner_sum_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSInnerSumImpl for $be {}
    };
}
pub use crate::impl_ckks_inner_sum_derived_full;
