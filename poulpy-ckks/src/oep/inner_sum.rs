use crate::CKKSResult as Result;

use poulpy_core::{
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey},
    oep::GLWETraceImpl,
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    oep::derived::inner_sum::{
        ckks_inner_sum_assign_derived, ckks_inner_sum_assign_tmp_bytes_derived, ckks_inner_sum_into_derived,
        ckks_inner_sum_into_tmp_bytes_derived, ckks_replicate_assign_derived, ckks_replicate_into_derived,
    },
};

/// Extension point of [`crate::api::CKKSInnerSumOps`]. Every method is provided, composed over the selected copy,
/// poulpy-core's trace and the division by a power of two; an override, such as one that fuses each level's key
/// switch and addition, reproduces it bit for bit.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures. See `poulpy-hal/docs/backend_safety_contract.md`.
pub unsafe trait CKKSInnerSumImpl: super::CKKSCopyImpl + super::CKKSPow2Impl + GLWETraceImpl {
    fn ckks_inner_sum_into_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_inner_sum_into_tmp_bytes_derived(module, res, a, key)
    }

    fn ckks_inner_sum_assign_tmp_bytes_impl<R, K>(module: &Module<Self>, res: &R, key: &K) -> usize
    where
        R: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_inner_sum_assign_tmp_bytes_derived(module, res, key)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the sum's operands, batch, count and normalization, keys and scratch"
    )]
    fn ckks_inner_sum_into_impl<Dst, Src, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        ckks_inner_sum_into_derived(module, dst, src, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_inner_sum_assign_impl<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_inner_sum_assign_derived(module, dst, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_replicate_into_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_inner_sum_into_tmp_bytes_derived(module, res, a, key)
    }

    fn ckks_replicate_assign_tmp_bytes_impl<R, K>(module: &Module<Self>, res: &R, key: &K) -> usize
    where
        R: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_inner_sum_assign_tmp_bytes_derived(module, res, key)
    }

    fn ckks_replicate_into_impl<Dst, Src, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        ckks_replicate_into_derived(module, dst, src, log_batch, log_count, keys, scratch)
    }

    fn ckks_replicate_assign_impl<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_replicate_assign_derived(module, dst, log_batch, log_count, keys, scratch)
    }
}

/// Selects the provided inner sum and replicate for `$be`.
#[macro_export]
macro_rules! impl_ckks_inner_sum_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSInnerSumImpl for $be {}
    };
}
pub use crate::impl_ckks_inner_sum_reference;
