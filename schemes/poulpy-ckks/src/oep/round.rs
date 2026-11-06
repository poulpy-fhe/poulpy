use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// The rounding family of [`CKKSRoundOps`](crate::api::CKKSRoundOps).
///
/// # Safety
///
/// Both methods reproduce
/// [`CKKSRoundReference::ckks_round_reference`](crate::reference::round::CKKSRoundReference::ckks_round_reference)
/// and its query bit for bit, including validation, metadata and canonical
/// flags, with no more scratch than `ckks_round_tmp_bytes` reports. An override
/// chooses only its kernels and scratch within its own query. Memory, layout,
/// synchrony and aliasing obligations are those of
/// `poulpy-hal/docs/backend_safety_contract.md`.
pub unsafe trait CKKSRoundImpl: Backend {
    fn ckks_round_tmp_bytes<R, A>(module: &Module<Self>, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds;

    fn ckks_round<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Backend correction used internally by shaped precision reduction.
///
/// # Safety
///
/// For `U = profile.coefficients()` and `g = U(Y)`, write
/// `h = centered(U^-1 * low mod 2^lambda)` to `h`, `(low - g*h) / 2^lambda`
/// to limb zero of `carry`, and `-low` to limb one. Products are in the ring
/// of `profile.ring()`: negacyclic of degree `N` with `Y = X^(N/(2R))` on the
/// standard ring, and on the conjugate-invariant ring of degree `N` those of
/// `Z[X]/(X^(2N)+1)` with `Y = X^(N/R)`, on the invariant coefficients.
/// Inverse multiplication wraps modulo `2^64`. `low` and `h` have one column
/// and one limb, `carry` one column and two limbs, all at the same degree, a
/// multiple of `profile.period()`. The caller provides
/// `0 < lambda <= profile.max_pass_bits()`, so the correction fits signed
/// 64-bit words. Read every input before overwriting it. `low` is unchanged.
/// No scratch beyond the supplied buffers is needed.
pub unsafe trait CKKSShapedRoundingImpl: Backend {
    fn ckks_shaped_round_correction(
        module: &Module<Self>,
        carry: &mut VecZnxBackendMut<'_, Self>,
        h: &mut VecZnxBackendMut<'_, Self>,
        low: &VecZnxBackendRef<'_, Self>,
        profile: &crate::shaping::ShapingProfile,
        lambda: usize,
    );
}

/// Implements [`CKKSRoundImpl`] for `$be` by forwarding to
/// [`CKKSRoundReference`](crate::reference::round::CKKSRoundReference); `$be`
/// implements [`CKKSShapedRoundingImpl`] itself.
#[macro_export]
macro_rules! impl_ckks_round_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSRoundImpl for $be {
            fn ckks_round_tmp_bytes<R, A>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
            {
                $crate::reference::round::CKKSRoundReference::ckks_round_tmp_bytes_reference(module, res, a)
            }

            fn ckks_round<R, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::round::CKKSRoundReference::ckks_round_reference(module, res, a, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_round_reference_full;
