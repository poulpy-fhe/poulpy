use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard, VecZnxBackendMut, VecZnxBackendRef};

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
pub unsafe trait CKKSRoundImpl<Rg: Ring = Standard>: Backend {
    fn ckks_round_tmp_bytes<R, A>(module: &Module<Self>, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds;

    fn ckks_round<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>) -> Result<()>
    where
        R: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self, Rg> + CKKSCtBounds;
}

/// Backend correction used internally by shaped precision reduction.
///
/// # Safety
///
/// For `U = profile.coefficients()` and `g = U(Y)`, write
/// `h = centered(U^-1 * low mod 2^lambda)` to `h`, `(low - g*h) / 2^lambda`
/// to limb zero of `carry`, and `-low` to limb one. Products are in the ring
/// of `profile.ring()`: negacyclic of degree `N` with `Y = X^(N/(2R))` on the
/// standard ring, and on the conjugate-invariant ring `Rg` of degree `N` those of
/// `Z[X]/(X^(2N)+1)` with `Y = X^(N/R)`, on the invariant coefficients.
/// Inverse multiplication wraps modulo `2^64`. `low` and `h` have one column
/// and one limb, `carry` one column and two limbs, all at the same degree, a
/// multiple of `profile.period()`. The caller provides
/// `0 < lambda <= profile.max_pass_bits()`, so the correction fits signed
/// 64-bit words. Read every input before overwriting it. `low` is unchanged.
/// No scratch beyond the supplied buffers is needed.
pub unsafe trait CKKSShapedRoundingImpl<Rg: Ring = Standard>: Backend {
    fn ckks_shaped_round_correction(
        module: &Module<Self>,
        carry: &mut VecZnxBackendMut<'_, Self, Rg>,
        h: &mut VecZnxBackendMut<'_, Self, Rg>,
        low: &VecZnxBackendRef<'_, Self, Rg>,
        profile: &crate::shaping::ShapingProfile,
        lambda: usize,
    );
}

/// Implements [`CKKSRoundImpl`] for `$be` by forwarding to
/// [`CKKSRoundReference`](crate::reference::round::CKKSRoundReference), on both
/// rings from `($be)` and on one from `($be, $ring)`; `$be` implements
/// [`CKKSShapedRoundingImpl`] of each ring itself.
#[macro_export]
macro_rules! impl_ckks_round_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_round_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_round_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSRoundImpl<$ring> for $be {
            fn ckks_round_tmp_bytes<R, A>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
            {
                $crate::reference::round::CKKSRoundReference::<Self, $ring>::ckks_round_tmp_bytes_reference(module, res, a)
            }

            fn ckks_round<R, A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + $crate::CKKSCtBounds,
            {
                $crate::reference::round::CKKSRoundReference::<Self, $ring>::ckks_round_reference(module, res, a, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_round_reference_full;
