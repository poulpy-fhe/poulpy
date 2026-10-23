use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Both methods reproduce [`crate::reference::round::ckks_round_into_reference`]
/// and its query, including validation, metadata and canonical flags. An
/// override may use different kernels and scratch within its own query.
pub unsafe trait CKKSRoundImpl: CKKSShapedRoundingImpl {
    fn ckks_round_tmp_bytes_impl<Dst, Src>(module: &Module<Self>, dst: &Dst, src: &Src) -> usize
    where
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds;

    fn ckks_round_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Backend correction used internally by shaped precision reduction.
///
/// # Safety
///
/// For `g = U(X^(N/(2R)))`, write `h = centered(U^-1 * low mod 2^lambda)`
/// to `h`, `(low - g*h) / 2^lambda` to limb zero of `carry`, and `-low` to
/// limb one. Products are negacyclic; inverse multiplication wraps modulo
/// `2^64`. `low` and `h` have one column and one limb, `carry` one column and
/// two limbs, all at the same degree divisible by `2R`. The caller provides
/// `0 < lambda <= profile.max_pass_bits()`, so the correction fits
/// signed 64-bit words. Read every input before overwriting it; `low` is
/// unchanged. No scratch beyond the supplied buffers is needed.
pub unsafe trait CKKSShapedRoundingImpl: Backend {
    fn ckks_shaped_round_correction_impl(
        module: &Module<Self>,
        carry: &mut VecZnxBackendMut<'_, Self>,
        h: &mut VecZnxBackendMut<'_, Self>,
        low: &VecZnxBackendRef<'_, Self>,
        profile: &crate::shaping::ShapingProfile,
        lambda: usize,
    );
}

/// Implements this contract with the backend-independent reference algorithm.
#[macro_export]
macro_rules! impl_ckks_round_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSRoundImpl for $be {
            fn ckks_round_tmp_bytes_impl<Dst, Src>(module: &::poulpy_hal::layouts::Module<Self>, dst: &Dst, src: &Src) -> usize
            where
                Dst: $crate::CKKSCtBounds,
                Src: $crate::CKKSCtBounds,
            {
                $crate::reference::round::ckks_round_tmp_bytes_reference(module, dst, src)
            }

            fn ckks_round_into_impl<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::round::ckks_round_into_reference(module, dst, src, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_round_reference;
