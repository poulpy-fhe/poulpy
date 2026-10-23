//! CPU coefficient correction for shaped CKKS rounding.

use poulpy_ckks::shaping::ShapingProfile;
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut};

/// Writes the two correction limbs and the centered shaping preimage.
pub fn shaped_round_correction<BE>(
    carry: &mut VecZnxBackendMut<'_, BE>,
    h: &mut VecZnxBackendMut<'_, BE>,
    low: &VecZnxBackendRef<'_, BE>,
    profile: &ShapingProfile,
    lambda: usize,
) where
    BE: Backend<ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert!(
        low.n() == h.n() && low.n() == carry.n(),
        "ckks_shaped_round_correction: degrees differ"
    );
    assert!(
        low.cols() == 1 && low.size() == 1 && h.cols() == 1 && h.size() == 1 && carry.cols() == 1 && carry.size() == 2,
        "ckks_shaped_round_correction: invalid buffers"
    );
    assert!(
        low.n().is_multiple_of(2 << profile.log_r()) && lambda > 0 && lambda <= profile.max_pass_bits(),
        "ckks_shaped_round_correction: invalid shaping depth or degree"
    );
    let c = low.at(0, 0);
    profile.shaped_multiple_host(
        low.n() >> (profile.log_r() + 1),
        lambda,
        c,
        h.at_mut(0, 0),
        carry.at_mut(0, 0),
    );
    for (kappa, &cj) in carry.at_mut(0, 0).iter_mut().zip(c) {
        *kappa = (cj - *kappa) >> lambda;
    }
    for (neg, &cj) in carry.at_mut(0, 1).iter_mut().zip(c) {
        *neg = -cj;
    }
}

/// Selects the CPU shaping correction for a host-visible backend.
#[macro_export]
macro_rules! impl_cpu_ckks_shaped_rounding {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSShapedRoundingImpl for $be {
            fn ckks_shaped_round_correction_impl(
                _module: &::poulpy_hal::layouts::Module<Self>,
                carry: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
                h: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
                low: &::poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
                profile: &::poulpy_ckks::shaping::ShapingProfile,
                lambda: usize,
            ) {
                $crate::ckks_round::shaped_round_correction::<Self>(carry, h, low, profile, lambda);
            }
        }
    };
}
