use poulpy_ckks::shaping::ShapingProfile;
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut};

pub(crate) fn shaped_round_correction<BE>(
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
    let n = low.n();
    assert!(n == h.n() && n == carry.n(), "ckks_shaped_round_correction: degrees differ");
    assert!(
        low.cols() == 1 && low.size() == 1 && h.cols() == 1 && h.size() == 1 && carry.cols() == 1 && carry.size() == 2,
        "ckks_shaped_round_correction: invalid buffers"
    );
    assert!(
        n.is_multiple_of(2 << profile.log_r()) && lambda > 0 && lambda <= profile.max_pass_bits(),
        "ckks_shaped_round_correction: invalid shaping depth or degree"
    );
    let stride = n >> (profile.log_r() + 1);
    let mask = (1u64 << lambda) - 1;
    let c = low.at(0, 0);
    let h = h.at_mut(0, 0);
    for (i, out) in h.iter_mut().enumerate() {
        let mut sum = 0u64;
        for (j, &coefficient) in profile.inverse().iter().enumerate() {
            let shift = j * stride;
            let term = coefficient.wrapping_mul(c[(i + n - shift) % n] as u64);
            sum = if i >= shift {
                sum.wrapping_add(term)
            } else {
                sum.wrapping_sub(term)
            };
        }
        *out = ((sum & mask) as i64) << (64 - lambda) >> (64 - lambda);
    }
    for (i, out) in carry.at_mut(0, 0).iter_mut().enumerate() {
        let mut product = 0i128;
        for (j, &coefficient) in profile.coefficients().iter().enumerate() {
            let shift = j * stride;
            let term = coefficient as i128 * h[(i + n - shift) % n] as i128;
            product += if i >= shift { term } else { -term };
        }
        *out = ((c[i] as i128 - product) >> lambda) as i64;
    }
    for (out, &value) in carry.at_mut(0, 1).iter_mut().zip(c) {
        *out = -value;
    }
}
