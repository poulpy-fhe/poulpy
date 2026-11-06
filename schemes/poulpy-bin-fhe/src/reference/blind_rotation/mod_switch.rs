use crate::blind_rotation::LookUpTableRotationDirection;
use poulpy_core::layouts::{LWEInfos, LWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module};

/// Independently callable canonical implementation of
/// [`BlindRotationModSwitch::blind_rotation_mod_switch`](crate::api::BlindRotationModSwitch::blind_rotation_mod_switch),
/// selected by `impl_blind_rotation_mod_switch_reference_full!`.
///
/// Downloads the body and mask and modulus-switches all normalized signed limbs.
/// The result is rounded to the nearest integer modulo `modulus`, with ties
/// towards positive infinity. Left rotation negates the complete coefficient
/// before rounding.
///
/// # Panics
///
/// If `modulus` is not a power of two in `[2, 2^63)`, `res` does not hold `lwe.n() + 1` values or the radix of `lwe` is
/// outside `[1, 63]`; `res` is then untouched.
pub fn blind_rotation_mod_switch_reference<BE, A>(
    _module: &Module<BE>,
    modulus: usize,
    direction: LookUpTableRotationDirection,
    res: &mut [i64],
    lwe: &A,
) where
    BE: Backend<ZnxWord = i64>,
    A: LWEToBackendRef<BE> + LWEInfos,
{
    assert!(
        modulus >= 2 && modulus.is_power_of_two() && modulus <= i64::MAX as usize,
        "the modulus is not a power of two in [2, 2^63)"
    );
    assert!(
        res.len() == lwe.n().as_usize() + 1,
        "the output length differs from the LWE dimension plus one"
    );
    let lwe = lwe.to_backend_ref();
    let base2k = lwe.base2k().as_usize();
    assert!((1..=63).contains(&base2k), "the LWE radix is outside [1, 63]");
    let body_bytes = BE::vec_znx_bytes_of(lwe.body().n(), lwe.body().cols(), lwe.body().size());
    let mask_bytes = BE::vec_znx_bytes_of(lwe.mask().n(), lwe.mask().cols(), lwe.mask().size());
    let mut body = vec![0u8; body_bytes];
    let mut mask = vec![0u8; mask_bytes];
    BE::copy_view_to_host(&BE::region_ref(lwe.body().data(), 0, body_bytes), &mut body);
    BE::copy_view_to_host(&BE::region_ref(lwe.mask().data(), 0, mask_bytes), &mut mask);
    let word = |bytes: &[u8], index: usize| {
        let start = index * core::mem::size_of::<i64>();
        i64::from_ne_bytes(bytes[start..start + core::mem::size_of::<i64>()].try_into().unwrap())
    };
    let size = lwe.size();
    let n = lwe.n().as_usize();
    for (col, output) in res.iter_mut().enumerate() {
        *output = switch_coefficient(modulus, base2k, size, direction, |limb| {
            if col == 0 {
                word(&body, limb)
            } else {
                word(&mask, limb * n + col - 1)
            }
        });
    }
}

fn switch_coefficient(
    modulus: usize,
    base2k: usize,
    size: usize,
    direction: LookUpTableRotationDirection,
    limb: impl Fn(usize) -> i64,
) -> i64 {
    let bits = modulus.ilog2() as usize;
    let take = size.min((bits + 1).div_ceil(base2k));
    let sign = match direction {
        LookUpTableRotationDirection::Left => -1i128,
        LookUpTableRotationDirection::Right => 1,
    };
    let mut value = 0i128;
    for i in 0..take {
        value = (value << base2k) + sign * i128::from(limb(i));
    }
    let precision = take * base2k;
    let rounded = if precision <= bits {
        value << (bits - precision)
    } else {
        let shift = precision - bits;
        let half = 1i128 << (shift - 1);
        let tail_negative = (take..size)
            .map(|i| sign * i128::from(limb(i)))
            .find(|&x| x != 0)
            .is_some_and(|x| x < 0);
        // Signed lower limbs decide which side of an exact midpoint the
        // complete coefficient lies on, even when the prefix is identical.
        let below_tie = value.rem_euclid(1i128 << shift) == half && tail_negative;
        ((value + half) >> shift) - i128::from(below_tie)
    };
    (rounded & (modulus as i128 - 1)) as i64
}

/// Reference body of the [`BlindRotationModSwitchImpl`](crate::oep::BlindRotationModSwitchImpl) method, which
/// [`impl_blind_rotation_mod_switch_reference_full!`](crate::impl_blind_rotation_mod_switch_reference_full) forwards to.
pub trait BlindRotationModSwitchReference<BE: Backend> {
    fn blind_rotation_mod_switch_reference<A>(
        &self,
        modulus: usize,
        direction: LookUpTableRotationDirection,
        res: &mut [i64],
        lwe: &A,
    ) where
        A: LWEToBackendRef<BE> + LWEInfos;
}

impl<BE: Backend<ZnxWord = i64>> BlindRotationModSwitchReference<BE> for Module<BE> {
    fn blind_rotation_mod_switch_reference<A>(
        &self,
        modulus: usize,
        direction: LookUpTableRotationDirection,
        res: &mut [i64],
        lwe: &A,
    ) where
        A: LWEToBackendRef<BE> + LWEInfos,
    {
        blind_rotation_mod_switch_reference(self, modulus, direction, res, lwe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modulus_switch_rounds_complete_signed_coefficients() {
        for direction in [LookUpTableRotationDirection::Left, LookUpTableRotationDirection::Right] {
            let sign = if matches!(direction, LookUpTableRotationDirection::Left) {
                -1i128
            } else {
                1
            };
            for first in -8..8i64 {
                for second in -8..8i64 {
                    for third in -8..8i64 {
                        let limbs = [first, second, third];
                        let value = sign * i128::from(first * 256 + second * 16 + third);
                        for modulus in [2usize, 8, 32, 128, 1024, 4096] {
                            let expected = ((value * modulus as i128 + 2048) >> 12) & (modulus as i128 - 1);
                            assert_eq!(switch_coefficient(modulus, 4, 3, direction, |i| limbs[i]), expected as i64);
                        }
                    }
                }
            }
        }
    }
}
