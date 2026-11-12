use poulpy_ckks::shaping::{ShapingProfile, conjugate_invariant};
use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, Ring, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut};

pub(crate) fn shaped_round_correction<BE, Rg>(
    carry: &mut VecZnxBackendMut<'_, BE, Rg>,
    h: &mut VecZnxBackendMut<'_, BE, Rg>,
    low: &VecZnxBackendRef<'_, BE, Rg>,
    profile: &ShapingProfile<Rg>,
    lambda: usize,
) where
    BE: Backend<ZnxWord = i64>,
    Rg: Ring,
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
        n.is_multiple_of(profile.period()) && lambda > 0 && lambda <= profile.max_pass_bits(),
        "ckks_shaped_round_correction: invalid shaping depth or degree"
    );
    let c = low.at(0, 0);
    // The shift and the inverse, recomputed from `U` rather than read from the profile under test.
    let (coefficients, inverse) = shaping_and_inverse(profile);
    // The conjugate-invariant ring computes in `Z[X]/(X^(2n)+1)` on the
    // unfolded coefficients, which the products keep invariant.
    let unfold = |x: &[i64]| -> Vec<i64> {
        if !conjugate_invariant::<Rg>() {
            return x.to_vec();
        }
        let mut ambient = vec![0; 2 * n];
        ambient[..n].copy_from_slice(x);
        for j in 1..n {
            ambient[2 * n - j] = -x[j];
        }
        ambient
    };
    let ambient_c = unfold(c);
    let m = ambient_c.len();
    let stride = m / coefficients.len();
    let mask = (1u64 << lambda) - 1;
    let h = h.at_mut(0, 0);
    for (i, out) in h.iter_mut().enumerate() {
        let mut sum = 0u64;
        for (j, &coefficient) in inverse.iter().enumerate() {
            let shift = j * stride;
            let term = coefficient.wrapping_mul(ambient_c[(i + m - shift) % m] as u64);
            sum = if i >= shift {
                sum.wrapping_add(term)
            } else {
                sum.wrapping_sub(term)
            };
        }
        *out = ((sum & mask) as i64) << (64 - lambda) >> (64 - lambda);
    }
    let ambient_h = unfold(h);
    for (i, out) in carry.at_mut(0, 0).iter_mut().enumerate() {
        let mut product = 0i128;
        for (j, &coefficient) in coefficients.iter().enumerate() {
            let shift = j * stride;
            let term = coefficient as i128 * ambient_h[(i + m - shift) % m] as i128;
            product += if i >= shift { term } else { -term };
        }
        *out = ((c[i] as i128 - product) >> lambda) as i64;
    }
    for (out, &value) in carry.at_mut(0, 1).iter_mut().zip(c) {
        *out = -value;
    }
}

/// `Y^s*U` and its inverse modulo `2^64` in `Z[Y]/(Y^(2R)+1)`, from the layout's `U`: on the
/// conjugate-invariant ring, the smallest `s` whose `Y^s*U` is invariant (`v_R = 0` and
/// `v_(2R-j) = -v_j`), then the inverse by Gaussian elimination on its negacyclic matrix.
fn shaping_and_inverse<Rg: Ring>(profile: &ShapingProfile<Rg>) -> (Vec<i64>, Vec<u64>) {
    let u: Vec<i64> = profile.unit().iter().map(|&c| c as i64).collect();
    let m = u.len();
    // `Y^s*U`: coefficient `j` of `U` moves to `j + s`, negated past `2R`.
    let shifted = |s: usize| -> Vec<i64> {
        let mut v = vec![0i64; m];
        for (j, &c) in u.iter().enumerate() {
            let t = j + s;
            if t < m {
                v[t] += c;
            } else {
                v[t - m] -= c;
            }
        }
        v
    };
    let invariant = |v: &[i64]| v[m / 2] == 0 && (1..m).all(|j| v[m - j] == -v[j]);
    let v = if conjugate_invariant::<Rg>() {
        (0..m)
            .map(shifted)
            .find(|v| invariant(v))
            .expect("the profile admits a conjugate-invariant shift")
    } else {
        u.clone()
    };
    (v.clone(), negacyclic_inverse(&v))
}

/// The first column of the inverse of the negacyclic matrix of `v` modulo `2^64`, by
/// Gauss-Jordan elimination with odd pivots, which a unit's matrix has at every step
/// since it is invertible modulo 2.
fn negacyclic_inverse(v: &[i64]) -> Vec<u64> {
    let m = v.len();
    // Column `j` of the matrix is `Y^j*v`, so entry `(i, j)` is `±v[i - j]`.
    let mut a: Vec<Vec<u64>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| {
                    if i >= j {
                        v[i - j] as u64
                    } else {
                        (v[i + m - j] as u64).wrapping_neg()
                    }
                })
                .collect()
        })
        .collect();
    let mut rhs = vec![0u64; m];
    rhs[0] = 1;
    for col in 0..m {
        let pivot = (col..m)
            .find(|&r| a[r][col] & 1 == 1)
            .expect("a unit's negacyclic matrix is invertible modulo 2");
        a.swap(col, pivot);
        rhs.swap(col, pivot);
        let inv = odd_inverse(a[col][col]);
        for x in a[col].iter_mut() {
            *x = x.wrapping_mul(inv);
        }
        rhs[col] = rhs[col].wrapping_mul(inv);
        let (pivot_row, pivot_rhs) = (a[col].clone(), rhs[col]);
        for (r, (row, value)) in a.iter_mut().zip(rhs.iter_mut()).enumerate() {
            let factor = row[col];
            if r != col && factor != 0 {
                for (x, &p) in row.iter_mut().zip(&pivot_row) {
                    *x = x.wrapping_sub(factor.wrapping_mul(p));
                }
                *value = value.wrapping_sub(factor.wrapping_mul(pivot_rhs));
            }
        }
    }
    rhs
}

/// The inverse of an odd word modulo `2^64`, its bits solved one at a time.
fn odd_inverse(a: u64) -> u64 {
    let mut x = 1u64;
    for bit in 1..64 {
        if a.wrapping_mul(x) >> bit & 1 == 1 {
            x |= 1 << bit;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_ckks::shaping::ShapingUnit;
    use poulpy_hal::layouts::{ConjugateInvariant, Standard};

    /// The independent shift and inverse agree with the profile for every unit on both rings.
    #[test]
    fn recomputed_shaping_matches_the_profile() {
        for unit in ShapingUnit::ALL {
            let layout = unit.layout(1).unwrap();
            matches_the_profile(&ShapingProfile::<Standard>::try_new(&layout).unwrap(), unit);
            matches_the_profile(&ShapingProfile::<ConjugateInvariant>::try_new(&layout).unwrap(), unit);
        }
    }

    fn matches_the_profile<Rg: Ring>(profile: &ShapingProfile<Rg>, unit: ShapingUnit) {
        let (coefficients, inverse) = shaping_and_inverse(profile);
        let want: Vec<i64> = profile.coefficients().iter().map(|&c| c as i64).collect();
        let invariant = conjugate_invariant::<Rg>();
        assert_eq!(coefficients, want, "{unit:?}, invariant = {invariant}");
        assert_eq!(inverse, profile.inverse(), "{unit:?}, invariant = {invariant}");
    }
}
