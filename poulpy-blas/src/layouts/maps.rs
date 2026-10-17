//! Diagonals of the per-matrix-slot linear maps, built on the host.

use std::collections::BTreeMap;

use num_traits::One;
use poulpy_core::layouts::{DiagonalArithmetic, Diagonals};

use super::grid::assert_tiles;

/// Diagonals over `slots` slots of the map whose output `(r, c)` in every matrix slot sums `w * in(r2, c2)` over `terms(r, c)`.
pub fn per_matrix_map<F, I>(slots: usize, log_d: usize, terms: impl Fn(usize, usize) -> I) -> Diagonals<F>
where
    F: DiagonalArithmetic + PartialEq,
    I: IntoIterator<Item = (usize, usize, F)>,
{
    per_lane_map(slots, log_d, 1, |_, r, c| terms(r, c))
}

/// Diagonals over `slots` slots of the map whose output `(r, c)` in matrix slot `s` sums `w * in(r2, c2)` over
/// `terms(s mod count, r, c)`, every diagonal stored at the smallest power-of-two period that holds them all.
pub fn per_lane_map<F, I>(slots: usize, log_d: usize, count: usize, terms: impl Fn(usize, usize, usize) -> I) -> Diagonals<F>
where
    F: DiagonalArithmetic + PartialEq,
    I: IntoIterator<Item = (usize, usize, F)>,
{
    let acc = accumulate(slots, log_d, count, terms);
    let period = acc.values().map(|values| min_period(values)).max().unwrap_or(1);
    truncate(slots, acc, period)
}

/// The values of every diagonal of [`per_lane_map`], `count d^2` slots each, by offset.
fn accumulate<F, I>(slots: usize, log_d: usize, count: usize, terms: impl Fn(usize, usize, usize) -> I) -> BTreeMap<i64, Vec<F>>
where
    F: DiagonalArithmetic,
    I: IntoIterator<Item = (usize, usize, F)>,
{
    let d = 1usize << log_d;
    assert_tiles(slots, d, count);
    let mut acc: BTreeMap<i64, Vec<F>> = BTreeMap::new();
    for s in 0..count {
        for r in 0..d {
            for c in 0..d {
                let t = r * d + c;
                for (r2, c2, w) in terms(s, r, c) {
                    // Offsets equal modulo `slots` are one diagonal: accumulate, never overwrite.
                    let offset = ((r2 * d + c2) as i64 - t as i64).rem_euclid(slots as i64);
                    acc.entry(offset)
                        .or_insert_with(|| vec![<F as DiagonalArithmetic>::zero(); count * d * d])[s * d * d + t]
                        .add_assign(&w);
                }
            }
        }
    }
    acc
}

/// Every diagonal of `acc` stored at `period`, which each of them repeats with.
fn truncate<F: DiagonalArithmetic>(slots: usize, acc: BTreeMap<i64, Vec<F>>, period: usize) -> Diagonals<F> {
    // One period for every diagonal is only a hedge: upstream re-encodes every diagonal of a map at the least common
    // multiple of their periods.
    let mut diagonals = Diagonals::new(slots);
    for (offset, mut values) in acc {
        values.truncate(period);
        diagonals.set(offset, values);
    }
    diagonals
}

/// The smallest power of two `p` dividing `v.len()` with `v[t] == v[t mod p]` for every `t`, else `v.len()`.
fn min_period<F: PartialEq>(v: &[F]) -> usize {
    (0..usize::BITS)
        .map(|e| 1usize << e)
        .take_while(|&p| v.len().is_multiple_of(p))
        .find(|&p| v.iter().enumerate().all(|(t, x)| *x == v[t % p]))
        .unwrap_or(v.len())
}

/// JKLS sigma: output `(r, c)` reads `(r, r + c mod d)`.
pub fn sigma<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| [(r, (r + c) % d, F::one())])
}

/// JKLS tau: output `(r, c)` reads `(r + c mod d, c)`.
pub fn tau<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| [((r + c) % d, c, F::one())])
}

/// JKLS tau on a replicated input, every matrix slot holding the same matrix: output `(r, c)` reads `(r + c mod d, c)`
/// through the rotation by `c d`, which wraps inside a block because the matrix slots repeat. `d` diagonals at the
/// offsets `c d`, each the mask of column `c`, where [`tau`] needs `2d - 1` once two matrix slots differ.
pub fn tau_rep<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    assert_tiles(slots, d, 1);
    let mut diagonals = Diagonals::new(slots);
    for c in 0..d {
        let mask = (0..d)
            .map(|col| {
                if col == c {
                    F::one()
                } else {
                    <F as DiagonalArithmetic>::zero()
                }
            })
            .collect();
        diagonals.set((c * d) as i64, mask);
    }
    diagonals
}

/// Transpose: output `(r, c)` reads `(c, r)`.
pub fn transpose<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize) -> Diagonals<F> {
    per_matrix_map(slots, log_d, |r, c| [(c, r, F::one())])
}

/// JKLS psi^k: output `(r, c)` reads `(r + k mod d, c)`.
pub fn rotate_rows<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| [((r + k) % d, c, F::one())])
}

/// psi^k on a replicated input, for `1 <= k < d`: the rotation by `k d`, one all-ones diagonal, which wraps inside a
/// block because the matrix slots repeat. Encoded at `log_delta = 0`, a product by 1, it spends no scale.
pub fn rotate_rows_rep<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    assert_tiles(slots, d, 1);
    assert!(0 < k && k < d, "rotate_rows_rep takes 1 <= k < d");
    let mut diagonals = Diagonals::new(slots);
    diagonals.set(((k << log_d) % slots) as i64, vec![F::one()]);
    diagonals
}

/// JKLS phi^k: output `(r, c)` reads `(r, c + k mod d)`.
pub fn rotate_cols<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| [(r, (c + k) % d, F::one())])
}

/// Upper half of psi^k: output `(r, c)` reads `(r + k, c)` when `r + k < d`, else 0.
pub fn rows_upper<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| (r + k < d).then(|| (r + k, c, F::one())))
}

/// Lower half of psi^k: output `(r, c)` reads `(r + k - d, c)` when `r + k >= d`, else 0.
pub fn rows_lower<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| (r + k >= d).then(|| (r + k - d, c, F::one())))
}

/// Left half of phi^k: output `(r, c)` reads `(r, c + k)` when `c + k < d`, else 0.
pub fn cols_left<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| (c + k < d).then(|| (r, c + k, F::one())))
}

/// Right half of phi^k: output `(r, c)` reads `(r, c + k - d)` when `c + k >= d`, else 0.
pub fn cols_right<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| (c + k >= d).then(|| (r, c + k - d, F::one())))
}

/// Right product in every matrix slot: output `(r, c)` of matrix slot `s` is `sum_l in(r, l) b(s mod count, l, c)`.
///
/// Its diagonals are stored at a period fixed by the structure, never by `b`'s values: `d` for one public matrix, whose
/// diagonals vary with the column alone, `count d^2` otherwise. Every diagonal offset is stored, zero or not.
pub fn right_mul<F: DiagonalArithmetic + PartialEq>(
    slots: usize,
    log_d: usize,
    count: usize,
    b: impl Fn(usize, usize, usize) -> F,
) -> Diagonals<F> {
    let d = 1usize << log_d;
    let b = &b;
    let acc = accumulate(slots, log_d, count, |s, r, c| (0..d).map(move |l| (r, l, b(s, l, c))));
    truncate(slots, acc, if count == 1 { d } else { count * d * d })
}

/// Left product in every matrix slot: output `(r, c)` of matrix slot `s` is `sum_l b(s mod count, r, l) in(l, c)`.
///
/// Its diagonals sit at the offsets `(l - r) d`, `2d - 1` of them, `d` when one matrix fills the slots, every one
/// stored, zero or not, at `count d^2` slots, a period fixed by the structure, never by `b`'s values.
pub fn left_mul<F: DiagonalArithmetic + PartialEq>(
    slots: usize,
    log_d: usize,
    count: usize,
    b: impl Fn(usize, usize, usize) -> F,
) -> Diagonals<F> {
    let d = 1usize << log_d;
    let b = &b;
    let acc = accumulate(slots, log_d, count, |s, r, c| (0..d).map(move |l| (l, c, b(s, r, l))));
    truncate(slots, acc, count * d * d)
}

/// [`left_mul`] of one public matrix on a replicated input, every matrix slot holding the same matrix: the rotation by
/// `-j d` equals the one by `(d - j) d` there, so the diagonals `j` and `j - d` merge. `d` diagonals at the offsets
/// `j d`, `0 <= j < d`, the entry of matrix slot position `(r, c)` being `b(0, r, (r + j) mod d)`, stored at `d^2` slots.
pub fn left_mul_rep<F: DiagonalArithmetic + PartialEq>(
    slots: usize,
    log_d: usize,
    b: impl Fn(usize, usize, usize) -> F,
) -> Diagonals<F> {
    let d = 1usize << log_d;
    assert_tiles(slots, d, 1);
    let mut diagonals = Diagonals::new(slots);
    for j in 0..d {
        let values = (0..d * d).map(|t| b(0, t / d, ((t / d) + j) % d)).collect();
        diagonals.set((j * d) as i64, values);
    }
    diagonals
}

/// JKLS stage 2 of the right product, applied to `x = sigma(A)`: output `(r, c)` of matrix slot `s` is
/// `sum_k x(r, (c + k) mod d) b(s mod count, (r + c + k) mod d, c)`, so that `right_jkls(sigma(A)) = A B`, where `B` is
/// `b(s mod count, ., .)` in matrix slot `s`.
///
/// Its `2d - 1` diagonals sit at sigma's offsets, `-(d-1)..=d-1`, every one stored, zero or not, at `count d^2` slots,
/// a period fixed by the structure, never by `b`'s values.
pub fn right_jkls<F: DiagonalArithmetic + PartialEq>(
    slots: usize,
    log_d: usize,
    count: usize,
    b: impl Fn(usize, usize, usize) -> F,
) -> Diagonals<F> {
    let d = 1usize << log_d;
    let b = &b;
    let acc = accumulate(slots, log_d, count, |s, r, c| {
        (0..d).map(move |k| (r, (c + k) % d, b(s, (r + c + k) % d, c)))
    });
    truncate(slots, acc, count * d * d)
}

/// JKLS stage 2 of the left product, applied to `y = tau(A)`: output `(r, c)` of matrix slot `s` is
/// `sum_k y((r + k) mod d, c) b(s mod count, r, (r + c + k) mod d)`, so that `left_jkls(tau(A)) = B A`, where `B` is
/// `b(s mod count, ., .)` in matrix slot `s`.
///
/// Its diagonals sit at tau's offsets, `j d` for `|j| < d`, `2d - 1` of them, `d` when one matrix fills the slots,
/// every one stored, zero or not, at `count d^2` slots.
pub fn left_jkls<F: DiagonalArithmetic + PartialEq>(
    slots: usize,
    log_d: usize,
    count: usize,
    b: impl Fn(usize, usize, usize) -> F,
) -> Diagonals<F> {
    let d = 1usize << log_d;
    let b = &b;
    let acc = accumulate(slots, log_d, count, |s, r, c| {
        (0..d).map(move |k| ((r + k) % d, c, b(s, r, (r + c + k) % d)))
    });
    truncate(slots, acc, count * d * d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_core::layouts::{Diagonals, Evaluate, LinearTransformationStrategy};

    const LOG_D: usize = 2;
    const D: usize = 1 << LOG_D;
    const MATS: usize = 4;
    const SLOTS: usize = MATS * D * D;

    fn input(seed: usize) -> Vec<f64> {
        (0..SLOTS).map(|t| ((t * t + seed) % 97) as f64 - 48.0).collect()
    }

    fn apply(map: &Diagonals<f64>, x: &[f64]) -> Vec<f64> {
        map.evaluate(x, LinearTransformationStrategy::Direct)
    }

    /// `out[s][r][c] = f(matrix s of x, r, c)`.
    fn per_matrix(x: &[f64], f: impl Fn(&[f64], usize, usize) -> f64) -> Vec<f64> {
        (0..SLOTS)
            .map(|t| {
                let s = t / (D * D);
                f(&x[s * D * D..(s + 1) * D * D], (t / D) % D, t % D)
            })
            .collect()
    }

    #[test]
    fn sigma_reads_row_shifted_by_row_index() {
        let x = input(1);
        assert_eq!(
            apply(&sigma(SLOTS, LOG_D), &x),
            per_matrix(&x, |m, r, c| m[r * D + (r + c) % D])
        );
    }

    #[test]
    fn tau_reads_column_shifted_by_column_index() {
        let x = input(2);
        assert_eq!(
            apply(&tau(SLOTS, LOG_D), &x),
            per_matrix(&x, |m, r, c| m[((r + c) % D) * D + c])
        );
    }

    #[test]
    fn transpose_swaps_indices() {
        let x = input(3);
        assert_eq!(apply(&transpose(SLOTS, LOG_D), &x), per_matrix(&x, |m, r, c| m[c * D + r]));
    }

    #[test]
    fn rotate_cols_and_rows_shift_by_k() {
        let x = input(4);
        for k in 0..D {
            assert_eq!(
                apply(&rotate_cols(SLOTS, LOG_D, k), &x),
                per_matrix(&x, |m, r, c| m[r * D + (c + k) % D])
            );
            assert_eq!(
                apply(&rotate_rows(SLOTS, LOG_D, k), &x),
                per_matrix(&x, |m, r, c| m[((r + k) % D) * D + c])
            );
        }
    }

    #[test]
    fn right_mul_multiplies_every_matrix_slot() {
        let x = input(5);
        let b: Vec<f64> = (0..D * D).map(|i| (i % 7) as f64 - 3.0).collect();
        let want = per_matrix(&x, |m, r, c| (0..D).map(|l| m[r * D + l] * b[l * D + c]).sum());
        assert_eq!(apply(&right_mul(SLOTS, LOG_D, 1, |_, l, c| b[l * D + c]), &x), want);
    }

    #[test]
    fn right_mul_reads_one_matrix_per_lane() {
        let x = input(10);
        let b = |m: usize, l: usize, c: usize| ((m * 5 + l * D + c) % 7) as f64 - 3.0;
        for count in [2, MATS] {
            let map = right_mul(SLOTS, LOG_D, count, b);
            let want: Vec<f64> = (0..SLOTS)
                .map(|t| {
                    let (s, r, c) = (t / (D * D), (t / D) % D, t % D);
                    (0..D).map(|l| x[s * D * D + r * D + l] * b(s % count, l, c)).sum()
                })
                .collect();
            assert_eq!(apply(&map, &x), want);
        }
    }

    /// The right product stores `d` slots per diagonal for one public matrix and `count d^2` for more, at every offset
    /// and whatever the entries: generic, identity, constant, zero, and every matrix equal. Its values stay exact.
    #[test]
    fn right_mul_stores_a_structural_period() {
        type Entry = fn(usize, usize, usize) -> f64;
        let variants: [(&str, Entry); 5] = [
            ("generic", |m, l, c| ((m * 5 + l * D + c) % 7) as f64 - 3.0),
            ("identity", |_, l, c| f64::from(l == c)),
            ("constant", |_, _, _| 2.0),
            ("zero", |_, _, _| 0.0),
            ("equal", |_, l, c| ((l * D + c) % 7) as f64 - 3.0),
        ];
        let x = input(12);
        let offsets = right_mul(SLOTS, LOG_D, 1, variants[0].1).indexes();
        assert_eq!(offsets.len(), 2 * D - 1);
        for count in [1, 2] {
            let period = if count == 1 { D } else { count * D * D };
            for (label, b) in variants {
                let map = right_mul(SLOTS, LOG_D, count, b);
                assert_eq!(map.indexes(), offsets, "{label} count={count}");
                for &i in &offsets {
                    assert_eq!(map.period(i), Some(period), "{label} count={count} offset={i}");
                }
                let want: Vec<f64> = (0..SLOTS)
                    .map(|t| {
                        let (s, r, c) = (t / (D * D), (t / D) % D, t % D);
                        (0..D).map(|l| x[s * D * D + r * D + l] * b(s % count, l, c)).sum()
                    })
                    .collect();
                assert_eq!(apply(&map, &x), want, "{label} count={count}");
            }
        }
    }

    /// The left product reads one public matrix per matrix slot, stores its `2d - 1` diagonals at the offsets `(l - r) d`
    /// and at `count d^2` slots each, whatever the entries, and its values stay exact.
    #[test]
    fn left_mul_stores_a_structural_period() {
        type Entry = fn(usize, usize, usize) -> f64;
        let variants: [(&str, Entry); 4] = [
            ("generic", |m, r, l| ((m * 5 + r * D + l) % 7) as f64 - 3.0),
            ("identity", |_, r, l| f64::from(r == l)),
            ("constant", |_, _, _| 2.0),
            ("zero", |_, _, _| 0.0),
        ];
        let mut offsets: Vec<i64> = (1 - D as i64..D as i64)
            .map(|j| (j * D as i64).rem_euclid(SLOTS as i64))
            .collect();
        offsets.sort_unstable();
        let x = input(13);
        for count in [1, 2, MATS] {
            for (label, b) in variants {
                let map = left_mul(SLOTS, LOG_D, count, b);
                assert_eq!(map.indexes(), offsets, "{label} count={count}");
                for &i in &offsets {
                    assert_eq!(map.period(i), Some(count * D * D), "{label} count={count} offset={i}");
                }
                let want: Vec<f64> = (0..SLOTS)
                    .map(|t| {
                        let (s, r, c) = (t / (D * D), (t / D) % D, t % D);
                        (0..D).map(|l| b(s % count, r, l) * x[s * D * D + l * D + c]).sum()
                    })
                    .collect();
                assert_eq!(apply(&map, &x), want, "{label} count={count}");
            }
        }
    }

    /// Entry `(l, c)` of public matrix `m`: an integer from -3 to 3, so seven values over a matrix's `D^2` positions,
    /// shifted from one matrix to the next.
    fn public(m: usize, l: usize, c: usize) -> f64 {
        ((m * 5 + l * D + c) % 7) as f64 - 3.0
    }

    /// On one matrix slot and on four, for every count dividing them: the JKLS stage-2 maps after sigma (tau) are the
    /// direct products, at sigma's (tau's) offsets, every one stored at `count d^2` slots.
    #[test]
    fn jkls_stage_two_after_sigma_or_tau_is_the_direct_product() {
        for (slots, counts) in [(D * D, vec![1]), (SLOTS, vec![1, 2, MATS])] {
            let x = &input(14)[..slots];
            for count in counts {
                let (right, left) = (
                    right_jkls(slots, LOG_D, count, public),
                    left_jkls(slots, LOG_D, count, public),
                );
                assert_eq!(
                    apply(&right, &apply(&sigma(slots, LOG_D), x)),
                    apply(&right_mul(slots, LOG_D, count, public), x),
                    "right slots={slots} count={count}"
                );
                assert_eq!(
                    apply(&left, &apply(&tau(slots, LOG_D), x)),
                    apply(&left_mul(slots, LOG_D, count, public), x),
                    "left slots={slots} count={count}"
                );
                assert_eq!(
                    right.indexes(),
                    sigma::<f64>(slots, LOG_D).indexes(),
                    "right offsets slots={slots}"
                );
                assert_eq!(
                    left.indexes(),
                    tau::<f64>(slots, LOG_D).indexes(),
                    "left offsets slots={slots}"
                );
                for map in [&right, &left] {
                    for &i in &map.indexes() {
                        assert_eq!(map.period(i), Some(count * D * D), "slots={slots} count={count} offset={i}");
                    }
                }
            }
        }
    }

    #[test]
    fn maps_store_one_minimal_period() {
        let period = |map: Diagonals<f64>| -> usize {
            let periods: Vec<usize> = map.indexes().iter().map(|&i| map.period(i).unwrap()).collect();
            assert!(periods.iter().all(|&p| p == periods[0]), "{periods:?}");
            periods[0]
        };
        let b = |m: usize, l: usize, c: usize| ((m * 5 + l * D + c) % 7) as f64 - 3.0;
        // One public matrix: every diagonal of the right product depends on the column only.
        assert_eq!(period(right_mul(SLOTS, LOG_D, 1, b)), D);
        for count in [2, MATS] {
            assert_eq!(period(right_mul(SLOTS, LOG_D, count, b)), count * D * D);
        }
        for k in 1..D {
            for map in [rotate_cols, cols_left, cols_right] {
                assert_eq!(period(map(SLOTS, LOG_D, k)), D, "k={k}");
            }
            for map in [rotate_rows, rows_upper, rows_lower] {
                assert_eq!(period(map(SLOTS, LOG_D, k)), D * D, "k={k}");
            }
        }
        for map in [sigma, tau, transpose] {
            assert_eq!(period(map(SLOTS, LOG_D)), D * D);
        }
    }

    #[test]
    fn half_maps_split_the_row_and_column_rotations() {
        let x = input(11);
        let add = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> { a.iter().zip(&b).map(|(u, v)| u + v).collect() };
        for k in 1..D {
            let upper = apply(&rows_upper(SLOTS, LOG_D, k), &x);
            let lower = apply(&rows_lower(SLOTS, LOG_D, k), &x);
            let pick = |m: &[f64], keep: bool, r2: usize, c2: usize| if keep { m[r2 * D + c2] } else { 0.0 };
            assert_eq!(upper, per_matrix(&x, |m, r, c| pick(m, r + k < D, (r + k) % D, c)));
            assert_eq!(lower, per_matrix(&x, |m, r, c| pick(m, r + k >= D, (r + k) % D, c)));
            assert_eq!(add(upper, lower), apply(&rotate_rows(SLOTS, LOG_D, k), &x));

            let left = apply(&cols_left(SLOTS, LOG_D, k), &x);
            let right = apply(&cols_right(SLOTS, LOG_D, k), &x);
            assert_eq!(left, per_matrix(&x, |m, r, c| pick(m, c + k < D, r, (c + k) % D)));
            assert_eq!(right, per_matrix(&x, |m, r, c| pick(m, c + k >= D, r, (c + k) % D)));
            assert_eq!(add(left, right), apply(&rotate_cols(SLOTS, LOG_D, k), &x));
        }
    }

    #[test]
    fn jkls_identity_holds_in_the_clear() {
        let (a, b) = (input(6), input(7));
        let sa = apply(&sigma(SLOTS, LOG_D), &a);
        let tb = apply(&tau(SLOTS, LOG_D), &b);
        let mut acc = vec![0.0; SLOTS];
        for k in 0..D {
            let pa = apply(&rotate_cols(SLOTS, LOG_D, k), &sa);
            let pb = apply(&rotate_rows(SLOTS, LOG_D, k), &tb);
            acc.iter_mut().zip(pa.iter().zip(&pb)).for_each(|(o, (x, y))| *o += x * y);
        }
        let want: Vec<f64> = (0..SLOTS)
            .map(|t| {
                let (s, r, c) = (t / (D * D), (t / D) % D, t % D);
                (0..D).map(|l| a[s * D * D + r * D + l] * b[s * D * D + l * D + c]).sum()
            })
            .collect();
        assert_eq!(acc, want);
    }

    #[test]
    fn one_matrix_slot_merges_wrapping_offsets() {
        let map = rotate_rows::<f64>(D * D, LOG_D, 1);
        assert_eq!(map.indexes().len(), 1);
        let x = &input(8)[..D * D];
        let want: Vec<f64> = (0..D * D).map(|t| x[((t / D + 1) % D) * D + t % D]).collect();
        assert_eq!(apply(&map, x), want);
    }

    /// On a replicated input, every matrix slot holding the same matrix, `tau_rep` is tau and `rotate_rows_rep(k)` is
    /// psi^k; on distinct matrix slots they differ. With one matrix filling the slots, `tau_rep` is tau.
    #[test]
    fn rep_maps_match_tau_and_psi_on_replicated_inputs() {
        let replicated: Vec<f64> = input(15)[..D * D].iter().copied().cycle().take(SLOTS).collect();
        let distinct = input(16);
        assert_eq!(
            apply(&tau_rep(SLOTS, LOG_D), &replicated),
            apply(&tau(SLOTS, LOG_D), &replicated)
        );
        assert_ne!(apply(&tau_rep(SLOTS, LOG_D), &distinct), apply(&tau(SLOTS, LOG_D), &distinct));
        for k in 1..D {
            let (rep, psi) = (rotate_rows_rep(SLOTS, LOG_D, k), rotate_rows(SLOTS, LOG_D, k));
            assert_eq!(apply(&rep, &replicated), apply(&psi, &replicated), "k={k}");
            assert_ne!(apply(&rep, &distinct), apply(&psi, &distinct), "k={k}");
        }
        assert_eq!(tau_rep::<f64>(D * D, LOG_D).indexes(), tau::<f64>(D * D, LOG_D).indexes());
    }

    /// On a replicated input, `left_mul_rep` is `left_mul` of one public matrix; on distinct matrix slots they differ.
    /// With one matrix filling the slots they are the same map, at the same offsets.
    #[test]
    fn left_mul_rep_matches_left_mul_on_replicated_inputs() {
        let replicated: Vec<f64> = input(17)[..D * D].iter().copied().cycle().take(SLOTS).collect();
        let distinct = input(18);
        let b = |m: usize, r: usize, l: usize| public(m, r, l);
        let (rep, full) = (left_mul_rep(SLOTS, LOG_D, b), left_mul(SLOTS, LOG_D, 1, b));
        assert_eq!(apply(&rep, &replicated), apply(&full, &replicated));
        assert_ne!(apply(&rep, &distinct), apply(&full, &distinct));
        let offsets: Vec<i64> = (0..D as i64).map(|j| j * D as i64).collect();
        assert_eq!(rep.indexes(), offsets);
        for &i in &offsets {
            assert_eq!(rep.period(i), Some(D * D), "offset={i}");
        }
        let (rep, full) = (left_mul_rep(D * D, LOG_D, b), left_mul(D * D, LOG_D, 1, b));
        assert_eq!(rep.indexes(), full.indexes());
        let x = &input(19)[..D * D];
        assert_eq!(apply(&rep, x), apply(&full, x));
    }
}
