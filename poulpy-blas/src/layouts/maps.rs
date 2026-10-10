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
    // One period for every diagonal: prepared maps take the plaintext degree of their first diagonal.
    let period = acc.values().map(|values| min_period(values)).max().unwrap_or(1);
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

/// Transpose: output `(r, c)` reads `(c, r)`.
pub fn transpose<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize) -> Diagonals<F> {
    per_matrix_map(slots, log_d, |r, c| [(c, r, F::one())])
}

/// JKLS psi^k: output `(r, c)` reads `(r + k mod d, c)`.
pub fn rotate_rows<F: DiagonalArithmetic + One + PartialEq>(slots: usize, log_d: usize, k: usize) -> Diagonals<F> {
    let d = 1usize << log_d;
    per_matrix_map(slots, log_d, |r, c| [((r + k) % d, c, F::one())])
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
pub fn right_mul<F: DiagonalArithmetic + PartialEq>(
    slots: usize,
    log_d: usize,
    count: usize,
    b: impl Fn(usize, usize, usize) -> F,
) -> Diagonals<F> {
    let d = 1usize << log_d;
    let b = &b;
    per_lane_map(slots, log_d, count, |s, r, c| (0..d).map(move |l| (r, l, b(s, l, c))))
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
}
