//! Host index helpers between whole matrices and grids of `d x d` blocks.

/// One slot vector per block (row-major block order), each `slots` long: lane `s` holds block `(i, j)` of matrix
/// `s mod count`, read as `entry(s mod count, row, col)` with `row < rows * d`, `col < cols * d`; `count` divides
/// `slots / d^2`.
///
/// For a batch of `count` matrices, pass `slots = count * d^2` and encrypt each vector from a plaintext of that many
/// slots, as [`crate::layouts::BlasCiphertext::from_blocks`] describes; vectors of every slot encrypt the full batch.
///
/// # Panics
///
/// Panics if `count` matrices of `d x d` blocks do not tile `slots` slots, or if `rows * cols` overflows `usize`.
pub fn split_into_blocks<F: Copy>(
    rows: usize,
    cols: usize,
    log_d: usize,
    slots: usize,
    count: usize,
    entry: impl Fn(usize, usize, usize) -> F,
) -> Vec<Vec<F>> {
    let d = 1usize << log_d;
    assert_tiles(slots, d, count);
    (0..rows.checked_mul(cols).expect("rows * cols overflows usize"))
        .map(|b| {
            let (i, j) = (b / cols, b % cols);
            (0..slots)
                .map(|t| entry((t / (d * d)) % count, i * d + (t / d) % d, j * d + t % d))
                .collect()
        })
        .collect()
}

/// Panics unless `count` matrices of `d x d` blocks tile `slots` slots.
pub(crate) fn assert_tiles(slots: usize, d: usize, count: usize) {
    assert!(
        slots.is_multiple_of(d * d) && count > 0 && (slots / (d * d)).is_multiple_of(count),
        "{count} matrices do not tile {slots} slots of {d} x {d} blocks"
    );
}

/// Inverse of [`split_into_blocks`]: calls `write(s, row, col, value)` for the first `count` lanes.
pub fn join_blocks<F: Copy>(
    blocks: &[Vec<F>],
    rows: usize,
    cols: usize,
    log_d: usize,
    count: usize,
    mut write: impl FnMut(usize, usize, usize, F),
) {
    let d = 1usize << log_d;
    assert!(
        rows.checked_mul(cols) == Some(blocks.len()) && count > 0 && blocks.iter().all(|block| count * d * d <= block.len()),
        "a {rows} x {cols} grid of {count} matrices takes {rows} x {cols} blocks of at least {} slots",
        count * d * d
    );
    for (b, block) in blocks.iter().enumerate() {
        let (i, j) = (b / cols, b % cols);
        for (t, value) in block[..count * d * d].iter().enumerate() {
            write(t / (d * d), i * d + (t / d) % d, j * d + t % d, *value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::maps;
    use poulpy_core::layouts::{Diagonals, Evaluate, LinearTransformationStrategy};

    const LOG_D: usize = 2;
    const D: usize = 1 << LOG_D;
    const LANES: usize = 4;
    const SLOTS: usize = LANES * D * D;

    /// Lane `s`, entry `(r, c)` of a test matrix with `cols` columns.
    fn entry(seed: usize, cols: usize) -> impl Fn(usize, usize, usize) -> f64 {
        move |s, r, c| (((s * 31 + r) * cols + c + seed) % 23) as f64 - 11.0
    }

    fn apply(map: &Diagonals<f64>, x: &[f64]) -> Vec<f64> {
        map.evaluate(x, LinearTransformationStrategy::Direct)
    }

    fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// Every entry of the first `LANES` matrices written by `join_blocks`.
    fn join(blocks: &[Vec<f64>], rows: usize, cols: usize) -> Vec<f64> {
        let (w, h) = (cols * D, rows * D);
        let mut out = vec![f64::NAN; LANES * h * w];
        join_blocks(blocks, rows, cols, LOG_D, LANES, |s, r, c, v| out[(s * h + r) * w + c] = v);
        out
    }

    fn whole(rows: usize, cols: usize, f: impl Fn(usize, usize, usize) -> f64) -> Vec<f64> {
        let (w, h) = (cols * D, rows * D);
        (0..LANES * h * w).map(|t| f(t / (h * w), (t / w) % h, t % w)).collect()
    }

    #[test]
    fn split_and_join_round_trip() {
        let (rows, cols) = (2, 3);
        let a = entry(1, cols * D);
        let blocks = split_into_blocks(rows, cols, LOG_D, SLOTS, LANES, &a);
        assert_eq!(blocks.len(), rows * cols);
        assert!(blocks.iter().all(|b| b.len() == SLOTS));
        assert_eq!(join(&blocks, rows, cols), whole(rows, cols, &a));
    }

    #[test]
    fn fewer_matrices_repeat_across_lanes() {
        let (rows, cols, count) = (1, 2, 2);
        let a = entry(2, cols * D);
        let blocks = split_into_blocks(rows, cols, LOG_D, SLOTS, count, &a);
        assert_eq!(join(&blocks, rows, cols), whole(rows, cols, |s, r, c| a(s % count, r, c)));
    }

    #[test]
    fn grid_rotations_compose_from_half_maps() {
        let (rows, cols) = (2, 3);
        let a = entry(3, cols * D);
        let blocks = split_into_blocks(rows, cols, LOG_D, SLOTS, LANES, &a);
        let at = |i: usize, j: usize| &blocks[(i % rows) * cols + j % cols];
        for k in [0, 1, D, D + 3, 2 * D - 1] {
            let (q, k1) = (k / D, k % D);
            let rotated: Vec<Vec<f64>> = (0..rows * cols)
                .map(|b| {
                    let (i, j) = (b / cols, b % cols);
                    if k1 == 0 {
                        return at(i + q, j).clone();
                    }
                    add(
                        &apply(&maps::rows_upper(SLOTS, LOG_D, k1), at(i + q, j)),
                        &apply(&maps::rows_lower(SLOTS, LOG_D, k1), at(i + q + 1, j)),
                    )
                })
                .collect();
            let want = whole(rows, cols, |s, r, c| a(s, (r + k) % (rows * D), c));
            assert_eq!(join(&rotated, rows, cols), want, "rows k={k}");
        }
        for k in [0, 2, D, 2 * D + 1, 3 * D - 1] {
            let (q, k1) = (k / D, k % D);
            let rotated: Vec<Vec<f64>> = (0..rows * cols)
                .map(|b| {
                    let (i, j) = (b / cols, b % cols);
                    if k1 == 0 {
                        return at(i, j + q).clone();
                    }
                    add(
                        &apply(&maps::cols_left(SLOTS, LOG_D, k1), at(i, j + q)),
                        &apply(&maps::cols_right(SLOTS, LOG_D, k1), at(i, j + q + 1)),
                    )
                })
                .collect();
            let want = whole(rows, cols, |s, r, c| a(s, r, (c + k) % (cols * D)));
            assert_eq!(join(&rotated, rows, cols), want, "cols k={k}");
        }
    }

    #[test]
    fn block_jkls_product_holds_in_the_clear() {
        let (m, inner, n) = (2, 3, 2);
        let (a, b) = (entry(4, inner * D), entry(5, n * D));
        let blocks_a = split_into_blocks(m, inner, LOG_D, SLOTS, LANES, &a);
        let blocks_b = split_into_blocks(inner, n, LOG_D, SLOTS, LANES, &b);
        let mut acc = vec![vec![0.0; SLOTS]; m * n];
        for l in 0..inner {
            let s: Vec<Vec<f64>> = (0..m)
                .map(|i| apply(&maps::sigma(SLOTS, LOG_D), &blocks_a[i * inner + l]))
                .collect();
            let t: Vec<Vec<f64>> = (0..n)
                .map(|j| apply(&maps::tau(SLOTS, LOG_D), &blocks_b[l * n + j]))
                .collect();
            for k in 0..D {
                let p: Vec<Vec<f64>> = s.iter().map(|x| apply(&maps::rotate_cols(SLOTS, LOG_D, k), x)).collect();
                let q: Vec<Vec<f64>> = t.iter().map(|x| apply(&maps::rotate_rows(SLOTS, LOG_D, k), x)).collect();
                for (ij, out) in acc.iter_mut().enumerate() {
                    let (pi, qj) = (&p[ij / n], &q[ij % n]);
                    out.iter_mut().zip(pi.iter().zip(qj)).for_each(|(o, (x, y))| *o += x * y);
                }
            }
        }
        let want = whole(m, n, |s, r, c| (0..inner * D).map(|l| a(s, r, l) * b(s, l, c)).sum());
        assert_eq!(join(&acc, m, n), want);
    }
}
