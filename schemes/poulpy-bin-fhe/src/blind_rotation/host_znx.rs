//! Host-side scalar helpers for building lookup tables. Plain `i64` slices,
//! negacyclic convention `X^n = -1`, degree `n = res.len()` a power of two.

/// `res = src * X^p` in `Z[X]/(X^n + 1)`.
pub(crate) fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
    let n = res.len();
    debug_assert_eq!(n, src.len());
    debug_assert!(n.is_power_of_two());
    let mp_2n = (p & (2 * n as i64 - 1)) as usize; // -p mod 2n
    let mp_1n = mp_2n & (n - 1); // -p mod n
    let split = n - mp_1n;
    let neg_first = mp_2n < n;
    let (dst1, dst2) = res.split_at_mut(mp_1n);
    let (src1, src2) = src.split_at(split);
    if neg_first {
        dst1.iter_mut().zip(src2).for_each(|(d, s)| *d = -*s);
        dst2.copy_from_slice(src1);
    } else {
        dst1.copy_from_slice(src2);
        dst2.iter_mut().zip(src1).for_each(|(d, s)| *d = -*s);
    }
}

/// Ring switch by subsampling (`n_in > n_out`) or interleaved zero-insertion
/// (`n_in < n_out`); the two degrees must be powers of two dividing each other.
pub(crate) fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
    let (n_in, n_out) = (a.len(), res.len());
    debug_assert!(n_in.is_power_of_two() && n_out.is_power_of_two());
    if n_in == n_out {
        res.copy_from_slice(a);
        return;
    }
    let (gap_in, gap_out) = if n_in > n_out { (n_in / n_out, 1) } else { (1, n_out / n_in) };
    if n_in < n_out {
        res.fill(0);
    }
    res.iter_mut()
        .step_by(gap_out)
        .zip(a.iter().step_by(gap_in))
        .for_each(|(x, y)| *x = *y);
}

#[cfg(test)]
mod tests {
    use super::{znx_rotate, znx_switch_ring};

    #[test]
    fn rotate_by_one_is_negacyclic_shift() {
        let src = [1i64, 2, 3, 4];
        let mut res = [0i64; 4];
        znx_rotate(1, &mut res, &src);
        assert_eq!(res, [-4, 1, 2, 3]);
        znx_rotate(-1, &mut res, &src);
        assert_eq!(res, [2, 3, 4, -1]);
    }

    #[test]
    fn switch_ring_down_keeps_multiples_and_up_interleaves_zeros() {
        let mut down = [0i64; 2];
        znx_switch_ring(&mut down, &[1, 2, 3, 4]);
        assert_eq!(down, [1, 3]);
        let mut up = [9i64; 4];
        znx_switch_ring(&mut up, &[5, 6]);
        assert_eq!(up, [5, 0, 6, 0]);
    }
}
