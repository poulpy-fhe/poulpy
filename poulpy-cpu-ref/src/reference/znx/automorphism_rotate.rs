/// Fused automorphism + rotation: computes `res = X^k * auto(p, a)`.
///
/// The automorphism `X^i -> X^ip` and the rotation `* X^k` compose into a single
/// signed permutation on `Z_2n`: coefficient `a[i]` lands at position
/// `(i*p + k) mod 2n`, negated when it wraps past `n`. `p` must be odd
/// (invertible mod `2n`), in which case the map writes every output position
/// exactly once (no pre-zeroing of `res` required), exactly like
/// [`znx_automorphism_ref`](super::znx_automorphism_ref).
pub fn znx_automorphism_rotate_ref(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    let n: usize = res.len();
    if n == 0 {
        return;
    }

    let mask: usize = 2 * n - 1;
    let p_2n: usize = (p & mask as i64) as usize;

    // Starting position for i = 0 is the rotation offset (mod 2n).
    let mut pos: usize = (k & mask as i64) as usize;
    if pos < n {
        res[pos] = a[0]
    } else {
        res[pos - n] = -a[0]
    }

    for ai in a.iter().take(n).skip(1) {
        pos = (pos + p_2n) & mask;
        if pos < n { res[pos] = *ai } else { res[pos - n] = -*ai }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::znx::{ZnxRef, automorphism::znx_automorphism_ref, znx_rotate};

    /// The fused kernel must equal an automorphism by `p` followed by a rotation
    /// by `k` (two separate passes).
    #[test]
    fn fused_matches_compose() {
        let a: Vec<i64> = (1..=32).collect();
        let n: usize = a.len();

        for p in [1i64, 3, 5, -5, 7, -11, 13, -17] {
            for k in [0i64, 1, 3, 8, 31, -1, -7, 64, -40] {
                let mut fused: Vec<i64> = vec![0; n];
                znx_automorphism_rotate_ref(p, k, &mut fused, &a);

                let mut auto: Vec<i64> = vec![0; n];
                znx_automorphism_ref(p, &mut auto, &a);
                let mut compose: Vec<i64> = vec![0; n];
                znx_rotate::<ZnxRef>(k, &mut compose, &auto);

                assert_eq!(fused, compose, "mismatch for p={p}, k={k}");
            }
        }
    }
}
