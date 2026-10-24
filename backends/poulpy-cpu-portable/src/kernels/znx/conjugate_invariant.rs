//! Conjugate-invariant coefficient arithmetic.

use rand_distr::num_traits::ops::wrapping::WrappingNeg;

/// `X -> X^p`: the standard automorphism of the ambient degree-`2n` vector
/// `[a_0, .., a_{n-1}, 0, -a_{n-1}, .., -a_1]`, keeping its first `n` coefficients.
pub fn znx_automorphism_portable<T: Copy + WrappingNeg>(p: i64, res: &mut [T], a: &[T]) {
    let n = a.len();
    assert_eq!(res.len(), n);
    assert!(n.is_power_of_two());
    assert!(p & 1 == 1, "p must be odd, got {p}");
    let (n2, mask) = (2 * n, 4 * n - 1);
    let p_4n = p as usize & mask;
    res[0] = a[0];
    let mut k = 0;
    for j in 1..n2 {
        k = (k + p_4n) & mask;
        let (target, negate) = if k < n2 { (k, j > n) } else { (k - n2, j < n) };
        if target < n && j != n {
            let value = if j < n { a[j] } else { a[n2 - j] };
            res[target] = if negate { value.wrapping_neg() } else { value };
        }
    }
}
