/// Changes between invariant coefficients and a negacyclic NTT basis.
pub struct ConjugateInvariantNtt {
    factors: Vec<[u64; 2]>,
    n: usize,
}

impl ConjugateInvariantNtt {
    /// Direct and reflected coefficient multipliers, indexed by coefficient.
    pub fn factors(&self) -> &[[u64; 2]] {
        &self.factors
    }

    /// Uses a primitive `2^(max_log_n + 1)`-th root for the supplied prime.
    pub fn new(n: usize, q: u64, omega: u64, max_log_n: u32, inverse: bool) -> Self {
        assert!(n.is_power_of_two() && n <= (1usize << (max_log_n - 1)));
        let mul = |a: u64, b: u64| ((a as u128 * b as u128) % q as u128) as u64;
        let pow = |mut a: u64, mut e: u64| {
            let mut r = 1;
            while e != 0 {
                if e & 1 != 0 {
                    r = mul(r, a);
                }
                a = mul(a, a);
                e >>= 1;
            }
            r
        };
        let alpha = pow(omega, (1u64 << (max_log_n - 1)) / n as u64);
        let alpha_inv = pow(alpha, q - 2);
        // At x^n = i, the invariant basis becomes c[j] - i*c[n-j].
        let i = pow(alpha, n as u64);
        let half = q.div_ceil(2);
        let mut forward = 1;
        let mut backward = 1;
        let mut factors = vec![[1, 0]; n];
        for factors in factors.iter_mut().skip(1) {
            forward = mul(forward, alpha);
            backward = mul(backward, alpha_inv);
            *factors = if inverse {
                [mul(forward, half), q - mul(backward, half)]
            } else {
                [backward, q - mul(i, backward)]
            };
        }
        Self { factors, n }
    }

    /// Applies the basis change to one prime lane with the given coefficient stride.
    pub fn apply(&self, data: &mut [u64], stride: usize, q: u64) {
        let apply = |j: usize, a: u64, b: u64| {
            let [d, c] = self.factors[j];
            ((d as u128 * (a % q) as u128 + c as u128 * (b % q) as u128) % q as u128) as u64
        };
        data[0] %= q;
        for j in 1..=self.n / 2 {
            let other = self.n - j;
            let a = data[j * stride];
            let b = data[other * stride];
            data[j * stride] = apply(j, a, b);
            data[other * stride] = apply(other, b, a);
        }
    }
}

/// Permutes bit-reversed evaluations at the ambient roots with exponent 1 modulo 4.
pub fn ntt_automorphism_permutation(n: usize, p: i64) -> Vec<usize> {
    assert!(n.is_power_of_two());
    assert!(p & 1 == 1, "automorphism exponent must be odd");
    let bits = n.ilog2();
    let reverse = |x: usize| {
        if bits == 0 {
            0
        } else {
            x.reverse_bits() >> (usize::BITS - bits)
        }
    };
    let mask = 4 * n - 1;
    let p = p as usize & mask;
    (0..n)
        .map(|j| {
            let mut exponent = (p * (4 * reverse(j) + 1)) & mask;
            if exponent & 3 == 3 {
                exponent = 4 * n - exponent;
            }
            reverse((exponent - 1) / 4)
        })
        .collect()
}
