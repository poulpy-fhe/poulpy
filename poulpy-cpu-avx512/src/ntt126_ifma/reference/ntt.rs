//! Scalar reference NTT execution for the 3-prime IFMA backend.
//!
//! Pure-Rust mirror of the AVX-512-IFMA forward and inverse NTT kernels in
//! [`super::super::kernels`].  Used by the AVX-512 unit tests as the
//! correctness oracle for `ntt_avx512` / `intt_avx512`.
//!
//! See [`super::super::tables`] for the table layout this implementation
//! consumes — the IFMA-native lazy `[0, 4q)` arithmetic, the SoA twiddle
//! layout, and the `cond_sub_2q` / `harvey_modmul` helpers all live there.

use super::super::primes::PrimeSetNtt126Ifma;
use super::super::tables::{Ntt126IfmaTable, Ntt126IfmaTableInv, cond_sub_2q, harvey_modmul};

/// Forward NTT — scalar reference with IFMA-native lazy arithmetic.
///
/// Butterfly values live in `[0, 4q)`.  Diff path feeds directly into Harvey
/// without a pre-reduction; sum path keeps one `cond_sub(·, 4q)`.  A final
/// `cond_sub(·, 2q)` pass renormalises the output to `[0, 2q)` so downstream
/// consumers see the usual range.
pub fn ntt126_ifma_ref<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n <= 1 {
        return;
    }

    let q2 = &table.q2;
    let q4 = &table.q4;
    let mut seg_base = 0usize; // base offset (in u64) for current segment

    // ── Level 0: a[i] *= ω^i (Harvey multiply) ──────────────────────
    {
        let omega_base = seg_base;
        let quot_base = seg_base + 4 * n;
        for i in 0..n {
            for k in 0..3 {
                let a = data[4 * i + k];
                let omega = table.powomega[omega_base + 4 * i + k];
                let omega_quot = table.powomega[quot_base + 4 * i + k];
                data[4 * i + k] = harvey_modmul(a, omega, omega_quot, P::Q[k]);
            }
        }
        seg_base += 8 * n;
    }

    // ── Butterfly levels: nn = n, n/2, …, 2 (Cooley-Tukey DIT) ──────
    let mut nn = n;
    while nn >= 2 {
        let halfnn = nn / 2;

        if halfnn > 1 {
            let count = halfnn - 1;
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * count;

            let mut block_start = 0usize;
            while block_start < n {
                // i = 0: no twiddle multiply (both sides need cond_sub_4q)
                {
                    let p1 = 4 * block_start;
                    let p2 = 4 * (block_start + halfnn);
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b = data[p2 + k];
                        let sum = a + b;
                        let diff = a + q4[k] - b;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        data[p2 + k] = cond_sub_2q(diff, q4[k]);
                    }
                }

                // i = 1..halfnn-1: Harvey multiply absorbs the diff-path reduction
                for i in 1..halfnn {
                    let p1 = 4 * (block_start + i);
                    let p2 = 4 * (block_start + halfnn + i);
                    let tw_idx = i - 1;
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b = data[p2 + k];
                        let sum = a + b;
                        let diff = a + q4[k] - b;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        let omega = table.powomega[omega_base + 4 * tw_idx + k];
                        let omega_quot = table.powomega[quot_base + 4 * tw_idx + k];
                        data[p2 + k] = harvey_modmul(diff, omega, omega_quot, P::Q[k]);
                    }
                }

                block_start += nn;
            }

            seg_base += 8 * count;
        } else {
            // nn == 2: add/sub only, no twiddle
            let mut block_start = 0usize;
            while block_start < n {
                let p1 = 4 * block_start;
                let p2 = 4 * (block_start + 1);
                for k in 0..3 {
                    let a = data[p1 + k];
                    let b = data[p2 + k];
                    data[p1 + k] = cond_sub_2q(a + b, q4[k]);
                    data[p2 + k] = cond_sub_2q(a + q4[k] - b, q4[k]);
                }
                block_start += 2;
            }
        }

        nn /= 2;
    }

    // ── Final normalisation: [0, 4q) → [0, 2q) ──────────────────────
    for i in 0..n {
        for k in 0..3 {
            data[4 * i + k] = cond_sub_2q(data[4 * i + k], q2[k]);
        }
    }
}

/// Inverse NTT — scalar reference with IFMA-native lazy arithmetic.
///
/// Butterfly values live in `[0, 4q)`.  The final pointwise Harvey pass
/// reduces to `[0, 2q)` automatically, so no explicit renormalisation is
/// needed.
pub fn intt126_ifma_ref<P: PrimeSetNtt126Ifma>(table: &Ntt126IfmaTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n <= 1 {
        return;
    }

    let q4 = &table.q4;
    let mut seg_base = 0usize;

    // ── Butterfly levels: nn = 2, 4, …, n (Gentleman-Sande DIF) ─────
    let mut nn = 2usize;
    while nn <= n {
        let halfnn = nn / 2;

        if halfnn > 1 {
            let count = halfnn - 1;
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * count;

            let mut block_start = 0usize;
            while block_start < n {
                // i = 0: no twiddle
                {
                    let p1 = 4 * block_start;
                    let p2 = 4 * (block_start + halfnn);
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b = data[p2 + k];
                        let sum = a + b;
                        let diff = a + q4[k] - b;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        data[p2 + k] = cond_sub_2q(diff, q4[k]);
                    }
                }

                // i = 1..halfnn-1: twiddle on b BEFORE butterfly (b_raw ∈ [0, 4q)
                // fed directly into Harvey → bo ∈ [0, 2q); sum/diff use cond_sub_4q).
                for i in 1..halfnn {
                    let p1 = 4 * (block_start + i);
                    let p2 = 4 * (block_start + halfnn + i);
                    let tw_idx = i - 1;
                    for k in 0..3 {
                        let a = data[p1 + k];
                        let b_raw = data[p2 + k];
                        let omega = table.powomega[omega_base + 4 * tw_idx + k];
                        let omega_quot = table.powomega[quot_base + 4 * tw_idx + k];
                        let bo = harvey_modmul(b_raw, omega, omega_quot, P::Q[k]);
                        let sum = a + bo;
                        let diff = a + q4[k] - bo;
                        data[p1 + k] = cond_sub_2q(sum, q4[k]);
                        data[p2 + k] = cond_sub_2q(diff, q4[k]);
                    }
                }

                block_start += nn;
            }

            seg_base += 8 * count;
        } else {
            // nn == 2: add/sub only
            let mut block_start = 0usize;
            while block_start < n {
                let p1 = 4 * block_start;
                let p2 = 4 * (block_start + 1);
                for k in 0..3 {
                    let a = data[p1 + k];
                    let b = data[p2 + k];
                    data[p1 + k] = cond_sub_2q(a + b, q4[k]);
                    data[p2 + k] = cond_sub_2q(a + q4[k] - b, q4[k]);
                }
                block_start += 2;
            }
        }

        nn *= 2;
    }

    // ── Last pass: a[i] *= ω^{-i} / n (n entries, input ∈ [0, 4q), output ∈ [0, 2q)) ──
    {
        let omega_base = seg_base;
        let quot_base = seg_base + 4 * n;
        for i in 0..n {
            for k in 0..3 {
                let a = data[4 * i + k];
                let omega = table.powomega[omega_base + 4 * i + k];
                let omega_quot = table.powomega[quot_base + 4 * i + k];
                data[4 * i + k] = harvey_modmul(a, omega, omega_quot, P::Q[k]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::primes::Primes42;
    use super::super::arithmetic::{b_ntt126_ifma_from_znx64_ref, b_ntt126_ifma_to_znx128_ref};
    use super::*;

    #[test]
    fn ntt_intt_identity() {
        for log_n in 1..=10usize {
            let n = 1 << log_n;
            let fwd = Ntt126IfmaTable::<Primes42>::new(n);
            let inv = Ntt126IfmaTableInv::<Primes42>::new(n);

            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data = vec![0u64; 4 * n];
            b_ntt126_ifma_from_znx64_ref(n, &mut data, &coeffs);
            let data_orig = data.clone();

            ntt126_ifma_ref::<Primes42>(&fwd, &mut data);
            intt126_ifma_ref::<Primes42>(&inv, &mut data);

            for i in 0..n {
                for k in 0..3 {
                    let orig = data_orig[4 * i + k] % Primes42::Q[k];
                    let got = data[4 * i + k] % Primes42::Q[k];
                    assert_eq!(orig, got, "n={n} i={i} k={k}: mismatch after NTT+iNTT round-trip");
                }
            }
        }
    }

    #[test]
    fn ntt_convolution() {
        let n = 8usize;
        let fwd = Ntt126IfmaTable::<Primes42>::new(n);
        let inv = Ntt126IfmaTableInv::<Primes42>::new(n);

        let a: Vec<i64> = vec![1, 2, 0, 0, 0, 0, 0, 0];
        let b: Vec<i64> = vec![3, 4, 0, 0, 0, 0, 0, 0];

        let mut da = vec![0u64; 4 * n];
        let mut db = vec![0u64; 4 * n];
        b_ntt126_ifma_from_znx64_ref(n, &mut da, &a);
        b_ntt126_ifma_from_znx64_ref(n, &mut db, &b);

        ntt126_ifma_ref::<Primes42>(&fwd, &mut da);
        ntt126_ifma_ref::<Primes42>(&fwd, &mut db);

        // Pointwise multiply (mod each Q[k])
        let mut dc = vec![0u64; 4 * n];
        for i in 0..n {
            for k in 0..3 {
                let q = Primes42::Q[k];
                dc[4 * i + k] = ((da[4 * i + k] % q) as u128 * (db[4 * i + k] % q) as u128 % q as u128) as u64;
            }
        }

        intt126_ifma_ref::<Primes42>(&inv, &mut dc);

        let mut result = vec![0i128; n];
        b_ntt126_ifma_to_znx128_ref(n, &mut result, &dc);

        let expected: Vec<i128> = vec![3, 10, 8, 0, 0, 0, 0, 0];
        assert_eq!(result, expected, "NTT convolution mismatch");
    }
}
