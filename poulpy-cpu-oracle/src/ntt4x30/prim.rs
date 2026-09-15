//! Trait implementations for [`NTT4x30Oracle`] — primitive NTT-domain operations.
//!
//! Implements all `Ntt*` traits from [`crate::reference::ntt4x30`] for
//! [`NTT4x30Oracle`], delegating to the `*_ref` scalar functions.
//!
//! This mirrors `poulpy_cpu_oracle::fft64::reim` for the FFT64 backend.

use crate::reference::ntt4x30::{
    NttAdd, NttAddAssign, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateAssign, NttSub, NttSubAssign,
    NttSubNegateAssign, NttToZnx128, NttZero,
    arithmetic::{b_from_znx64_ref, b_to_znx128_ref},
    ntt::{NttTable, NttTableInv, intt_ref, ntt_ref},
    primes::{PrimeSet, Primes30},
};

use crate::NTT4x30Oracle;

impl NttDFTExecute<NttTable<Primes30>> for NTT4x30Oracle {
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        ntt_ref::<Primes30>(table, data);
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT4x30Oracle {
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        intt_ref::<Primes30>(table, data);
    }
}

impl NttFromZnx64 for NTT4x30Oracle {
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
        b_from_znx64_ref::<Primes30>(a.len(), res, a);
    }
}

impl NttToZnx128 for NTT4x30Oracle {
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        b_to_znx128_ref::<Primes30>(divisor_is_n, res, a);
    }
}

impl NttAdd for NTT4x30Oracle {
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        for (i, r) in res.iter_mut().enumerate() {
            let q = Primes30::Q[i % 4] as u64;
            *r = (a[i] % q + b[i] % q) % q;
        }
    }
}

impl NttAddAssign for NTT4x30Oracle {
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Primes30::Q.map(u64::from).iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = (res[idx] % q_s + a[idx] % q_s) % q_s;
            }
        }
    }
}

impl NttSub for NTT4x30Oracle {
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Primes30::Q.map(u64::from).iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = (a[idx] % q_s + (q_s - b[idx] % q_s)) % q_s;
            }
        }
    }
}

impl NttSubAssign for NTT4x30Oracle {
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Primes30::Q.map(u64::from).iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = (res[idx] % q_s + (q_s - a[idx] % q_s)) % q_s;
            }
        }
    }
}

impl NttSubNegateAssign for NTT4x30Oracle {
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Primes30::Q.map(u64::from).iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = (a[idx] % q_s + (q_s - res[idx] % q_s)) % q_s;
            }
        }
    }
}

impl NttNegate for NTT4x30Oracle {
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Primes30::Q.map(u64::from).iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = (q_s - a[idx] % q_s) % q_s;
            }
        }
    }
}

impl NttNegateAssign for NTT4x30Oracle {
    fn ntt_negate_assign(res: &mut [u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Primes30::Q.map(u64::from).iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = (q_s - res[idx] % q_s) % q_s;
            }
        }
    }
}

impl NttZero for NTT4x30Oracle {
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT4x30Oracle {
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}
