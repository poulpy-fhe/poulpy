//! Direct evaluation oracles for negacyclic transform tables over `Z/q`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use dashu_int::{UBig, fast_div::ConstDivisor, ops::BitTest};

use crate::{api::ZqNew, bit_reverse};

/// `(n, q, psi)` cases: `q = p^e` for a prime `p = 1 mod 2n` and `psi` a primitive
/// `2n`-th root modulo `q`, from 17 bits up to 186 bits, degrees 1 to 1024.
fn cases() -> Vec<(usize, UBig, UBig)> {
    const F4: u64 = 65537;
    const P57: u64 = 29 * (1 << 57) + 1;
    [
        (0, F4, 1),
        (1, F4, 2),
        (4, F4, 1),
        (6, F4, 3),
        (10, F4, 2),
        (8, P57, 1),
        (5, P57, 3),
    ]
    .into_iter()
    .map(|(log_n, p, e)| {
        let (q, psi) = prime_power_root(log_n, &UBig::from(p), e);
        (1 << log_n, q, psi)
    })
    .collect()
}

/// `p^e` and a primitive `2^(log_n + 1)`-th root of unity modulo it, for a prime
/// `p = 1 mod 2^(log_n + 1)`: `a^((p - 1) / 2n)` for the first non-residue `a`, lifted to
/// `p^e` as its `p^(e - 1)`-th power.
fn prime_power_root(log_n: usize, p: &UBig, e: usize) -> (UBig, UBig) {
    let ring = ConstDivisor::new(p.clone());
    let exponent: UBig = (p - UBig::ONE) >> (log_n + 1);
    let minus_one = ring.reduce(p - UBig::ONE);
    let root: UBig = (2u32..)
        .map(|a| ring.reduce(a).pow(&exponent))
        .find(|w| w.pow(&(UBig::ONE << log_n)) == minus_one)
        .expect("a prime has a quadratic non-residue")
        .residue();
    let q: UBig = p.pow(e);
    let psi: UBig = ConstDivisor::new(q.clone()).reduce(root).pow(&p.pow(e - 1)).residue();
    (q, psi)
}

/// Deterministic residues in `[0, q)`.
fn residues(n: usize, seed: u64, q: &UBig) -> Vec<UBig> {
    let mut state: u64 = seed;
    (0..n)
        .map(|_| {
            let words: usize = q.bit_len().div_ceil(64) + 1;
            let value: UBig = (0..words).fold(UBig::ZERO, |acc, _| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (acc << 64) + UBig::from(state)
            });
            value % q
        })
        .collect()
}

/// The definition of `zq_ntt`, by Horner evaluation at `psi^(2 rev_n(r) + 1)`.
fn direct_ntt(a: &[UBig], q: &UBig, psi: &UBig) -> Vec<UBig> {
    let ring = ConstDivisor::new(q.clone());
    let n: usize = a.len();
    (0..n)
        .map(|r| {
            let x = ring.reduce(psi.clone()).pow(&UBig::from(2 * bit_reverse(r, n) + 1));
            a.iter()
                .rev()
                .fold(ring.reduce(0u8), |acc, c| acc * &x + ring.reduce(c.clone()))
                .residue()
        })
        .collect()
}

/// `a * b` in `(Z/q)[X]/(X^n + 1)`.
fn negacyclic_product(a: &[UBig], b: &[UBig], q: &UBig) -> Vec<UBig> {
    let n: usize = a.len();
    let mut res: Vec<UBig> = vec![UBig::ZERO; n];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let term: UBig = (x * y) % q;
            let k: usize = (i + j) % n;
            res[k] = if i + j < n {
                (&res[k] + term) % q
            } else {
                (&res[k] + q - term) % q
            };
        }
    }
    res
}

/// Checks the forward transform against its definition, the inverse as its exact inverse,
/// and the negacyclic convolution property, from 17-bit to 186-bit moduli.
pub fn test_zq_ntt<T: ZqNew<UBig>>() {
    for (n, q, psi) in cases() {
        let table = T::new(n, &q, &psi);
        assert_eq!(table.n(), n);
        let a: Vec<UBig> = residues(n, 1, &q);
        let b: Vec<UBig> = residues(n, 2, &q);

        let mut a_hat: Vec<UBig> = a.clone();
        table.zq_ntt(&mut a_hat);
        assert_eq!(a_hat, direct_ntt(&a, &q, &psi), "n={n}, q={q}: forward");
        let mut roundtrip: Vec<UBig> = a_hat.clone();
        table.zq_intt(&mut roundtrip);
        assert_eq!(roundtrip, a, "n={n}, q={q}: inverse");

        let mut b_hat: Vec<UBig> = b.clone();
        table.zq_ntt(&mut b_hat);
        let mut product: Vec<UBig> = a_hat.iter().zip(&b_hat).map(|(x, y)| (x * y) % &q).collect();
        table.zq_intt(&mut product);
        assert_eq!(product, negacyclic_product(&a, &b, &q), "n={n}, q={q}: convolution");
    }
}

/// Checks that the tested implementation `BT` returns the residues of the reference `BR`,
/// forward and inverse, on the cases of [`test_zq_ntt`] and on edge inputs: zeros, `q - 1`
/// everywhere, an impulse.
pub fn test_zq_ntt_parity<BR: ZqNew<UBig>, BT: ZqNew<UBig>>() {
    for (n, q, psi) in cases() {
        let (table_ref, table_test) = (BR::new(n, &q, &psi), BT::new(n, &q, &psi));
        let mut impulse: Vec<UBig> = vec![UBig::ZERO; n];
        impulse[n / 2] = UBig::ONE;
        let inputs: [Vec<UBig>; 4] = [vec![UBig::ZERO; n], vec![&q - UBig::ONE; n], impulse, residues(n, 3, &q)];
        for (sample, input) in inputs.iter().enumerate() {
            let (mut want, mut have) = (input.clone(), input.clone());
            table_ref.zq_ntt(&mut want);
            table_test.zq_ntt(&mut have);
            assert_eq!(have, want, "n={n}, q={q}, sample={sample}: forward");
            let (mut want, mut have) = (input.clone(), input.clone());
            table_ref.zq_intt(&mut want);
            table_test.zq_intt(&mut have);
            assert_eq!(have, want, "n={n}, q={q}, sample={sample}: inverse");
        }
    }
}

/// Checks that both transforms refuse a slice whose length is not the table degree and leave
/// it unchanged: empty, a smaller power of two, one element short or over, twice the degree.
pub fn test_zq_ntt_rejects_wrong_length<T: ZqNew<UBig>>() {
    let (q, psi) = prime_power_root(4, &UBig::from(65537u32), 1);
    let table = T::new(16, &q, &psi);
    for len in [0, 8, 15, 17, 32] {
        let original: Vec<UBig> = residues(len, 4, &q);
        let mut data: Vec<UBig> = original.clone();
        assert!(
            catch_unwind(AssertUnwindSafe(|| table.zq_ntt(&mut data))).is_err(),
            "length {len}: forward"
        );
        assert_eq!(data, original, "length {len}: forward wrote");
        assert!(
            catch_unwind(AssertUnwindSafe(|| table.zq_intt(&mut data))).is_err(),
            "length {len}: inverse"
        );
        assert_eq!(data, original, "length {len}: inverse wrote");
    }
}

/// Checks that construction panics outside the domain of [`ZqNew::new`]: a degree that is
/// not a positive power of two, an even modulus or one below three, and a root whose
/// `n`-th power is not `-1`.
pub fn test_zq_new_rejects<T: ZqNew<UBig>>() {
    let (q, psi) = prime_power_root(4, &UBig::from(65537u32), 1);
    let rejects = |n: usize, q: UBig, psi: UBig| std::panic::catch_unwind(move || T::new(n, &q, &psi)).is_err();
    assert!(!rejects(16, q.clone(), psi.clone()), "valid parameters");
    assert!(rejects(0, q.clone(), psi.clone()), "zero degree");
    assert!(rejects(12, q.clone(), psi.clone()), "degree not a power of two");
    assert!(rejects(16, &q + UBig::ONE, psi.clone()), "even modulus");
    assert!(rejects(16, UBig::ONE, UBig::ZERO), "modulus one");
    assert!(rejects(16, q.clone(), (&psi * &psi) % &q), "root of order 16");
    assert!(rejects(32, q, psi), "root of order 32 at degree 32");
}
