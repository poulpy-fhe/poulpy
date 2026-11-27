//! Direct evaluation oracles for negacyclic transform tables over `Z/q`.

use std::{
    fmt::Debug,
    panic::{AssertUnwindSafe, catch_unwind},
};

use dashu_int::{UBig, fast_div::ConstDivisor, ops::BitTest};

use crate::{api::ZqNew, bit_reverse};

/// A residue word of a transform table, which the suites convert their cases to and from.
pub trait ZqWord: Clone + PartialEq + Debug {
    /// Whether the word holds every residue modulo `q`.
    fn holds(q: &UBig) -> bool;

    /// `value`, held by the word.
    fn from_ubig(value: &UBig) -> Self;

    fn to_ubig(&self) -> UBig;
}

impl ZqWord for UBig {
    fn holds(_q: &UBig) -> bool {
        true
    }

    fn from_ubig(value: &UBig) -> Self {
        value.clone()
    }

    fn to_ubig(&self) -> UBig {
        self.clone()
    }
}

/// `L` little-endian 64-bit words.
impl<const L: usize> ZqWord for [u64; L] {
    fn holds(q: &UBig) -> bool {
        q.bit_len() <= 64 * L
    }

    fn from_ubig(value: &UBig) -> Self {
        let mut words: [u64; L] = [0; L];
        words[..value.as_words().len()].copy_from_slice(value.as_words());
        words
    }

    fn to_ubig(&self) -> UBig {
        UBig::from_words(self)
    }
}

fn words<W: ZqWord>(values: &[UBig]) -> Vec<W> {
    values.iter().map(W::from_ubig).collect()
}

fn ubigs<W: ZqWord>(values: &[W]) -> Vec<UBig> {
    values.iter().map(W::to_ubig).collect()
}

/// `(n, q, psi)` cases: `q = p^e` for a prime `p = 1 mod 2n` and `psi` a primitive
/// `2n`-th root modulo `q`, from 17 bits up to 192 bits, degrees 1 to 1024. The powers of
/// `G64 = 2^64 - 2^32 + 1` fill whole 64-bit words.
fn cases() -> Vec<(usize, UBig, UBig)> {
    const F4: u64 = 65537;
    const P57: u64 = 29 * (1 << 57) + 1;
    const G64: u64 = 0xFFFF_FFFF_0000_0001;
    [
        (0, F4, 1),
        (1, F4, 2),
        (4, F4, 1),
        (6, F4, 3),
        (10, F4, 2),
        (8, P57, 1),
        (5, P57, 3),
        (6, G64, 1),
        (4, G64, 3),
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

/// Checks the forward transform on residues `W` against its definition, the inverse as its
/// exact inverse, and the negacyclic convolution property, on the moduli from 17 to 192 bits
/// that `W` holds.
pub fn test_zq_ntt<W: ZqWord, T: ZqNew<W>>() {
    let cases: Vec<(usize, UBig, UBig)> = cases().into_iter().filter(|(_, q, _)| W::holds(q)).collect();
    assert!(!cases.is_empty(), "the word holds no modulus of the suite");
    for (n, q, psi) in cases {
        let table = T::new(n, &W::from_ubig(&q), &W::from_ubig(&psi));
        assert_eq!(table.n(), n);
        let a: Vec<UBig> = residues(n, 1, &q);
        let b: Vec<UBig> = residues(n, 2, &q);

        let mut a_hat: Vec<W> = words(&a);
        table.zq_ntt(&mut a_hat);
        let a_hat: Vec<UBig> = ubigs(&a_hat);
        assert_eq!(a_hat, direct_ntt(&a, &q, &psi), "n={n}, q={q}: forward");
        let mut roundtrip: Vec<W> = words(&a_hat);
        table.zq_intt(&mut roundtrip);
        assert_eq!(ubigs(&roundtrip), a, "n={n}, q={q}: inverse");

        let mut b_hat: Vec<W> = words(&b);
        table.zq_ntt(&mut b_hat);
        let products: Vec<UBig> = a_hat.iter().zip(ubigs(&b_hat)).map(|(x, y)| (x * y) % &q).collect();
        let mut product: Vec<W> = words(&products);
        table.zq_intt(&mut product);
        assert_eq!(ubigs(&product), negacyclic_product(&a, &b, &q), "n={n}, q={q}: convolution");
    }
}

/// Checks that the tested implementation `BT`, on residues `WT`, returns the residues of the
/// reference `BR`, on residues `WR`, forward and inverse, on the cases of [`test_zq_ntt`] both
/// words hold and on edge inputs: zeros, `q - 1` everywhere, an impulse.
pub fn test_zq_ntt_parity<WR: ZqWord, BR: ZqNew<WR>, WT: ZqWord, BT: ZqNew<WT>>() {
    let cases: Vec<(usize, UBig, UBig)> = cases().into_iter().filter(|(_, q, _)| WR::holds(q) && WT::holds(q)).collect();
    assert!(!cases.is_empty(), "the words hold no modulus of the suite");
    for (n, q, psi) in cases {
        let table_ref = BR::new(n, &WR::from_ubig(&q), &WR::from_ubig(&psi));
        let table_test = BT::new(n, &WT::from_ubig(&q), &WT::from_ubig(&psi));
        let mut impulse: Vec<UBig> = vec![UBig::ZERO; n];
        impulse[n / 2] = UBig::ONE;
        let inputs: [Vec<UBig>; 4] = [vec![UBig::ZERO; n], vec![&q - UBig::ONE; n], impulse, residues(n, 3, &q)];
        for (sample, input) in inputs.iter().enumerate() {
            let (mut want, mut have): (Vec<WR>, Vec<WT>) = (words(input), words(input));
            table_ref.zq_ntt(&mut want);
            table_test.zq_ntt(&mut have);
            assert_eq!(ubigs(&have), ubigs(&want), "n={n}, q={q}, sample={sample}: forward");
            let (mut want, mut have): (Vec<WR>, Vec<WT>) = (words(input), words(input));
            table_ref.zq_intt(&mut want);
            table_test.zq_intt(&mut have);
            assert_eq!(ubigs(&have), ubigs(&want), "n={n}, q={q}, sample={sample}: inverse");
        }
    }
}

/// Checks that both transforms refuse a slice whose length is not the table degree and leave
/// it unchanged: empty, a smaller power of two, one element short or over, twice the degree.
pub fn test_zq_ntt_rejects_wrong_length<W: ZqWord, T: ZqNew<W>>() {
    let (q, psi) = prime_power_root(4, &UBig::from(65537u32), 1);
    let table = T::new(16, &W::from_ubig(&q), &W::from_ubig(&psi));
    for len in [0, 8, 15, 17, 32] {
        let original: Vec<W> = words(&residues(len, 4, &q));
        let mut data: Vec<W> = original.clone();
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
pub fn test_zq_new_rejects<W: ZqWord, T: ZqNew<W>>() {
    let (q, psi) = prime_power_root(4, &UBig::from(65537u32), 1);
    let rejects = |n: usize, q: UBig, psi: UBig| {
        let (q, psi): (W, W) = (W::from_ubig(&q), W::from_ubig(&psi));
        catch_unwind(AssertUnwindSafe(move || T::new(n, &q, &psi))).is_err()
    };
    assert!(!rejects(16, q.clone(), psi.clone()), "valid parameters");
    assert!(rejects(0, q.clone(), psi.clone()), "zero degree");
    assert!(rejects(12, q.clone(), psi.clone()), "degree not a power of two");
    assert!(rejects(16, &q + UBig::ONE, psi.clone()), "even modulus");
    assert!(rejects(16, UBig::ONE, UBig::ZERO), "modulus one");
    assert!(rejects(16, q.clone(), (&psi * &psi) % &q), "root of order 16");
    assert!(rejects(32, q, psi), "root of order 32 at degree 32");
}
