use super::{ZqINTTTable, ZqNTTTable};

/// Forward negacyclic NTT over `Z/q` in place: Cooley-Tukey, natural order in, bit-reversed
/// order out.
///
/// For `m = 1, 2, ..., n/2`, with `t = n / 2m`, block `i < m` applies the butterfly
/// `(u, v) -> (u + w v, u - w v)` to the pairs `(j, j + t)`, `2 i t <= j < (2 i + 1) t`,
/// with `w = psi^(rev_n(m + i))`, held as `twiddles[m + i]` in Montgomery form. Residues stay
/// in `[0, q)`, on the stack: the transform allocates nothing.
///
/// # Panics
///
/// When the length of `data` is not the table degree.
pub fn zq_ntt_portable<const L: usize>(table: &ZqNTTTable<[u64; L]>, data: &mut [[u64; L]]) {
    let (q, q_neg_inv, twiddles) = (table.q(), table.q_neg_inv(), table.twiddles());
    let n: usize = data.len();
    assert!(n == twiddles.len(), "zq ntt: data length must equal the degree");
    let mut t: usize = n;
    let mut m: usize = 1;
    while m < n {
        t /= 2;
        for i in 0..m {
            let w: &[u64; L] = &twiddles[m + i];
            let start: usize = 2 * i * t;
            for j in start..start + t {
                let v: [u64; L] = mont_mul(&data[j + t], w, q, q_neg_inv);
                let u: [u64; L] = data[j];
                data[j + t] = sub_mod(&u, &v, q);
                data[j] = add_mod(&u, &v, q);
            }
        }
        m *= 2;
    }
}

/// Inverse negacyclic NTT over `Z/q` in place, normalized: Gentleman-Sande, bit-reversed
/// order in, natural order out.
///
/// For `m = n, n/2, ..., 2`, with `h = m/2` and `t = n / m`, block `i < h` applies the
/// butterfly `(u, v) -> (u + v, (u - v) w)` with `w = psi^(-rev_n(h + i))`, held as
/// `twiddles[h + i]` in Montgomery form. The last pass, `h = 1`, folds in `n^(-1)`: it applies
/// `(u, v) -> ((u + v) n_inv, (u - v) last)` with `last = psi^(-1) n_inv`. Residues stay in
/// `[0, q)`, on the stack: the transform allocates nothing.
///
/// # Panics
///
/// When the length of `data` is not the table degree.
pub fn zq_intt_portable<const L: usize>(table: &ZqINTTTable<[u64; L]>, data: &mut [[u64; L]]) {
    let (q, q_neg_inv, twiddles) = (table.q(), table.q_neg_inv(), table.twiddles());
    let (n_inv, last) = (table.n_inv(), table.last());
    let n: usize = data.len();
    assert!(n == twiddles.len(), "zq ntt: data length must equal the degree");
    let mut t: usize = 1;
    let mut m: usize = n;
    while m > 2 {
        let h: usize = m / 2;
        for i in 0..h {
            let w: &[u64; L] = &twiddles[h + i];
            let start: usize = 2 * i * t;
            for j in start..start + t {
                let (u, v): ([u64; L], [u64; L]) = (data[j], data[j + t]);
                data[j + t] = mont_mul(&sub_mod(&u, &v, q), w, q, q_neg_inv);
                data[j] = add_mod(&u, &v, q);
            }
        }
        t *= 2;
        m = h;
    }
    if n >= 2 {
        for j in 0..t {
            let (u, v): ([u64; L], [u64; L]) = (data[j], data[j + t]);
            data[j + t] = mont_mul(&sub_mod(&u, &v, q), last, q, q_neg_inv);
            data[j] = mont_mul(&add_mod(&u, &v, q), n_inv, q, q_neg_inv);
        }
    }
}

/// `a b 2^(-64 L) mod q` for `a, b` in `[0, q)`, in `[0, q)`, by word-serial Montgomery
/// multiplication (CIOS): `q_neg_inv = -q^(-1) mod 2^64`. The running sum stays below `2q`,
/// in `L` words and a carry word.
#[inline(always)]
fn mont_mul<const L: usize>(a: &[u64; L], b: &[u64; L], q: &[u64; L], q_neg_inv: u64) -> [u64; L] {
    let mut t: [u64; L] = [0; L];
    let mut top: u64 = 0;
    for &b_i in b {
        let mut carry: u64 = 0;
        for (t_j, &a_j) in t.iter_mut().zip(a) {
            let s: u128 = *t_j as u128 + a_j as u128 * b_i as u128 + carry as u128;
            (*t_j, carry) = (s as u64, (s >> 64) as u64);
        }
        let s: u128 = top as u128 + carry as u128;
        let (sum_top, over): (u64, u64) = (s as u64, (s >> 64) as u64);
        let m: u64 = t[0].wrapping_mul(q_neg_inv);
        let mut carry: u64 = ((t[0] as u128 + m as u128 * q[0] as u128) >> 64) as u64;
        for j in 1..L {
            let s: u128 = t[j] as u128 + m as u128 * q[j] as u128 + carry as u128;
            (t[j - 1], carry) = (s as u64, (s >> 64) as u64);
        }
        let s: u128 = sum_top as u128 + carry as u128;
        t[L - 1] = s as u64;
        top = over + (s >> 64) as u64;
    }
    if top != 0 || !less(&t, q) { sub(&t, q).0 } else { t }
}

/// `u + v mod q` for `u, v` in `[0, q)`.
#[inline(always)]
fn add_mod<const L: usize>(u: &[u64; L], v: &[u64; L], q: &[u64; L]) -> [u64; L] {
    let (sum, carry): ([u64; L], bool) = add(u, v);
    if carry || !less(&sum, q) { sub(&sum, q).0 } else { sum }
}

/// `u - v mod q` for `u, v` in `[0, q)`.
#[inline(always)]
fn sub_mod<const L: usize>(u: &[u64; L], v: &[u64; L], q: &[u64; L]) -> [u64; L] {
    let (diff, borrow): ([u64; L], bool) = sub(u, v);
    if borrow { add(&diff, q).0 } else { diff }
}

/// `a + b` and its carry out.
#[inline(always)]
fn add<const L: usize>(a: &[u64; L], b: &[u64; L]) -> ([u64; L], bool) {
    let mut res: [u64; L] = [0; L];
    let mut carry: bool = false;
    for j in 0..L {
        let (s, c1) = a[j].overflowing_add(b[j]);
        let (s, c2) = s.overflowing_add(carry as u64);
        (res[j], carry) = (s, c1 | c2);
    }
    (res, carry)
}

/// `a - b` and its borrow out.
#[inline(always)]
fn sub<const L: usize>(a: &[u64; L], b: &[u64; L]) -> ([u64; L], bool) {
    let mut res: [u64; L] = [0; L];
    let mut borrow: bool = false;
    for j in 0..L {
        let (d, b1) = a[j].overflowing_sub(b[j]);
        let (d, b2) = d.overflowing_sub(borrow as u64);
        (res[j], borrow) = (d, b1 | b2);
    }
    (res, borrow)
}

/// `a < b`, little-endian words.
#[inline(always)]
fn less<const L: usize>(a: &[u64; L], b: &[u64; L]) -> bool {
    a.iter().rev().cmp(b.iter().rev()).is_lt()
}
