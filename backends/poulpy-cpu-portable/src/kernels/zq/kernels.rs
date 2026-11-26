use dashu_int::UBig;

use super::{ZqINTTTable, ZqNTTTable};

/// Forward negacyclic NTT over `Z/q` in place: Cooley-Tukey, natural order in, bit-reversed
/// order out.
///
/// For `m = 1, 2, ..., n/2`, with `t = n / 2m`, block `i < m` applies the butterfly
/// `(u, v) -> (u + w v, u - w v)` to the pairs `(j, j + t)`, `2 i t <= j < (2 i + 1) t`,
/// with `w = twiddles[m + i] = psi^(rev_n(m + i))`. Residues stay in `[0, q)`.
///
/// # Panics
///
/// When the length of `data` is not the table degree.
pub fn zq_ntt_portable(table: &ZqNTTTable<UBig>, data: &mut [UBig]) {
    let (q, twiddles): (&UBig, &[UBig]) = (table.q(), table.twiddles());
    let n: usize = data.len();
    assert!(n == twiddles.len(), "zq ntt: data length must equal the degree");
    let mut t: usize = n;
    let mut m: usize = 1;
    while m < n {
        t /= 2;
        for i in 0..m {
            let w: &UBig = &twiddles[m + i];
            let start: usize = 2 * i * t;
            for j in start..start + t {
                let v: UBig = (&data[j + t] * w) % q;
                let u: UBig = std::mem::take(&mut data[j]);
                data[j + t] = sub_mod(&u, &v, q);
                data[j] = add_mod(u, &v, q);
            }
        }
        m *= 2;
    }
}

/// Inverse negacyclic NTT over `Z/q` in place, normalized: Gentleman-Sande, bit-reversed
/// order in, natural order out.
///
/// For `m = n, n/2, ..., 2`, with `h = m/2` and `t = n / m`, block `i < h` applies the
/// butterfly `(u, v) -> (u + v, (u - v) w)` with `w = twiddles[h + i] = psi^(-rev_n(h + i))`.
/// The last pass, `h = 1`, folds in `n^(-1)`: it applies
/// `(u, v) -> ((u + v) n_inv, (u - v) last)` with `last = twiddles[1] n_inv`. Residues stay
/// in `[0, q)`.
///
/// # Panics
///
/// When the length of `data` is not the table degree.
pub fn zq_intt_portable(table: &ZqINTTTable<UBig>, data: &mut [UBig]) {
    let (q, twiddles): (&UBig, &[UBig]) = (table.q(), table.twiddles());
    let (n_inv, last): (&UBig, &UBig) = (table.n_inv(), table.last());
    let n: usize = data.len();
    assert!(n == twiddles.len(), "zq ntt: data length must equal the degree");
    let mut t: usize = 1;
    let mut m: usize = n;
    while m > 2 {
        let h: usize = m / 2;
        for i in 0..h {
            let w: &UBig = &twiddles[h + i];
            let start: usize = 2 * i * t;
            for j in start..start + t {
                let u: UBig = std::mem::take(&mut data[j]);
                let v: UBig = std::mem::take(&mut data[j + t]);
                data[j + t] = (sub_mod(&u, &v, q) * w) % q;
                data[j] = add_mod(u, &v, q);
            }
        }
        t *= 2;
        m = h;
    }
    if n >= 2 {
        for j in 0..t {
            let u: UBig = std::mem::take(&mut data[j]);
            let v: UBig = std::mem::take(&mut data[j + t]);
            data[j + t] = (sub_mod(&u, &v, q) * last) % q;
            data[j] = (add_mod(u, &v, q) * n_inv) % q;
        }
    }
}

/// `u + v mod q` for `u, v` in `[0, q)`.
#[inline(always)]
fn add_mod(u: UBig, v: &UBig, q: &UBig) -> UBig {
    let sum: UBig = u + v;
    if sum >= *q { sum - q } else { sum }
}

/// `u - v mod q` for `u, v` in `[0, q)`.
#[inline(always)]
fn sub_mod(u: &UBig, v: &UBig, q: &UBig) -> UBig {
    if u >= v { u - v } else { u + q - v }
}
