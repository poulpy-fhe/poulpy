use dashu_int::{UBig, fast_div::ConstDivisor, ops::BitTest};
use poulpy_hal::bit_reverse;

use super::{ZqDFTExecute, zq_ntt_portable};

/// Portable executor of the forward transform: [`zq_ntt_portable`].
pub struct ZqNTTPortable;

impl<const L: usize> ZqDFTExecute<ZqNTTTable<[u64; L]>, [u64; L]> for ZqNTTPortable {
    fn zq_dft_execute(table: &ZqNTTTable<[u64; L]>, data: &mut [[u64; L]]) {
        zq_ntt_portable(table, data);
    }
}

/// Precomputed table of the forward negacyclic NTT of degree `n` modulo `q` for the root
/// `psi`, on residues of `L` little-endian 64-bit words: `twiddles[rev_n(k)] = psi^k` for
/// `k < n`, in Montgomery form `psi^k 2^(64 L) mod q`, and `q_neg_inv = -q^(-1) mod 2^64`.
pub struct ZqNTTTable<W> {
    q: W,
    q_neg_inv: u64,
    twiddles: Vec<W>,
}

impl<const L: usize> ZqNTTTable<[u64; L]> {
    /// Builds the table.
    ///
    /// # Panics
    ///
    /// Outside the domain of [`ZqNew::new`](poulpy_hal::api::ZqNew::new).
    pub fn new(n: usize, q: &[u64; L], psi: &[u64; L]) -> Self {
        let (q_big, psi_big): (UBig, UBig) = (UBig::from_words(q), UBig::from_words(psi));
        check_zq_parameters(n, &q_big, &psi_big);
        let ring: ConstDivisor = ConstDivisor::new(q_big);
        Self {
            q: *q,
            q_neg_inv: neg_inv_word(q[0]),
            twiddles: montgomery_words(&ring, bit_reversed_powers(&ring, &psi_big, n)),
        }
    }

    pub fn n(&self) -> usize {
        self.twiddles.len()
    }

    pub fn q(&self) -> &[u64; L] {
        &self.q
    }

    /// `-q^(-1) mod 2^64`.
    pub fn q_neg_inv(&self) -> u64 {
        self.q_neg_inv
    }

    /// The twiddles, in Montgomery form.
    pub fn twiddles(&self) -> &[[u64; L]] {
        &self.twiddles
    }
}

/// Panics unless `n` is a positive power of two, `q` is odd and greater than one, and
/// `psi^n = -1 mod q`.
pub(super) fn check_zq_parameters(n: usize, q: &UBig, psi: &UBig) {
    assert!(n.is_power_of_two(), "zq ntt: degree must be a positive power of two");
    assert!(*q > UBig::ONE && q.bit(0), "zq ntt: modulus must be odd and greater than one");
    let ring: ConstDivisor = ConstDivisor::new(q.clone());
    assert!(
        ring.reduce(psi.clone()).pow(&UBig::from(n)) == ring.reduce(q - UBig::ONE),
        "zq ntt: root must satisfy root^n = -1"
    );
}

/// `res[rev_n(k)] = base^k mod q` for `k < n`.
pub(super) fn bit_reversed_powers(ring: &ConstDivisor, base: &UBig, n: usize) -> Vec<UBig> {
    let base = ring.reduce(base.clone());
    let mut power = ring.reduce(1u8);
    let mut res: Vec<UBig> = vec![UBig::ZERO; n];
    for k in 0..n {
        res[bit_reverse(k, n)] = power.residue();
        power *= &base;
    }
    res
}

/// The residues `values` modulo the odd `q` of `ring` in Montgomery form, `x 2^(64 L) mod q`,
/// as `L` words.
pub(super) fn montgomery_words<const L: usize>(ring: &ConstDivisor, values: Vec<UBig>) -> Vec<[u64; L]> {
    values
        .into_iter()
        .map(|value| {
            let residue: UBig = ring.reduce(value << (64 * L)).residue();
            let mut words: [u64; L] = [0; L];
            words[..residue.as_words().len()].copy_from_slice(residue.as_words());
            words
        })
        .collect()
}

/// `-q^(-1) mod 2^64` for an odd `q`, by Newton's iteration, each step doubling the correct
/// low bits of the inverse from the three of `q` itself.
pub(super) fn neg_inv_word(q: u64) -> u64 {
    let mut inv: u64 = q;
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}
