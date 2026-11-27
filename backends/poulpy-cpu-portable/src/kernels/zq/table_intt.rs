use dashu_int::{UBig, fast_div::ConstDivisor};

use super::{
    ZqDFTExecute,
    table_ntt::{bit_reversed_powers, check_zq_parameters, montgomery_words, neg_inv_word},
    zq_intt_portable,
};

/// Portable executor of the inverse transform: [`zq_intt_portable`].
pub struct ZqINTTPortable;

impl<const L: usize> ZqDFTExecute<ZqINTTTable<[u64; L]>, [u64; L]> for ZqINTTPortable {
    fn zq_dft_execute(table: &ZqINTTTable<[u64; L]>, data: &mut [[u64; L]]) {
        zq_intt_portable(table, data);
    }
}

/// Precomputed table of the normalized inverse negacyclic NTT of degree `n` modulo `q` for
/// the root `psi`, on residues of `L` little-endian 64-bit words: `twiddles[rev_n(k)] =
/// psi^(-k)` for `k < n`, `n_inv = n^(-1) mod q`, and `last = twiddles[1] n_inv`, the twiddle
/// of the last pass with `n^(-1)` folded in (`n_inv` when `n = 1`), all in Montgomery form
/// `x 2^(64 L) mod q`, and `q_neg_inv = -q^(-1) mod 2^64`.
pub struct ZqINTTTable<W> {
    q: W,
    q_neg_inv: u64,
    twiddles: Vec<W>,
    n_inv: W,
    last: W,
}

impl<const L: usize> ZqINTTTable<[u64; L]> {
    /// Builds the table.
    ///
    /// # Panics
    ///
    /// Outside the domain of [`ZqNew::new`](poulpy_hal::api::ZqNew::new).
    pub fn new(n: usize, q: &[u64; L], psi: &[u64; L]) -> Self {
        let (q_big, psi_big): (UBig, UBig) = (UBig::from_words(q), UBig::from_words(psi));
        check_zq_parameters(n, &q_big, &psi_big);
        let ring: ConstDivisor = ConstDivisor::new(q_big);
        let psi_inv: UBig = ring.reduce(psi_big).pow(&UBig::from(2 * n - 1)).residue();
        let twiddles: Vec<UBig> = bit_reversed_powers(&ring, &psi_inv, n);
        let n_inv = ring
            .reduce(UBig::from(n))
            .inv()
            .expect("an odd modulus makes a power-of-two degree invertible");
        let last: UBig = if n >= 2 {
            (n_inv.clone() * ring.reduce(twiddles[1].clone())).residue()
        } else {
            n_inv.residue()
        };
        let [n_inv, last]: [[u64; L]; 2] = montgomery_words(&ring, vec![n_inv.residue(), last])
            .try_into()
            .expect("two residues");
        Self {
            q: *q,
            q_neg_inv: neg_inv_word(q[0]),
            twiddles: montgomery_words(&ring, twiddles),
            n_inv,
            last,
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

    /// `n^(-1) mod q`, in Montgomery form.
    pub fn n_inv(&self) -> &[u64; L] {
        &self.n_inv
    }

    /// The twiddle of the last pass with `n^(-1)` folded in, in Montgomery form.
    pub fn last(&self) -> &[u64; L] {
        &self.last
    }
}
