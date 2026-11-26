use dashu_int::{UBig, fast_div::ConstDivisor};

use super::{
    ZqDFTExecute,
    table_ntt::{bit_reversed_powers, check_zq_parameters},
    zq_intt_portable,
};

/// Portable executor of the inverse transform: [`zq_intt_portable`].
pub struct ZqINTTPortable;

impl ZqDFTExecute<ZqINTTTable<UBig>, UBig> for ZqINTTPortable {
    fn zq_dft_execute(table: &ZqINTTTable<UBig>, data: &mut [UBig]) {
        zq_intt_portable(table, data);
    }
}

/// Precomputed table of the normalized inverse negacyclic NTT of degree `n` modulo `q` for
/// the root `psi`: `twiddles[rev_n(k)] = psi^(-k)` for `k < n`, `n_inv = n^(-1) mod q`, and
/// `last = twiddles[1] n_inv`, the twiddle of the last pass with `n^(-1)` folded in
/// (`n_inv` when `n = 1`).
pub struct ZqINTTTable<W> {
    q: W,
    twiddles: Vec<W>,
    n_inv: W,
    last: W,
}

impl ZqINTTTable<UBig> {
    /// Builds the table.
    ///
    /// # Panics
    ///
    /// Outside the domain of [`ZqNew::new`](poulpy_hal::api::ZqNew::new).
    pub fn new(n: usize, q: &UBig, psi: &UBig) -> Self {
        check_zq_parameters(n, q, psi);
        let ring: ConstDivisor = ConstDivisor::new(q.clone());
        let psi_inv: UBig = ring.reduce(psi.clone()).pow(&UBig::from(2 * n - 1)).residue();
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
        Self {
            q: q.clone(),
            twiddles,
            n_inv: n_inv.residue(),
            last,
        }
    }

    pub fn n(&self) -> usize {
        self.twiddles.len()
    }

    pub fn q(&self) -> &UBig {
        &self.q
    }

    pub fn twiddles(&self) -> &[UBig] {
        &self.twiddles
    }

    pub fn n_inv(&self) -> &UBig {
        &self.n_inv
    }

    pub fn last(&self) -> &UBig {
        &self.last
    }
}
