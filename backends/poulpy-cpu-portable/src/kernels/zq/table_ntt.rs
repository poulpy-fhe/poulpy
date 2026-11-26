use dashu_int::{UBig, fast_div::ConstDivisor, ops::BitTest};
use poulpy_hal::bit_reverse;

use super::{ZqDFTExecute, zq_ntt_portable};

/// Portable executor of the forward transform: [`zq_ntt_portable`].
pub struct ZqNTTPortable;

impl ZqDFTExecute<ZqNTTTable<UBig>, UBig> for ZqNTTPortable {
    fn zq_dft_execute(table: &ZqNTTTable<UBig>, data: &mut [UBig]) {
        zq_ntt_portable(table, data);
    }
}

/// Precomputed table of the forward negacyclic NTT of degree `n` modulo `q` for the root
/// `psi`: `twiddles[rev_n(k)] = psi^k` for `k < n`.
pub struct ZqNTTTable<W> {
    q: W,
    twiddles: Vec<W>,
}

impl ZqNTTTable<UBig> {
    /// Builds the table.
    ///
    /// # Panics
    ///
    /// Outside the domain of [`ZqNew::new`](poulpy_hal::api::ZqNew::new).
    pub fn new(n: usize, q: &UBig, psi: &UBig) -> Self {
        check_zq_parameters(n, q, psi);
        let ring: ConstDivisor = ConstDivisor::new(q.clone());
        Self {
            q: q.clone(),
            twiddles: bit_reversed_powers(&ring, psi, n),
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
