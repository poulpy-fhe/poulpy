use std::marker::PhantomData;

use poulpy_hal::api::{ZqINTT, ZqN, ZqNTT, ZqNew};

use super::{ZqDFTExecute, ZqINTTPortable, ZqINTTTable, ZqNTTPortable, ZqNTTTable};

/// Forward and inverse negacyclic NTT tables over `Z/q` for one degree, run by the
/// executors `FWD` and `INV`, on residues of type `W`: `[u64; L]`, `L` little-endian words,
/// for any odd modulus below `2^(64 L)`.
pub struct ZqPlan<W, FWD = ZqNTTPortable, INV = ZqINTTPortable> {
    ntt: ZqNTTTable<W>,
    intt: ZqINTTTable<W>,
    kernels: PhantomData<(FWD, INV)>,
}

impl<W, FWD, INV> ZqPlan<W, FWD, INV> {
    pub fn ntt(&self) -> &ZqNTTTable<W> {
        &self.ntt
    }

    pub fn intt(&self) -> &ZqINTTTable<W> {
        &self.intt
    }
}

impl<const L: usize, FWD, INV> ZqNew<[u64; L]> for ZqPlan<[u64; L], FWD, INV>
where
    FWD: ZqDFTExecute<ZqNTTTable<[u64; L]>, [u64; L]>,
    INV: ZqDFTExecute<ZqINTTTable<[u64; L]>, [u64; L]>,
{
    fn new(n: usize, q: &[u64; L], psi: &[u64; L]) -> Self {
        Self {
            ntt: ZqNTTTable::new(n, q, psi),
            intt: ZqINTTTable::new(n, q, psi),
            kernels: PhantomData,
        }
    }
}

impl<const L: usize, FWD, INV> ZqN for ZqPlan<[u64; L], FWD, INV> {
    fn n(&self) -> usize {
        self.ntt.n()
    }
}

impl<const L: usize, FWD, INV> ZqNTT<[u64; L]> for ZqPlan<[u64; L], FWD, INV>
where
    FWD: ZqDFTExecute<ZqNTTTable<[u64; L]>, [u64; L]>,
{
    fn zq_ntt(&self, data: &mut [[u64; L]]) {
        FWD::zq_dft_execute(&self.ntt, data);
    }
}

impl<const L: usize, FWD, INV> ZqINTT<[u64; L]> for ZqPlan<[u64; L], FWD, INV>
where
    INV: ZqDFTExecute<ZqINTTTable<[u64; L]>, [u64; L]>,
{
    fn zq_intt(&self, data: &mut [[u64; L]]) {
        INV::zq_dft_execute(&self.intt, data);
    }
}
