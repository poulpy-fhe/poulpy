use std::marker::PhantomData;

use dashu_int::UBig;
use poulpy_hal::api::{ZqINTT, ZqN, ZqNTT, ZqNew};

use super::{ZqDFTExecute, ZqINTTPortable, ZqINTTTable, ZqNTTPortable, ZqNTTTable};

/// Forward and inverse negacyclic NTT tables over `Z/q` for one degree, run by the
/// executors `FWD` and `INV`.
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

impl<FWD, INV> ZqNew<UBig> for ZqPlan<UBig, FWD, INV>
where
    FWD: ZqDFTExecute<ZqNTTTable<UBig>, UBig>,
    INV: ZqDFTExecute<ZqINTTTable<UBig>, UBig>,
{
    fn new(n: usize, q: &UBig, psi: &UBig) -> Self {
        Self {
            ntt: ZqNTTTable::new(n, q, psi),
            intt: ZqINTTTable::new(n, q, psi),
            kernels: PhantomData,
        }
    }
}

impl<FWD, INV> ZqN for ZqPlan<UBig, FWD, INV> {
    fn n(&self) -> usize {
        self.ntt.n()
    }
}

impl<FWD, INV> ZqNTT<UBig> for ZqPlan<UBig, FWD, INV>
where
    FWD: ZqDFTExecute<ZqNTTTable<UBig>, UBig>,
{
    fn zq_ntt(&self, data: &mut [UBig]) {
        FWD::zq_dft_execute(&self.ntt, data);
    }
}

impl<FWD, INV> ZqINTT<UBig> for ZqPlan<UBig, FWD, INV>
where
    INV: ZqDFTExecute<ZqINTTTable<UBig>, UBig>,
{
    fn zq_intt(&self, data: &mut [UBig]) {
        INV::zq_dft_execute(&self.intt, data);
    }
}
