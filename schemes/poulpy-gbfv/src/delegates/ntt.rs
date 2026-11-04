use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module};

use crate::{
    api::GBFVNttOps,
    layouts::{GBFVNttBufferToBackendMut, GBFVNttPlanToBackendRef},
    oep::GBFVNttImpl,
};

impl<BE: Backend + GBFVNttImpl> GBFVNttOps<BE> for Module<BE> {
    fn gbfv_ntt_forward_assign<P, V>(&self, plan: &P, values: &mut V) -> Result<()>
    where
        P: GBFVNttPlanToBackendRef<BE>,
        V: GBFVNttBufferToBackendMut<BE>,
    {
        BE::gbfv_ntt_forward_assign(self, plan, values)
    }

    fn gbfv_ntt_inverse_assign<P, V>(&self, plan: &P, values: &mut V) -> Result<()>
    where
        P: GBFVNttPlanToBackendRef<BE>,
        V: GBFVNttBufferToBackendMut<BE>,
    {
        BE::gbfv_ntt_inverse_assign(self, plan, values)
    }
}
