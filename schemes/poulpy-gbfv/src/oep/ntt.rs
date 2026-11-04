use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module};

use crate::layouts::{GBFVNttBufferToBackendMut, GBFVNttPlanToBackendRef};

/// Backend contract of [`GBFVNttOps`](crate::api::GBFVNttOps).
///
/// The canonical circuit is
/// [`GBFVNttReference`](crate::reference::ntt::GBFVNttReference): an override returns
/// the same residues and refuses the same operands before writing. Residue arithmetic,
/// layouts and scheduling are free.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit and refuse as
/// the reference does. The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GBFVNttImpl: Backend {
    /// Reproduces
    /// [`gbfv_ntt_forward_assign_reference`](crate::reference::ntt::GBFVNttReference::gbfv_ntt_forward_assign_reference)
    /// bit for bit.
    fn gbfv_ntt_forward_assign<P, V>(module: &Module<Self>, plan: &P, values: &mut V) -> Result<()>
    where
        P: GBFVNttPlanToBackendRef<Self>,
        V: GBFVNttBufferToBackendMut<Self>;

    /// Reproduces
    /// [`gbfv_ntt_inverse_assign_reference`](crate::reference::ntt::GBFVNttReference::gbfv_ntt_inverse_assign_reference)
    /// bit for bit.
    fn gbfv_ntt_inverse_assign<P, V>(module: &Module<Self>, plan: &P, values: &mut V) -> Result<()>
    where
        P: GBFVNttPlanToBackendRef<Self>,
        V: GBFVNttBufferToBackendMut<Self>;
}

/// Selects the reference implementation of `GBFVNttImpl` for `$be`, a backend whose
/// borrowed views are host bytes.
#[macro_export]
macro_rules! impl_gbfv_ntt_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GBFVNttImpl for $be {
            fn gbfv_ntt_forward_assign<P, V>(
                module: &::poulpy_hal::layouts::Module<Self>,
                plan: &P,
                values: &mut V,
            ) -> $crate::GBFVResult<()>
            where
                P: $crate::layouts::GBFVNttPlanToBackendRef<Self>,
                V: $crate::layouts::GBFVNttBufferToBackendMut<Self>,
            {
                $crate::reference::ntt::GBFVNttReference::gbfv_ntt_forward_assign_reference(module, plan, values)
            }

            fn gbfv_ntt_inverse_assign<P, V>(
                module: &::poulpy_hal::layouts::Module<Self>,
                plan: &P,
                values: &mut V,
            ) -> $crate::GBFVResult<()>
            where
                P: $crate::layouts::GBFVNttPlanToBackendRef<Self>,
                V: $crate::layouts::GBFVNttBufferToBackendMut<Self>,
            {
                $crate::reference::ntt::GBFVNttReference::gbfv_ntt_inverse_assign_reference(module, plan, values)
            }
        }
    };
}
pub use crate::impl_gbfv_ntt_reference_full;
