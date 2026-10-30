use crate::blind_rotation::LookupTable;
use poulpy_hal::layouts::{Backend, Module, Standard};

/// Backend implementation contract for [`LookupTableFactory`](crate::api::LookupTableFactory), on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit and refuse as the reference does.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait LookupTableFactoryImpl: Backend<Ring = Standard> {
    /// Reproduces [`lookup_table_set_reference`](crate::reference::blind_rotation::lookup_table_set_reference) bit for
    /// bit; free in kernels.
    fn lookup_table_set(module: &Module<Self>, k: usize, res: &mut LookupTable<Self::OwnedBuf, Self::ZnxWord>, f: &[i64]);

    /// Reproduces [`lookup_table_rotate_assign_reference`](crate::reference::blind_rotation::lookup_table_rotate_assign_reference) bit
    /// for bit; free in kernels.
    fn lookup_table_rotate_assign(module: &Module<Self>, k: i64, res: &mut LookupTable<Self::OwnedBuf, Self::ZnxWord>);
}

/// Selects canonical LUT encoding and rotation through HAL operations.
#[macro_export]
macro_rules! impl_lookup_table_factory_reference_full {
    ($backend:ty) => {
        unsafe impl $crate::oep::LookupTableFactoryImpl for $backend {
            fn lookup_table_set(
                module: &poulpy_hal::layouts::Module<Self>,
                k: usize,
                res: &mut $crate::blind_rotation::LookupTable<Self::OwnedBuf, Self::ZnxWord>,
                f: &[i64],
            ) {
                <::poulpy_hal::layouts::Module<$backend> as $crate::reference::blind_rotation::LookupTableFactoryReference<
                    $backend,
                >>::lookup_table_set_reference(module, k, res, f)
            }

            fn lookup_table_rotate_assign(
                module: &poulpy_hal::layouts::Module<Self>,
                k: i64,
                res: &mut $crate::blind_rotation::LookupTable<Self::OwnedBuf, Self::ZnxWord>,
            ) {
                <::poulpy_hal::layouts::Module<$backend> as $crate::reference::blind_rotation::LookupTableFactoryReference<
                    $backend,
                >>::lookup_table_rotate_assign_reference(module, k, res)
            }
        }
    };
}
