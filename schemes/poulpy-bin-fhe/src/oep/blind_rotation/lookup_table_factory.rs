use crate::blind_rotation::LookupTable;
use poulpy_hal::layouts::{Backend, Module, Standard};

/// Backend implementation of [`LookupTableFactory`](crate::api::LookupTableFactory), on the standard ring only.
///
/// # Safety
/// Implementations must preserve the canonical LUT encoding, coefficient and
/// polynomial ordering, precision, rotation direction, and drift metadata.
/// Writes must stay within the destination's allocated backend storage.
pub unsafe trait LookupTableFactoryImpl: Backend<Ring = Standard> {
    /// Encodes host function samples into the backend-owned LUT.
    fn lookup_table_set(module: &Module<Self>, k: usize, res: &mut LookupTable<Self::OwnedBuf, Self::ZnxWord>, f: &[i64]);

    /// Rotates the backend-owned LUT in its extended negacyclic domain.
    fn lookup_table_rotate(module: &Module<Self>, k: i64, res: &mut LookupTable<Self::OwnedBuf, Self::ZnxWord>);
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
                $crate::reference::blind_rotation::lookup_table_set_reference(module, k, res, f)
            }

            fn lookup_table_rotate(
                module: &poulpy_hal::layouts::Module<Self>,
                k: i64,
                res: &mut $crate::blind_rotation::LookupTable<Self::OwnedBuf, Self::ZnxWord>,
            ) {
                $crate::reference::blind_rotation::lookup_table_rotate_reference(module, k, res)
            }
        }
    };
}
