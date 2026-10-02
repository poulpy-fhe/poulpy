use crate::blind_rotation::LookupTable;
use poulpy_hal::layouts::{Backend, Module};

/// Backend implementation of [`LookupTableFactory`](crate::api::LookupTableFactory).
///
/// # Safety
/// Implementations must preserve the canonical LUT encoding, coefficient and
/// polynomial ordering, precision, rotation direction, and drift metadata.
/// Writes must stay within the destination's allocated backend storage.
pub unsafe trait LookupTableFactoryImpl: Backend {
    /// Encodes host function samples into the backend-owned LUT.
    fn lookup_table_set(module: &Module<Self>, res: &mut LookupTable<Self::OwnedBuf, Self::ZnxWord>, f: &[i64], k: usize);

    /// Rotates the backend-owned LUT in its extended negacyclic domain.
    fn lookup_table_rotate(module: &Module<Self>, k: i64, res: &mut LookupTable<Self::OwnedBuf, Self::ZnxWord>);
}

/// Selects canonical LUT encoding and rotation through HAL operations.
#[macro_export]
macro_rules! impl_bin_fhe_lookup_table_reference {
    ($backend:ty) => {
        unsafe impl $crate::oep::LookupTableFactoryImpl for $backend {
            fn lookup_table_set(
                module: &poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::blind_rotation::LookupTable<Self::OwnedBuf, Self::ZnxWord>,
                f: &[i64],
                k: usize,
            ) {
                $crate::reference::blind_rotation::lookup_table_set_ref(module, res, f, k)
            }

            fn lookup_table_rotate(
                module: &poulpy_hal::layouts::Module<Self>,
                k: i64,
                res: &mut $crate::blind_rotation::LookupTable<Self::OwnedBuf, Self::ZnxWord>,
            ) {
                $crate::reference::blind_rotation::lookup_table_rotate_ref(module, k, res)
            }
        }
    };
}
