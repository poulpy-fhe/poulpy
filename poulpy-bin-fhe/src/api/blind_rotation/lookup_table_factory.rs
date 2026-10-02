use crate::blind_rotation::LookupTable;
use poulpy_hal::layouts::{Data, ZnxWord};

/// Public helpers for constructing and rotating a [`LookupTable`].
///
/// Module calls dispatch through [`LookupTableFactoryImpl`](crate::oep::LookupTableFactoryImpl),
/// allowing the backend to select LUT encoding and rotation implementations.
/// [`LookupTable::set`] is a convenience wrapper around this trait.
pub trait LookupTableFactory<D: Data, W: ZnxWord> {
    /// Encode the function `f` into `res`, scaling by the appropriate power of
    /// the decomposition base so that the most significant limb carries the
    /// message.
    ///
    /// `k` is the message-bit count (e.g., 1 for a binary-valued LUT).
    /// `f` must be nonempty and have length at most the module's ring degree,
    /// including when `res` uses an extended LUT domain.
    fn lookup_table_set(&self, res: &mut LookupTable<D, W>, f: &[i64], k: usize);

    /// Rotate the lookup table in-place by `k` positions in the ring
    /// `Z[X] / (X^{domain_size} + 1)`.
    fn lookup_table_rotate(&self, k: i64, res: &mut LookupTable<D, W>);
}
