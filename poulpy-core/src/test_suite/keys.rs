use poulpy_hal::{
    layouts::{FillUniform, HostDataMut, ZnxWord},
    source::Source,
};

use crate::layouts::GGLWE;

/// Fills `key` from `source`, one draw per digit a `stride`-strided read
/// reaches, in digit order; every row no digit maps to is poisoned from an
/// unrelated stream.
///
/// Calling this on a stored key at its stride and on its coarse twin at stride
/// 1, each with an identically seeded `source`, makes the shared digits
/// byte-identical without copying anything: the two keys are interchangeable
/// exactly where the coarsening says they are and nowhere else.
pub fn fill_by_digit<D: HostDataMut, W: ZnxWord>(key: &mut GGLWE<D, W>, stride: usize, source: &mut Source) {
    let log_bound: usize = key.base2k.into();
    let (rows, cols_in) = (key.data().rows(), key.data().cols_in());
    let mut poison: Source = Source::new([0xFFu8; 32]);
    for row in 0..rows {
        for col in 0..cols_in {
            let stream = if (row + 1).is_multiple_of(stride) {
                &mut *source
            } else {
                &mut poison
            };
            key.at_mut(row, col).fill_uniform(log_bound, stream);
        }
    }
}
