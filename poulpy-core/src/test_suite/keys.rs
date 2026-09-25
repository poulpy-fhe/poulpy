use crate::layouts::GLWEToBackendMut;
use crate::layouts::LWEInfos;
use poulpy_hal::api::VecZnxFillUniformSource;
use poulpy_hal::{
    layouts::{Backend, Module},
    source::Source,
};

use crate::{
    GLWEMaskFill,
    layouts::{GGLWEAtViewMut, GGLWEInfos},
};

/// Fills `key` from `source`, one draw per digit a `stride`-strided read
/// reaches, in digit order; every row no digit maps to is poisoned from an
/// unrelated stream.
///
/// Calling this on a stored key at its stride and on its coarse twin at stride
/// 1, each with an identically seeded `source`, makes the shared digits
/// byte-identical without copying anything: the two keys are interchangeable
/// exactly where the coarsening says they are and nowhere else.
pub fn fill_by_digit<BE: Backend, K>(module: &Module<BE>, key: &mut K, stride: usize, source: &mut Source)
where
    Module<BE>: GLWEMaskFill<BE> + VecZnxFillUniformSource<BE>,
    K: GGLWEAtViewMut<BE> + GGLWEInfos,
{
    let base2k: usize = key.base2k().into();
    let (rows, cols_in) = (key.dnum().as_usize(), key.rank_in().as_usize());
    let mut poison: Source = Source::new([0xFFu8; 32]);
    for row in 0..rows {
        for col in 0..cols_in {
            let stream = if (row + 1).is_multiple_of(stride) {
                &mut *source
            } else {
                &mut poison
            };
            module.vec_znx_fill_uniform_source(
                base2k,
                key.k().as_usize(),
                GLWEToBackendMut::<BE>::to_backend_mut(&mut key.at_view_mut(row, col)).data_mut(),
                0,
                stream,
            );
            module.fill_glwe_mask_from_source(&mut key.at_view_mut(row, col), stream);
        }
    }
}
