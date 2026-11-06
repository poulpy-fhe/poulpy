use poulpy_hal::{
    layouts::{Backend, Module},
    source::Source,
};

use crate::{
    GLWEFill,
    layouts::{
        GGLWEAtViewMut, GGLWEInfos, GetAutomorphismKey, GetTensorKey, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
    },
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
    Module<BE>: GLWEFill<BE>,
    K: GGLWEAtViewMut<BE> + GGLWEInfos,
{
    let (rows, cols_in) = (key.dnum().as_usize(), key.rank_in().as_usize());
    let mut poison: Source = Source::new([0xFFu8; 32]);
    for row in 0..rows {
        for col in 0..cols_in {
            let stream = if (row + 1).is_multiple_of(stride) {
                &mut *source
            } else {
                &mut poison
            };
            module.glwe_fill_from_source(&mut key.at_view_mut(row, col), stream);
        }
    }
}

/// The automorphism keys and the tensor key of one key set, the keys of a transform of ciphertext diagonals.
pub struct TransformKeys<'a, A, T> {
    pub automorphism: &'a A,
    pub tensor: &'a T,
}

impl<BE: Backend, A: GetAutomorphismKey<BE>, T> GetAutomorphismKey<BE> for TransformKeys<'_, A, T> {
    fn lookup_automorphism_key(&self, p: i64, k: TorusPrecision) -> crate::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.automorphism.lookup_automorphism_key(p, k)
    }
}

impl<BE: Backend, A, T: GetTensorKey<BE>> GetTensorKey<BE> for TransformKeys<'_, A, T> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> crate::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        self.tensor.lookup_tensor_key(k)
    }
}
