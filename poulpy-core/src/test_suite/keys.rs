//! A key source that coarsens what it holds.
//!
//! The shape a caller writes to put a policy in front of its key set: the
//! provider decides which key a request gets and which decomposition it is read
//! through, and nothing above it has a say.

use std::collections::HashMap;

use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, ZnxWord},
    source::Source,
};

use crate::{
    error::{CoreError, Result},
    layouts::{
        Dsize, GGLWE, GetAutomorphismKey, GetTensorKey, TorusPrecision,
        prepared::{
            GGLWEPrepared, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef,
            GLWETensorKeyPrepared, GLWETensorKeyPreparedBackendRef,
        },
    },
};

/// Answers from `keys`, every key read through `dsize`.
pub struct AtDsize<'a, K>(pub &'a K, pub Dsize);

impl<BE: Backend, D: Data> GetAutomorphismKey<BE> for AtDsize<'_, HashMap<i64, GLWEAutomorphismKeyPrepared<D, BE>>>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToBackendRef<BE>,
{
    fn get_automorphism_key(&self, p: i64, _k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.0
            .get(&p)
            .ok_or(CoreError::GGLWEKeyUse {
                op: "get_automorphism_key",
                detail: format!("no automorphism key for p={p}"),
            })?
            .with_dsize(self.1)
    }
}

impl<BE: Backend, D: Data> GetAutomorphismKey<BE> for AtDsize<'_, GLWEAutomorphismKeyPrepared<D, BE>>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToBackendRef<BE>,
{
    fn get_automorphism_key(&self, _p: i64, _k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.0.with_dsize(self.1)
    }
}

impl<BE: Backend, D: Data> GetTensorKey<BE> for AtDsize<'_, GLWETensorKeyPrepared<D, BE>>
where
    GGLWEPrepared<D, BE>: GGLWEPreparedToBackendRef<BE>,
{
    fn get_tensor_key(&self, _k: TorusPrecision) -> Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        self.0.with_dsize(self.1)
    }
}

/// Fills `key` so that every digit a `stride`-strided read reaches is drawn
/// from a source seeded by its digit index, and every row no digit maps to is
/// poisoned from an unrelated source.
///
/// `salt` separates keys that must differ from one another, such as the
/// rotations of a trace.
///
/// Filling a stored key at its stride and its coarse twin at stride 1 makes the
/// shared digits byte-identical without copying anything, so the two keys are
/// interchangeable exactly where the coarsening says they are and nowhere else.
pub fn fill_by_digit<D: HostDataMut, W: ZnxWord>(key: &mut GGLWE<D, W>, stride: usize, salt: u64) {
    let log_bound: usize = key.base2k.into();
    let (rows, cols_in) = (key.data().rows(), key.data().cols_in());
    let mut poison: Source = Source::new([0xFFu8; 32]);
    for row in 0..rows {
        for col in 0..cols_in {
            if (row + 1).is_multiple_of(stride) {
                let mut seed: [u8; 32] = [0u8; 32];
                seed[0..8].copy_from_slice(&(((row + 1) / stride - 1) as u64).to_le_bytes());
                seed[8..16].copy_from_slice(&(col as u64).to_le_bytes());
                seed[16..24].copy_from_slice(&salt.to_le_bytes());
                key.at_mut(row, col).fill_uniform(log_bound, &mut Source::new(seed));
            } else {
                key.at_mut(row, col).fill_uniform(log_bound, &mut poison);
            }
        }
    }
}
