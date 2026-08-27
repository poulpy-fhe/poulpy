//! Key lookup.
//!
//! A caller names a function and the precision it is about to use the key at;
//! the source answers with the backend view of a prepared key, which may be a
//! coarser reading of it ([`GGLWEPrepared::with_dsize`]). Which key and which
//! decomposition is the source's business: a map answers with what it stores,
//! and a caller wanting other rules implements the trait itself.

use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Data};

use crate::{
    error::{CoreError, Result},
    layouts::{
        TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
            GLWETensorKeyPrepared, GLWETensorKeyPreparedBackendRef, GLWETensorKeyPreparedToBackendRef,
        },
    },
};

/// Automorphism key for Galois element `p`, at the precision `k` it will be
/// used at.
pub trait GetAutomorphismKey<BE: Backend> {
    fn get_automorphism_key(&self, p: i64, k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>>;
}

/// Tensor (relinearization) key, at the precision `k` it will be used at.
pub trait GetTensorKey<BE: Backend> {
    fn get_tensor_key(&self, k: TorusPrecision) -> Result<GLWETensorKeyPreparedBackendRef<'_, BE>>;
}

/// A map answers with each key as stored.
impl<BE: Backend, K: GLWEAutomorphismKeyPreparedToBackendRef<BE>> GetAutomorphismKey<BE> for HashMap<i64, K> {
    fn get_automorphism_key(&self, p: i64, _k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.get(&p)
            .map(GLWEAutomorphismKeyPreparedToBackendRef::to_backend_ref)
            .ok_or(CoreError::GGLWEKeyUse {
                op: "get_automorphism_key",
                detail: format!("no automorphism key for p={p}"),
            })
    }
}

/// A bare key answers "this key, as stored, for every rotation".
impl<D: Data, BE: Backend> GetAutomorphismKey<BE> for GLWEAutomorphismKeyPrepared<D, BE>
where
    Self: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
{
    fn get_automorphism_key(&self, _p: i64, _k: TorusPrecision) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        Ok(self.to_backend_ref())
    }
}

/// A bare key is its own source, as stored.
impl<D: Data, BE: Backend> GetTensorKey<BE> for GLWETensorKeyPrepared<D, BE>
where
    Self: GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn get_tensor_key(&self, _k: TorusPrecision) -> Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        Ok(self.to_backend_ref())
    }
}
