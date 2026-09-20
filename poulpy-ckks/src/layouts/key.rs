use super::CKKSRing;
use crate::CKKSResult;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::Backend;
use std::{collections::HashMap, ops::Deref};

/// A key or immutable key provider bound to its evaluation ring.
#[derive(Clone)]
pub struct CKKSKey<K> {
    inner: K,
    ring: CKKSRing,
}

impl<K> CKKSKey<K> {
    /// Imports a raw key. The caller supplies its generation and preparation ring.
    pub fn from_raw_parts(inner: K, ring: CKKSRing) -> CKKSResult<Self>
    where
        K: LWEInfos,
    {
        ring.check(
            "key import",
            CKKSRing {
                kind: ring.kind,
                n: inner.n(),
            },
        )?;
        Ok(Self { inner, ring })
    }

    /// Imports a provider whose every returned key must belong to `ring`.
    /// The caller guarantees this invariant, including for lazily prepared keys.
    pub fn from_raw_provider(inner: K, ring: CKKSRing) -> Self {
        Self { inner, ring }
    }

    /// Ring used to generate or prepare the key material.
    pub fn key_ring(&self) -> CKKSRing {
        self.ring
    }

    /// Borrows the underlying Core representation.
    pub fn as_core(&self) -> &K {
        &self.inner
    }

    /// Discards the CKKS provenance and returns the Core representation.
    pub fn into_core(self) -> K {
        self.inner
    }
}

impl<K> CKKSKey<HashMap<i64, K>> {
    /// Decomposes the collection, preserving the ring on every key.
    pub fn into_keys(self) -> HashMap<i64, CKKSKey<K>> {
        self.inner
            .into_iter()
            .map(|(p, inner)| (p, CKKSKey { inner, ring: self.ring }))
            .collect()
    }

    /// Builds an immutable provider, rejecting any key from a different ring.
    pub fn from_keys(keys: HashMap<i64, CKKSKey<K>>, ring: CKKSRing) -> CKKSResult<Self> {
        for key in keys.values() {
            ring.check("key collection", key.ring)?;
        }
        Ok(Self {
            inner: keys.into_iter().map(|(p, key)| (p, key.inner)).collect(),
            ring,
        })
    }
}

impl<K> Deref for CKKSKey<K> {
    type Target = K;
    fn deref(&self) -> &K {
        &self.inner
    }
}

impl<K> CKKSKey<K> {
    /// Prepares the secret after validating its ring against the module.
    pub fn prepare_secret<BE: Backend>(
        &self,
        module: &poulpy_hal::layouts::Module<BE>,
    ) -> CKKSResult<CKKSKey<GLWESecretPrepared<BE::OwnedBuf, BE>>>
    where
        K: GLWESecretToBackendRef<BE> + GLWEInfos + poulpy_core::GetDistribution,
        poulpy_hal::layouts::Module<BE>: GLWESecretPreparedFactory<BE>,
    {
        crate::api::CKKSModuleInfos::ckks_ring(module).check("secret preparation", self.ring)?;
        let mut prepared = module.glwe_secret_prepared_alloc_from_infos(&self.inner);
        module.glwe_secret_prepare(&mut prepared, &self.inner);
        CKKSKey::from_raw_parts(prepared, self.ring)
    }
    /// Prepares the tensor key after validating its ring against the module.
    pub fn prepare_tensor<BE: Backend>(
        &self,
        module: &poulpy_hal::layouts::Module<BE>,
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) -> CKKSResult<CKKSKey<GLWETensorKeyPrepared<BE::OwnedBuf, BE>>>
    where
        K: GGLWEToBackendRef<BE> + GGLWEInfos,
        poulpy_hal::layouts::Module<BE>: GLWETensorKeyPreparedFactory<BE>,
    {
        crate::api::CKKSModuleInfos::ckks_ring(module).check("tensor-key preparation", self.ring)?;
        let mut prepared = module.alloc_tensor_key_prepared_from_infos(&self.inner);
        module.prepare_tensor_key(&mut prepared, &self.inner, scratch);
        CKKSKey::from_raw_parts(prepared, self.ring)
    }
    /// Prepares the automorphism key after validating its ring against the module.
    pub fn prepare_automorphism<BE: Backend>(
        &self,
        module: &poulpy_hal::layouts::Module<BE>,
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) -> CKKSResult<CKKSKey<GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>>
    where
        K: GGLWEToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        poulpy_hal::layouts::Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE>,
    {
        crate::api::CKKSModuleInfos::ckks_ring(module).check("automorphism-key preparation", self.ring)?;
        let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&self.inner);
        module.glwe_automorphism_key_prepare(&mut prepared, &self.inner, scratch);
        CKKSKey::from_raw_parts(prepared, self.ring)
    }
    /// Prepares the switching key after validating its ring against the module.
    pub fn prepare_switching<BE: Backend>(
        &self,
        module: &poulpy_hal::layouts::Module<BE>,
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) -> CKKSResult<CKKSKey<GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>>>
    where
        K: GGLWEToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
        poulpy_hal::layouts::Module<BE>: GLWESwitchingKeyPreparedFactory<BE>,
    {
        crate::api::CKKSModuleInfos::ckks_ring(module).check("switching-key preparation", self.ring)?;
        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(&self.inner);
        module.glwe_switching_key_prepare(&mut prepared, &self.inner, scratch);
        CKKSKey::from_raw_parts(prepared, self.ring)
    }
}
