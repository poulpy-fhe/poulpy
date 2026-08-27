//! Precision-driven key lookup.
//!
//! A caller names a function and the exact precision it is about to work at;
//! the helper owns the [`GGLWEKeyUsePolicy`](super::GGLWEKeyUsePolicy) and
//! answers with a complete physical key and the effective `dsize` to use it
//! through. The `dsize` may differ from the key's native one, so resolution,
//! scratch and execution must all be given the returned value.
//!
//! The layout twins answer the same question without a backend, for scratch
//! planning. [`GGLWEKeyRegistry::try_map_values`](super::GGLWEKeyRegistry::try_map_values)
//! keeps a layout registry and its prepared registry selecting the same
//! physical key.

use std::collections::HashMap;
use std::hash::Hash;

use poulpy_hal::layouts::Backend;

use crate::{
    error::Result,
    layouts::gglwe_key_use::err,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEKeyRegistry, GGLWEPreparedBackendRef, GLWEInfos, GetGaloisElement,
        LWEInfos, Rank, TorusPrecision,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};

/// Automorphism key for Galois element `p` at exact precision `k`.
pub trait GLWEAutomorphismKeyHelper<K> {
    fn get_automorphism_key_for(&self, p: i64, k: TorusPrecision) -> Result<(&K, Dsize)>;
}

/// Backend-free twin of [`GLWEAutomorphismKeyHelper`] for scratch planning.
pub trait GLWEAutomorphismKeyLayoutHelper<L: GGLWEInfos> {
    fn get_automorphism_key_layout_for(&self, p: i64, k: TorusPrecision) -> Result<(&L, Dsize)>;
}

/// Relinearization key at exact precision `k`.
///
/// The key type is associated rather than a parameter: a source answers with
/// one kind of key, and a bare key is a source of itself, so a parameter would
/// leave it ambiguous at every call site.
pub trait GLWERelinearizationKeyHelper {
    type Key;
    fn get_relinearization_key_for(&self, k: TorusPrecision) -> Result<(&Self::Key, Dsize)>;
}

/// Backend-free twin of [`GLWERelinearizationKeyHelper`] for scratch planning.
pub trait GLWERelinearizationKeyLayoutHelper {
    type Layout: GGLWEInfos;
    fn get_relinearization_key_layout_for(&self, k: TorusPrecision) -> Result<(&Self::Layout, Dsize)>;
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyHelper<K> for GGLWEKeyRegistry<i64, K> {
    fn get_automorphism_key_for(&self, p: i64, k: TorusPrecision) -> Result<(&K, Dsize)> {
        self.key_for(&p, k)
    }
}

impl<L: GGLWEInfos> GLWEAutomorphismKeyLayoutHelper<L> for GGLWEKeyRegistry<i64, L> {
    fn get_automorphism_key_layout_for(&self, p: i64, k: TorusPrecision) -> Result<(&L, Dsize)> {
        self.key_for(&p, k)
    }
}

impl<K: GGLWEInfos> GLWERelinearizationKeyHelper for GGLWEKeyRegistry<(), K> {
    type Key = K;

    fn get_relinearization_key_for(&self, k: TorusPrecision) -> Result<(&K, Dsize)> {
        self.key_for(&(), k)
    }
}

impl<L: GGLWEInfos> GLWERelinearizationKeyLayoutHelper for GGLWEKeyRegistry<(), L> {
    type Layout = L;

    fn get_relinearization_key_layout_for(&self, k: TorusPrecision) -> Result<(&L, Dsize)> {
        self.key_for(&(), k)
    }
}

/// Adapter over a single key, so a caller holding one key still goes through
/// the policy instead of silently using the key's native `dsize`.
pub struct GGLWESingleKey<Id, K> {
    registry: GGLWEKeyRegistry<Id, K>,
}

impl<Id: Clone + Eq + Hash, K: GGLWEInfos> GGLWESingleKey<Id, K> {
    /// Registers `key` under `id` and compiles dispatch for `policy`.
    pub fn new(id: Id, key: K, policy: super::GGLWEKeyUsePolicy) -> Result<Self> {
        let mut builder = super::GGLWEKeyRegistryBuilder::new();
        builder.register(id, key)?;
        Ok(Self {
            registry: builder.finish(policy)?,
        })
    }

    pub fn registry(&self) -> &GGLWEKeyRegistry<Id, K> {
        &self.registry
    }
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyHelper<K> for GGLWESingleKey<i64, K> {
    fn get_automorphism_key_for(&self, p: i64, k: TorusPrecision) -> Result<(&K, Dsize)> {
        self.registry.key_for(&p, k)
    }
}

impl<K: GGLWEInfos> GLWERelinearizationKeyHelper for GGLWESingleKey<(), K> {
    type Key = K;

    fn get_relinearization_key_for(&self, k: TorusPrecision) -> Result<(&K, Dsize)> {
        self.registry.key_for(&(), k)
    }
}

impl<L: GGLWEInfos> GLWEAutomorphismKeyLayoutHelper<L> for GGLWESingleKey<i64, L> {
    fn get_automorphism_key_layout_for(&self, p: i64, k: TorusPrecision) -> Result<(&L, Dsize)> {
        self.registry.key_for(&p, k)
    }
}

impl<L: GGLWEInfos> GLWERelinearizationKeyLayoutHelper for GGLWESingleKey<(), L> {
    type Layout = L;

    fn get_relinearization_key_layout_for(&self, k: TorusPrecision) -> Result<(&L, Dsize)> {
        self.registry.key_for(&(), k)
    }
}

/// A plain map answers with each key used through its own decomposition, which
/// is what callers holding no policy expect.
impl<K: GGLWEInfos> GLWEAutomorphismKeyHelper<K> for HashMap<i64, K> {
    fn get_automorphism_key_for(&self, p: i64, _k: TorusPrecision) -> Result<(&K, Dsize)> {
        let key: &K = self
            .get(&p)
            .ok_or_else(|| err("get_automorphism_key_for", format!("no automorphism key for p={p}")))?;
        Ok((key, key.effective_dsize()))
    }
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyLayoutHelper<K> for HashMap<i64, K> {
    fn get_automorphism_key_layout_for(&self, p: i64, k: TorusPrecision) -> Result<(&K, Dsize)> {
        self.get_automorphism_key_for(p, k)
    }
}

/// A bare layout plans as "the same shape for every rotation, used natively",
/// which is what scratch sizing assumed before keys could differ per rotation.
impl<L: GGLWEInfos> GLWEAutomorphismKeyLayoutHelper<L> for L {
    fn get_automorphism_key_layout_for(&self, _p: i64, _k: TorusPrecision) -> Result<(&L, Dsize)> {
        Ok((self, self.effective_dsize()))
    }
}

/// A bare key is its own source, used through its own decomposition, which is
/// what callers holding no policy expect.
impl<K: GGLWEInfos> GLWERelinearizationKeyHelper for K {
    type Key = K;

    fn get_relinearization_key_for(&self, _k: TorusPrecision) -> Result<(&K, Dsize)> {
        Ok((self, self.effective_dsize()))
    }
}

impl<L: GGLWEInfos> GLWERelinearizationKeyLayoutHelper for L {
    type Layout = L;

    fn get_relinearization_key_layout_for(&self, _k: TorusPrecision) -> Result<(&L, Dsize)> {
        Ok((self, self.effective_dsize()))
    }
}

/// A key together with the decomposition it is being used through.
///
/// Forwards every layout accessor to the key it borrows, except
/// [`GGLWEInfos::effective_dsize`]. Operations therefore take it wherever they
/// take a key, with no signature of their own to change. It holds a borrow and
/// a scalar: no rows are copied, projected or materialized.
pub struct GGLWEKeyUse<'a, K> {
    key: &'a K,
    effective_dsize: Dsize,
}

impl<'a, K> GGLWEKeyUse<'a, K> {
    pub fn key(&self) -> &'a K {
        self.key
    }
}

impl<K: LWEInfos> LWEInfos for GGLWEKeyUse<'_, K> {
    fn n(&self) -> Degree {
        self.key.n()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }

    fn size(&self) -> usize {
        self.key.size()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }
}

impl<K: GLWEInfos> GLWEInfos for GGLWEKeyUse<'_, K> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<K: GGLWEInfos> GGLWEInfos for GGLWEKeyUse<'_, K> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.key.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    /// The only accessor that is not a forward.
    fn effective_dsize(&self) -> Dsize {
        self.effective_dsize
    }
}

impl<K: GetGaloisElement> GetGaloisElement for GGLWEKeyUse<'_, K> {
    fn p(&self) -> i64 {
        self.key.p()
    }
}

impl<BE: Backend, K: GGLWEPreparedToBackendRef<BE>> GGLWEPreparedToBackendRef<BE> for GGLWEKeyUse<'_, K> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, BE> {
        GGLWEPreparedToBackendRef::to_backend_ref(self.key)
    }
}

impl<BE: Backend, K: GLWETensorKeyPreparedToBackendRef<BE>> GLWETensorKeyPreparedToBackendRef<BE> for GGLWEKeyUse<'_, K> {
    fn to_backend_ref(&self) -> GLWETensorKeyPreparedBackendRef<'_, BE> {
        GLWETensorKeyPreparedToBackendRef::to_backend_ref(self.key)
    }
}

/// Pairs any key with an effective `dsize`.
pub trait WithEffectiveDsize: Sized {
    fn with_dsize(&self, effective_dsize: Dsize) -> GGLWEKeyUse<'_, Self> {
        GGLWEKeyUse {
            key: self,
            effective_dsize,
        }
    }
}

impl<T: GGLWEInfos> WithEffectiveDsize for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::{Base2K, Degree, Dnum, GGLWEKeyRegistryBuilder, GGLWEKeyUsePolicy, GGLWELayout, Rank, TorusPrecision};

    const N: Degree = Degree(1024);
    const K: u32 = 8;

    fn layout(dsize: u32, dnum: u32) -> GGLWELayout {
        GGLWELayout {
            n: N,
            base2k: Base2K(K),
            dnum: Dnum(dnum),
            k_aux: TorusPrecision(dsize * K + N.log2() as u32),
            rank_in: Rank(1),
            rank_out: Rank(1),
            dsize: Dsize(dsize),
        }
    }

    fn policy() -> GGLWEKeyUsePolicy {
        GGLWEKeyUsePolicy::new(Base2K(K), Box::new([Dsize(1), Dsize(1), Dsize(8), Dsize(8), Dsize(16)])).unwrap()
    }

    // Acceptance 12: the helper answers with a physical key and the effective
    // dsize, driven by exact `k`.
    #[test]
    fn automorphism_helper_returns_effective_dsize() {
        let mut builder: GGLWEKeyRegistryBuilder<i64, GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        builder.register(-5, layout(8, 4)).unwrap();
        let registry = builder.finish(policy()).unwrap();

        // Same physical key, two precisions, two effective decompositions.
        let (key, d) = registry.get_automorphism_key_for(-5, TorusPrecision(2 * K)).unwrap();
        assert_eq!((key.dsize, d), (Dsize(8), Dsize(8)));
        let (key, d) = registry.get_automorphism_key_for(-5, TorusPrecision(4 * K)).unwrap();
        assert_eq!((key.dsize, d), (Dsize(8), Dsize(16)));
        // An unregistered rotation is an error, not a fallback.
        assert!(registry.get_automorphism_key_for(-7, TorusPrecision(2 * K)).is_err());
    }

    // A single-key adapter applies the policy rather than its native dsize.
    #[test]
    fn single_key_adapter_applies_the_policy() {
        let single: GGLWESingleKey<(), GGLWELayout> = GGLWESingleKey::new((), layout(8, 4), policy()).unwrap();
        let (key, d) = single.get_relinearization_key_for(TorusPrecision(4 * K)).unwrap();
        assert_eq!(key.dsize, Dsize(8));
        assert_eq!(d, Dsize(16));
    }

    // Acceptance 11: layout planning and key lookup select the same physical key.
    #[test]
    fn layout_helper_tracks_the_key_helper() {
        let mut builder: GGLWEKeyRegistryBuilder<i64, GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        builder.register(-5, layout(8, 4)).unwrap();
        builder.register(-5, layout(3, 4)).unwrap();
        let registry = builder.finish(policy()).unwrap();
        let mapped = registry.try_map_values(|_, key| Ok(*key)).unwrap();

        for k in [2 * K, 3 * K, 4 * K] {
            let (key, d) = registry.get_automorphism_key_for(-5, TorusPrecision(k)).unwrap();
            let (layout, layout_d) = mapped.get_automorphism_key_layout_for(-5, TorusPrecision(k)).unwrap();
            assert_eq!((key.gglwe_layout(), d), (layout.gglwe_layout(), layout_d));
        }
    }
}
