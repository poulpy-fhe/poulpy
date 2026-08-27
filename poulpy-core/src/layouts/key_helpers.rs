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

use std::hash::Hash;

use crate::{
    error::Result,
    layouts::{Dsize, GGLWEInfos, GGLWEKeyRegistry, TorusPrecision},
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
pub trait GLWERelinearizationKeyHelper<K> {
    fn get_relinearization_key_for(&self, k: TorusPrecision) -> Result<(&K, Dsize)>;
}

/// Backend-free twin of [`GLWERelinearizationKeyHelper`] for scratch planning.
pub trait GLWERelinearizationKeyLayoutHelper<L: GGLWEInfos> {
    fn get_relinearization_key_layout_for(&self, k: TorusPrecision) -> Result<(&L, Dsize)>;
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

impl<K: GGLWEInfos> GLWERelinearizationKeyHelper<K> for GGLWEKeyRegistry<(), K> {
    fn get_relinearization_key_for(&self, k: TorusPrecision) -> Result<(&K, Dsize)> {
        self.key_for(&(), k)
    }
}

impl<L: GGLWEInfos> GLWERelinearizationKeyLayoutHelper<L> for GGLWEKeyRegistry<(), L> {
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

impl<K: GGLWEInfos> GLWERelinearizationKeyHelper<K> for GGLWESingleKey<(), K> {
    fn get_relinearization_key_for(&self, k: TorusPrecision) -> Result<(&K, Dsize)> {
        self.registry.key_for(&(), k)
    }
}

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
