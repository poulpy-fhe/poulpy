use poulpy_hal::layouts::{Backend, Data};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{Base2K, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GLWEInfos, LWEInfos, Rank, TorusPrecision};

/// DFT-domain (prepared) variant of [`GGLWEToGGSWKey`](crate::layouts::GGLWEToGGSWKey).
///
/// Stores a collection of [`GGLWEPrepared`] matrices (one per rank element)
/// with polynomials in the frequency domain of the backend's DFT/NTT transform,
/// enabling O(N log N) polynomial multiplication. Used for GGLWE-to-GGSW
/// key-switching operations.
///
/// Requires `rank_in == rank_out`. Tied to a specific backend via `BE: Backend`.
pub struct GGLWEToGGSWKeyPrepared<D: Data, BE: Backend> {
    pub(crate) keys: Vec<GGLWEPrepared<D, BE>>,
}

/// Provides LWE-level parameter accessors, delegating to the first key element.
impl<D: Data, BE: Backend> LWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        crate::layouts::common_noise(self.keys.iter().map(crate::layouts::LWEInfos::noise))
    }

    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn max_size(&self) -> usize {
        self.keys[0].max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }
}

/// Provides the GLWE rank, derived from the output rank.
impl<D: Data, BE: Backend> GLWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

/// Provides GGLWE-specific parameter accessors. Note that `rank_in == rank_out` for this type.
impl<D: Data, BE: Backend> GGLWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.keys[0].k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.keys[0].k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.keys[0].k_offset()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }

    fn stride(&self) -> usize {
        self.keys[0].stride()
    }
}

impl<D: Data, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    /// Returns a mutable reference to the `i`-th prepared GGLWE key element.
    ///
    /// The `i`-th element corresponds to `GGLWEPrepared_s([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])`.
    ///
    /// # Panics
    ///
    /// If `i >= rank`.
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWEPrepared<D, BE> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: Data, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    /// Returns a reference to the `i`-th prepared GGLWE key element.
    ///
    /// The `i`-th element corresponds to `GGLWEPrepared_s([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])`.
    ///
    /// # Panics
    ///
    /// If `i >= rank`.
    pub fn at(&self, i: usize) -> &GGLWEPrepared<D, BE> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

pub type GGLWEToGGSWKeyPreparedBackendRef<'a, B> = GGLWEToGGSWKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GGLWEToGGSWKeyPreparedBackendMut<'a, B> = GGLWEToGGSWKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GGLWEToGGSWKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GGLWEToGGSWKeyPreparedToBackendRef<B> for GGLWEToGGSWKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyPreparedBackendRef<'_, B> {
        GGLWEToGGSWKeyPrepared {
            keys: self.keys.iter().map(|c| c.to_backend_ref()).collect(),
        }
    }
}

pub trait GGLWEToGGSWKeyPreparedToBackendMut<B: Backend> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GGLWEToGGSWKeyPreparedToBackendMut<B> for GGLWEToGGSWKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        for key in &mut self.keys {
            key.set_noise(metadata.clone());
        }
    }

    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyPreparedBackendMut<'_, B> {
        GGLWEToGGSWKeyPrepared {
            keys: self.keys.iter_mut().map(|c| c.to_backend_mut()).collect(),
        }
    }
}
