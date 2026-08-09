use poulpy_hal::{
    api::VmpPrepare,
    layouts::{Backend, Data, HostDataMut, Module, ScratchArena, VmpPMat},
};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GGLWEToGGSWKeyCore, GGLWEToGGSWKeyToBackendRef,
    GLWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of [`GGLWEToGGSWKey`](crate::layouts::GGLWEToGGSWKey).
///
/// Stores a collection of [`GGLWEPrepared`] matrices (one per rank element)
/// with polynomials in the frequency domain of the backend's DFT/NTT transform,
/// enabling O(N log N) polynomial multiplication. Used for GGLWE-to-GGSW
/// key-switching operations.
///
/// Requires `rank_in == rank_out`. Tied to a specific backend via `BE: Backend`.
/// This is [`GGLWEToGGSWKeyCore`] over a `VmpPMat` payload; the `Infos` traits
/// come from the payload-generic impls there.
pub type GGLWEToGGSWKeyPrepared<D, BE> = GGLWEToGGSWKeyCore<VmpPMat<D, <BE as Backend>::DftWord, BE>>;

/// Factory trait for allocating and preparing [`GGLWEToGGSWKeyPrepared`] instances.
pub trait GGLWEToGGSWKeyPreparedFactory<BE: Backend> {
    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] matching the parameters of `infos`.
    ///
    /// Panics if `rank_in != rank_out`.
    fn gglwe_to_ggsw_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos;

    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    ///
    /// Creates `rank` prepared GGLWE matrices, one per secret-key component.
    fn gglwe_to_ggsw_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE>;

    /// Returns the byte size required to store a [`GGLWEToGGSWKeyPrepared`] matching `infos`.
    fn bytes_of_gglwe_to_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Returns the byte size required to store a [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    fn bytes_of_gglwe_to_ggsw(&self, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize;

    /// Returns the scratch-space bytes needed by [`gglwe_to_ggsw_key_prepare`](Self::gglwe_to_ggsw_key_prepare).
    fn gglwe_to_ggsw_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Transforms a standard [`GGLWEToGGSWKey`] into the DFT domain, writing into `res`.
    ///
    /// Iterates over each key element and prepares it individually.
    fn gglwe_to_ggsw_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToGGSWKeyToBackendRef<BE>;
}

impl<BE: Backend> GGLWEToGGSWKeyPreparedFactory<BE> for Module<BE>
where
    Self: GGLWEPreparedFactory<BE>,
{
    fn gglwe_to_ggsw_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        self.gglwe_to_ggsw_key_prepared_alloc(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn gglwe_to_ggsw_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> {
        GGLWEToGGSWKeyPrepared {
            keys: (0..rank.as_usize())
                .map(|_| self.gglwe_prepared_alloc(base2k, dnum, dsize, k_aux, rank, rank))
                .collect(),
        }
    }

    fn bytes_of_gglwe_to_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        self.bytes_of_gglwe_to_ggsw(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn bytes_of_gglwe_to_ggsw(&self, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        rank.as_usize() * self.gglwe_prepared_bytes_of(base2k, dnum, dsize, k_aux, rank, rank)
    }

    fn gglwe_to_ggsw_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.gglwe_prepare_tmp_bytes(infos);
        lvl_0
    }

    fn gglwe_to_ggsw_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToGGSWKeyToBackendRef<BE>,
    {
        let needed = {
            let res_infos = res.to_backend_mut();
            self.gglwe_to_ggsw_key_prepare_tmp_bytes(&res_infos)
        };
        assert!(
            scratch.available() >= needed,
            "scratch.available(): {} < GGLWEToGGSWKeyPreparedFactory::gglwe_to_ggsw_key_prepare_tmp_bytes: {}",
            scratch.available(),
            needed
        );

        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.keys.len(), other.keys.len());
        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.vmp_prepare(&mut a.data, &b.data, &mut scratch.borrow());
        }
    }
}

// module-only API: allocation, sizing, and preparation are provided by
// `GGLWEToGGSWKeyPreparedFactory` on `Module`.

impl<D: HostDataMut, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    /// Returns a mutable reference to the `i`-th prepared GGLWE key element.
    ///
    /// The `i`-th element corresponds to `GGLWEPrepared_s([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])`.
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWEPrepared<D, BE> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: Data, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
    /// Returns a reference to the `i`-th prepared GGLWE key element.
    ///
    /// The `i`-th element corresponds to `GGLWEPrepared_s([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])`.
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
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GGLWEToGGSWKeyPreparedToBackendMut<B> for GGLWEToGGSWKeyPrepared<D, B>
where
    GGLWEPrepared<D, B>: GGLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyPreparedBackendMut<'_, B> {
        GGLWEToGGSWKeyPrepared {
            keys: self.keys.iter_mut().map(|c| c.to_backend_mut()).collect(),
        }
    }
}
