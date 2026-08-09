use poulpy_hal::layouts::VmpPMatToBackendMut;
use poulpy_hal::layouts::VmpPMatToBackendRef;
use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes, VmpZero},
    layouts::{Backend, HostDataRef, Module, ScratchArena, VmpPMat},
};

use crate::layouts::GGSWCore;
use crate::layouts::{Base2K, Dnum, Dsize, GGSWInfos, GGSWToBackendRef, GetDegree, LWEInfos, Rank, TorusPrecision};

/// DFT-domain (prepared) variant of [`GGSW`].
///
/// Stores the GGSW gadget matrix with polynomials in the frequency domain
/// of the backend's DFT/NTT transform, enabling O(N log N) polynomial
/// operations. Tied to a specific backend via `B: Backend`.
/// This is [`GGSWCore`] over a `VmpPMat` payload: the same semantic object as a
/// coefficient-domain GGSW, in the prepared domain. `LWEInfos` / `GLWEInfos` /
/// `GGSWInfos` all come from the payload-generic impls on `GGSWCore`.
pub type GGSWPrepared<D, B> = GGSWCore<VmpPMat<D, <B as Backend>::DftWord, B>>;

pub type GGSWPreparedBackendRef<'a, B> = GGSWPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GGSWPreparedBackendMut<'a, B> = GGSWPrepared<<B as Backend>::BufMut<'a>, B>;

/// Trait for allocating and preparing DFT-domain GGSW ciphertexts.
pub trait GGSWPreparedFactory<B: Backend>
where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf + VmpPrepareTmpBytes + VmpPrepare<B> + VmpZero<B>,
{
    /// Allocates a new prepared GGSW with the given parameters.
    fn ggsw_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GGSWPrepared<B::OwnedBuf, B> {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        GGSWPrepared {
            data: self.vmp_pmat_alloc(dnum.into(), (rank + 1).into(), (rank + 1).into(), size),
            base2k,
            dsize,
            k_aux,
        }
    }

    fn ggsw_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGSWPrepared<B::OwnedBuf, B>
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.ggsw_prepared_alloc(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn ggsw_prepared_bytes_of(&self, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        self.bytes_of_vmp_pmat(dnum.into(), (rank + 1).into(), (rank + 1).into(), size)
    }

    fn ggsw_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.ggsw_prepared_bytes_of(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn ggsw_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        let lvl_0: usize = self.vmp_prepare_tmp_bytes(
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        lvl_0
    }
    fn ggsw_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, B>)
    where
        R: GGSWPreparedToBackendMut<B>,
        O: GGSWToBackendRef<B>,
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        assert_eq!(res.n(), self.ring_degree());
        assert_eq!(other.n(), self.ring_degree());
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.dsize, other.dsize);
        assert!(
            scratch.available() >= self.ggsw_prepare_tmp_bytes(&res),
            "scratch.available(): {} < GGSWPreparedFactory::ggsw_prepare_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_prepare_tmp_bytes(&res)
        );
        self.vmp_prepare(&mut res.data, &other.data, scratch);
    }

    fn ggsw_zero<R>(&self, res: &mut R)
    where
        R: GGSWPreparedToBackendMut<B>,
    {
        let mut res = res.to_backend_mut();
        self.vmp_zero(&mut res.data);
    }
}

impl<B: Backend> GGSWPreparedFactory<B> for Module<B> where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf + VmpPrepareTmpBytes + VmpPrepare<B> + VmpZero<B>
{
}

// module-only API: allocation/size helpers are provided by `GGSWPreparedFactory` on `Module`.

impl<D: HostDataRef, B: Backend> GGSWPrepared<D, B> {
    pub fn data(&self) -> &VmpPMat<D, B::DftWord, B> {
        &self.data
    }
}

// module-only API: preparation sizing is provided by `GGSWPreparedFactory` on `Module`.

// module-only API: preparation and zeroing are provided by `GGSWPreparedFactory` on `Module`.

pub trait GGSWPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, B>;
}

impl<B: Backend> GGSWPreparedToBackendRef<B> for GGSWPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, B> {
        GGSWPrepared {
            base2k: self.base2k,
            k_aux: self.k_aux,
            dsize: self.dsize,
            data: self.data.to_backend_ref(),
        }
    }
}

pub trait GGSWPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, B>;
}

impl<B: Backend> GGSWPreparedToBackendMut<B> for GGSWPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, B> {
        GGSWPrepared {
            base2k: self.base2k,
            k_aux: self.k_aux,
            dsize: self.dsize,
            data: self.data.to_backend_mut(),
        }
    }
}
