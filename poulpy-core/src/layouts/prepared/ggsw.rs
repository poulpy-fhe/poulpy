use poulpy_hal::{
    api::{ScratchAvailable, VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes, VmpZero},
    layouts::{
        Backend, Data, HostDataRef, Module, ScratchArena, VmpPMat, VmpPMatReborrowBackendRef, VmpPMatToBackendMut,
        VmpPMatToBackendRef, vmp_pmat_backend_mut_from_mut, vmp_pmat_backend_ref_from_ref,
    },
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSWInfos, GGSWToBackendRef, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of [`GGSW`].
///
/// Stores the GGSW gadget matrix with polynomials in the frequency domain
/// of the backend's DFT/NTT transform, enabling O(N log N) polynomial
/// operations. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GGSWPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub type GGSWPreparedBackendRef<'a, B> = GGSWPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GGSWPreparedBackendMut<'a, B> = GGSWPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, B: Backend> LWEInfos for GGSWPrepared<D, B> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data, B: Backend> LWEInfos for &GGSWPrepared<D, B> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GGSWPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data, B: Backend> GLWEInfos for &GGSWPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data, B: Backend> GGSWInfos for GGSWPrepared<D, B> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: Data, B: Backend> GGSWInfos for &GGSWPrepared<D, B> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

/// Trait for allocating and preparing DFT-domain GGSW ciphertexts.
pub trait GGSWPreparedFactory<B: Backend>
where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf + VmpPrepareTmpBytes + VmpPrepare<B> + VmpZero<B>,
{
    /// Allocates a new prepared GGSW with the given parameters.
    fn ggsw_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> GGSWPrepared<B::OwnedBuf, B> {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        GGSWPrepared {
            data: self.vmp_pmat_alloc(
                dnum.into(),
                (rank + 1).into(),
                (rank + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
            ),
            base2k,
            dsize,
        }
    }

    fn ggsw_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGSWPrepared<B::OwnedBuf, B>
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.ggsw_prepared_alloc(infos.base2k(), infos.max_k(), infos.dnum(), infos.dsize(), infos.rank())
    }

    fn ggsw_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        self.bytes_of_vmp_pmat(dnum.into(), (rank + 1).into(), (rank + 1).into(), size)
    }

    fn ggsw_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.ggsw_prepared_bytes_of(infos.base2k(), infos.max_k(), infos.dnum(), infos.dsize(), infos.rank())
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
    fn ggsw_prepare<'s, R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'s, B>)
    where
        R: GGSWPreparedToBackendMut<B>,
        O: GGSWToBackendRef<B>,
        ScratchArena<'s, B>: ScratchAvailable,
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
    pub fn data(&self) -> &VmpPMat<D, B> {
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
            dsize: self.dsize,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<'b, B: Backend + 'b> GGSWPreparedToBackendRef<B> for &GGSWPrepared<B::BufRef<'b>, B> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, B> {
        GGSWPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: vmp_pmat_backend_ref_from_ref::<B>(&self.data),
        }
    }
}

impl<'b, B: Backend + 'b> GGSWPreparedToBackendRef<B> for &mut GGSWPrepared<B::BufMut<'b>, B> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, B> {
        GGSWPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: self.data.reborrow_backend_ref(),
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
            dsize: self.dsize,
            data: self.data.to_backend_mut(),
        }
    }
}

impl<'b, B: Backend + 'b> GGSWPreparedToBackendMut<B> for &mut GGSWPrepared<B::BufMut<'b>, B> {
    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, B> {
        GGSWPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: vmp_pmat_backend_mut_from_mut::<B>(&mut self.data),
        }
    }
}
