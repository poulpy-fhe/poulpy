use poulpy_hal::{
    api::{ScratchAvailable, VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{
        Backend, Data, Module, ScratchArena, VmpPMat, VmpPMatReborrowBackendRef, VmpPMatToBackendMut, VmpPMatToBackendRef,
        vmp_pmat_backend_mut_from_mut, vmp_pmat_backend_ref_from_ref,
    },
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEToBackendRef, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision,
};

/// DFT-domain (prepared) variant of [`GGLWE`].
///
/// Stores the gadget GLWE matrix with polynomials in the frequency domain
/// of the backend's DFT/NTT transform, enabling O(N log N) polynomial
/// multiplication. The underlying data is held as a [`VmpPMat`], which
/// represents a prepared matrix suitable for vector-matrix products.
///
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GGLWEPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B>,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub type GGLWEPreparedBackendRef<'a, B> = GGLWEPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GGLWEPreparedBackendMut<'a, B> = GGLWEPrepared<<B as Backend>::BufMut<'a>, B>;

/// Provides LWE-level parameter accessors (degree, base2k, precision, size).
impl<D: Data, B: Backend> LWEInfos for GGLWEPrepared<D, B> {
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

/// Provides the GLWE rank, derived from the output rank.
impl<D: Data, B: Backend> GLWEInfos for GGLWEPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

/// Provides GGLWE-specific parameter accessors (input/output rank, dsize, dnum).
impl<D: Data, B: Backend> GGLWEInfos for GGLWEPrepared<D, B> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

/// Factory trait for allocating and preparing [`GGLWEPrepared`] instances.
///
/// Requires the backend module to support VMP prepared-matrix allocation,
/// byte-size queries, and the prepare transform.
pub trait GGLWEPreparedFactory<BE: Backend>
where
    Self: GetDegree + VmpPMatAlloc<BE> + VmpPMatBytesOf + VmpPrepare<BE> + VmpPrepareTmpBytes,
{
    /// Allocates a new [`GGLWEPrepared`] with the given parameters.
    ///
    /// Panics if `dnum * dsize > ceil(k / base2k)`.
    fn gglwe_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWEPrepared<BE::OwnedBuf, BE> {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        GGLWEPrepared {
            data: self.vmp_pmat_alloc(dnum.into(), rank_in.into(), (rank_out + 1).into(), size),
            base2k,
            dsize,
        }
    }

    /// Allocates a new [`GGLWEPrepared`] matching the parameters of `infos`.
    fn gglwe_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGLWEPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.gglwe_prepared_alloc(
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the byte size required to store a [`GGLWEPrepared`] with the given parameters.
    fn gglwe_prepared_bytes_of(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        self.bytes_of_vmp_pmat(dnum.into(), rank_in.into(), (rank_out + 1).into(), size)
    }

    /// Returns the byte size required to store a [`GGLWEPrepared`] matching `infos`.
    fn gglwe_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.ring_degree(), infos.n());
        self.gglwe_prepared_bytes_of(
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the scratch-space bytes needed by [`gglwe_prepare`](Self::gglwe_prepare).
    fn gglwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = self.vmp_prepare_tmp_bytes(
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        lvl_0
    }

    /// Transforms a standard [`GGLWE`] into the DFT domain, writing the result into `res`.
    ///
    /// Both `res` and `other` must share the same ring degree, base2k, precision, and dsize.
    fn gglwe_prepare<'s, R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchAvailable,
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.n(), self.ring_degree());
        assert_eq!(other.n(), self.ring_degree());
        assert_eq!(res.base2k, other.base2k);
        assert_eq!(res.size(), other.size());
        assert_eq!(res.dsize, other.dsize);
        assert!(
            scratch.available() >= self.gglwe_prepare_tmp_bytes(&res),
            "scratch.available(): {} < GGLWEPreparedFactory::gglwe_prepare_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_prepare_tmp_bytes(&res)
        );
        self.vmp_prepare(&mut res.data, &other.data, scratch);
    }
}

impl<BE: Backend> GGLWEPreparedFactory<BE> for Module<BE> where
    Module<BE>: GetDegree + VmpPMatAlloc<BE> + VmpPMatBytesOf + VmpPrepare<BE> + VmpPrepareTmpBytes
{
}

// module-only API: allocation/size helpers are provided by `GGLWEPreparedFactory` on `Module`.

// module-only API: preparation is provided by `GGLWEPreparedFactory` on `Module`.

pub trait GGLWEPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B>;
}

impl<B: Backend> GGLWEPreparedToBackendRef<B> for GGLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<'b, B: Backend + 'b> GGLWEPreparedToBackendRef<B> for &GGLWEPrepared<B::BufRef<'b>, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: vmp_pmat_backend_ref_from_ref::<B>(&self.data),
        }
    }
}

impl<'b, B: Backend + 'b> GGLWEPreparedToBackendRef<B> for &mut GGLWEPrepared<B::BufMut<'b>, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: self.data.reborrow_backend_ref(),
        }
    }
}

pub trait GGLWEPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B>;
}

impl<B: Backend> GGLWEPreparedToBackendMut<B> for GGLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        GGLWEPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: self.data.to_backend_mut(),
        }
    }
}

impl<'b, B: Backend + 'b> GGLWEPreparedToBackendMut<B> for &mut GGLWEPrepared<B::BufMut<'b>, B> {
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        GGLWEPrepared {
            base2k: self.base2k,
            dsize: self.dsize,
            data: vmp_pmat_backend_mut_from_mut::<B>(&mut self.data),
        }
    }
}
