use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{Backend, Data, Module, ScratchArena, VmpPMat, VmpPMatBackendRef},
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
#[derive(PartialEq)]
pub struct GGLWEPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B::DftWord, B>,
    pub(crate) k_aux: TorusPrecision,
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

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum(), self.dsize, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.base2k, self.dnum(), self.dsize, self.k_aux)
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
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

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
    fn gglwe_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GGLWEPrepared<BE::OwnedBuf, BE> {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        GGLWEPrepared {
            data: self.vmp_pmat_alloc(dnum.into(), rank_in.into(), (rank_out + 1).into(), size),
            base2k,
            dsize,
            k_aux,
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
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    /// Returns the byte size required to store a [`GGLWEPrepared`] with the given parameters.
    fn gglwe_prepared_bytes_of(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

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
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank_in(),
            infos.rank_out(),
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
    fn gglwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>,
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

/// Read-only access to the prepared VMP matrix stored inside a GGLWE-like key.
///
/// This is intentionally narrower than exposing the prepared key internals as
/// mutable state. `VmpPMat` is the backend-prepared representation obtained from
/// a coefficient-domain matrix; callers that need different contents should
/// rebuild and prepare a new key/matrix instead of modifying this view.
///
/// The PIR collapse precompute uses this to run a specialized `1 x 1` VMP over
/// fixed mask data while still reusing the same prepared matrix representation
/// as the generic key-switch pipeline.
pub trait GGLWEPreparedVmpPMatRef<B: Backend> {
    /// Returns an immutable backend-native view of the underlying prepared VMP
    /// matrix.
    fn vmp_pmat_backend_ref(&self) -> VmpPMatBackendRef<'_, B>;
}

impl<B: Backend> GGLWEPreparedToBackendRef<B> for GGLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B> {
        GGLWEPrepared {
            base2k: self.base2k,
            k_aux: self.k_aux,
            dsize: self.dsize,
            data: self.data.to_backend_ref::<B>(),
        }
    }
}

impl<B: Backend> GGLWEPreparedVmpPMatRef<B> for GGLWEPrepared<B::OwnedBuf, B> {
    fn vmp_pmat_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        self.data.to_backend_ref::<B>()
    }
}

pub trait GGLWEPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B>;
}

impl<B: Backend> GGLWEPreparedToBackendMut<B> for GGLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, B> {
        GGLWEPrepared {
            base2k: self.base2k,
            k_aux: self.k_aux,
            dsize: self.dsize,
            data: self.data.to_backend_mut::<B>(),
        }
    }
}
