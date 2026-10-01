use poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxCopy, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{
        Backend, Data, MatZnxToBackendMut, MatZnxToBackendRef, Module, ScratchArena, mat_znx_at_backend_mut_from_mut,
        mat_znx_at_backend_ref_from_ref,
    },
};

use crate::layouts::prepared::{GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef};
use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GGLWEToBackendRef, GLWEInfos, LWEInfos, Rank,
    TorusPrecision,
};

/// [`GLWETensorKey`](crate::layouts::GLWETensorKey) prepared for
/// [`GGSWExpandRows`](crate::api::GGSWExpandRows): one [`GGLWEPrepared`] per
/// rank element, whose input column `j` of key `i` encrypts `s[i]*s[j]`.
///
/// Each key has `rank_in == rank_out`, so expanding a GGSW column is one
/// vector-matrix product. Tied to a specific backend via `BE: Backend`.
pub struct GGLWEToGGSWKeyPrepared<D: Data, BE: Backend> {
    pub(crate) keys: Vec<GGLWEPrepared<D, BE>>,
}

/// Provides LWE-level parameter accessors, delegating to the first key element.
impl<D: Data, BE: Backend> LWEInfos for GGLWEToGGSWKeyPrepared<D, BE> {
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

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }

    fn stride(&self) -> usize {
        self.keys[0].stride()
    }
}

/// Factory trait for allocating and preparing [`GGLWEToGGSWKeyPrepared`] instances.
pub trait GGLWEToGGSWKeyPreparedFactory<BE: Backend> {
    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] for the tensor key layout
    /// `infos`; only its rank, gadget and precision are read.
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

    /// Prepares the [`GLWETensorKey`](crate::layouts::GLWETensorKey) `tsk` into `res`:
    /// input column `j` of key `i` is the tensor key entry of `s[min(i, j)]*s[max(i, j)]`.
    fn gglwe_to_ggsw_key_prepare<R, O>(&self, res: &mut R, tsk: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE> + GGLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyPreparedFactory<BE> for Module<BE>
where
    Self: GGLWEPreparedFactory<BE> + VecZnxCopy<BE>,
{
    fn gglwe_to_ggsw_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
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
        self.bytes_of_gglwe_to_ggsw(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }

    fn bytes_of_gglwe_to_ggsw(&self, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        rank.as_usize() * self.gglwe_prepared_bytes_of(base2k, dnum, dsize, k_aux, rank, rank)
    }

    fn gglwe_to_ggsw_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let rank: usize = infos.rank_out().as_usize();
        let key = BE::bytes_of_mat_znx(infos.n().as_usize(), infos.dnum().as_usize(), rank, rank + 1, infos.size());
        BE::scratch_aligned(key) + self.vmp_prepare_tmp_bytes(infos.dnum().as_usize(), rank, rank + 1, infos.size())
    }

    fn gglwe_to_ggsw_key_prepare<R, O>(&self, res: &mut R, tsk: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        let rank: usize = tsk.rank_out().as_usize();
        assert_eq!(
            tsk.rank_in().as_usize(),
            (rank * (rank + 1) / 2).max(1),
            "tsk must be a GLWETensorKey"
        );
        let needed = self.gglwe_to_ggsw_key_prepare_tmp_bytes(tsk);
        assert!(
            scratch.available() >= needed,
            "scratch.available(): {} < GGLWEToGGSWKeyPreparedFactory::gglwe_to_ggsw_key_prepare_tmp_bytes: {}",
            scratch.available(),
            needed
        );

        let mut res = res.to_backend_mut();
        let tsk = tsk.to_backend_ref();
        assert_eq!(res.keys.len(), rank);

        let (mut key, mut scratch_1) =
            scratch
                .borrow()
                .take_mat_znx_scratch(tsk.n().as_usize(), tsk.dnum().as_usize(), rank, rank + 1, tsk.size());
        for (i, res_i) in res.keys.iter_mut().enumerate() {
            {
                let mut key_mut = key.to_backend_mut();
                for row in 0..tsk.dnum().as_usize() {
                    for j in 0..rank {
                        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                        let src = mat_znx_at_backend_ref_from_ref::<BE>(&tsk.data, row, lo * rank + hi - lo * (lo + 1) / 2);
                        let mut dst = mat_znx_at_backend_mut_from_mut::<BE>(&mut key_mut, row, j);
                        for col in 0..rank + 1 {
                            self.vec_znx_copy(&mut dst, col, &src, col);
                        }
                    }
                }
            }
            self.vmp_prepare(&mut res_i.data, &key.to_backend_ref(), &mut scratch_1.borrow());
        }
    }
}

// module-only API: allocation, sizing, and preparation are provided by
// `GGLWEToGGSWKeyPreparedFactory` on `Module`.

impl<D: Data, BE: Backend> GGLWEToGGSWKeyPrepared<D, BE> {
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
