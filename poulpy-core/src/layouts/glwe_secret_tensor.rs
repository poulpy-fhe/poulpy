use poulpy_hal::layouts::VecZnxBigToBackendRef;
use poulpy_hal::layouts::VecZnxDftToBackendMut;
use poulpy_hal::layouts::VecZnxDftToBackendRef;
use poulpy_hal::layouts::{VecZnxBigToBackendMut, ZnxWord};
use poulpy_hal::{
    api::{
        ModuleN, SvpApplyPPolDftToDft, SvpPreparePPol, VecZnxBigAlloc, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, Data, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ScratchOwned,
        SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef, VecZnxDftOwned, ZnxView, ZnxViewMut,
        scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
};

use crate::{
    GetDistribution, GetDistributionMut, ScratchArenaTakeCore,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, GLWESecretPreparedFactory, GLWESecretToBackendRef, LWEInfos, Rank},
};

/// Number of distinct unordered secret-key products for a rank-`r` tensor key.
/// Depends only on the rank: neither the storage nor the coefficient word.
pub(crate) fn pairs(rank: usize) -> usize {
    (((rank + 1) * rank) >> 1).max(1)
}

/// Tensor of a [`GLWESecret`]: the `(rank + 1) * rank / 2` distinct products
/// `s_i * s_j` of a base secret `(s_0, ..., s_{rank-1})`, e.g.
/// `(1, s_0, s_1)^(x)2 = (s_0^2, s_0*s_1, s_1^2)`.
///
/// Note that `dist` is the tag of the *base* secret, not of the stored
/// products: the coefficients held here are neither ternary nor binary, and
/// no [`Distribution`] variant describes them. The tag is kept because the
/// products' own statistics are a closed-form function of the base
/// distribution (see [`Distribution`] for the variance).
pub struct GLWESecretTensor<D: Data, W: ZnxWord> {
    pub(crate) data: ScalarZnx<D, W>,
    pub(crate) rank: Rank,
    /// Distribution of the base secret this tensor was derived from, *not*
    /// of the `s_i * s_j` coefficients stored in `data`.
    pub(crate) dist: Distribution,
}

impl<W: ZnxWord> GLWESecretTensor<Vec<u8>, W> {}

impl<D: Data, W: ZnxWord> GetDistribution for GLWESecretTensor<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWESecretTensor<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord> GLWESecretTensor<D, W> {
    pub fn data(&self) -> &ScalarZnx<D, W> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ScalarZnx<D, W> {
        &mut self.data
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWESecretTensor<D, W> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        1
    }

    fn k(&self) -> super::TorusPrecision {
        unimplemented!("this method is not defined on secrets")
    }
}

impl<D: HostDataRef, W: ZnxWord> GLWESecretTensor<D, W> {
    pub fn at(&self, mut i: usize, mut j: usize) -> ScalarZnx<&[u8], W> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank().into();
        ScalarZnx::from_data(
            bytemuck::cast_slice(self.data.at(i * rank + j - (i * (i + 1) / 2), 0)),
            self.n().into(),
            1,
        )
    }
}

impl<D: HostDataMut, W: ZnxWord> GLWESecretTensor<D, W> {
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> ScalarZnx<&mut [u8], W> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank().into();
        let n = self.n().into();
        ScalarZnx::from_data(
            bytemuck::cast_slice_mut(self.data.at_mut(i * rank + j - (i * (i + 1) / 2), 0)),
            n,
            1,
        )
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWESecretTensor<D, W> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

pub type GLWESecretTensorBackendRef<'a, BE> = GLWESecretTensor<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type GLWESecretTensorBackendMut<'a, BE> = GLWESecretTensor<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

/// Backend view of a tensor secret.
///
/// Deliberately distinct from [`GLWESecretToBackendRef`]: a tensor secret is
/// not interchangeable with the base secret it was derived from, so it is
/// not accepted by APIs that ask for a [`GLWESecret`].
pub trait GLWESecretTensorToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWESecretTensorBackendRef<'_, BE>;
}

impl<BE: Backend> GLWESecretTensorToBackendRef<BE> for GLWESecretTensor<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GLWESecretTensorBackendRef<'_, BE> {
        GLWESecretTensor {
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            rank: self.rank,
            dist: self.dist,
        }
    }
}

/// Mutable backend view of a tensor secret. See [`GLWESecretTensorToBackendRef`].
pub trait GLWESecretTensorToBackendMut<BE: Backend>: GLWESecretTensorToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorBackendMut<'_, BE>;
}

impl<BE: Backend> GLWESecretTensorToBackendMut<BE> for GLWESecretTensor<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GLWESecretTensorBackendMut<'_, BE> {
        let rank = self.rank;
        GLWESecretTensor {
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as poulpy_hal::layouts::ScalarZnxToBackendMut<BE>>::to_backend_mut(
                &mut self.data,
            ),
            rank,
            dist: self.dist,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWESecretTensor<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecretTensor {
            data: ScalarZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>, W>::bytes_of(
                    n.into(),
                    pairs(rank.into()),
                )),
                n.into(),
                pairs(rank.into()),
            ),
            rank,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), pairs(infos.rank().into()).into())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::<Vec<u8>, W>::bytes_of(n.into(), pairs(rank.into()))
    }
}

// module-only API: secret tensor preparation is provided by `GLWESecretTensorFactory` on `Module`.

pub trait GLWESecretTensorFactory<BE: Backend> {
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize;

    /// Fills `res` with the pairwise products `s_i * s_j` of the base secret
    /// `other`.
    ///
    /// `res` inherits `other`'s [`Distribution`] tag. That tag keeps
    /// describing the base secret: the products written into `res` follow a
    /// different (derived) distribution, which no tag variant encodes.
    fn glwe_secret_tensor_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWESecretTensorToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWESecretTensorFactory<BE> for Module<BE>
where
    Self: ModuleN
        + GLWESecretPreparedFactory<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyPPolDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes,
{
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize {
        self.glwe_secret_prepared_bytes_of(rank)
    }

    fn glwe_secret_tensor_prepare<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWESecretTensorToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        A: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let res = &mut res.to_backend_mut();
        let a = a.to_backend_ref();

        // `res.rank()` is the rank of the base secret the tensor is derived
        // from; its column count is `pairs(rank)`.
        assert_eq!(res.rank(), a.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_secret_tensor_prepare_tmp_bytes(a.rank()),
            "scratch.available(): {} < GLWESecretTensorFactory::glwe_secret_tensor_prepare_tmp_bytes: {}",
            scratch.available(),
            self.glwe_secret_tensor_prepare_tmp_bytes(a.rank())
        );

        let rank: usize = a.rank().into();

        let scratch = scratch.borrow();
        let (mut a_prepared, _scratch_1) = scratch.take_glwe_secret_prepared_scratch(self, rank.into());
        {
            let mut a_prepared_data = a_prepared.data.reborrow_backend_mut();
            for i in 0..rank {
                self.svp_prepare_ppol(&mut a_prepared_data, i, a.data(), i);
            }
        }
        a_prepared.dist = *a.dist();

        let base2k: usize = 17;

        let mut a_dft = VecZnxDftOwned::<BE>::alloc(self.n(), rank, 1);
        let a_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(a.data());
        for i in 0..rank {
            let mut a_dft_backend = a_dft.to_backend_mut();
            self.vec_znx_dft_apply(1, 0, &mut a_dft_backend, i, &a_backend_vec, i);
        }

        let mut a_ij_dft = VecZnxDftOwned::<BE>::alloc(self.n(), 1, 1);
        let a_prepared_backend_ref = a_prepared.data.reborrow_backend_ref();
        let mut a_ij_big_backend = self.vec_znx_big_alloc(1, 1);
        let mut norm_scratch = ScratchOwned {
            data: BE::alloc_bytes(self.vec_znx_big_normalize_tmp_bytes()),
            _phantom: std::marker::PhantomData,
        };
        // Tag of the base secret `a`, carried over as-is: the products below
        // are not distributed like `a`, but their statistics derive from it.
        res.dist = *a.dist();
        let mut res_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(res.data_mut());

        // sk_tensor = sk (x) sk
        // For example: (s0, s1) (x) (s0, s1) = (s0^2, s0s1, s1^2)
        for i in 0..rank {
            for j in i..rank {
                let idx: usize = i * rank + j - (i * (i + 1) / 2);
                let a_dft_ref = a_dft.to_backend_ref();
                {
                    let mut a_ij_dft_backend = a_ij_dft.to_backend_mut();
                    self.svp_apply_ppol_dft_to_dft(&mut a_ij_dft_backend, 0, &a_prepared_backend_ref, j, &a_dft_ref, i);
                }
                {
                    let mut a_ij_big = a_ij_big_backend.to_backend_mut();
                    let mut a_ij_dft = a_ij_dft.to_backend_mut();
                    self.vec_znx_idft_apply_tmpa(&mut a_ij_big, 0, &mut a_ij_dft, 0);
                }
                {
                    let a_ij_big = a_ij_big_backend.to_backend_ref();
                    self.vec_znx_big_normalize(
                        &mut res_backend,
                        base2k,
                        0,
                        idx,
                        &a_ij_big,
                        base2k,
                        0,
                        &mut norm_scratch.arena(),
                    );
                }
            }
        }
    }
}
