use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDft, SvpPrepare, VecZnxBigAlloc, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, Data, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ScratchOwned,
        SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDft,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, ZnxView, ZnxViewMut, scalar_znx_as_vec_znx_backend_mut_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
};

use crate::{
    GetDistribution, GetDistributionMut, ScratchArenaTakeCore,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWESecret, GLWESecretBackendMut, GLWESecretBackendRef, GLWESecretPreparedFactory,
        GLWESecretToBackendMut, GLWESecretToBackendRef, LWEInfos, Rank,
    },
};

pub struct GLWESecretTensor<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) rank: Rank,
    pub(crate) dist: Distribution,
}

impl GLWESecretTensor<Vec<u8>> {
    pub(crate) fn pairs(rank: usize) -> usize {
        (((rank + 1) * rank) >> 1).max(1)
    }
}

impl<D: Data> GetDistribution for GLWESecretTensor<D> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data> GetDistributionMut for GLWESecretTensor<D> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data> LWEInfos for GLWESecretTensor<D> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        1
    }
}

impl<D: Data> LWEInfos for &mut GLWESecretTensor<D> {
    fn base2k(&self) -> Base2K {
        (**self).base2k()
    }

    fn n(&self) -> Degree {
        (**self).n()
    }

    fn size(&self) -> usize {
        (**self).size()
    }
}

impl<D: HostDataRef> GLWESecretTensor<D> {
    pub fn at(&self, mut i: usize, mut j: usize) -> ScalarZnx<&[u8]> {
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

impl<D: HostDataMut> GLWESecretTensor<D> {
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> ScalarZnx<&mut [u8]> {
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

impl<D: Data> GLWEInfos for GLWESecretTensor<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data> GLWEInfos for &mut GLWESecretTensor<D> {
    fn rank(&self) -> Rank {
        (**self).rank()
    }
}

impl<BE: Backend> GLWESecretToBackendRef<BE> for GLWESecretTensor<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            data: <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            dist: self.dist,
        }
    }
}

impl<'b, BE: Backend + 'b> GLWESecretToBackendRef<BE> for &mut GLWESecretTensor<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            data: ScalarZnx::from_data(BE::view_ref_mut(&self.data.data), self.data.n(), self.data.cols()),
            dist: self.dist,
        }
    }
}

impl<BE: Backend> GLWESecretToBackendMut<BE> for GLWESecretTensor<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        GLWESecret {
            data: <ScalarZnx<BE::OwnedBuf> as poulpy_hal::layouts::ScalarZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
            dist: self.dist,
        }
    }
}

impl<'b, BE: Backend + 'b> GLWESecretToBackendMut<BE> for &mut GLWESecretTensor<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        let n = self.data.n();
        let cols = self.data.cols();
        GLWESecret {
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.data.data), n, cols),
            dist: self.dist,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GLWESecretTensor<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecretTensor {
            data: ScalarZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>>::bytes_of(
                    n.into(),
                    Self::pairs(rank.into()),
                )),
                n.into(),
                Self::pairs(rank.into()),
            ),
            rank,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), Self::pairs(infos.rank().into()).into())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::bytes_of(n.into(), Self::pairs(rank.into()))
    }
}

// module-only API: secret tensor preparation is provided by `GLWESecretTensorFactory` on `Module`.

pub trait GLWESecretTensorFactory<BE: Backend> {
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize;

    fn glwe_secret_tensor_prepare<'s, R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> GLWESecretTensorFactory<BE> for Module<BE>
where
    Self: ModuleN
        + GLWESecretPreparedFactory<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDft<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize {
        self.glwe_secret_prepared_bytes_of(rank)
    }

    fn glwe_secret_tensor_prepare<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        A: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res = &mut res.to_backend_mut();
        let a = a.to_backend_ref();

        assert_eq!(res.rank(), GLWESecretTensor::pairs(a.rank().into()) as u32);
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
                self.svp_prepare(&mut a_prepared_data, i, &a.data, i);
            }
        }
        a_prepared.dist = *a.dist();

        let base2k: usize = 17;

        let mut a_dft = VecZnxDft::<BE::OwnedBuf, BE>::alloc(self.n(), rank, 1);
        let a_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(&a.data);
        for i in 0..rank {
            let mut a_dft_backend = a_dft.to_backend_mut();
            self.vec_znx_dft_apply(1, 0, &mut a_dft_backend, i, &a_backend_vec, i);
        }

        let mut a_ij_dft = VecZnxDft::<BE::OwnedBuf, BE>::alloc(self.n(), 1, 1);
        let a_prepared_backend_ref = a_prepared.data.reborrow_backend_ref();
        let mut a_ij_big_backend = self.vec_znx_big_alloc(1, 1);
        let mut norm_scratch = ScratchOwned {
            data: BE::alloc_bytes(self.vec_znx_big_normalize_tmp_bytes()),
            _phantom: std::marker::PhantomData,
        };
        let mut res_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut res.data);

        // sk_tensor = sk (x) sk
        // For example: (s0, s1) (x) (s0, s1) = (s0^2, s0s1, s1^2)
        for i in 0..rank {
            for j in i..rank {
                let idx: usize = i * rank + j - (i * (i + 1) / 2);
                let a_dft_ref = a_dft.to_backend_ref();
                {
                    let mut a_ij_dft_backend = a_ij_dft.to_backend_mut();
                    self.svp_apply_dft_to_dft(&mut a_ij_dft_backend, 0, &a_prepared_backend_ref, j, &a_dft_ref, i);
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

        res.dist = *a.dist();
    }
}
