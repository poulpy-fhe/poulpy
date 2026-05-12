use poulpy_hal::{
    layouts::{
        Backend, Data, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendMut, ScalarZnxToBackendRef, TransferFrom,
        ZnxZero,
    },
    source::Source,
};

use crate::{
    GetDistribution,
    api::ModuleTransfer,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank},
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWESecretLayout {
    pub n: Degree,
    pub rank: Rank,
}

impl LWEInfos for GLWESecretLayout {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        1
    }
}
impl GLWEInfos for GLWESecretLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESecret<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) dist: Distribution,
}

pub type GLWESecretBackendRef<'a, BE> = GLWESecret<<BE as Backend>::BufRef<'a>>;
pub type GLWESecretBackendMut<'a, BE> = GLWESecret<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for GLWESecret<D> {
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

impl<D: Data> LWEInfos for &mut GLWESecret<D> {
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

impl<D: Data> GetDistribution for GLWESecret<D> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data> GLWEInfos for GLWESecret<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

impl<D: Data> GLWEInfos for &mut GLWESecret<D> {
    fn rank(&self) -> Rank {
        (**self).rank()
    }
}

impl<D: HostDataRef> GLWESecret<D> {
    /// Copies this secret's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GLWESecret<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_glwe_secret(self)
    }
}

impl<D: Data> GLWESecret<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWESecret<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let n = self.data.n();
        let cols = self.data.cols();
        let data = self.data.data;
        GLWESecret {
            data: ScalarZnx::from_data(data, n, cols),
            dist: self.dist,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GLWESecret<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecret {
            data: ScalarZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>>::bytes_of(n.into(), rank.into())),
                n.into(),
                rank.into(),
            ),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.rank())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::bytes_of(n.into(), rank.into())
    }
}

impl<D: HostDataMut> GLWESecret<D> {
    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_ternary_prob(i, prob, source);
        });
        self.dist = Distribution::TernaryProb(prob);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_ternary_hw(i, hw, source);
        });
        self.dist = Distribution::TernaryFixed(hw);
    }

    pub fn fill_binary_prob(&mut self, prob: f64, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_binary_prob(i, prob, source);
        });
        self.dist = Distribution::BinaryProb(prob);
    }

    pub fn fill_binary_hw(&mut self, hw: usize, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_binary_hw(i, hw, source);
        });
        self.dist = Distribution::BinaryFixed(hw);
    }

    pub fn fill_binary_block(&mut self, block_size: usize, source: &mut Source) {
        (0..self.rank().into()).for_each(|i| {
            self.data.fill_binary_block(i, block_size, source);
        });
        self.dist = Distribution::BinaryBlock(block_size);
    }

    pub fn fill_zero(&mut self) {
        self.data.zero();
        self.dist = Distribution::ZERO;
    }
}

pub trait GLWESecretToBackendMut<BE: Backend>: GLWESecretToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE>;
}

impl<BE: Backend> GLWESecretToBackendMut<BE> for GLWESecret<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        GLWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> GLWESecretToBackendMut<BE> for &mut GLWESecret<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        let n = self.data.n();
        let cols = self.data.cols();
        GLWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.data.data), n, cols),
        }
    }
}

pub trait GLWESecretToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE>;
}

impl<BE: Backend> GLWESecretToBackendRef<BE> for GLWESecret<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            data: <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            dist: self.dist,
        }
    }
}

impl<'b, BE: Backend + 'b> GLWESecretToBackendRef<BE> for &GLWESecret<BE::BufRef<'b>> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            data: ScalarZnx::from_data(BE::view_ref(&self.data.data), self.data.n(), self.data.cols()),
            dist: self.dist,
        }
    }
}

impl<'b, BE: Backend + 'b> GLWESecretToBackendRef<BE> for &mut GLWESecret<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            data: ScalarZnx::from_data(BE::view_ref_mut(&self.data.data), self.data.n(), self.data.cols()),
            dist: self.dist,
        }
    }
}
