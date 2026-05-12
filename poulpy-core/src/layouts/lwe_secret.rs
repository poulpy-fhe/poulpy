use poulpy_hal::{
    layouts::{
        Backend, Data, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendMut, ScalarZnxToBackendRef, TransferFrom,
        ZnxView, ZnxZero,
    },
    source::Source,
};

use crate::{
    GetDistribution,
    api::ModuleTransfer,
    dist::Distribution,
    layouts::{Base2K, Degree, LWEInfos},
};

pub struct LWESecret<D: Data> {
    pub(crate) data: ScalarZnx<D>,
    pub(crate) dist: Distribution,
}

pub type LWESecretBackendRef<'a, BE> = LWESecret<<BE as Backend>::BufRef<'a>>;
pub type LWESecretBackendMut<'a, BE> = LWESecret<<BE as Backend>::BufMut<'a>>;

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl LWESecret<Vec<u8>> {
    pub(crate) fn alloc(n: Degree) -> Self {
        LWESecret {
            data: ScalarZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>>::bytes_of(n.into(), 1)),
                n.into(),
                1,
            ),
            dist: Distribution::NONE,
        }
    }
}

impl<D: HostDataRef> LWESecret<D> {
    /// Copies this secret's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> LWESecret<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_lwe_secret(self)
    }
}

impl<D: Data> LWESecret<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWESecret<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let n = self.data.n();
        let cols = self.data.cols();
        let data = self.data.data;
        LWESecret {
            data: ScalarZnx::from_data(data, n, cols),
            dist: self.dist,
        }
    }
}

impl<D: HostDataRef> GetDistribution for LWESecret<D> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: HostDataRef> LWESecret<D> {
    pub fn raw(&self) -> &[i64] {
        self.data.at(0, 0)
    }

    pub fn dist(&self) -> Distribution {
        self.dist
    }

    pub fn data(&self) -> &ScalarZnx<D> {
        &self.data
    }
}

impl<D: Data> LWEInfos for LWESecret<D> {
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

impl<D: HostDataMut> LWESecret<D> {
    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        self.data.fill_ternary_prob(0, prob, source);
        self.dist = Distribution::TernaryProb(prob);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        self.data.fill_ternary_hw(0, hw, source);
        self.dist = Distribution::TernaryFixed(hw);
    }

    pub fn fill_binary_prob(&mut self, prob: f64, source: &mut Source) {
        self.data.fill_binary_prob(0, prob, source);
        self.dist = Distribution::BinaryProb(prob);
    }

    pub fn fill_binary_hw(&mut self, hw: usize, source: &mut Source) {
        self.data.fill_binary_hw(0, hw, source);
        self.dist = Distribution::BinaryFixed(hw);
    }

    pub fn fill_binary_block(&mut self, block_size: usize, source: &mut Source) {
        self.data.fill_binary_block(0, block_size, source);
        self.dist = Distribution::BinaryBlock(block_size);
    }

    pub fn fill_zero(&mut self) {
        self.data.zero();
        self.dist = Distribution::ZERO;
    }
}

pub trait LWESecretToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE>;
}

impl<BE: Backend> LWESecretToBackendRef<BE> for LWESecret<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> LWESecretToBackendRef<BE> for &LWESecret<BE::BufRef<'b>> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_ref(&self.data.data), self.data.n(), self.data.cols()),
        }
    }
}

impl<'b, BE: Backend + 'b> LWESecretToBackendRef<BE> for &mut LWESecret<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_ref_mut(&self.data.data), self.data.n(), self.data.cols()),
        }
    }
}

pub trait LWESecretToBackendMut<BE: Backend>: LWESecretToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE>;
}

impl<BE: Backend> LWESecretToBackendMut<BE> for LWESecret<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> LWESecretToBackendMut<BE> for &mut LWESecret<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE> {
        let n = self.data.n();
        let cols = self.data.cols();
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.data.data), n, cols),
        }
    }
}
