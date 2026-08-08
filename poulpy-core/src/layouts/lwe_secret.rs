use poulpy_hal::layouts::ZnxWord;
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

pub struct LWESecret<D: Data, W: ZnxWord> {
    pub(crate) data: ScalarZnx<D, W>,
    pub(crate) dist: Distribution,
}

pub type LWESecretBackendRef<'a, BE> = LWESecret<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type LWESecretBackendMut<'a, BE> = LWESecret<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> LWESecret<Vec<u8>, W> {
    pub(crate) fn alloc(n: Degree) -> Self {
        LWESecret {
            data: ScalarZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>, W>::bytes_of(n.into(), 1)),
                n.into(),
                1,
            ),
            dist: Distribution::NONE,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord> LWESecret<D, W> {
    /// Copies this secret's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> LWESecret<To::OwnedBuf, To::ZnxWord>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
        To: Backend<ZnxWord = W>,
        To: TransferFrom<BE>,
    {
        dst.upload_lwe_secret(self)
    }
}

impl<D: Data, W: ZnxWord> LWESecret<D, W> {
    pub fn data(&self) -> &ScalarZnx<D, W> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ScalarZnx<D, W> {
        &mut self.data
    }

    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWESecret<To::OwnedBuf, To::ZnxWord>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
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

impl<D: HostDataRef, W: ZnxWord> GetDistribution for LWESecret<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: HostDataRef, W: ZnxWord> LWESecret<D, W> {
    /// Borrows the secret's coefficients in the layout's own word.
    pub fn raw(&self) -> &[W] {
        self.data.at(0, 0)
    }

    pub fn dist(&self) -> Distribution {
        self.dist
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for LWESecret<D, W> {
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

impl<D: HostDataMut, W: ZnxWord> LWESecret<D, W> {
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

impl<BE: Backend> LWESecretToBackendRef<BE> for LWESecret<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> LWESecretToBackendRef<BE> for &LWESecret<BE::BufRef<'b>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWESecretBackendRef<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_ref(&self.data.data), self.data.n(), self.data.cols()),
        }
    }
}

impl<'b, BE: Backend + 'b> LWESecretToBackendRef<BE> for &mut LWESecret<BE::BufMut<'b>, BE::ZnxWord> {
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

impl<BE: Backend> LWESecretToBackendMut<BE> for LWESecret<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE> {
        LWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> LWESecretToBackendMut<BE> for &mut LWESecret<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> LWESecretBackendMut<'_, BE> {
        let n = self.data.n();
        let cols = self.data.cols();
        LWESecret {
            dist: self.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.data.data), n, cols),
        }
    }
}
