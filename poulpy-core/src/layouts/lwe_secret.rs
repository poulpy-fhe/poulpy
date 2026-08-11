use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::{
    api::{
        ScalarZnxFillBinaryBlockSourceBackend, ScalarZnxFillBinaryHwSourceBackend, ScalarZnxFillBinaryProbSourceBackend,
        ScalarZnxFillTernaryHwSourceBackend, ScalarZnxFillTernaryProbSourceBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, Data, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendMut, ScalarZnxToBackendRef, TransferFrom, ZnxView,
        scalar_znx_as_vec_znx_backend_mut_from_mut,
    },
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::ModuleTransfer,
    dist::Distribution,
    layouts::{Base2K, Degree, LWEInfos},
};

pub struct LWESecret<D: Data, W: ZnxWord> {
    pub(crate) data: ScalarZnx<D, W>,
    /// Distribution the base secret was sampled from. When this secret is
    /// obtained by flattening a GLWE secret, the tag is that of the source
    /// key's polynomial components and is not rescaled by the rank.
    /// See [`Distribution`].
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

impl<D: Data, W: ZnxWord> GetDistributionMut for LWESecret<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
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

/// Secret-key sampling, dispatched to the backend.
///
/// The LWE counterpart of [`GLWESecretSampling`](crate::layouts::GLWESecretSampling):
/// sampling is routed through the `ScalarZnxFill*` extension points so a
/// backend can substitute its own implementation, rather than being a
/// host-memory method on the layout.
pub trait LWESecretSampling<BE: Backend> {
    /// Ternary `{-1, 0, 1}` coefficients, each non-zero with probability `prob`.
    fn lwe_secret_fill_ternary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Ternary `{-1, 0, 1}` coefficients with exactly `hw` non-zero entries.
    fn lwe_secret_fill_ternary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Binary `{0, 1}` coefficients, each set with probability `prob`.
    fn lwe_secret_fill_binary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Binary `{0, 1}` coefficients with exactly `hw` ones.
    fn lwe_secret_fill_binary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Binary coefficients with at most one `1` per block of `block_size`.
    fn lwe_secret_fill_binary_block<S>(&self, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// All-zero secret, tagged [`Distribution::ZERO`] (debug / testing only).
    fn lwe_secret_fill_zero<S>(&self, sk: &mut S)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;
}

impl<BE: Backend> LWESecretSampling<BE> for Module<BE>
where
    Self: ScalarZnxFillTernaryProbSourceBackend<BE>
        + ScalarZnxFillTernaryHwSourceBackend<BE>
        + ScalarZnxFillBinaryProbSourceBackend<BE>
        + ScalarZnxFillBinaryHwSourceBackend<BE>
        + ScalarZnxFillBinaryBlockSourceBackend<BE>
        + VecZnxZeroBackend<BE>,
{
    fn lwe_secret_fill_ternary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        {
            let mut sk_backend = sk.to_backend_mut();
            self.scalar_znx_fill_ternary_prob_source_backend(&mut sk_backend.data, 0, prob, source);
        }
        *sk.dist_mut() = Distribution::TernaryProb(prob);
    }

    fn lwe_secret_fill_ternary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        {
            let mut sk_backend = sk.to_backend_mut();
            self.scalar_znx_fill_ternary_hw_source_backend(&mut sk_backend.data, 0, hw, source);
        }
        *sk.dist_mut() = Distribution::TernaryFixed(hw);
    }

    fn lwe_secret_fill_binary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        {
            let mut sk_backend = sk.to_backend_mut();
            self.scalar_znx_fill_binary_prob_source_backend(&mut sk_backend.data, 0, prob, source);
        }
        *sk.dist_mut() = Distribution::BinaryProb(prob);
    }

    fn lwe_secret_fill_binary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        {
            let mut sk_backend = sk.to_backend_mut();
            self.scalar_znx_fill_binary_hw_source_backend(&mut sk_backend.data, 0, hw, source);
        }
        *sk.dist_mut() = Distribution::BinaryFixed(hw);
    }

    fn lwe_secret_fill_binary_block<S>(&self, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        {
            let mut sk_backend = sk.to_backend_mut();
            self.scalar_znx_fill_binary_block_source_backend(&mut sk_backend.data, 0, block_size, source);
        }
        *sk.dist_mut() = Distribution::BinaryBlock(block_size);
    }

    fn lwe_secret_fill_zero<S>(&self, sk: &mut S)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        {
            let mut sk_backend = sk.to_backend_mut();
            let mut sk_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_backend.data);
            self.vec_znx_zero_backend(&mut sk_vec, 0);
        }
        *sk.dist_mut() = Distribution::ZERO;
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
