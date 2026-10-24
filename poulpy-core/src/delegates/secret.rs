use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{
        GLWESecretPreparedFactory, GLWESecretSampling, GLWESecretTensorFactory, GLWESecretTensorPreparedFactory,
        LWESecretSampling, SecretConversion,
    },
    layouts::{
        Degree, GLWEInfos, GLWESecret, GLWESecretPreparedToBackendMut, GLWESecretTensorPreparedToBackendMut,
        GLWESecretTensorToBackendMut, GLWESecretTensorToBackendRef, GLWESecretToBackendMut, GLWESecretToBackendRef, LWESecret,
        LWESecretToBackendMut, LWESecretToBackendRef, Rank,
    },
    oep::SecretImpl,
};

macro_rules! impl_secret_delegate {
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + SecretImpl,
        {
            $($body)+
        }
    };
}

impl_secret_delegate!(
    GLWESecretSampling<BE>,
    fn glwe_secret_fill_ternary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_secret_fill_ternary_prob(self, sk, prob, source)
    },
    fn glwe_secret_fill_ternary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_secret_fill_ternary_hw(self, sk, hw, source)
    },
    fn glwe_secret_fill_binary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_secret_fill_binary_prob(self, sk, prob, source)
    },
    fn glwe_secret_fill_binary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_secret_fill_binary_hw(self, sk, hw, source)
    },
    fn glwe_secret_fill_binary_block<S>(&self, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_secret_fill_binary_block(self, sk, block_size, source)
    },
    fn glwe_secret_fill_zero<S>(&self, sk: &mut S)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_secret_fill_zero(self, sk)
    }
);

impl_secret_delegate!(
    LWESecretSampling<BE>,
    fn lwe_secret_fill_ternary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        BE::lwe_secret_fill_ternary_prob(self, sk, prob, source)
    },
    fn lwe_secret_fill_ternary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        BE::lwe_secret_fill_ternary_hw(self, sk, hw, source)
    },
    fn lwe_secret_fill_binary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        BE::lwe_secret_fill_binary_prob(self, sk, prob, source)
    },
    fn lwe_secret_fill_binary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        BE::lwe_secret_fill_binary_hw(self, sk, hw, source)
    },
    fn lwe_secret_fill_binary_block<S>(&self, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        BE::lwe_secret_fill_binary_block(self, sk, block_size, source)
    },
    fn lwe_secret_fill_zero<S>(&self, sk: &mut S)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut,
    {
        BE::lwe_secret_fill_zero(self, sk)
    }
);

impl_secret_delegate!(
    GLWESecretPreparedFactory<BE>,
    fn glwe_secret_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWESecretToBackendRef<BE> + GetDistribution,
    {
        BE::glwe_secret_prepare(self, res, other)
    }
);

impl_secret_delegate!(
    GLWESecretTensorPreparedFactory<BE>,
    fn glwe_secret_tensor_prepared_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretTensorPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWESecretTensorToBackendRef<BE> + GetDistribution,
    {
        BE::glwe_secret_tensor_prepared_prepare(self, res, other)
    }
);

impl_secret_delegate!(
    GLWESecretTensorFactory<BE>,
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize {
        BE::glwe_secret_tensor_prepare_tmp_bytes(self, rank)
    },
    fn glwe_secret_tensor_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWESecretTensorToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::glwe_secret_tensor_prepare(self, res, other, scratch)
    }
);

impl_secret_delegate!(
    SecretConversion<BE>,
    fn glwe_secret_from_lwe_secret<S>(&self, src: &S) -> GLWESecret<BE::OwnedBuf, BE::ZnxWord>
    where
        S: LWESecretToBackendRef<BE>,
    {
        BE::glwe_secret_from_lwe_secret(self, src)
    },
    fn lwe_secret_from_glwe_secret_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::lwe_secret_from_glwe_secret_tmp_bytes(self, infos)
    },
    fn lwe_secret_from_glwe_secret<S>(
        &self,
        src: &S,
        lwe_n: Degree,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> LWESecret<BE::OwnedBuf, BE::ZnxWord>
    where
        S: GLWESecretToBackendRef<BE>,
    {
        BE::lwe_secret_from_glwe_secret(self, src, lwe_n, scratch)
    }
);
