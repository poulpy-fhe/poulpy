//! Secret fills composed from the selected sampler's draws and the selected `vec_znx_zero`, and the
//! secret preparations and the GLWE secret from an LWE secret, one selected HAL operation per column.
use poulpy_hal::{
    layouts::{Module, scalar_znx_as_vec_znx_backend_mut_from_mut},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::ScalarZnxFillDistribution,
    dist::Distribution,
    layouts::{
        GLWEInfos, GLWESecret, GLWESecretPreparedToBackendMut, GLWESecretTensorPreparedToBackendMut,
        GLWESecretTensorToBackendRef, GLWESecretToBackendMut, GLWESecretToBackendRef, LWEInfos, LWESecretToBackendMut,
        LWESecretToBackendRef, ModuleCoreAlloc, Rank,
    },
    oep::SecretImpl,
};

pub(crate) fn glwe_secret_prepare_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: SecretImpl,
    R: GLWESecretPreparedToBackendMut<BE> + GetDistributionMut,
    O: GLWESecretToBackendRef<BE> + GetDistribution,
{
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        assert!(
            res.rank() == other.rank(),
            "glwe_secret_prepare: prepared secret rank differs from the source secret rank"
        );
        for i in 0..res.rank().into() {
            BE::svp_prepare(module, &mut res.data, i, &other.data, i);
        }
    }
    *res.dist_mut() = *other.dist();
}

pub(crate) fn glwe_secret_tensor_prepared_prepare_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: SecretImpl,
    R: GLWESecretTensorPreparedToBackendMut<BE> + GetDistributionMut,
    O: GLWESecretTensorToBackendRef<BE> + GetDistribution,
{
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        assert!(
            res.rank == other.rank,
            "glwe_secret_tensor_prepared_prepare: prepared tensor rank differs from the source tensor rank"
        );
        let cols: usize = other.data.cols();
        assert!(
            res.data.cols() == cols,
            "glwe_secret_tensor_prepared_prepare: prepared tensor column count differs from the source tensor"
        );
        for i in 0..cols {
            BE::svp_prepare(module, &mut res.data, i, &other.data, i);
        }
    }
    *res.dist_mut() = *other.dist();
}

pub(crate) fn glwe_secret_from_lwe_secret_derived<BE, S>(module: &Module<BE>, src: &S) -> GLWESecret<BE::OwnedBuf, BE::ZnxWord>
where
    BE: SecretImpl,
    S: LWESecretToBackendRef<BE>,
{
    let src = src.to_backend_ref();
    assert!(
        src.n().as_usize() == module.n(),
        "secret degree differs from the module degree"
    );
    let mut res = module.glwe_secret_alloc(Rank(1));
    {
        let mut res_ref = GLWESecretToBackendMut::<BE>::to_backend_mut(&mut res);
        BE::scalar_znx_automorphism(module, -1, res_ref.data_mut(), 0, src.data(), 0);
    }
    res.dist = src.dist;
    res
}

pub(crate) fn glwe_secret_fill_ternary_prob_derived<BE, S>(module: &Module<BE>, sk: &mut S, prob: f64, source: &mut Source)
where
    BE: SecretImpl,
    S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    glwe_secret_fill(module, sk, Distribution::TernaryProb(prob), source)
}

pub(crate) fn glwe_secret_fill_ternary_hw_derived<BE, S>(module: &Module<BE>, sk: &mut S, hw: usize, source: &mut Source)
where
    BE: SecretImpl,
    S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    glwe_secret_fill(module, sk, Distribution::TernaryFixed(hw), source)
}

pub(crate) fn glwe_secret_fill_binary_prob_derived<BE, S>(module: &Module<BE>, sk: &mut S, prob: f64, source: &mut Source)
where
    BE: SecretImpl,
    S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    glwe_secret_fill(module, sk, Distribution::BinaryProb(prob), source)
}

pub(crate) fn glwe_secret_fill_binary_hw_derived<BE, S>(module: &Module<BE>, sk: &mut S, hw: usize, source: &mut Source)
where
    BE: SecretImpl,
    S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    glwe_secret_fill(module, sk, Distribution::BinaryFixed(hw), source)
}

pub(crate) fn glwe_secret_fill_binary_block_derived<BE, S>(
    module: &Module<BE>,
    sk: &mut S,
    block_size: usize,
    source: &mut Source,
) where
    BE: SecretImpl,
    S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    glwe_secret_fill(module, sk, Distribution::BinaryBlock(block_size), source)
}

pub(crate) fn glwe_secret_fill_zero_derived<BE, S>(module: &Module<BE>, sk: &mut S)
where
    BE: SecretImpl,
    S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    let rank: usize = sk.rank().into();
    {
        let mut sk_backend = sk.to_backend_mut();
        let mut sk_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_backend.data);
        for i in 0..rank {
            BE::vec_znx_zero(module, &mut sk_vec, i);
        }
    }
    *sk.dist_mut() = Distribution::ZERO;
}

pub(crate) fn lwe_secret_fill_ternary_prob_derived<BE, S>(module: &Module<BE>, sk: &mut S, prob: f64, source: &mut Source)
where
    BE: SecretImpl,
    S: LWESecretToBackendMut<BE> + GetDistributionMut,
{
    lwe_secret_fill(module, sk, Distribution::TernaryProb(prob), source)
}

pub(crate) fn lwe_secret_fill_ternary_hw_derived<BE, S>(module: &Module<BE>, sk: &mut S, hw: usize, source: &mut Source)
where
    BE: SecretImpl,
    S: LWESecretToBackendMut<BE> + GetDistributionMut,
{
    lwe_secret_fill(module, sk, Distribution::TernaryFixed(hw), source)
}

pub(crate) fn lwe_secret_fill_binary_prob_derived<BE, S>(module: &Module<BE>, sk: &mut S, prob: f64, source: &mut Source)
where
    BE: SecretImpl,
    S: LWESecretToBackendMut<BE> + GetDistributionMut,
{
    lwe_secret_fill(module, sk, Distribution::BinaryProb(prob), source)
}

pub(crate) fn lwe_secret_fill_binary_hw_derived<BE, S>(module: &Module<BE>, sk: &mut S, hw: usize, source: &mut Source)
where
    BE: SecretImpl,
    S: LWESecretToBackendMut<BE> + GetDistributionMut,
{
    lwe_secret_fill(module, sk, Distribution::BinaryFixed(hw), source)
}

pub(crate) fn lwe_secret_fill_binary_block_derived<BE, S>(module: &Module<BE>, sk: &mut S, block_size: usize, source: &mut Source)
where
    BE: SecretImpl,
    S: LWESecretToBackendMut<BE> + GetDistributionMut,
{
    lwe_secret_fill(module, sk, Distribution::BinaryBlock(block_size), source)
}

pub(crate) fn lwe_secret_fill_zero_derived<BE, S>(module: &Module<BE>, sk: &mut S)
where
    BE: SecretImpl,
    S: LWESecretToBackendMut<BE> + GetDistributionMut,
{
    {
        let mut sk_backend = sk.to_backend_mut();
        let mut sk_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_backend.data);
        BE::vec_znx_zero(module, &mut sk_vec, 0);
    }
    *sk.dist_mut() = Distribution::ZERO;
}

/// Draws every column of `sk` from `dist` through the selected sampler and tags it.
fn glwe_secret_fill<BE, S>(module: &Module<BE>, sk: &mut S, dist: Distribution, source: &mut Source)
where
    BE: SecretImpl,
    S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
{
    let rank: usize = sk.rank().into();
    {
        let mut sk_backend = sk.to_backend_mut();
        for i in 0..rank {
            module.scalar_znx_fill_distribution(&mut sk_backend.data, i, dist, source);
        }
    }
    *sk.dist_mut() = dist;
}

/// Draws the single column of `sk` from `dist` through the selected sampler and tags it.
fn lwe_secret_fill<BE, S>(module: &Module<BE>, sk: &mut S, dist: Distribution, source: &mut Source)
where
    BE: SecretImpl,
    S: LWESecretToBackendMut<BE> + GetDistributionMut,
{
    {
        let mut sk_backend = sk.to_backend_mut();
        module.scalar_znx_fill_distribution(&mut sk_backend.data, 0, dist, source);
    }
    *sk.dist_mut() = dist;
}
