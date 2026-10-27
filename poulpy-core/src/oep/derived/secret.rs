//! Secret fills composed from the selected sampler's draws and the selected `vec_znx_zero`.
use poulpy_hal::{
    layouts::{Module, scalar_znx_as_vec_znx_backend_mut_from_mut},
    source::Source,
};

use crate::{
    GetDistributionMut,
    api::ScalarZnxFillDistribution,
    dist::Distribution,
    layouts::{GLWEInfos, GLWESecretToBackendMut, LWESecretToBackendMut},
    oep::SecretImpl,
};

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
