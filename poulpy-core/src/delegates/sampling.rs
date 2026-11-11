use poulpy_hal::{
    layouts::{Backend, Module, Ring, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut},
    source::Source,
};

use crate::{
    Distribution, Noise,
    api::{ScalarZnxFillDistribution, VecZnxAddNoise, VecZnxBigAddNoise},
    oep::SamplingImpl,
};

impl<BE, Rg: Ring> ScalarZnxFillDistribution<BE, Rg> for Module<BE>
where
    BE: Backend + SamplingImpl<Rg>,
{
    fn scalar_znx_fill_distribution(
        &self,
        res: &mut ScalarZnxBackendMut<'_, BE, Rg>,
        res_col: usize,
        dist: Distribution,
        source: &mut Source,
    ) {
        BE::scalar_znx_fill_distribution(self, res, res_col, dist, source.new_seed());
    }
}

impl<BE, Rg: Ring> VecZnxAddNoise<BE, Rg> for Module<BE>
where
    BE: Backend + SamplingImpl<Rg>,
{
    fn vec_znx_add_noise(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE, Rg>,
        res_col: usize,
        noise: Noise,
        source: &mut Source,
    ) {
        BE::vec_znx_add_noise(self, base2k, k, res, res_col, noise, source.new_seed());
    }
}

impl<BE, Rg: Ring> VecZnxBigAddNoise<BE, Rg> for Module<BE>
where
    BE: Backend + SamplingImpl<Rg>,
{
    fn vec_znx_big_add_noise(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBigBackendMut<'_, BE, Rg>,
        res_col: usize,
        noise: Noise,
        source: &mut Source,
    ) {
        BE::vec_znx_big_add_noise(self, base2k, k, res, res_col, noise, source.new_seed());
    }
}
