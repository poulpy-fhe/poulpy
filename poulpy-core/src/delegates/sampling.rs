use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut},
    source::Source,
};

use crate::{
    Distribution, Noise,
    api::{ScalarZnxFillDistribution, VecZnxAddNoise, VecZnxBigAddNoise},
    oep::SamplingImpl,
};

impl<BE> ScalarZnxFillDistribution<BE> for Module<BE>
where
    BE: Backend + SamplingImpl,
{
    fn scalar_znx_fill_distribution(
        &self,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        dist: Distribution,
        source: &mut Source,
    ) {
        BE::scalar_znx_fill_distribution(self, res, res_col, dist, source.new_seed());
    }
}

impl<BE> VecZnxAddNoise<BE> for Module<BE>
where
    BE: Backend + SamplingImpl,
{
    fn vec_znx_add_noise(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise: Noise,
        source: &mut Source,
    ) {
        BE::vec_znx_add_noise(self, base2k, k, res, res_col, noise, source.new_seed());
    }
}

impl<BE> VecZnxBigAddNoise<BE> for Module<BE>
where
    BE: Backend + SamplingImpl,
{
    fn vec_znx_big_add_noise(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        noise: Noise,
        source: &mut Source,
    ) {
        BE::vec_znx_big_add_noise(self, base2k, k, res, res_col, noise, source.new_seed());
    }
}
