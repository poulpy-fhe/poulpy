use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut},
    source::Source,
};

use crate::{
    Distribution, NoiseInfos,
    api::{ScalarZnxFillDistribution, VecZnxAddNormal, VecZnxBigAddNormal},
    oep::SamplingImpl,
};

impl<BE> ScalarZnxFillDistribution<BE> for Module<BE>
where
    BE: Backend + SamplingImpl<BE>,
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

impl<BE> VecZnxAddNormal<BE> for Module<BE>
where
    BE: Backend + SamplingImpl<BE>,
{
    fn vec_znx_add_normal(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise: NoiseInfos,
        source: &mut Source,
    ) {
        BE::vec_znx_add_normal(self, base2k, res, res_col, noise, source.new_seed());
    }
}

impl<BE> VecZnxBigAddNormal<BE> for Module<BE>
where
    BE: Backend + SamplingImpl<BE>,
{
    fn vec_znx_big_add_normal(
        &self,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        noise: NoiseInfos,
        source: &mut Source,
    ) {
        BE::vec_znx_big_add_normal(self, base2k, res, res_col, noise, source.new_seed());
    }
}
