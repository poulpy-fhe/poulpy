use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxBackendMut},
    source::Source,
};

use crate::{Distribution, api::ScalarZnxFillDistribution, oep::SamplingImpl};

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
