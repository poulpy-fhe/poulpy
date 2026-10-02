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

impl<BE> VecZnxAddNormal<BE> for Module<BE>
where
    BE: Backend + SamplingImpl,
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
    BE: Backend + SamplingImpl,
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

impl<BE> crate::VecZnxAddSmudging<BE> for Module<BE>
where
    BE: Backend + crate::oep::SmudgingSamplingImpl,
{
    fn vec_znx_add_smudging(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise: crate::SmudgingNoise,
        source: &mut Source,
    ) {
        use poulpy_hal::layouts::ZnxWord;
        assert!(res.n() == self.n(), "invalid smudging: degree differs from the module's");
        assert!(res_col < res.cols(), "invalid smudging: column outside the destination");
        assert!(
            base2k > 0 && base2k <= BE::ZnxWord::BITS - 2,
            "invalid smudging: radix outside the coefficient headroom"
        );
        let capacity = res
            .size()
            .checked_mul(base2k)
            .expect("invalid smudging: destination precision overflow");
        assert!(k <= capacity, "invalid smudging: precision outside the destination");
        noise.assert_valid_for(base2k, k);
        BE::vec_znx_add_smudging(self, base2k, k, res, res_col, noise, source.new_seed());
    }
}
