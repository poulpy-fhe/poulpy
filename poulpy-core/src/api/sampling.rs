use poulpy_hal::{
    layouts::{Backend, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut},
    source::Source,
};

use crate::{Distribution, NoiseInfos};

/// Overwrites one column of a `ScalarZnx` with a sample of `dist`, drawn in
/// place by the backend.
///
/// See [`SamplingImpl`](crate::oep::SamplingImpl) for the distributions and the
/// treatment of `NONE` / `ENCAPSULATED`.
pub trait ScalarZnxFillDistribution<BE: Backend> {
    fn scalar_znx_fill_distribution(
        &self,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        dist: Distribution,
        source: &mut Source,
    );
}

/// Adds a bounded discrete Gaussian scaled by `2^-noise.k` to a `VecZnx` column.
///
/// See [`SamplingImpl`](crate::oep::SamplingImpl) for the exact placement and
/// distribution. `res` is left un-normalized.
pub trait VecZnxAddNormal<BE: Backend> {
    fn vec_znx_add_normal(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise: NoiseInfos,
        source: &mut Source,
    );
}

/// Adds a bounded discrete Gaussian scaled by `2^-noise.k` to a `VecZnxBig` column.
///
/// See [`SamplingImpl`](crate::oep::SamplingImpl) for the exact placement and
/// distribution. `res` is left un-normalized.
pub trait VecZnxBigAddNormal<BE: Backend> {
    fn vec_znx_big_add_normal(
        &self,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        noise: NoiseInfos,
        source: &mut Source,
    );
}
