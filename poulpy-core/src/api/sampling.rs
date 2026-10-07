use poulpy_hal::{
    layouts::{Backend, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut},
    source::Source,
};

use crate::{Distribution, Noise};

/// Overwrites one column of a `ScalarZnx` with a sample of `dist`, drawn in
/// place by the backend.
///
/// See [`Distribution`] for distribution semantics and
/// [`SamplingImpl`](crate::oep::SamplingImpl) for the
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

/// Adds a sample of [`Noise`] scaled by `2^-k` to a `VecZnx` column.
///
/// See [`SamplingImpl`](crate::oep::SamplingImpl) for the exact placement and
/// distribution. `res` is left un-normalized.
pub trait VecZnxAddNoise<BE: Backend> {
    fn vec_znx_add_noise(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise: Noise,
        source: &mut Source,
    );
}

/// Adds a sample of [`Noise`] scaled by `2^-k` to a `VecZnxBig` column.
///
/// See [`SamplingImpl`](crate::oep::SamplingImpl) for the exact placement and
/// distribution. `res` is left un-normalized.
pub trait VecZnxBigAddNoise<BE: Backend> {
    fn vec_znx_big_add_noise(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        noise: Noise,
        source: &mut Source,
    );
}
