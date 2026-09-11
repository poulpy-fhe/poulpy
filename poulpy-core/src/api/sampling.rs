use poulpy_hal::{
    layouts::{Backend, ScalarZnxBackendMut},
    source::Source,
};

use crate::Distribution;

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
