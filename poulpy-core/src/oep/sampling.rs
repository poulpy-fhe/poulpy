use poulpy_hal::layouts::{Backend, Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut};

use crate::{Distribution, Noise};

/// Backend-provided secret and noise sampling.
///
/// Scalar sampling overwrites the chosen column with the requested secret
/// distribution; `NONE` and `ENCAPSULATED` must panic. Noise sampling adds an
/// integer distributed according to [`Noise`], multiplied by `2^-k`, to the
/// chosen column. Digits use radix `2^base2k`; padding below precision `k`
/// stays zero. A sample that fits in one digit changes only the last limb.
/// Wider samples use balanced digits over `ceil(k / base2k)` limbs. Results
/// are left unnormalized and are interpreted modulo the destination torus.
///
/// Each delegate derives a private child seed. Fixed seeds are reproducible
/// within a backend, but distinct backends may use different random streams.
/// Every backend must implement these methods, including full-width noise.
/// A Gaussian with bound `B = floor(cutoff_factor * sigma) <= 64` may use a
/// 128-bit cumulative table with statistical distance at most `B * 2^-128`.
/// Larger Gaussians and uniform noise must be sampled exactly.
///
/// # Safety
/// Implementations must preserve the requested distribution and write only
/// within the selected column and precision. Invalid noise or placement
/// parameters must panic before writing. The scheme's security relies on
/// these distribution and precision guarantees.
pub unsafe trait SamplingImpl: Backend {
    fn scalar_znx_fill_distribution(
        module: &Module<Self>,
        res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    );

    fn vec_znx_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    );

    fn vec_znx_big_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    );
}
