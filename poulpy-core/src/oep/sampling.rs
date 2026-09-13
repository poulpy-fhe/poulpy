use poulpy_hal::layouts::{Backend, Module, ScalarZnxBackendMut};

use crate::Distribution;

/// Backend-provided sampling of secret distributions.
///
/// `scalar_znx_fill_distribution` overwrites column `res_col` of `res` with a
/// sample of `dist`, drawn by the backend from its own stream seeded with
/// `seed`. `TernaryFixed(h)` places exactly `h` non-zero coefficients in
/// `{-1, +1}` at uniform positions, `TernaryProb(p)` makes each coefficient
/// non-zero with total probability `p`, `BinaryFixed(h)` and `BinaryProb(p)`
/// are their `{0, 1}` counterparts, `BinaryBlock(b)` gives each block of `b`
/// coefficients at most one `1`, and `ZERO` zeroes the column. `NONE` and
/// `ENCAPSULATED` are not sampleable: callers reject them before reaching the
/// seam, implementations panic on them.
///
/// The [`ScalarZnxFillDistribution`] delegate derives one seed per call with
/// [`Source::new_seed`](poulpy_hal::source::Source::new_seed), so for a fixed
/// seed the output is per-backend, not cross-backend.
///
/// There is **no default body**: `poulpy-core` has no reference body to offer
/// here. Every other family's reference body composes HAL operations; drawing
/// from a [`Distribution`] is not such a composition, and a backend's buffers
/// are opaque to generic code, so only the backend can produce the values.
/// Everything `poulpy-core` samples comes through here: the long-lived secret
/// keys ([`GLWESecretSampling`] / [`LWESecretSampling`]) and the ephemeral
/// secret of public-key encryption.
///
/// # Safety
/// Implementations must write only within column `res_col` of `res` and must
/// preserve the distribution above: the scheme's security rests on it.
///
/// [`ScalarZnxFillDistribution`]: crate::ScalarZnxFillDistribution
/// [`GLWESecretSampling`]: crate::layouts::GLWESecretSampling
/// [`LWESecretSampling`]: crate::layouts::LWESecretSampling
pub unsafe trait SamplingImpl<BE: Backend>: Backend {
    fn scalar_znx_fill_distribution(
        module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    );
}
