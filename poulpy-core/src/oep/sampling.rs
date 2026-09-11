use poulpy_hal::layouts::{Backend, Module, ScalarZnxBackendMut};

use crate::Distribution;

/// Backend-provided sampling of secret distributions.
///
/// `scalar_znx_fill_distribution` overwrites column `res_col` of `res` with a
/// sample of `dist`, drawn by the backend from its own stream seeded with
/// `seed`. The distributions are those of the host `ScalarZnx::fill_*` methods:
/// `TernaryFixed(h)` is `fill_ternary_hw`, `TernaryProb(p)` is
/// `fill_ternary_prob`, `BinaryFixed(h)` is `fill_binary_hw`, `BinaryProb(p)`
/// is `fill_binary_prob`, `BinaryBlock(b)` is `fill_binary_block`, and `ZERO`
/// zeroes the column. `NONE` and `ENCAPSULATED` are not sampleable: callers
/// reject them before reaching the seam, implementations panic on them.
///
/// The [`ScalarZnxFillDistribution`] delegate derives one seed per call with
/// [`Source::new_seed`](poulpy_hal::source::Source::new_seed), so for a fixed
/// seed the output is per-backend, not cross-backend.
///
/// There is **no default body**: the ephemeral secret of public-key encryption
/// is sampled on every call, in place, in backend memory. A host default would
/// sample on the host and upload, which is the round trip this seam exists to
/// forbid. Long-lived secret keys are the one place `poulpy-core` still samples
/// on the host ([`GLWESecretSampling`] / [`LWESecretSampling`]): once per key,
/// into an owned buffer.
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
