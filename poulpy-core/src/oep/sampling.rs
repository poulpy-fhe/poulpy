use poulpy_hal::layouts::{Backend, Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut};

use crate::{Distribution, NoiseInfos};

/// Backend-provided sampling of secret distributions and of Gaussian noise.
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
/// `vec_znx_add_normal` and `vec_znx_big_add_normal` add `e * 2^-noise.k` to
/// column `res_col` of `res`, with `e = round(N(0, noise.sigma))` resampled
/// while `|e| > noise.bound`. In base `2^base2k` that lands in limb
/// `ceil(noise.k / base2k) - 1`, shifted left by the
/// `(limb + 1) * base2k - noise.k` unused low bits of that limb
/// ([`NoiseInfos::target_limb_and_shift`]). `res` is left un-normalized.
///
/// The backend draws its own stream from `seed` — a fixed seed therefore gives
/// a per-backend output, not a cross-backend one. The
/// [`ScalarZnxFillDistribution`], [`VecZnxAddNormal`] and [`VecZnxBigAddNormal`]
/// delegates derive one seed per call with
/// [`Source::new_seed`](poulpy_hal::source::Source::new_seed).
///
/// There is **no default body**: the ephemeral secret of public-key encryption
/// and the noise of every encryption are sampled in place, in backend memory. A
/// host default would sample on the host and upload, which is the round trip
/// this seam exists to forbid. Long-lived secret keys are the one place
/// `poulpy-core` still samples on the host ([`GLWESecretSampling`] /
/// [`LWESecretSampling`]): once per key, into an owned buffer.
///
/// # Safety
/// Implementations must write only within column `res_col` of `res` — for the
/// noise methods only in the target limb —, must panic rather than truncate
/// when `ceil(log2(noise.bound)) >= 64`, and must preserve the distributions
/// above: the scheme's security rests on them.
///
/// [`ScalarZnxFillDistribution`]: crate::ScalarZnxFillDistribution
/// [`VecZnxAddNormal`]: crate::VecZnxAddNormal
/// [`VecZnxBigAddNormal`]: crate::VecZnxBigAddNormal
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

    fn vec_znx_add_normal(
        module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    );

    fn vec_znx_big_add_normal(
        module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    );
}
