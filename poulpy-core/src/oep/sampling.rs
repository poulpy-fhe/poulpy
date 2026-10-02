use poulpy_hal::layouts::{Backend, Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut};

use crate::{Distribution, NoiseInfos};

/// Backend-provided sampling of secret distributions and of Gaussian noise.
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
/// `vec_znx_add_normal` and `vec_znx_big_add_normal` add `e * 2^-noise.k` to
/// column `res_col` of `res`, with `e = round(N(0, noise.sigma))` resampled
/// while `|e| > noise.bound`. In base `2^base2k` that lands in limb
/// `ceil(noise.k / base2k) - 1`, shifted left by the
/// `(limb + 1) * base2k - noise.k` unused low bits of that limb
/// ([`NoiseInfos::target_limb_and_shift`]). `res` is left un-normalized.
///
/// The backend draws its own stream from `seed`: a fixed seed therefore gives
/// a per-backend output, not a cross-backend one. The
/// [`ScalarZnxFillDistribution`], [`VecZnxAddNormal`] and [`VecZnxBigAddNormal`]
/// delegates derive one seed per call with
/// [`Source::new_seed`](poulpy_hal::source::Source::new_seed).
///
/// There is **no default body**: `poulpy-core` has no reference body to offer
/// here. Every other family's reference body composes HAL operations; drawing
/// from a [`Distribution`] or a discrete Gaussian is not such a composition,
/// and a backend's buffers are opaque to generic code, so only the backend can
/// produce the values. Everything `poulpy-core` samples comes through here:
/// the long-lived secret keys ([`GLWESecretSampling`] / [`LWESecretSampling`]),
/// the ephemerals of public-key encryption, and the encryption noise.
///
/// # Safety
/// Implementations must write only within column `res_col` of `res`, and for
/// the noise methods only in the target limb. They must panic rather than
/// truncate when `ceil(log2(noise.bound)) >= 64`, and must preserve the
/// distributions above: the scheme's security rests on them.
///
/// [`ScalarZnxFillDistribution`]: crate::ScalarZnxFillDistribution
/// [`VecZnxAddNormal`]: crate::VecZnxAddNormal
/// [`VecZnxBigAddNormal`]: crate::VecZnxBigAddNormal
/// [`GLWESecretSampling`]: crate::layouts::GLWESecretSampling
/// [`LWESecretSampling`]: crate::layouts::LWESecretSampling
pub unsafe trait SamplingImpl: Backend {
    fn scalar_znx_fill_distribution(
        module: &Module<Self>,
        res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    );

    fn vec_znx_add_normal(
        module: &Module<Self>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    );

    fn vec_znx_big_add_normal(
        module: &Module<Self>,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    );
}

/// Optional backend sampling of full-precision bounded smudging noise.
///
/// Coefficients are independent integer samples from [`crate::SmudgingNoise`],
/// then scaled by `2^-k`, the destination's precision, and decomposed into
/// balanced radix digits. The distribution is on complete integers, not on
/// independently sampled limbs.
/// A backend may select this extension without changing ordinary encryption's
/// [`SamplingImpl`] contract. Fixed seeds reproduce draws within that backend.
///
/// # Safety
/// The selected input column has canonical balanced digits. Implementations
/// must validate degree, column, radix, precision and noise bounds before
/// mutation, and add at most one bounded balanced noise digit per coefficient
/// and limb.
/// They must not alter other columns, discard low noise bits, truncate samples
/// to a machine word, or bias rejection sampling by an unaccounted runtime cap.
/// Padding below `k` in the added polynomial is zero. Noise must follow
/// the declared conditional Gaussian or exact uniform distribution, assuming
/// uniform private source bits. Big integer scalar temporaries may allocate;
/// no scratch arena is required. The reference CPU algorithm has variable runtime.
pub unsafe trait SmudgingSamplingImpl: Backend {
    fn vec_znx_add_smudging(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        noise: crate::SmudgingNoise,
        seed: [u8; 32],
    );
}
