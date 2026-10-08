//! Checks for the [`SamplingImpl`](crate::oep::SamplingImpl) seam.
//!
//! The suite checks secret-law moments, bounded noise distributions, full-width
//! placement, untouched storage and independence from destination shape. Two
//! Gaussian additions also check the mean and standard deviation on the supplied
//! module, with tolerances scaled to its degree.

use dashu_int::{IBig, UBig};
use poulpy_hal::{AlignedBuf, alloc_aligned};
use std::f64::consts::SQRT_2;

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallAssign,
    },
    layouts::{
        Backend, Module, ScalarZnx, ScratchOwned, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxOwned,
        ZnxView, ZnxViewMut, ZnxWord,
    },
    source::Source,
    test_suite::{
        TestBackend, TestParams, alloc_host_vec_znx, download_scalar_znx, download_vec_znx, scalar_znx_backend_mut,
        upload_scalar_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref,
    },
};

use crate::{ComponentNoise, Distribution, Noise, ScalarZnxFillDistribution, VecZnxAddNoise, VecZnxBigAddNoise};

const BASE2K: usize = 17;
const SIZE: usize = 5;
const COLS: usize = 2;
const SENTINEL: i64 = 7;

/// Uploads a two-column `ScalarZnx` holding the sentinel everywhere, fills
/// column 1 through the seam, downloads, and returns `(column 0, column 1)`.
fn fill_and_download<BE: TestBackend>(
    module: &Module<BE>,
    dist: Distribution,
    source: &mut Source,
) -> (Vec<BE::ZnxWord>, Vec<BE::ZnxWord>)
where
    Module<BE>: ScalarZnxFillDistribution<BE>,
{
    let mut host: ScalarZnx<AlignedBuf, BE::ZnxWord> = ScalarZnx::from_data(
        alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, BE::ZnxWord>::bytes_of(module.n(), COLS)),
        module.n(),
        COLS,
    );
    for col in 0..COLS {
        host.at_mut(col, 0).fill(<BE::ZnxWord as ZnxWord>::from_i64(SENTINEL));
    }
    let mut backend = upload_scalar_znx::<BE>(&host);
    module.scalar_znx_fill_distribution(&mut scalar_znx_backend_mut::<BE>(&mut backend), 1, dist, source);
    let out = download_scalar_znx::<BE>(&backend);
    (out.at(0, 0).to_vec(), out.at(1, 0).to_vec())
}

pub fn test_scalar_znx_fill_distribution<BE: TestBackend>(_params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ScalarZnxFillDistribution<BE>,
{
    let n: usize = module.n();
    let word = |v: i64| <<BE as Backend>::ZnxWord as ZnxWord>::from_i64(v);
    let (zero, one, minus_one, sentinel) = (word(0), word(1), word(-1), word(SENTINEL));
    let ternary = |w: &BE::ZnxWord| *w == zero || *w == one || *w == minus_one;
    let binary = |w: &BE::ZnxWord| *w == zero || *w == one;
    let hw: usize = n / 4;
    let block_size: usize = 8;
    assert_eq!(n % block_size, 0);

    let cases: [Distribution; 6] = [
        Distribution::TernaryFixed(hw),
        Distribution::TernaryProb(0.5),
        Distribution::BinaryFixed(hw),
        Distribution::BinaryProb(0.5),
        Distribution::BinaryBlock(block_size),
        Distribution::ZERO,
    ];

    for dist in cases {
        let mut source: Source = Source::new([1u8; 32]);
        let (untouched, col) = fill_and_download(module, dist, &mut source);
        assert!(untouched.iter().all(|w| *w == sentinel), "{dist:?}: column 0 was written");
        assert_eq!(col.len(), n);
        let non_zero: usize = col.iter().filter(|w| **w != zero).count();
        match dist {
            Distribution::TernaryFixed(h) => {
                assert!(col.iter().all(ternary), "{dist:?}: value outside {{-1, 0, 1}}");
                assert_eq!(non_zero, h, "{dist:?}: hamming weight");
            }
            Distribution::TernaryProb(_) => {
                assert!(col.iter().all(ternary), "{dist:?}: value outside {{-1, 0, 1}}");
                assert!(
                    (n / 4..=3 * n / 4).contains(&non_zero),
                    "{dist:?}: {non_zero} non-zero out of {n} at p = 0.5"
                );
            }
            Distribution::BinaryFixed(h) => {
                assert!(col.iter().all(binary), "{dist:?}: value outside {{0, 1}}");
                assert_eq!(non_zero, h, "{dist:?}: hamming weight");
            }
            Distribution::BinaryProb(_) => {
                assert!(col.iter().all(binary), "{dist:?}: value outside {{0, 1}}");
                assert!(
                    (n / 4..=3 * n / 4).contains(&non_zero),
                    "{dist:?}: {non_zero} non-zero out of {n} at p = 0.5"
                );
            }
            Distribution::BinaryBlock(b) => {
                assert!(col.iter().all(binary), "{dist:?}: value outside {{0, 1}}");
                assert!(
                    col.chunks(b).all(|chunk| chunk.iter().filter(|w| **w == one).count() <= 1),
                    "{dist:?}: a block holds more than one 1"
                );
                assert!(non_zero > 0, "{dist:?}: all zero");
            }
            Distribution::ZERO => assert_eq!(non_zero, 0, "{dist:?}: not zero"),
            Distribution::NONE | Distribution::ENCAPSULATED(_) => unreachable!(),
        }

        if !matches!(dist, Distribution::ZERO) {
            // Same seed, same column; a second draw from the same source differs.
            let (_, again) = fill_and_download(module, dist, &mut Source::new([1u8; 32]));
            assert_eq!(again, col, "{dist:?}: not a function of the seed");
            let (_, next) = fill_and_download(module, dist, &mut source);
            assert_ne!(next, col, "{dist:?}: the source did not advance");
        }
    }
}

const NOISE_K: usize = 2 * BASE2K - 3;
fn noise_infos() -> Noise {
    Noise::ENCRYPTION
}

/// Four standard errors of a sample standard deviation over `n` draws. Wide
/// enough that a correct sampler never trips it, narrow enough that a wrong
/// scale does.
fn std_tolerance(n: usize, expected: f64) -> f64 {
    4.0 * expected / (2.0 * n as f64).sqrt()
}

/// Asserts the shape of `a` after two additions into `col_i`.
fn assert_two_additions(a: &VecZnxOwned<i64>, col_i: usize, k: usize) {
    let zero: Vec<i64> = vec![0; a.n()];
    let k_f64: f64 = (1u64 << k as u64) as f64;
    for col_j in 0..COLS {
        if col_j != col_i {
            for limb_i in 0..SIZE {
                assert_eq!(a.at(col_j, limb_i), zero);
            }
            continue;
        }
        let std: f64 = a.stats(BASE2K, col_i).std() * k_f64;
        let want: f64 = 3.2 * SQRT_2;
        assert!((std - want).abs() < std_tolerance(a.n(), want), "std={std} ~!= {want}");
        let mean = a.stats(BASE2K, col_i).mean() * k_f64;
        assert!(mean.abs() < 6.0 * want / (a.n() as f64).sqrt(), "mean={mean}");
        let limb = k.div_ceil(BASE2K) - 1;
        let shift = (limb + 1) * BASE2K - k;
        let low_mask = (1i64 << shift) - 1;
        assert!(a.at(col_i, limb).iter().all(|value| value & low_mask == 0));
    }
}

pub fn test_vec_znx_add_noise<BE: TestBackend>(_params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxAddNoise<BE>,
{
    let noise = noise_infos();
    let mut source: Source = Source::new([0u8; 32]);

    for k in [2 * BASE2K - 3, 2 * BASE2K + 1] {
        for col_i in 0..COLS {
            let mut a = upload_vec_znx::<BE>(&alloc_host_vec_znx::<BE>(module.n(), COLS, SIZE));
            module.vec_znx_add_noise(BASE2K, k, &mut vec_znx_backend_mut::<BE>(&mut a), col_i, noise, &mut source);
            module.vec_znx_add_noise(BASE2K, k, &mut vec_znx_backend_mut::<BE>(&mut a), col_i, noise, &mut source);
            assert_two_additions(&download_vec_znx::<BE>(&a), col_i, k);
        }
    }
}

pub fn test_vec_znx_big_add_noise<BE: TestBackend>(_params: &TestParams, module: &Module<BE>)
where
    Module<BE>:
        VecZnxBigAddNoise<BE> + VecZnxAlloc<BE> + VecZnxBigAlloc<BE> + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let noise = noise_infos();
    let mut source: Source = Source::new([2u8; 32]);
    let mut scratch = ScratchOwned::<BE>::alloc(module.vec_znx_big_normalize_tmp_bytes());

    for k in [2 * BASE2K - 3, 2 * BASE2K + 1] {
        for col_i in 0..COLS {
            let mut a: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(module.n(), COLS, SIZE);
            module.vec_znx_big_add_noise(BASE2K, k, &mut a.to_backend_mut(), col_i, noise, &mut source);
            module.vec_znx_big_add_noise(BASE2K, k, &mut a.to_backend_mut(), col_i, noise, &mut source);

            // The digits are only readable once normalized back into a `VecZnx`.
            let mut res = module.vec_znx_alloc(module.n(), COLS, SIZE);
            for col_j in 0..COLS {
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res),
                    BASE2K,
                    SIZE * BASE2K,
                    0,
                    col_j,
                    &a.to_backend_ref(),
                    BASE2K,
                    col_j,
                    &mut scratch.borrow(),
                );
            }
            assert_two_additions(&download_vec_znx::<BE>(&res), col_i, k);
        }
    }
}

/// Same-backend Gaussian reproducibility, source advancement and preservation.
/// Random streams deliberately are not compared between distinct backends.
fn gaussian_reproducibility<BE: TestBackend>(module: &Module<BE>)
where
    Module<BE>: VecZnxAddNoise<BE>
        + VecZnxBigAddNoise<BE>
        + VecZnxAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let noise = noise_infos();
    for big in [false, true] {
        let draw = |source: &mut Source| {
            let mut result = module.vec_znx_alloc(module.n(), COLS, SIZE);
            if big {
                let mut a = module.vec_znx_big_alloc(module.n(), COLS, SIZE);
                module.vec_znx_big_add_noise(BASE2K, NOISE_K, &mut a.to_backend_mut(), 1, noise, source);
                let mut scratch = ScratchOwned::<BE>::alloc(module.vec_znx_big_normalize_tmp_bytes());
                for col in 0..COLS {
                    module.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BE>(&mut result),
                        BASE2K,
                        SIZE * BASE2K,
                        0,
                        col,
                        &a.to_backend_ref(),
                        BASE2K,
                        col,
                        &mut scratch.borrow(),
                    );
                }
            } else {
                module.vec_znx_add_noise(BASE2K, NOISE_K, &mut vec_znx_backend_mut::<BE>(&mut result), 1, noise, source);
            }
            download_vec_znx::<BE>(&result)
        };
        let mut first = Source::new([91; 32]);
        let mut repeat = Source::new([91; 32]);
        let a = draw(&mut first);
        let b = draw(&mut repeat);
        assert_eq!(a, b, "Gaussian same-backend reproducibility (big={big})");
        assert_eq!(first.new_seed(), repeat.new_seed(), "Gaussian source consumption (big={big})");
        assert_ne!(a, draw(&mut first), "Gaussian source did not advance (big={big})");
        for limb in 0..SIZE {
            assert!(a.at(0, limb).iter().all(|value| *value == 0));
        }
    }
}

/// Checks the moments used by fresh-noise planning with at least 2^14 draws.
fn secret_moments<BE: TestBackend>(module: &Module<BE>)
where
    Module<BE>: ScalarZnxFillDistribution<BE>,
{
    for dist in [
        Distribution::TernaryProb(0.25),
        Distribution::BinaryProb(0.25),
        Distribution::BinaryBlock(3),
        Distribution::BinaryBlock(8),
        Distribution::TernaryFixed(module.n() / 4),
    ] {
        // Scalar buffers do not require a power-of-two degree. Exercise a
        // whole number of blocks for the non-power-of-two block law.
        let n = if let Distribution::BinaryBlock(b) = dist {
            module.n() / b * b
        } else {
            module.n()
        };
        let host = ScalarZnx::from_data(alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, i64>::bytes_of(n, 1)), n, 1);
        let mut out = upload_scalar_znx::<BE>(&host);
        let mut source = Source::new([81; 32]);
        let mut sum = 0i64;
        let mut squares = 0usize;
        let mut empty = 0usize;
        let mut blocks = 0usize;
        let draws = (1usize << 14).div_ceil(n);
        for _ in 0..draws {
            module.scalar_znx_fill_distribution(&mut scalar_znx_backend_mut::<BE>(&mut out), 0, dist, &mut source);
            let host = download_scalar_znx::<BE>(&out);
            for &value in host.at(0, 0) {
                assert!((-1..=1).contains(&value));
                sum += value;
                squares += (value * value) as usize;
            }
            if let Distribution::BinaryBlock(b) = dist {
                for block in host.at(0, 0).chunks_exact(b) {
                    assert!(block.iter().all(|&x| x == 0 || x == 1));
                    let weight = block.iter().sum::<i64>();
                    assert!(weight <= 1);
                    empty += usize::from(weight == 0);
                    blocks += 1;
                }
            }
        }
        let count = (draws * n) as f64;
        let law = ComponentNoise::from_secret(dist, 0).secret_distribution();
        let mean = law.coefficient_mean(n).unwrap();
        let second = law.coefficient_second_moment(n).unwrap();
        assert!((sum as f64 / count - mean).abs() < 7.0 / count.sqrt(), "{dist:?}: mean");
        assert!(
            (squares as f64 / count - second).abs() < 7.0 / count.sqrt(),
            "{dist:?}: second moment"
        );
        if let Distribution::BinaryBlock(b) = dist {
            let p = 1.0 / (b + 1) as f64;
            assert!((empty as f64 / blocks as f64 - p).abs() < 7.0 * (p * (1.0 - p) / blocks as f64).sqrt());
        }
    }
}

fn reconstruct(a: &VecZnxOwned<i64>, col: usize, i: usize, base: usize, k: usize) -> IBig {
    let active = k.div_ceil(base);
    let padding = active * base - k;
    let raw = (0..active).fold(IBig::ZERO, |raw, limb| (raw << base) + a.at(col, limb)[i]);
    assert_eq!(&raw % (IBig::ONE << padding), IBig::ZERO, "nonzero precision padding");
    raw >> padding
}

/// Full-width distribution checks run on the supplied module, without relying
/// on its particular mapping from seeds to integer samples.
pub fn test_full_width_noise<BE: TestBackend>(module: &Module<BE>)
where
    Module<BE>: VecZnxAddNoise<BE>,
{
    for k in [173usize, 170] {
        let active = k.div_ceil(BASE2K);
        let padding = active * BASE2K - k;
        for noise in [
            Noise::Uniform { bits: 1 },
            Noise::Uniform { bits: BASE2K - padding },
            Noise::Uniform {
                bits: BASE2K - padding + 1,
            },
            Noise::Uniform { bits: 135 },
            Noise::Gaussian { sigma: 2f64.powi(132) },
            Noise::Gaussian { sigma: 1.0 },
            Noise::Gaussian { sigma: 1.5 },
            Noise::Gaussian { sigma: 16.75 },
        ] {
            let bound = match noise {
                Noise::Uniform { bits } => UBig::ONE << (bits - 1),
                Noise::Gaussian { sigma } if sigma > 1e30 => (UBig::ONE << 132) * 6u8,
                Noise::Gaussian { sigma: 1.0 } => UBig::from(6u8),
                Noise::Gaussian { sigma: 1.5 } => UBig::from(9u8),
                Noise::Gaussian { .. } => UBig::from(100u8),
            };
            let pmf = match noise {
                Noise::Gaussian { sigma } if sigma < 17.0 => Some((sigma, usize::try_from(&bound).unwrap())),
                _ => None,
            };
            let rounds = if pmf.is_some() {
                (1usize << 14).div_ceil(module.n())
            } else {
                256usize.div_ceil(module.n())
            };
            let mut histogram = vec![0usize; pmf.map_or(0, |(_, b)| 2 * b + 1)];
            let mut negative = 0usize;
            let mut positive = 0usize;
            let mut wide = 0usize;
            let mut low_bits = [0usize; 8];
            let mut source = Source::new([64; 32]);
            for _ in 0..rounds {
                let mut host = alloc_host_vec_znx::<BE>(module.n(), 2, active + 1);
                for limb in 0..=active {
                    host.at_mut(0, limb).fill(91);
                }
                host.at_mut(1, active).fill(37);
                let mut out = upload_vec_znx::<BE>(&host);
                module.vec_znx_add_noise(BASE2K, k, &mut vec_znx_backend_mut::<BE>(&mut out), 1, noise, &mut source);
                let got = download_vec_znx::<BE>(&out);
                for i in 0..got.n() {
                    let sample = reconstruct(&got, 1, i, BASE2K, k);
                    if matches!(noise, Noise::Uniform { .. }) {
                        assert!(sample < IBig::from(bound.clone()), "uniform upper endpoint is excluded");
                    }
                    negative += usize::from(sample < IBig::ZERO);
                    positive += usize::from(sample > IBig::ZERO);
                    if let Some((_, b)) = pmf {
                        histogram[(i64::try_from(&sample).unwrap() + b as i64) as usize] += 1;
                    }
                    let (_, magnitude) = sample.into_parts();
                    assert!(magnitude <= bound, "{noise:?}: support");
                    wide += usize::from(magnitude > UBig::ONE << 128);
                    low_bits[usize::try_from(&magnitude & UBig::from(7u8)).unwrap()] += 1;
                    for limb in 0..active {
                        assert!((-(1 << (BASE2K - 1))..1 << (BASE2K - 1)).contains(&got.at(1, limb)[i]));
                    }
                }
                for limb in 0..=active {
                    assert!(got.at(0, limb).iter().all(|&x| x == 91));
                }
                assert!(got.at(1, active).iter().all(|&x| x == 37));
            }
            let count = (rounds * module.n()) as f64;
            if let Some((sigma, b)) = pmf {
                let weights: Vec<_> = (-(b as i64)..=b as i64)
                    .map(|z| (-(z as f64).powi(2) / (2.0 * sigma * sigma)).exp())
                    .collect();
                let total: f64 = weights.iter().sum();
                for (actual, weight) in histogram.into_iter().zip(weights) {
                    let p = weight / total;
                    assert!(
                        (actual as f64 - count * p).abs() < 7.0 * (count * p * (1.0 - p)).sqrt() + 8.0,
                        "{noise:?}: PMF"
                    );
                }
                assert!((negative as f64 - positive as f64).abs() < 7.0 * count.sqrt());
            } else {
                assert!((negative as f64 - count / 2.0).abs() < 7.0 * (count / 4.0).sqrt());
            }
            if bound > UBig::ONE << 128 {
                assert!(wide as f64 > count * 0.85);
                assert!(low_bits.iter().all(|&x| x as f64 > count / 32.0));
            }
        }
    }
}

fn stream_independence<BE: TestBackend>(module: &Module<BE>)
where
    Module<BE>: ScalarZnxFillDistribution<BE>
        + VecZnxAddNoise<BE>
        + VecZnxBigAddNoise<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigSubSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    for dist in [Distribution::TernaryProb(0.25), Distribution::BinaryBlock(8)] {
        let draw = |cols, col| {
            let mut host = ScalarZnx::from_data(
                alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, i64>::bytes_of(module.n(), cols)),
                module.n(),
                cols,
            );
            for i in 0..cols {
                host.at_mut(i, 0).fill(9);
            }
            let mut out = upload_scalar_znx::<BE>(&host);
            module.scalar_znx_fill_distribution(
                &mut scalar_znx_backend_mut::<BE>(&mut out),
                col,
                dist,
                &mut Source::new([28; 32]),
            );
            download_scalar_znx::<BE>(&out).at(col, 0).to_vec()
        };
        assert_eq!(draw(1, 0), draw(3, 2));
    }
    for big in [false, true] {
        for k in [2 * BASE2K - 3, 2 * BASE2K + 1] {
            for noise in [
                Noise::ENCRYPTION,
                Noise::Uniform { bits: 24 },
                Noise::Gaussian { sigma: 16.75 },
            ] {
                let active = k.div_ceil(BASE2K);
                let draw = |cols, size, col| {
                    let initial = if cols > 1 { 11 } else { 0 };
                    let mut host = alloc_host_vec_znx::<BE>(module.n(), cols, size);
                    for limb in 0..active {
                        host.at_mut(col, limb).fill(initial);
                    }
                    let mut out = upload_vec_znx::<BE>(&host);
                    let mut source = Source::new([29; 32]);
                    let mut residual = initial;
                    if big {
                        // Preload the same digits and remove them after sampling: the sampler only adds.
                        let mut wide = module.vec_znx_big_alloc(module.n(), cols, size);
                        module.vec_znx_big_from_small(&mut wide.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&out), col);
                        module.vec_znx_big_add_noise(BASE2K, k, &mut wide.to_backend_mut(), col, noise, &mut source);
                        module.vec_znx_big_sub_small_assign(
                            &mut wide.to_backend_mut(),
                            col,
                            &vec_znx_backend_ref::<BE>(&out),
                            col,
                        );
                        residual = 0;
                        let mut scratch = ScratchOwned::<BE>::alloc(module.vec_znx_big_normalize_tmp_bytes());
                        module.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BE>(&mut out),
                            BASE2K,
                            size * BASE2K,
                            0,
                            col,
                            &wide.to_backend_ref(),
                            BASE2K,
                            col,
                            &mut scratch.borrow(),
                        );
                    } else {
                        module.vec_znx_add_noise(BASE2K, k, &mut vec_znx_backend_mut::<BE>(&mut out), col, noise, &mut source);
                    }
                    let host = download_vec_znx::<BE>(&out);
                    (0..active)
                        .flat_map(|limb| host.at(col, limb).iter().map(|&digit| digit - residual))
                        .collect::<Vec<_>>()
                };
                assert_eq!(draw(1, active, 0), draw(3, active + 2, 2), "big={big}, {noise:?}, k={k}");
            }
        }
    }
}

/// Runs the distribution and seeded-stream contracts independently on each
/// backend, on the modules the suite supplies. The noise bounds follow their
/// degree, see the module documentation.
pub fn test_sampling_contract<BR: TestBackend, BT: TestBackend>(
    params: &TestParams,
    _: &crate::test_suite::parity::ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
) where
    Module<BR>: ScalarZnxFillDistribution<BR>
        + VecZnxAddNoise<BR>
        + VecZnxBigAddNoise<BR>
        + VecZnxAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmall<BR>
        + VecZnxBigSubSmallAssign<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: ScalarZnxFillDistribution<BT>
        + VecZnxAddNoise<BT>
        + VecZnxBigAddNoise<BT>
        + VecZnxAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmall<BT>
        + VecZnxBigSubSmallAssign<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    test_scalar_znx_fill_distribution(params, r);
    test_scalar_znx_fill_distribution(params, t);
    test_vec_znx_add_noise(params, r);
    test_vec_znx_add_noise(params, t);
    test_vec_znx_big_add_noise(params, r);
    test_vec_znx_big_add_noise(params, t);
    gaussian_reproducibility(r);
    gaussian_reproducibility(t);
    secret_moments(r);
    secret_moments(t);
    test_full_width_noise(r);
    test_full_width_noise(t);
    stream_independence(r);
    stream_independence(t);
}
