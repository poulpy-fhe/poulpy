//! Checks for the [`SamplingImpl`](crate::oep::SamplingImpl) seam.
//!
//! The noise checks add two Gaussians into one column, then assert: the other
//! column is untouched, the sum has standard deviation `sigma * sqrt(2)` at
//! torus precision `2^-k`, and the unused low bits of the target limb are zero.
//! The module degree has to be at least 4096 for the 0.1 tolerance on the
//! standard deviation to hold, so those two are registered by hand next to a
//! suite that runs that wide, not from `core_backend_test_suite!` (degree 256).

use std::f64::consts::SQRT_2;

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes},
    layouts::{
        Backend, Module, ScratchOwned, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxOwned, ZnxView,
        ZnxViewMut, ZnxWord,
    },
    source::Source,
    test_suite::{
        TestBackend, TestParams, alloc_host_vec_znx, download_scalar_znx, download_vec_znx, scalar_znx_backend_mut,
        upload_scalar_znx, upload_vec_znx, vec_znx_backend_mut,
    },
};

use crate::{Distribution, NoiseInfos, ScalarZnxFillDistribution, VecZnxAddNormal, VecZnxBigAddNormal, scalar_znx_host_zeroed};

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
    let mut host = scalar_znx_host_zeroed::<BE::ZnxWord>(module.n(), COLS);
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

fn noise_infos() -> NoiseInfos {
    NoiseInfos::new(2 * BASE2K - 3, 3.2, 6.0 * 3.2).unwrap()
}

/// Asserts the shape of `a` after two additions into `col_i`.
fn assert_two_additions(a: &VecZnxOwned<i64>, col_i: usize, noise: NoiseInfos) {
    let zero: Vec<i64> = vec![0; a.n()];
    let k_f64: f64 = (1u64 << noise.k as u64) as f64;
    for col_j in 0..COLS {
        if col_j != col_i {
            for limb_i in 0..SIZE {
                assert_eq!(a.at(col_j, limb_i), zero);
            }
            continue;
        }
        let std: f64 = a.stats(BASE2K, col_i).std() * k_f64;
        assert!(
            (std - noise.sigma * SQRT_2).abs() < 0.1,
            "std={std} ~!= {}",
            noise.sigma * SQRT_2
        );
        let (limb, shift) = noise.target_limb_and_shift(BASE2K);
        let low_mask = (1i64 << shift) - 1;
        assert!(a.at(col_i, limb).iter().all(|value| value & low_mask == 0));
    }
}

pub fn test_vec_znx_add_normal<BE: TestBackend>(_params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxAddNormal<BE>,
{
    let noise = noise_infos();
    let mut source: Source = Source::new([0u8; 32]);

    for col_i in 0..COLS {
        let mut a = upload_vec_znx::<BE>(&alloc_host_vec_znx::<BE>(module.n(), COLS, SIZE));
        module.vec_znx_add_normal(BASE2K, &mut vec_znx_backend_mut::<BE>(&mut a), col_i, noise, &mut source);
        module.vec_znx_add_normal(BASE2K, &mut vec_znx_backend_mut::<BE>(&mut a), col_i, noise, &mut source);
        assert_two_additions(&download_vec_znx::<BE>(&a), col_i, noise);
    }
}

pub fn test_vec_znx_big_add_normal<BE: TestBackend>(_params: &TestParams, module: &Module<BE>)
where
    Module<BE>:
        VecZnxBigAddNormal<BE> + VecZnxAlloc<BE> + VecZnxBigAlloc<BE> + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let noise = noise_infos();
    let mut source: Source = Source::new([2u8; 32]);
    let mut scratch = ScratchOwned::<BE>::alloc(module.vec_znx_big_normalize_tmp_bytes());

    for col_i in 0..COLS {
        let mut a: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(COLS, SIZE);
        module.vec_znx_big_add_normal(BASE2K, &mut a.to_backend_mut(), col_i, noise, &mut source);
        module.vec_znx_big_add_normal(BASE2K, &mut a.to_backend_mut(), col_i, noise, &mut source);

        // The digits are only readable once normalized back into a `VecZnx`.
        let mut res = module.vec_znx_alloc(COLS, SIZE);
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
        assert_two_additions(&download_vec_znx::<BE>(&res), col_i, noise);
    }
}
