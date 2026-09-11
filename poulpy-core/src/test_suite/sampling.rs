//! Checks for the [`SamplingImpl`](crate::oep::SamplingImpl) seam.

use poulpy_hal::{
    layouts::{Backend, Module, ZnxView, ZnxViewMut, ZnxWord},
    source::Source,
    test_suite::{TestBackend, TestParams, download_scalar_znx, scalar_znx_backend_mut, upload_scalar_znx},
};

use crate::{Distribution, ScalarZnxFillDistribution, scalar_znx_host_zeroed};

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
