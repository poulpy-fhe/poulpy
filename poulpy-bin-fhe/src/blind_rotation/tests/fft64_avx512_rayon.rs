use poulpy_cpu_avx512::{FFT64Avx512, FFT64Avx512Rayon};
use poulpy_hal::layouts::Module;

use crate::blind_rotation::{CGGI, tests::test_suite::generic_blind_rotation::test_blind_rotation};

#[test]
fn block_binary() {
    let module: Module<FFT64Avx512Rayon> = Module::new(512);
    test_blind_rotation::<CGGI, _, FFT64Avx512Rayon>(&module, 224, 7, 1);
}

#[test]
fn block_binary_matches_serial() {
    let serial: Module<FFT64Avx512> = Module::new(512);
    let rayon: Module<FFT64Avx512Rayon> = Module::new(512);
    let expected = test_blind_rotation::<CGGI, _, FFT64Avx512>(&serial, 224, 7, 1);
    let actual = test_blind_rotation::<CGGI, _, FFT64Avx512Rayon>(&rayon, 224, 7, 1);
    assert_eq!(actual, expected);
}
