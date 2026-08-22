use poulpy_cpu_avx::{FFT64Avx, FFT64AvxRayon};
use poulpy_hal::layouts::Module;

use crate::blind_rotation::{CGGI, tests::test_suite::generic_blind_rotation::test_blind_rotation};

#[test]
fn block_binary_matches_serial() {
    let serial = Module::<FFT64Avx>::new(512);
    let rayon = Module::<FFT64AvxRayon>::new(512);
    let expected = test_blind_rotation::<CGGI, _, FFT64Avx>(&serial, 224, 7, 1);
    let actual = test_blind_rotation::<CGGI, _, FFT64AvxRayon>(&rayon, 224, 7, 1);
    assert_eq!(actual, expected);
}
