use poulpy_cpu_arm::{FFT64Neon, FFT64NeonRayon};
use poulpy_hal::layouts::Module;

use crate::blind_rotation::{CGGI, tests::test_suite::generic_blind_rotation::test_blind_rotation};

#[test]
fn block_binary_matches_serial() {
    let serial = Module::<FFT64Neon>::new(512);
    let rayon = Module::<FFT64NeonRayon>::new(512);
    let expected = test_blind_rotation::<CGGI, _, FFT64Neon>(&serial, 224, 7, 1);
    let actual = test_blind_rotation::<CGGI, _, FFT64NeonRayon>(&rayon, 224, 7, 1);
    assert_eq!(actual, expected);
}
