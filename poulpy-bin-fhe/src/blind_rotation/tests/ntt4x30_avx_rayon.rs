use poulpy_cpu_avx::{NTT4x30Avx, NTT4x30AvxRayon};
use poulpy_hal::layouts::Module;

use crate::blind_rotation::{CGGI, tests::test_suite::generic_blind_rotation::test_blind_rotation};

#[test]
fn block_binary_matches_serial() {
    let serial = Module::<NTT4x30Avx>::new(512);
    let rayon = Module::<NTT4x30AvxRayon>::new(512);
    let expected = test_blind_rotation::<CGGI, _, NTT4x30Avx>(&serial, 224, 7, 1);
    let actual = test_blind_rotation::<CGGI, _, NTT4x30AvxRayon>(&rayon, 224, 7, 1);
    assert_eq!(actual, expected);
}
