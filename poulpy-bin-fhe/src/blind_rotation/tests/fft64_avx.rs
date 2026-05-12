use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::layouts::Module;

use crate::blind_rotation::{
    CGGI,
    tests::test_suite::{
        generic_blind_rotation::test_blind_rotation,
        generic_lut::{test_lut_extended, test_lut_standard},
    },
};

#[test]
fn lut_standard() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(32);
    test_lut_standard(&module);
}

#[test]
fn lut_extended() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(32);
    test_lut_extended(&module);
}

#[test]
fn standard() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(512);
    test_blind_rotation::<CGGI, _, FFT64Avx>(&module, 224, 1, 1);
}

#[test]
fn block_binary() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(512);
    test_blind_rotation::<CGGI, _, FFT64Avx>(&module, 224, 7, 1);
}

#[test]
fn block_binary_extended() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(512);
    test_blind_rotation::<CGGI, _, FFT64Avx>(&module, 224, 7, 2);
}
