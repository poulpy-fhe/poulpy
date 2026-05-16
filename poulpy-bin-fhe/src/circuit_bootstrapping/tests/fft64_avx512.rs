use poulpy_cpu_avx512::FFT64Avx512;
use poulpy_hal::layouts::Module;

use crate::{
    blind_rotation::CGGI,
    circuit_bootstrapping::tests::circuit_bootstrapping::{
        test_circuit_bootstrapping_to_constant, test_circuit_bootstrapping_to_exponent,
    },
};

#[test]
fn to_constant_cggi() {
    let module: Module<FFT64Avx512> = Module::<FFT64Avx512>::new(256);
    test_circuit_bootstrapping_to_constant::<FFT64Avx512, _, CGGI>(&module);
}

#[test]
fn to_exponent_cggi() {
    let module: Module<FFT64Avx512> = Module::<FFT64Avx512>::new(256);
    test_circuit_bootstrapping_to_exponent::<FFT64Avx512, _, CGGI>(&module);
}
