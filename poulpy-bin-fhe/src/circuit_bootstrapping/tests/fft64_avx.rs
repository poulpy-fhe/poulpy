use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::layouts::Module;

use crate::{
    blind_rotation::CGGI,
    circuit_bootstrapping::tests::circuit_bootstrapping::{
        test_circuit_bootstrapping_to_constant, test_circuit_bootstrapping_to_exponent,
    },
};

#[test]
fn to_constant_cggi() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(256);
    test_circuit_bootstrapping_to_constant::<FFT64Avx, _, CGGI>(&module);
}

#[test]
fn to_exponent_cggi() {
    let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(256);
    test_circuit_bootstrapping_to_exponent::<FFT64Avx, _, CGGI>(&module);
}
