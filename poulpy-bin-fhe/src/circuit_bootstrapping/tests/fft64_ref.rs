use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::layouts::Module;

use crate::{
    blind_rotation::CGGI,
    circuit_bootstrapping::tests::circuit_bootstrapping::{
        test_circuit_bootstrapping_to_constant, test_circuit_bootstrapping_to_exponent,
    },
};

#[test]
fn to_constant_cggi() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(256);
    test_circuit_bootstrapping_to_constant::<FFT64Ref, _, CGGI>(&module);
}

#[test]
fn to_exponent_cggi() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(256);
    test_circuit_bootstrapping_to_exponent::<FFT64Ref, _, CGGI>(&module);
}
