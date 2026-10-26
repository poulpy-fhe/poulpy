use std::panic::{AssertUnwindSafe, catch_unwind};

use poulpy_hal::layouts::{Backend, CnvPVecL, DataView};

use crate::{
    ComponentNoise,
    layouts::{Base2K, GLWE, GLWEPlaintextMeta, GLWETensor, TorusPrecision},
    test_suite::decryption::panic_message,
};

/// Runs `call` on `state` and asserts it panics with exactly `message` and leaves `snapshot(state)` as it was.
pub(crate) fn assert_refusal<S, V: PartialEq>(
    message: &str,
    state: &mut S,
    snapshot: impl Fn(&S) -> V,
    call: impl FnOnce(&mut S),
) {
    let before = snapshot(state);
    let err = catch_unwind(AssertUnwindSafe(|| call(state))).expect_err("the call was accepted");
    assert_eq!(panic_message(&*err), message);
    assert!(snapshot(state) == before, "a refused call wrote into its destination");
}

/// Data and metadata of a GLWE: everything a refused call must leave alone.
pub(crate) type GLWEState = (
    Vec<u8>,
    Option<ComponentNoise>,
    bool,
    Base2K,
    TorusPrecision,
    Option<GLWEPlaintextMeta>,
);

pub(crate) fn glwe_state<BE: Backend>(glwe: &GLWE<BE::OwnedBuf, BE::ZnxWord>) -> GLWEState {
    (
        BE::to_host_bytes(glwe.data.data()),
        glwe.noise.clone(),
        glwe.canonical,
        glwe.base2k,
        glwe.k,
        glwe.plaintext_meta,
    )
}

pub(crate) fn glwe_tensor_state<BE: Backend>(
    tensor: &GLWETensor<BE::OwnedBuf, BE::ZnxWord>,
) -> (Vec<u8>, Base2K, TorusPrecision) {
    (BE::to_host_bytes(tensor.data.data()), tensor.base2k, tensor.k)
}

pub(crate) fn cnv_pvec_left_state<BE: Backend>(pvec: &CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>) -> (Vec<u8>, usize, usize, usize) {
    (BE::to_host_bytes(pvec.data()), pvec.n(), pvec.cols(), pvec.size())
}
