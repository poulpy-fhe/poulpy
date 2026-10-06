//! Explicit opt-ins for comparing independently implemented encryption paths.

use poulpy_hal::layouts::Module;

use crate::{DFTFamily, Oracle, OracleRing};

/// Constructs an oracle module that copies the current controlled sampling scope's draws.
///
/// Only this module's core samplers change. Its arithmetic remains the oracle's
/// independent implementation, and ordinary oracle modules retain their independent
/// samplers, including when used concurrently with this module. The caller must use
/// it inside [`poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples`].
/// Drawing outside such a scope panics.
pub fn controlled_sampling_module<F: DFTFamily, R: OracleRing>(n: u64) -> Module<Oracle<F, R>> {
    let module = Module::<Oracle<F, R>>::new(n);
    // The newly allocated handle has not been shared. Set its immutable sampling
    // policy before returning the module to its owner or any worker threads.
    unsafe { (*module.ptr()).controlled_sampling = true };
    module
}
