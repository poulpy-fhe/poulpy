use anyhow::Result;
use poulpy_core::{GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Scratch};

use crate::layouts::CKKSCiphertext;

/// CKKS rescaling and level-alignment APIs.
///
/// Rescale lowers `log_budget` by shifting the torus representation. Align
/// equalizes the `log_budget` of two ciphertexts by rescaling the one with
/// more remaining capacity.
pub trait CKKSRescaleOps<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_rescale_into`].
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    /// Rescales a ciphertext in place by `k` bits.
    ///
    /// Errors include `InsufficientHomomorphicCapacity` if `k` exceeds the
    /// available `log_budget`.
    fn ckks_rescale_assign(&self, ct: &mut CKKSCiphertext<impl DataMut>, k: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Computes a rescaled copy of `src` into `dst`.
    ///
    /// Errors include `InsufficientHomomorphicCapacity`.
    fn ckks_rescale_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Rescales either `a` or `b` in place so both ciphertexts end up with the
    /// same `log_budget`.
    ///
    /// Errors propagate from the underlying rescale operation.
    fn ckks_align_assign(
        &self,
        a: &mut CKKSCiphertext<impl DataMut>,
        b: &mut CKKSCiphertext<impl DataMut>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Returns scratch bytes required by [`Self::ckks_align_assign`].
    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;
}
