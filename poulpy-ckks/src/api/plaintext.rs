use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::GLWEToBackendMut;

use crate::{CKKSInfos, SetCKKSInfos};

/// Plaintext ZNX extraction.
///
/// Extracts the plaintext polynomial from a CKKS ciphertext without
/// decryption — useful for zero-ciphertext operations or debugging.
///
/// # Metadata
///
/// ```text
/// log_delta_out  = src.log_delta
/// log_budget_out = src.log_budget
/// ```
pub trait CKKSPlaintextVecOps<BE: Backend> {
    fn ckks_extract_pt_tmp_bytes(&self) -> usize;

    /// Extracts the ZNX plaintext polynomial from `src` into `dst`.
    fn ckks_extract_pt<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos + LWEInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;
}
