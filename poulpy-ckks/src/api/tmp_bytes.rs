use crate::{CKKSCtBounds, CKKSInfos};
use poulpy_core::layouts::GGLWEInfos;
use poulpy_hal::layouts::Backend;

/// Helpers that return the maximum scratch size needed across broad CKKS
/// operation sets, including the selected copy implementation.
///
/// Pass layouts covering the widest live ciphertext and plaintext operands in
/// the intended pipeline. These are shared bounds for polynomial evaluation
/// and homomorphic DFT compositions; specialized implementations of those
/// contracts must fit the same bound. EvalMod has its own selected query because
/// its working precision depends on the compiled plan.
pub trait CKKSAllOpsTmpBytes<BE: Backend> {
    /// Returns a scratch size large enough for the common CKKS workflow using
    /// ciphertext ops, plaintext ops, encryption/decryption, multiplication,
    /// and tensor-key setup, including one-shot polynomial input copies.
    fn ckks_all_ops_tmp_bytes<C, T, P>(&self, ct_infos: &C, tsk_infos: &T, pt_prec: &P) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos;

    /// Returns a scratch size large enough for [`Self::ckks_all_ops_tmp_bytes`]
    /// plus automorphism-key setup, rotation, conjugation, and prepared or
    /// streamed homomorphic DFT evaluation.
    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A, P>(&self, ct_infos: &C, tsk_infos: &T, atk_infos: &A, pt_prec: &P) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos;
}
