use crate::CKKSResult as Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSLayout,
    layouts::{CKKSCiphertextOwned, CKKSFoldKeys},
    reference::fold::CKKSFoldRing,
};

/// Folds CKKS ciphertexts into the standard ciphertexts of this module's degree
/// that a bootstrap refreshes, and unfolds the refreshed ones.
///
/// The strategy follows the inputs: complex standard inputs are packed alone and
/// real standard inputs in pairs `x + i·y`; conjugate-invariant inputs are unfolded
/// to the standard ring of twice their degree, then paired. Ring packing merges
/// `g = N/n` of the resulting ciphertexts of degree `n` as `Σ_j X^j·ct_j(X^g)` and
/// switches them to the bootstrap secret with the ring-switch keys of `keys`;
/// unfolding switches back, splits and converts. Inputs of `input_module` must
/// share their layout, scale and sparsity, and outputs their layout.
pub trait CKKSFoldOps<BE: Backend> {
    /// Number of ciphertexts `ins` folds into.
    fn ckks_fold_count<IN>(&self, input_module: &Module<IN>, ins: &[CKKSCiphertextOwned<IN>]) -> usize
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>;

    /// Layout of the ciphertexts that inputs like `ct_in` fold into.
    fn ckks_fold_layout<IN, C, K>(&self, input_module: &Module<IN>, ct_in: &C, keys: &K) -> CKKSLayout
    where
        IN: Backend,
        C: CKKSCtBounds,
        K: CKKSFoldKeys<BE, IN>;

    /// Scratch bound of [`Self::ckks_fold`] and [`Self::ckks_unfold`] for outputs
    /// like `ct_out` and inputs like `ct_in`.
    fn ckks_fold_tmp_bytes<IN, C1, C2, K>(&self, input_module: &Module<IN>, ct_out: &C1, ct_in: &C2, keys: &K) -> usize
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        K: CKKSFoldKeys<BE, IN>;

    /// Folds `ins` into `folded`, which holds [`Self::ckks_fold_count`] ciphertexts
    /// of [`Self::ckks_fold_layout`].
    fn ckks_fold<IN, K>(
        &self,
        input_module: &Module<IN>,
        folded: &mut [CKKSCiphertextOwned<BE>],
        ins: &[CKKSCiphertextOwned<IN>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        K: CKKSFoldKeys<BE, IN>;

    /// Unfolds `refreshed`, the refreshed ciphertexts folded from `ins`, into `outs`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_unfold<IN, K>(
        &self,
        input_module: &Module<IN>,
        outs: &mut [CKKSCiphertextOwned<IN>],
        refreshed: &[CKKSCiphertextOwned<BE>],
        ins: &[CKKSCiphertextOwned<IN>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<BE, IN>,
        K: CKKSFoldKeys<BE, IN>;
}
