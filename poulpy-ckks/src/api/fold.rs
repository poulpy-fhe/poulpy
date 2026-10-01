use crate::CKKSResult as Result;
use poulpy_core::layouts::{Degree, GGLWEInfos, GLWELayout, GetAutomorphismKey, prepared::GGLWEPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Ring, ScratchArena};

use crate::{
    CKKSCtBounds,
    layouts::{CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSRingCiphertext},
};

/// Planning queries of [`CKKSFoldOps`], from layouts only and for inputs of any ring:
/// the allocation layout of the folded ciphertexts and the scratch bound. `degree`
/// is the degree the batch folds into, the bootstrap degree, at most the module's.
pub trait CKKSFoldLayoutOps<BE: Backend> {
    /// Allocation layout of the ciphertexts of `degree` that inputs like `ct_in`
    /// fold into with keys like `keys`. The fold sets CKKS metadata from the batch:
    /// merging clears sparsity, and merging or pairing sets complex slots.
    fn ckks_fold_layout<C>(&self, ct_in: &C, degree: Degree, keys: &CKKSFoldKeysLayout) -> GLWELayout
    where
        C: CKKSCtBounds;

    /// Scratch bound of [`CKKSFoldOps::ckks_fold`] and [`CKKSFoldOps::ckks_unfold`] for
    /// outputs like `ct_out`, inputs like `ct_in` of any ring folded into `degree`, and
    /// keys like `keys`.
    fn ckks_fold_tmp_bytes<C1, C2>(&self, ct_out: &C1, ct_in: &C2, degree: Degree, keys: &CKKSFoldKeysLayout) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;
}

/// Folds CKKS ciphertexts of ring `R` into the standard ciphertexts of the bootstrap
/// degree that a bootstrap refreshes, and unfolds the refreshed ones. The bootstrap
/// degree is the degree of the folded ciphertexts, at most the module's.
///
/// The reference implementation supports `Standard` inputs: complex inputs are
/// packed alone, and real inputs of the bootstrap degree in pairs `x + i·y`.
/// Conjugate-invariant inputs can first be embedded with
/// [`ckks_ci_embed`](crate::api::CKKSCIRingMapOps::ckks_ci_embed), then folded as
/// standard inputs; after unfolding,
/// [`ckks_ci_trace`](crate::api::CKKSCIRingMapOps::ckks_ci_trace) maps them back.
/// Ring packing merges `g = N/n` of the resulting ciphertexts of degree `n` as
/// `Σ_j X^j·ct_j(X^g)` and
/// switches them to the bootstrap secret with an inbound ring-switch key; unfolding
/// switches back with an outbound one, splits and converts. Inputs under the
/// bootstrap secret at its degree need no ring-switch key. Inputs must share their
/// layout, scale and sparsity; outputs their layout, and are labeled like the inputs.
pub trait CKKSFoldOps<BE: Backend, R: Ring> {
    /// Number of ciphertexts of `degree` that `ins` folds into.
    fn ckks_fold_count(&self, ins: &[CKKSRingCiphertext<BE, R>], degree: Degree) -> usize;

    /// Galois elements of the input-secret automorphism keys [`Self::ckks_unfold`] may
    /// use on inputs like `ct_in`, from its metadata alone: `−1` splits pairs of real
    /// inputs, and one element per level of sparsity splits sparse inputs, at the degree
    /// of their packed parts. The caller names the input ring, as in
    /// `CKKSFoldOps::<_, Standard>::ckks_unfold_galois_elements(&module, &layout)`.
    fn ckks_unfold_galois_elements<C>(&self, ct_in: &C) -> Vec<i64>
    where
        C: CKKSCtBounds;

    /// Folds `ins` into `folded`, which holds [`Self::ckks_fold_count`] ciphertexts
    /// of [`CKKSFoldLayoutOps::ckks_fold_layout`], whose degree is the fold's.
    ///
    /// `inbound` switches the packed inputs from their secret to the bootstrap secret:
    /// the prepared [`RingSwitchKeys::inbound`](crate::layouts::RingSwitchKeys::inbound)
    /// of [`RingSwitchKeysLayout::generate`](crate::layouts::RingSwitchKeysLayout::generate),
    /// whose gadget covers the input width. Inputs already under the bootstrap secret
    /// at its degree pass `None`. Any prepared gadget key fits `S`: the fold checks its
    /// layout ([`GGLWEInfos`]) and hands its backend view
    /// ([`GGLWEPreparedToBackendRef`]) to the key switch.
    fn ckks_fold<S>(
        &self,
        folded: &mut [CKKSCiphertextOwned<BE>],
        ins: &[CKKSRingCiphertext<BE, R>],
        inbound: Option<&S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Unfolds `folded`, the ciphertexts of [`Self::ckks_fold`] once bootstrapped, into
    /// `outs`, labeled like the inputs they were folded from: their slot kinds give the
    /// pairs, their scale and sparsity the split. Each output must have a requested
    /// width `k` at least as large as every refreshed ciphertext. `folded` is
    /// overwritten. Invalid layouts and missing or incompatible keys are rejected
    /// before any ciphertext is changed.
    ///
    /// `outbound` switches `folded` back from the bootstrap secret to the input secret:
    /// the prepared [`RingSwitchKeys::outbound`](crate::layouts::RingSwitchKeys::outbound),
    /// whose gadget covers the width of `folded`; `None` when [`Self::ckks_fold`] took
    /// none. `automorphisms`, keys of the input secret for the elements of
    /// [`Self::ckks_unfold_galois_elements`], split real pairs and sparse
    /// inputs, the sparse ones decomposing `log_sparsity` bits beyond the width of
    /// `folded`; `None` when there are neither.
    fn ckks_unfold<S, H>(
        &self,
        outs: &mut [CKKSRingCiphertext<BE, R>],
        folded: &mut [CKKSCiphertextOwned<BE>],
        outbound: Option<&S>,
        automorphisms: Option<&H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GetAutomorphismKey<BE>;
}
