//! PaCo slot-fold composites `Tr_{a→b}` / `Pr_{a→b}` and the conjugate-rotate
//! Galois element.
//!
//! [`Tr_{a→b}`](PaCoSlotOps::ckks_slot_trace_assign) folds slots
//! `i, i+b, i+2b, …, i+a−b` by **addition** (`log(a/b)` fused
//! automorphism-add steps, zero budget consumed).
//! [`Pr_{a→b}`](PaCoSlotOps::ckks_slot_product_assign) folds them by
//! **multiplication** (`log(a/b)` rotate-and-relinearize steps,
//! `log(a/b) · log_delta` budget bits consumed) — this is seqPaCo line 11, the
//! simulation of the decryption sum's modular additions as products on the
//! unit circle, i.e. the EvalMod replacement.
//!
//! The trace fold uses the same fused primitive as the core
//! [`glwe_trace`](poulpy_core::GLWETrace) cascade
//! (`glwe_automorphism_add_assign`, `ct += σ(ct)` with no temporary
//! ciphertext), but is deliberately **not** `glwe_trace` itself: that op
//! always folds to the top of the automorphism tower and halves per level
//! (the normalized subring trace), whereas PaCo's folds need an arbitrary
//! upper bound `a` (its truncated mid-pipeline folds are not subgroup traces
//! at all) and the un-normalized sum.
//!
//! Both are pure compositions of existing `Module` ops (rotate, add, mul), so
//! no backend extension point (`oep`) is involved: the trait has a single
//! blanket implementation, mirroring
//! [`CKKSLinearTransformationOps`](crate::api::CKKSLinearTransformationOps).
//!
//! Conjugation-with-rotation (seqPaCo line 8, Galois element `−5^k`) needs no
//! new operation at all: the automorphism appliers use whatever key they are
//! handed, so an automorphism key generated for the signed Galois element
//! `−5^k` applied through
//! [`CKKSConjugateOps`](crate::api::CKKSConjugateOps) performs
//! `conj(rotate(·, k))` in a single keyswitch.

use crate::CKKSAtkBounds;
use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWEAutomorphism,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, CyclotomicOrder, Module, ScratchArena, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, SetCKKSInfos,
    api::{CKKSMulOps, CKKSRotateOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};

/// The rotation amounts of a `Tr_{a→b}` / `Pr_{a→b}` fold, in application
/// order: `a/2, a/4, …, b` (`log(a/b)` entries; empty when `a == b`). Use with
/// [`galois_elements_from_rotations`](poulpy_hal::layouts::galois_elements_from_rotations)
/// to enumerate the automorphism keys a fold requires.
pub(crate) fn fold_rotations(a: usize, b: usize) -> Vec<i64> {
    (0..(a / b).trailing_zeros()).map(|l| (a >> (l + 1)) as i64).collect()
}

/// Reverses the low `log_p` bits of `x`, keeping the higher bits — the tiled
/// permutation `P` of the `BitRevLow` slot-order convention (the paper's
/// extended bit-reversal `Π^{(C/2)}` for `log_p = log(C/2)`). Self-inverse;
/// the identity for `log_p ∈ {0, 1}`. Because it only touches bits below
/// `log_p`, it commutes with every stride that is a multiple of `2^{log_p}` —
/// the fold rotations, the ψ-pairing offset `C`, and the μ/η mask strata.
pub(crate) fn ext_bitrev_low(x: usize, log_p: usize) -> usize {
    if log_p == 0 {
        return x;
    }
    let mask = (1usize << log_p) - 1;
    let rev = (x & mask).reverse_bits() >> (usize::BITS as usize - log_p);
    (x & !mask) | rev
}

/// Galois element of the fused conjugate-and-rotate-by-`k` automorphism
/// (`X ↦ X^{−5^k}`), in the crate's signed encoding (negative = composed with
/// conjugation). `k = 0` degenerates to plain conjugation (`−1`). The
/// automorphism appliers use whatever key they are handed, so a key generated
/// for this element applied through
/// [`CKKSConjugateOps`](crate::api::CKKSConjugateOps) performs
/// `conj(rotate(·, k))` in a single keyswitch — the ψ-pairing's fast tail.
pub(crate) fn conj_rotate_galois_element(k: i64, cyclotomic_order: i64) -> i64 {
    -galois_element(k, cyclotomic_order)
}

/// The PaCo slot-fold composites. Blanket-implemented on `Module<BE>`.
pub trait PaCoSlotOps<BE: Backend> {
    /// `Tr_{a→b}` in place: slot `i` becomes `Σ_j ct[i + j·b]` for
    /// `j ∈ [0, a/b)` (indices mod `N/2`). One fused `ct += σ(ct)` keyswitch
    /// per level; consumes no budget. `keys` must contain the automorphism
    /// keys for `fold_rotations``(a, b)`.
    fn ckks_slot_trace_assign<Dst, H, K>(
        &self,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `Pr_{a→b}` in place: slot `i` becomes `Π_j ct[i + j·b]` for
    /// `j ∈ [0, a/b)` (indices mod `N/2`). Rotate-and-multiply with
    /// relinearization via `tsk`; consumes `log(a/b) · log_delta` budget bits.
    fn ckks_slot_product_assign<Dst, H, K, T>(
        &self,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}

impl<BE: Backend> PaCoSlotOps<BE> for Module<BE>
where
    Module<BE>: CKKSRotateOps<BE> + CKKSMulOps<BE> + CKKSModuleAlloc<BE> + GLWEAutomorphism<BE> + CyclotomicOrder + ModuleN,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    fn ckks_slot_trace_assign<Dst, H, K>(
        &self,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let order = self.cyclotomic_order();
        for rot in checked_fold_rotations(self, a, b)? {
            let key =
                keys.get_automorphism_key(galois_element(rot, order))
                    .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                        op: "paco_slot_trace",
                        rotation: rot,
                    })?;
            self.glwe_automorphism_add_assign(ct, key, scratch);
        }
        Ok(())
    }

    fn ckks_slot_product_assign<Dst, H, K, T>(
        &self,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let rotations = checked_fold_rotations(self, a, b)?;
        let mut tmp = self.ckks_ciphertext_alloc_from_infos(ct);
        for k in rotations {
            self.ckks_rotate_into(&mut tmp, ct, k, keys, scratch)?;
            self.ckks_mul_assign(ct, &tmp, tsk, scratch)?;
        }
        Ok(())
    }
}

/// Validates the fold bounds against the module's slot count and returns the
/// rotation schedule.
fn checked_fold_rotations<M: ModuleN>(module: &M, a: usize, b: usize) -> Result<Vec<i64>> {
    ckks_ensure!(
        a.is_power_of_two() && b.is_power_of_two() && b <= a,
        "invalid PaCo fold ({a} → {b})",
    );
    ckks_ensure!(
        a <= module.n() / 2,
        "fold period a = {a} exceeds the slot count {}",
        module.n() / 2
    );
    Ok(fold_rotations(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_rotation_schedules() {
        assert_eq!(fold_rotations(64, 8), vec![32, 16, 8]);
        assert_eq!(fold_rotations(128, 64), vec![64]);
        assert_eq!(fold_rotations(16, 16), Vec::<i64>::new());
    }
}
