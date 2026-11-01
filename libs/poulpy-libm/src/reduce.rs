//! Slot reductions.

use anyhow::{Result, ensure};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos, LWEInfos,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSModuleInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSRotateOps},
    layouts::ScratchArenaTakeCKKSCiphertext,
};

use crate::comparison::{CKKSComparisonOps, SignComposite};

/// Full-slot sum and extrema.
///
/// Each reduction folds the slots by rotations of `1, 2, 4, ...` on a scratch accumulator with the layout of `input`,
/// looking up the keys of [`Self::ckks_sum_slots_galois_elements`], then copies the result to `res`.
pub trait CKKSReductionOps<BE: Backend> {
    /// Galois elements of the rotation keys [`Self::ckks_sum_slots`], [`Self::ckks_fmax_slots`],
    /// [`Self::ckks_fmin_slots`] and [`CKKSSoftmaxOps::ckks_softmax`](crate::softmax::CKKSSoftmaxOps::ckks_softmax)
    /// look up for an input with the layout of `input`, in lookup order.
    fn ckks_sum_slots_galois_elements<C>(&self, input: &C) -> Vec<i64>
    where
        C: CKKSCtBounds;

    /// Scratch bytes for [`Self::ckks_sum_slots`].
    fn ckks_sum_slots_tmp_bytes<R, I, A>(&self, res: &R, input: &I, keys: &A) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos;

    /// Sums all dense slots and broadcasts the result.
    ///
    /// # Metadata
    ///
    /// The rotations and sums run at the layout of `input`, then `ckks_copy` writes the result to `res`.
    ///
    /// # Errors
    ///
    /// An error for a sparse input, and the errors of the rotations (a missing key included), the sums and the final
    /// `ckks_copy`. The fold runs in scratch, so `res` is untouched on `Err`.
    fn ckks_sum_slots<R, I, H>(&self, res: &mut R, input: &I, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// Scratch bytes for [`Self::ckks_fmax_slots`].
    fn ckks_fmax_slots_tmp_bytes<R, I, P, T, K>(&self, res: &R, input: &I, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// Reduces `fmax` over all dense slots and broadcasts the result.
    ///
    /// `keys` answers the rotations and the conjugation, Galois element `-1`, of the comparisons.
    ///
    /// # Metadata
    ///
    /// The rotations and comparisons run at the layout of `input`, then `ckks_copy` writes the result to `res`.
    ///
    /// # Errors
    ///
    /// An error for a sparse input, and the errors of the rotations, of
    /// [`CKKSComparisonOps::ckks_fmax`] and of the final `ckks_copy`. The fold runs in scratch, so `res` is untouched on
    /// `Err`.
    fn ckks_fmax_slots<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;

    /// Scratch bytes for [`Self::ckks_fmin_slots`].
    fn ckks_fmin_slots_tmp_bytes<R, I, P, T, K>(&self, res: &R, input: &I, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// Reduces `fmin` over all dense slots and broadcasts the result.
    ///
    /// # Metadata
    ///
    /// As [`Self::ckks_fmax_slots`].
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_fmax_slots`], with [`CKKSComparisonOps::ckks_fmin`]. `res` is untouched on `Err`.
    fn ckks_fmin_slots<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>;
}

impl<BE: Backend<Ring = Standard>> CKKSReductionOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSComparisonOps<BE> + CKKSCopyOps<BE> + CKKSRotateOps<BE> + GLWEBytesOf<BE>,
{
    fn ckks_sum_slots_galois_elements<C>(&self, input: &C) -> Vec<i64>
    where
        C: CKKSCtBounds,
    {
        slot_shifts(self, input)
            .map(|shift| self.ckks_galois_element(shift))
            .collect()
    }

    fn ckks_sum_slots_tmp_bytes<R, I, A>(&self, res: &R, input: &I, keys: &A) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
    {
        // The accumulator and the rotation, each with the layout of `input`.
        2 * BE::scratch_aligned(self.glwe_bytes_of_from_infos(input))
            + self
                .ckks_copy_tmp_bytes(input, input)
                .max(self.ckks_rotate_tmp_bytes(input, keys))
                .max(self.ckks_add_tmp_bytes(input.size()))
                .max(self.ckks_copy_tmp_bytes(res, input))
    }

    fn ckks_sum_slots<R, I, H>(&self, res: &mut R, input: &I, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        ensure_dense("ckks_sum_slots", input)?;
        scratch.scope(|scratch_local| {
            let (mut acc, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
            let (mut rotated, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
            self.ckks_copy(&mut acc, input, &mut scratch_local)?;
            for shift in slot_shifts(self, input) {
                self.ckks_rotate_into(shift, &mut rotated, &acc, keys, &mut scratch_local)?;
                self.ckks_add_assign(&mut acc, &rotated, &mut scratch_local)?;
            }
            self.ckks_copy(res, &acc, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_fmax_slots_tmp_bytes<R, I, P, T, K>(&self, res: &R, input: &I, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        extrema_slots_tmp_bytes(self, res, input, coeff_prec, tsk, keys)
    }

    fn ckks_fmax_slots<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        extrema_slots(self, true, res, input, composite, tsk, keys, scratch)
    }

    fn ckks_fmin_slots_tmp_bytes<R, I, P, T, K>(&self, res: &R, input: &I, coeff_prec: &P, tsk: &T, keys: &K) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        extrema_slots_tmp_bytes(self, res, input, coeff_prec, tsk, keys)
    }

    fn ckks_fmin_slots<R, I, F, P, H, K>(
        &self,
        res: &mut R,
        input: &I,
        composite: &SignComposite<F, P>,
        tsk: &H,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
    {
        extrema_slots(self, false, res, input, composite, tsk, keys, scratch)
    }
}

fn ensure_dense(op: &'static str, input: &impl CKKSCtBounds) -> Result<()> {
    ensure!(input.log_sparsity() == 0, "{op}: sparse packing is not supported");
    Ok(())
}

/// Rotations `1, 2, 4, ...` below the slot count of `input`.
fn slot_shifts<BE: Backend>(module: &Module<BE>, input: &impl CKKSCtBounds) -> impl Iterator<Item = i64> {
    let log_slots = module.ckks_slots_of_degree(input.n().as_usize()).ilog2();
    (0..log_slots).map(|i| 1i64 << i)
}

/// Query of [`extrema_slots`]: the accumulator, the rotation and the comparison, each with the layout of `input`.
fn extrema_slots_tmp_bytes<BE, R, I, P, T, K>(module: &Module<BE>, res: &R, input: &I, coeff_prec: &P, tsk: &T, keys: &K) -> usize
where
    BE: Backend,
    Module<BE>: CKKSComparisonOps<BE> + CKKSCopyOps<BE> + CKKSRotateOps<BE> + GLWEBytesOf<BE>,
    R: CKKSCtBounds,
    I: CKKSCtBounds,
    P: CKKSInfos + LWEInfos,
    T: GGLWEInfos,
    K: GGLWEInfos,
{
    3 * BE::scratch_aligned(module.glwe_bytes_of_from_infos(input))
        + module
            .ckks_copy_tmp_bytes(input, input)
            .max(module.ckks_rotate_tmp_bytes(input, keys))
            .max(module.ckks_comparison_tmp_bytes(input, input, input, coeff_prec, tsk, keys))
            .max(module.ckks_copy_tmp_bytes(res, input))
}

#[allow(clippy::too_many_arguments)]
fn extrema_slots<BE, R, I, F, P, H, K>(
    module: &Module<BE>,
    max: bool,
    res: &mut R,
    input: &I,
    composite: &SignComposite<F, P>,
    tsk: &H,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSComparisonOps<BE> + CKKSCopyOps<BE> + CKKSRotateOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
    H: GetTensorKey<BE>,
    K: GetAutomorphismKey<BE>,
{
    ensure_dense(if max { "ckks_fmax_slots" } else { "ckks_fmin_slots" }, input)?;
    scratch.scope(|scratch_local| {
        let (mut acc, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        let (mut rotated, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        let (mut next, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        module.ckks_copy(&mut acc, input, &mut scratch_local)?;
        for shift in slot_shifts(module, input) {
            module.ckks_rotate_into(shift, &mut rotated, &acc, keys, &mut scratch_local)?;
            if max {
                module.ckks_fmax(&mut next, &acc, &rotated, composite, tsk, keys, &mut scratch_local)?;
            } else {
                module.ckks_fmin(&mut next, &acc, &rotated, composite, tsk, keys, &mut scratch_local)?;
            }
            module.ckks_copy(&mut acc, &next, &mut scratch_local)?;
        }
        module.ckks_copy(res, &acc, &mut scratch_local)?;
        Ok(())
    })
}
