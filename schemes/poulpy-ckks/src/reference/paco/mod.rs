//! PaCo bootstrapping: the sequential and parallel branch schedules over the selected CKKS and core operations, their
//! preflight and scratch sizing, and the slot folds.

use std::borrow::Borrow;

use crate::CKKSResult as Result;
use poulpy_core::{
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey},
    oep::{AutomorphismImpl, GLWEKeyswitchImpl, GLWERotateImpl},
};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::PaCoScalar,
    layouts::{CKKSCiphertextOwned, PaCoContext, PaCoKeys, PaCoWorker},
    oep::{
        CKKSAddImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSEncodingImpl, CKKSLinearTransformationImpl, CKKSMulImpl,
        CKKSPaCoCoeffEncodingImpl, CKKSPaCoImpl, CKKSRotateImpl, CKKSSubImpl,
    },
};

mod bootstrap;
mod parallel;
mod preflight;
mod slot;

/// The families the PaCo drivers and their queries compose.
pub(super) trait PaCoBackend:
    CKKSPaCoImpl
    + CKKSPaCoCoeffEncodingImpl
    + CKKSMulImpl
    + CKKSAddImpl
    + CKKSSubImpl
    + CKKSConjugateImpl
    + CKKSCopyImpl
    + CKKSLinearTransformationImpl
    + GLWERotateImpl
    + GLWEKeyswitchImpl
{
}

impl<BE> PaCoBackend for BE where
    BE: CKKSPaCoImpl
        + CKKSPaCoCoeffEncodingImpl
        + CKKSMulImpl
        + CKKSAddImpl
        + CKKSSubImpl
        + CKKSConjugateImpl
        + CKKSCopyImpl
        + CKKSLinearTransformationImpl
        + GLWERotateImpl
        + GLWEKeyswitchImpl
{
}

/// Reference bodies of [`CKKSPaCoImpl`], forwarded by `impl_ckks_paco_reference_full!`. The drivers reach the slot
/// folds and their queries through the selected [`CKKSPaCoImpl`] hooks and the coefficient encoding through the selected
/// [`CKKSPaCoCoeffEncodingImpl`].
pub trait CKKSPaCoReference<BE: Backend>: Borrow<Module<BE>> {
    /// The runtime validation of the direct drivers, refusing as they do, then the branch output the sequential
    /// driver carves plus one branch at its working width, with the shifted input, accumulator and temporary it
    /// carves and the largest stage, the coefficient encoding included. Every parallel worker arena must provide at
    /// least the branch part.
    fn ckks_paco_bootstrap_direct_tmp_bytes_reference<F, K, Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
    ) -> Result<usize>
    where
        BE: CKKSPaCoImpl
            + CKKSPaCoCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSConjugateImpl
            + CKKSCopyImpl
            + CKKSLinearTransformationImpl
            + GLWERotateImpl
            + GLWEKeyswitchImpl,
        F: PaCoScalar,
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        preflight::ckks_paco_bootstrap_direct_tmp_bytes(self.borrow(), output, input, context, keys)
    }

    /// The runtime validation of the encapsulated drivers, refusing without an encapsulation key as they do, then the
    /// switched input they carve plus the larger of the direct bound and poulpy-core's `glwe_keyswitch_tmp_bytes` of
    /// the dense-to-PaCo switch. Parallel worker arenas need only the direct bound.
    fn ckks_paco_bootstrap_tmp_bytes_reference<F, K, Src>(
        &self,
        output: &CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
    ) -> Result<usize>
    where
        BE: CKKSPaCoImpl
            + CKKSPaCoCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSConjugateImpl
            + CKKSCopyImpl
            + CKKSLinearTransformationImpl
            + GLWERotateImpl
            + GLWEKeyswitchImpl,
        F: PaCoScalar,
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        preflight::ckks_paco_bootstrap_tmp_bytes(self.borrow(), output, input, context, keys)
    }

    /// The runtime validation (the plan, keys, layouts, the branch schedule of the input's sparsity, and an arena
    /// holding the scratch the call needs) refuses before any write. Then branch 0 into `output`, and each later
    /// branch, in increasing order, into one branch output carved once and added to `output` with `ckks_add_assign`,
    /// the recombined sparsity recorded last. A branch is the selected `ckks_paco_coeff_encodings` of the input, the
    /// blind rotation (`ckks_mul_pt_vec_into` of the first bootstrapping ciphertext, `ckks_mul_pt_vec_into` and
    /// `ckks_add_assign` of the other three), the selected `ckks_slot_trace_assign`, each CoeffsToSlots factor by
    /// `ckks_eval_linear_transformation_self_assign`, the psi tail (conjugation and two transforms added, or
    /// conjugate-rotate added and masked with `ckks_mul_pt_vec_assign`), the selected `ckks_slot_product_assign`, the
    /// imaginary-part extraction (`ckks_conjugate_into`, `ckks_sub_assign`), each SlotsToCoeffs factor, the scale
    /// relabel, and `ckks_copy` into the destination. A shifted branch runs on a `ckks_copy` of the input rotated by
    /// poulpy-core's `glwe_rotate_assign`, and its result is rotated back.
    fn ckks_paco_bootstrap_direct_into_reference<F, K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSPaCoImpl
            + CKKSPaCoCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSConjugateImpl
            + CKKSCopyImpl
            + CKKSLinearTransformationImpl
            + GLWERotateImpl
            + GLWEKeyswitchImpl,
        F: PaCoScalar,
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        parallel::ckks_paco_bootstrap_direct_into(self.borrow(), output, input, context, keys, scratch)
    }

    /// The runtime validation (the plan, keys, layouts, the branch schedule of the input's sparsity, and an arena
    /// holding the scratch the call needs) refuses before any write. Then poulpy-core's `glwe_keyswitch` of `input`
    /// into a carved ciphertext under the encapsulation key, once, carrying the input's metadata, and
    /// [`Self::ckks_paco_bootstrap_direct_into_reference`]'s schedule on it.
    fn ckks_paco_bootstrap_into_reference<F, K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSPaCoImpl
            + CKKSPaCoCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSConjugateImpl
            + CKKSCopyImpl
            + CKKSLinearTransformationImpl
            + GLWERotateImpl
            + GLWEKeyswitchImpl,
        F: PaCoScalar,
        K: PaCoKeys<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        parallel::ckks_paco_bootstrap_into(self.borrow(), output, input, context, keys, scratch)
    }

    /// The runtime validation (the plan, keys, layouts, the branch schedule of the input's sparsity, and an arena
    /// holding the scratch the call needs) refuses before any write. Every worker must also have the module degree of
    /// the plan and the scratch of one branch, refusing before any write. Branch 0 runs on the caller and the others
    /// on `workers`, in round-robin over bounded one-slot channels, and each result is added to `output` with
    /// `ckks_add_assign` in increasing branch order, the recombined sparsity recorded last; without a background
    /// worker it is [`Self::ckks_paco_bootstrap_direct_into_reference`]'s schedule, whose branch it runs.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_direct_into_reference<F, K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        workers: &mut [PaCoWorker<BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSPaCoImpl
            + CKKSPaCoCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSConjugateImpl
            + CKKSCopyImpl
            + CKKSLinearTransformationImpl
            + GLWERotateImpl
            + GLWEKeyswitchImpl,
        F: PaCoScalar,
        K: PaCoKeys<BE> + Sync,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    {
        parallel::ckks_paco_bootstrap_parallel_direct_into(self.borrow(), output, input, context, keys, workers, scratch)
    }

    /// The runtime validation (the plan, keys, layouts, the branch schedule of the input's sparsity, and an arena
    /// holding the scratch the call needs) refuses before any write. Then poulpy-core's `glwe_keyswitch` of `input`
    /// once on the caller into a carved ciphertext under the encapsulation key, carrying the input's metadata, and
    /// [`Self::ckks_paco_bootstrap_parallel_direct_into_reference`]'s schedule on it.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_into_reference<F, K, Src>(
        &self,
        output: &mut CKKSCiphertextOwned<BE>,
        input: &Src,
        context: &PaCoContext<BE, F>,
        keys: &K,
        workers: &mut [PaCoWorker<BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSPaCoImpl
            + CKKSPaCoCoeffEncodingImpl
            + CKKSEncodingImpl<F>
            + CKKSMulImpl
            + CKKSAddImpl
            + CKKSSubImpl
            + CKKSConjugateImpl
            + CKKSCopyImpl
            + CKKSLinearTransformationImpl
            + GLWERotateImpl
            + GLWEKeyswitchImpl,
        F: PaCoScalar,
        K: PaCoKeys<BE> + Sync,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        parallel::ckks_paco_bootstrap_parallel_into(self.borrow(), output, input, context, keys, workers, scratch)
    }

    /// poulpy-core's `glwe_automorphism_tmp_bytes` of `ct_infos` onto itself under `key_infos`.
    fn ckks_slot_trace_assign_tmp_bytes_reference<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        BE: AutomorphismImpl,
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        slot::ckks_slot_trace_assign_tmp_bytes(self.borrow(), ct_infos, key_infos)
    }

    /// A fold that is not a pair of powers of two with `b <= a` at most the slot count is `InvalidOperand`. Then the
    /// automorphism key of every rotation of the fold schedule at `ct.k()` (`MissingAutomorphismKey`) and an arena of
    /// [`Self::ckks_slot_trace_assign_tmp_bytes_reference`] at the widest key layout are checked before any write, and,
    /// for each rotation, poulpy-core's `glwe_automorphism_add_assign`. This is `Tr_{a→b}` in place, slot `i` becoming
    /// `Σ_j ct[i + j·b]` for `j ∈ [0, a/b)`, consuming no budget. The caller records metadata.
    fn ckks_slot_trace_assign_reference<Dst, H>(
        &self,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: AutomorphismImpl,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        slot::ckks_slot_trace_assign(self.borrow(), ct, a, b, keys, scratch)
    }

    /// The bytes of a rotated ciphertext shaped like `ct_infos` plus the larger of the selected `ckks_rotate_tmp_bytes`
    /// and `ckks_mul_tmp_bytes` (all three operands at `ct_infos`, under `tsk_infos`).
    fn ckks_slot_product_assign_tmp_bytes_reference<C, K, T>(&self, ct_infos: &C, key_infos: &K, tsk_infos: &T) -> usize
    where
        BE: CKKSRotateImpl + CKKSMulImpl,
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        slot::ckks_slot_product_assign_tmp_bytes(self.borrow(), ct_infos, key_infos, tsk_infos)
    }

    /// A fold that is not a pair of powers of two with `b <= a` at most the slot count is `InvalidOperand`, and
    /// `a == b` leaves `ct` unchanged. Then, for every rotation of the fold schedule, its automorphism key and the
    /// tensor key at the width the products before it leave `ct` at (`MissingAutomorphismKey`,
    /// `MissingRelinearizationKey`), the budget of every product (`MultiplicationPrecisionUnderflow`) and an arena of
    /// [`Self::ckks_slot_product_assign_tmp_bytes_reference`] at the widest key layouts are checked before any write.
    /// Then, for each rotation `k`, the selected `ckks_rotate_into` of `ct` into a `ct`-shaped scratch temporary and
    /// `ckks_mul_assign` of the temporary into `ct` under `tsk`. This is `Pr_{a→b}` in place, slot `i` becoming
    /// `Π_j ct[i + j·b]` for `j ∈ [0, a/b)`, consuming `log(a/b) · log_delta` budget bits. The metadata is the one its
    /// constituents record.
    fn ckks_slot_product_assign_reference<Dst, H, TH>(
        &self,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        tsk: &TH,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSRotateImpl + CKKSMulImpl,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
        TH: GetTensorKey<BE>,
    {
        slot::ckks_slot_product_assign(self.borrow(), ct, a, b, keys, tsk, scratch)
    }
}

impl<BE: Backend> CKKSPaCoReference<BE> for Module<BE> {}
