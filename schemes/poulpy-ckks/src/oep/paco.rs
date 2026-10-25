use crate::CKKSResult as Result;
use crate::layouts::CKKSPlaintextOwned;
use poulpy_core::{
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey},
    oep::{AutomorphismImpl, GLWEKeyswitchImpl, GLWERotateImpl, LinearTransformationImpl},
};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, Module, ScratchArena, ScratchOwned, Standard},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::PaCoScalar,
    layouts::{CKKSCiphertextOwned, PaCoContext, PaCoKeys, PaCoWorker},
    oep::{
        CKKSAddImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSEncodingImpl, CKKSLinearTransformationImpl, CKKSMulImpl,
        CKKSRotateImpl, CKKSSubImpl,
        derived::paco::{
            ckks_paco_bootstrap_direct_into_derived, ckks_paco_bootstrap_direct_tmp_bytes_derived,
            ckks_paco_bootstrap_into_derived, ckks_paco_bootstrap_parallel_direct_into_derived,
            ckks_paco_bootstrap_parallel_into_derived, ckks_paco_bootstrap_tmp_bytes_derived, ckks_slot_product_assign_derived,
            ckks_slot_product_assign_tmp_bytes_derived, ckks_slot_trace_assign_derived, ckks_slot_trace_assign_tmp_bytes_derived,
        },
    },
};

/// Backend extension point for the PaCo coefficient encoding: the ciphertext
/// → four-β-plaintexts transformation consumed by PaCo's blind rotation.
///
/// Implementing this trait is the entry requirement for the PaCo
/// bootstrapping operations
/// ([`CKKSPaCoOps`](crate::api::CKKSPaCoOps)): a backend without a ready
/// coefficient-encoding implementation cannot bootstrap. The trait imposes no
/// FFT engine, encoder, host codec, or transfer capability: the encoding
/// method sees only the ciphertext, the compiled context (its validated plan
/// and radix), and the backend's own precomputed material, so a backend may
/// implement the whole step as one fused native kernel. This crate carries no implementation; the
/// reference implementation is [`crate::reference::encoding::paco_coeff_encodings_host`], and the complete
/// scheme definition of the step (host ciphertext → four host plaintexts) is
/// exposed as
/// [`paco_coeff_encodings_host`](crate::encoding::paco_coeff_encodings_host)
/// so host reference implementations derive from the same math.
///
/// # Safety
/// Implementations meet the contract their methods document, a numeric one with no bit-exact body, report for the
/// `_tmp_bytes` query a bound that covers the body, and refuse as documented. The memory, layout, synchrony and
/// aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSPaCoCoeffEncodingImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::ckks_paco_coeff_encodings`], the scalar buffers of native FFT kernels included. That
    /// method refuses a smaller arena with `CKKSError::Internal` before any work.
    fn ckks_paco_coeff_encodings_tmp_bytes<F>(module: &Module<Self>, context: &PaCoContext<Self, F>) -> Result<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>;

    /// No reference body is wired: this crate carries no implementation, and the host scheme definition
    /// [`paco_coeff_encodings_host`](crate::reference::encoding::paco_coeff_encodings_host) is the math, agreed with
    /// within CKKS precision, not bit for bit. Encodes the exhausted ciphertext's public coefficients as the four dense
    /// beta plaintexts consumed by PaCo blind rotation, within [`Self::ckks_paco_coeff_encodings_tmp_bytes`]. Preserves
    /// the plan's degree, radix, scale, budget, full-slot ordering and backend ownership, and rejects malformed
    /// ciphertext layouts before reading their active limbs. The beta packing agrees numerically with the `2C`-block
    /// coefficient-to-evaluation packing used by the sigma vectors of
    /// [`PaCoSecretSpec`](crate::layouts::PaCoSecretSpec) at key generation.
    fn ckks_paco_coeff_encodings<F, Src>(
        module: &Module<Self>,
        ciphertext: &Src,
        context: &PaCoContext<Self, F>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<[CKKSPlaintextOwned<Self>; 4]>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Backend contract of the PaCo bootstrapping operations
/// ([`CKKSPaCoOps`](crate::api::CKKSPaCoOps)) beyond the coefficient encoding, on the standard ring.
///
/// Every method defaults to the PaCo drivers over the selected CKKS operations, the
/// core rotation, automorphism, linear-transformation and key-switch families, and
/// [`CKKSPaCoCoeffEncodingImpl`]; an override reproduces them bit for bit within its
/// own query. The scalar `F` is a method parameter bounded by `Self: CKKSEncodingImpl<F>`.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSPaCoImpl:
    Backend<Ring = Standard>
    + CKKSPaCoCoeffEncodingImpl
    + CKKSMulImpl
    + CKKSAddImpl
    + CKKSSubImpl
    + CKKSConjugateImpl
    + CKKSCopyImpl
    + CKKSRotateImpl
    + CKKSLinearTransformationImpl
    + GLWERotateImpl
    + AutomorphismImpl
    + LinearTransformationImpl
    + GLWEKeyswitchImpl
{
    /// Default body: `ckks_paco_bootstrap_direct_tmp_bytes_derived`, the runtime validation of the direct drivers
    /// (refusing as they do), then the branch output the sequential driver carves plus one branch at its working width,
    /// with the shifted input, accumulator and temporary it carves and the largest stage, the coefficient encoding
    /// included. Scratch bound of [`Self::ckks_paco_bootstrap_direct_into`]; may report less than the reference, never
    /// less than the body takes. Every parallel worker arena must provide at least the branch part.
    fn ckks_paco_bootstrap_direct_tmp_bytes<F, K, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
    ) -> Result<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_paco_bootstrap_direct_tmp_bytes_derived(module, output, input, context, keys)
    }

    /// Default body: `ckks_paco_bootstrap_tmp_bytes_derived`, the runtime validation of the encapsulated drivers
    /// (refusing without an encapsulation key, as they do), then the switched input they carve plus the larger of the
    /// direct bound and the selected `glwe_keyswitch_tmp_bytes` of the dense-to-PaCo switch. Scratch bound of
    /// [`Self::ckks_paco_bootstrap_into`]; may report less than the reference, never less than the body takes. Parallel
    /// worker arenas need only the direct bound.
    fn ckks_paco_bootstrap_tmp_bytes<F, K, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
    ) -> Result<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_paco_bootstrap_tmp_bytes_derived(module, output, input, context, keys)
    }

    /// Default body: `ckks_paco_bootstrap_direct_into_derived`. The runtime validation (the plan, keys, layouts, the
    /// branch schedule of the input's sparsity, and an arena holding the scratch the call needs) refuses before any
    /// write. Then branch 0 into `output`, and each later branch, in increasing order, into one branch output carved
    /// once and added to `output` with `ckks_add_assign`, the recombined sparsity recorded last. A branch is
    /// [`CKKSPaCoCoeffEncodingImpl::ckks_paco_coeff_encodings`] of the input, the blind rotation
    /// (`ckks_mul_pt_vec_into` of the first bootstrapping ciphertext, `ckks_mul_pt_vec_into` and `ckks_add_assign` of
    /// the other three), [`Self::ckks_slot_trace_assign`], each CoeffsToSlots factor by
    /// `ckks_eval_linear_transformation_self_assign`, the psi tail (conjugation and two transforms added, or
    /// conjugate-rotate added and masked with `ckks_mul_pt_vec_assign`), [`Self::ckks_slot_product_assign`], the
    /// imaginary-part extraction (`ckks_conjugate_into`, `ckks_sub_assign`), each SlotsToCoeffs factor, the scale
    /// relabel, and `ckks_copy` into the destination. A shifted branch runs on a `ckks_copy` of the input rotated by
    /// `glwe_rotate_assign`, and its result is rotated back.
    fn ckks_paco_bootstrap_direct_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_paco_bootstrap_direct_into_derived(module, output, input, context, keys, scratch)
    }

    /// Default body: `ckks_paco_bootstrap_into_derived`. The runtime validation (the plan, keys, layouts, the branch
    /// schedule of the input's sparsity, and an arena holding the scratch the call needs) refuses before any write.
    /// Then the selected `glwe_keyswitch` of `input` into a carved ciphertext under the encapsulation key, once,
    /// carrying the input's metadata, and [`Self::ckks_paco_bootstrap_direct_into`]'s schedule on it.
    fn ckks_paco_bootstrap_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        ckks_paco_bootstrap_into_derived(module, output, input, context, keys, scratch)
    }

    /// Default body: `ckks_paco_bootstrap_parallel_direct_into_derived`. The runtime validation (the plan, keys,
    /// layouts, the branch schedule of the input's sparsity, and an arena holding the scratch the call needs) refuses
    /// before any write. Every worker must also have the module degree of the plan and the scratch of one branch,
    /// refusing before any write. Branch 0 runs on the caller and the others on `workers`, in round-robin over bounded
    /// one-slot channels, and each result is added to `output` with `ckks_add_assign` in increasing branch order, the
    /// recombined sparsity recorded last; without a background worker it is [`Self::ckks_paco_bootstrap_direct_into`]'s
    /// schedule. A branch is [`CKKSPaCoCoeffEncodingImpl::ckks_paco_coeff_encodings`] of the input, the blind rotation
    /// (`ckks_mul_pt_vec_into` of the first bootstrapping ciphertext, `ckks_mul_pt_vec_into` and `ckks_add_assign` of
    /// the other three), [`Self::ckks_slot_trace_assign`], each CoeffsToSlots factor by
    /// `ckks_eval_linear_transformation_self_assign`, the psi tail (conjugation and two transforms added, or
    /// conjugate-rotate added and masked with `ckks_mul_pt_vec_assign`), [`Self::ckks_slot_product_assign`], the
    /// imaginary-part extraction (`ckks_conjugate_into`, `ckks_sub_assign`), each SlotsToCoeffs factor, the scale
    /// relabel, and `ckks_copy` into the destination. A shifted branch runs on a `ckks_copy` of the input rotated by
    /// `glwe_rotate_assign`, and its result is rotated back.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_direct_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        workers: &mut [PaCoWorker<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self> + Sync,
        ScratchOwned<Self>: ScratchOwnedBorrow<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
    {
        ckks_paco_bootstrap_parallel_direct_into_derived(module, output, input, context, keys, workers, scratch)
    }

    /// Default body: `ckks_paco_bootstrap_parallel_into_derived`. The runtime validation (the plan, keys, layouts, the
    /// branch schedule of the input's sparsity, and an arena holding the scratch the call needs) refuses before any
    /// write. Then the selected `glwe_keyswitch` of `input` once on the caller into a carved ciphertext under the
    /// encapsulation key, carrying the input's metadata, and [`Self::ckks_paco_bootstrap_parallel_direct_into`]'s
    /// schedule on it.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caller-allocated parallel API explicitly carries output, input, context, keys, workers, and both scratch/schedule controls"
    )]
    fn ckks_paco_bootstrap_parallel_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        workers: &mut [PaCoWorker<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self> + Sync,
        ScratchOwned<Self>: ScratchOwnedBorrow<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
    {
        ckks_paco_bootstrap_parallel_into_derived(module, output, input, context, keys, workers, scratch)
    }

    /// Scratch bound of [`Self::ckks_slot_trace_assign`] under keys laid out like `key_infos`; may report less than the
    /// reference, never less than the body takes. Default body: `ckks_slot_trace_assign_tmp_bytes_derived`, the
    /// selected `glwe_automorphism_tmp_bytes` of `ct_infos` onto itself under `key_infos`.
    fn ckks_slot_trace_assign_tmp_bytes<C, K>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_slot_trace_assign_tmp_bytes_derived(module, ct_infos, key_infos)
    }

    /// Hook without an API twin. Default body: `ckks_slot_trace_assign_derived`. A fold that is not a pair of powers of
    /// two with `b <= a` at most the slot count is a refusal. Then, for each rotation of the fold schedule, the
    /// automorphism key of its Galois element at `ct.k()` (a missing key is `MissingAutomorphismKey`) and
    /// `glwe_automorphism_add_assign`. This is `Tr_{a→b}` in place, slot `i` becoming `Σ_j ct[i + j·b]` for `j ∈ [0,
    /// a/b)`, consuming no budget. The caller records metadata.
    fn ckks_slot_trace_assign<Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        ckks_slot_trace_assign_derived(module, ct, a, b, keys, scratch)
    }

    /// Default body: `ckks_slot_product_assign_tmp_bytes_derived`, the bytes of a rotated ciphertext shaped like
    /// `ct_infos` plus the larger of the selected `ckks_rotate_tmp_bytes` and `ckks_mul_tmp_bytes` (all three operands
    /// at `ct_infos`, under `tsk_infos`). Scratch bound of [`Self::ckks_slot_product_assign`] under rotation keys laid
    /// out like `key_infos`; may report less than the reference, never less than the body takes.
    fn ckks_slot_product_assign_tmp_bytes<C, K, T>(module: &Module<Self>, ct_infos: &C, key_infos: &K, tsk_infos: &T) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        ckks_slot_product_assign_tmp_bytes_derived(module, ct_infos, key_infos, tsk_infos)
    }

    /// Hook without an API twin. Default body: `ckks_slot_product_assign_derived`. A fold that is not a pair of powers
    /// of two with `b <= a` at most the slot count is a refusal, and `a == b` leaves `ct` unchanged. Then the
    /// automorphism key of the first rotation and the tensor key at `ct.k()` (`MissingAutomorphismKey`,
    /// `MissingRelinearizationKey`) and an arena of [`Self::ckks_slot_product_assign_tmp_bytes`] at their layouts
    /// are checked before any write, and, for each rotation `k` of the fold schedule,
    /// `ckks_rotate_into` of `ct` into a `ct`-shaped scratch temporary and `ckks_mul_assign` of the temporary into `ct`
    /// under `tsk`. This is `Pr_{a→b}` in place, slot `i` becoming `Π_j ct[i + j·b]` for `j ∈ [0, a/b)`, consuming
    /// `log(a/b) · log_delta` budget bits. The metadata is the one its constituents record.
    fn ckks_slot_product_assign<Dst, H, TH>(
        module: &Module<Self>,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        tsk: &TH,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
        TH: GetTensorKey<Self>,
    {
        ckks_slot_product_assign_derived(module, ct, a, b, keys, tsk, scratch)
    }
}

/// Selects the derived PaCo drivers of `CKKSPaCoImpl` for the standard backend `$be`.
#[macro_export]
macro_rules! impl_ckks_paco_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSPaCoImpl for $be {}
    };
}

pub use crate::impl_ckks_paco_derived_full;
