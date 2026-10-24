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
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSPaCoCoeffEncodingImpl: Backend<Ring = Standard> {
    /// Backend-native arena bytes required by coefficient embedding and β
    /// packing. This includes any scalar buffers used by native FFT kernels.
    /// [`Self::ckks_paco_coeff_encodings`] refuses a smaller arena with
    /// `CKKSError::Internal` before any work.
    fn ckks_paco_coeff_encodings_tmp_bytes<F>(module: &Module<Self>, context: &PaCoContext<Self, F>) -> Result<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>;

    /// Encodes the exhausted ciphertext's public coefficients as the four
    /// dense beta plaintexts consumed by PaCo blind rotation.
    ///
    /// Implementations must preserve the plan's degree, radix, scale, budget,
    /// full-slot ordering, and backend ownership, and must reject malformed
    /// ciphertext layouts before reading their active limbs. The β packing
    /// must agree numerically (within CKKS precision) with the `2C`-block
    /// coefficient→evaluation packing used by the σ vectors of
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
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand, worker and scratch region they touch.
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
    /// Scratch of the direct drivers after their runtime validation: the branch output the sequential driver
    /// carves, then one branch at its working width with the shifted input, accumulator and temporary it carves,
    /// the coefficient encoding included.
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

    /// Scratch of the encapsulated drivers: the switched input they carve, then the larger of the direct bound and
    /// the dense-to-PaCo key switch.
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

    /// Validates, then evaluates every branch in increasing order and adds it to `output`.
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

    /// Validates, switches `input` to the PaCo secret once, then runs the direct driver.
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

    /// Validates every worker's degree and scratch, then evaluates branch zero on the caller and the
    /// others on `workers`, recombined in increasing branch order.
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

    /// Validates, switches `input` to the PaCo secret once on the caller, then runs the parallel direct driver.
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

    /// Scratch of [`Self::ckks_slot_trace_assign`] on `ct_infos` under keys laid out like
    /// `key_infos`: the selected `glwe_automorphism_tmp_bytes`.
    fn ckks_slot_trace_assign_tmp_bytes<C, K>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        ckks_slot_trace_assign_tmp_bytes_derived(module, ct_infos, key_infos)
    }

    /// Hook without an API twin: `Tr_{a→b}` in place, slot `i` becoming `Σ_j ct[i + j·b]` for
    /// `j ∈ [0, a/b)`, one `glwe_automorphism_add_assign` per level under the keys of the fold
    /// rotations; consumes no budget. Its scratch is [`Self::ckks_slot_trace_assign_tmp_bytes`],
    /// which an override replaces with it.
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

    /// Scratch of [`Self::ckks_slot_product_assign`] on `ct_infos` under rotation keys laid out
    /// like `key_infos` and the tensor key `tsk_infos`: the rotated ciphertext it carves, shaped
    /// like `ct_infos`, plus the larger of the selected `ckks_rotate_tmp_bytes` and
    /// `ckks_mul_tmp_bytes`.
    fn ckks_slot_product_assign_tmp_bytes<C, K, T>(module: &Module<Self>, ct_infos: &C, key_infos: &K, tsk_infos: &T) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        ckks_slot_product_assign_tmp_bytes_derived(module, ct_infos, key_infos, tsk_infos)
    }

    /// Hook without an API twin: `Pr_{a→b}` in place, slot `i` becoming `Π_j ct[i + j·b]` for
    /// `j ∈ [0, a/b)`, one `ckks_rotate_into` and `ckks_mul_assign` per level; consumes
    /// `log(a/b) · log_delta` budget bits. Its scratch is
    /// [`Self::ckks_slot_product_assign_tmp_bytes`], which an override replaces with it.
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
