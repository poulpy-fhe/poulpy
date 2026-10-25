use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Module, ModulePlanCache};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, SetCKKSInfos,
    api::CKKSEncodingScalar,
    layouts::{
        CKKSEncodingBufferBackendMut, CKKSEncodingBufferBackendRef, CKKSEncodingBufferToBackendMut,
        CKKSEncodingBufferToBackendRef,
    },
    oep::derived::encoding::{
        ckks_coeffs_to_slots_assign_derived, ckks_decode_coeffs_into_derived, ckks_decode_slots_into_derived,
        ckks_encode_coeffs_into_derived, ckks_encode_slots_assign_into_derived, ckks_slots_to_coeffs_assign_derived,
    },
};

/// Backend extension point for the CKKS encoding operations: four required
/// kernels (the coefficient codecs and the slot transforms under their plans) and
/// the API operations derived from them.
///
/// Every scalar operand is backend-resident. A device implementation can run
/// the permutation, FFT, and quantization kernels directly on arena-carved
/// device memory; host slices appear only in public convenience helpers.
///
/// The canonical circuit is defined by [`crate::reference::encoding`]. An
/// override must produce the same bytes at the selected scalar precision as
/// every other backend: the fused butterflies of the portable negacyclic FFT on
/// the correctly rounded twiddles of
/// [`CKKSFloat::ckks_root_of_unity`](crate::numerics::CKKSFloat::ckks_root_of_unity),
/// with no other fused or reassociated operation, and the conversions of
/// [`CKKSFloat::ckks_quantize`](crate::numerics::CKKSFloat::ckks_quantize) and
/// [`CKKSFloat::ckks_dequantize`](crate::numerics::CKKSFloat::ckks_dequantize).
/// Layouts and scheduling are free. The fixtures of
/// `ckks_determinism_test_suite!` check the result. See the CKKS encoding section of
/// `docs/guides/backends.md`. Plans and staging are backend-owned, and these
/// signatures require no host access.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSEncodingImpl<F: CKKSEncodingScalar>: Backend {
    /// Opaque backend-specific plan family for precision `F`.
    type Plans: Send + Sync + 'static;

    /// The plan cache of `module`, where the plans of [`Self::ckks_encoding_plans_create`] are created once per
    /// precision. Backend-owned state with no reference body.
    fn ckks_encoding_plan_cache(module: &Module<Self>) -> &ModulePlanCache;

    /// Builds the transform-plan family for every supported power-of-two dimension up to `module.max_n()`.
    /// Backend-owned plans with no reference body: what they must compute is fixed by
    /// [`Self::ckks_slots_to_coeffs_assign_with_plan`] and [`Self::ckks_coeffs_to_slots_assign_with_plan`].
    fn ckks_encoding_plans_create(module: &Module<Self>) -> Result<Self::Plans>;

    /// Hook without an API twin. Reproduces
    /// [`encode_coeffs_into_host`](crate::reference::encoding::encode_coeffs_into_host) bit for bit, including
    /// rounding, one value per coefficient; free in kernels and scratch use. The caller,
    /// [`Self::ckks_encode_coeffs_into`], validated the operands. Rejects non-finite inputs and rounded values outside
    /// the codec's signed integer range before writing the plaintext. Preserves the input coefficients and plaintext
    /// metadata, except that invariant modules mark the slots [`SlotsKind::Real`](crate::SlotsKind::Real).
    fn ckks_quantize_coeffs_into<P>(
        module: &Module<Self>,
        pt: &mut P,
        coeffs: &CKKSEncodingBufferBackendRef<'_, Self, F>,
    ) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos;

    /// Hook without an API twin. Reproduces
    /// [`decode_coeffs_into_host`](crate::reference::encoding::decode_coeffs_into_host) bit for bit; free in kernels
    /// and scratch use. The caller, [`Self::ckks_decode_coeffs_into`], validated the operands. Preserves the plaintext
    /// and its metadata.
    fn ckks_dequantize_coeffs_into<P>(
        module: &Module<Self>,
        pt: &P,
        coeffs: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos;

    /// Hook without an API twin. Reproduces
    /// [`EncodingPermutation::slots_to_coeffs_assign`](crate::reference::encoding::EncodingPermutation::slots_to_coeffs_assign)
    /// bit for bit (permutation and IFFT; the determinism fixtures pin the bytes across backends), in place under the
    /// cached `plans`, on a buffer [`Self::ckks_slots_to_coeffs_assign`] validated; free in layouts and scheduling.
    /// Invariant modules discard imaginary slots and store the independent coefficients in the first half of the
    /// buffer.
    fn ckks_slots_to_coeffs_assign_with_plan(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;

    /// Hook without an API twin. Reproduces
    /// [`EncodingPermutation::coeffs_to_slots_assign`](crate::reference::encoding::EncodingPermutation::coeffs_to_slots_assign)
    /// bit for bit (FFT and permutation; the determinism fixtures pin the bytes across backends), in place under the
    /// cached `plans`, on a buffer [`Self::ckks_coeffs_to_slots_assign`] validated; free in layouts and scheduling.
    /// Invariant modules read independent coefficients from the first half and return zero imaginary parts.
    fn ckks_coeffs_to_slots_assign_with_plan(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;

    /// Default body: `ckks_encode_slots_assign_into_derived`. A buffer length that is not a power of two between 2 and
    /// twice the maximum slot count, a plaintext of nonzero rank or above the module degree, or a coefficient count
    /// different from the plaintext degree is a refusal. Then [`Self::ckks_slots_to_coeffs_assign`] on the buffer and
    /// [`Self::ckks_encode_coeffs_into`] of its leading coefficients.
    fn ckks_encode_slots_assign_into<P, C>(module: &Module<Self>, pt: &mut P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_encode_slots_assign_into_derived(module, pt, slots)
    }

    /// Default body: `ckks_decode_slots_into_derived`, the checks of [`Self::ckks_encode_slots_assign_into`], then
    /// [`Self::ckks_decode_coeffs_into`] into the leading coefficients of the buffer and
    /// [`Self::ckks_coeffs_to_slots_assign`] on it.
    fn ckks_decode_slots_into<P, C>(module: &Module<Self>, pt: &P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_decode_slots_into_derived(module, pt, slots)
    }

    /// Default body: `ckks_encode_coeffs_into_derived`. A plaintext of nonzero rank or above the module degree, or a
    /// coefficient count different from the plaintext degree, is a refusal. Then [`Self::ckks_quantize_coeffs_into`].
    fn ckks_encode_coeffs_into<P, C>(module: &Module<Self>, pt: &mut P, coeffs: &C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendRef<Self, F>,
    {
        ckks_encode_coeffs_into_derived(module, pt, coeffs)
    }

    /// Default body: `ckks_decode_coeffs_into_derived`, the checks of [`Self::ckks_encode_coeffs_into`], then
    /// [`Self::ckks_dequantize_coeffs_into`].
    fn ckks_decode_coeffs_into<P, C>(module: &Module<Self>, pt: &P, coeffs: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_decode_coeffs_into_derived(module, pt, coeffs)
    }

    /// Default body: `ckks_slots_to_coeffs_assign_derived`. A buffer length that is not a power of two between 2 and
    /// twice the maximum slot count is a refusal. Then [`Self::ckks_slots_to_coeffs_assign_with_plan`] under the plans
    /// of [`Self::ckks_encoding_plan_cache`], created once by [`Self::ckks_encoding_plans_create`].
    fn ckks_slots_to_coeffs_assign<C>(module: &Module<Self>, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_slots_to_coeffs_assign_derived(module, values)
    }

    /// Default body: `ckks_coeffs_to_slots_assign_derived`, the check of [`Self::ckks_slots_to_coeffs_assign`], then
    /// [`Self::ckks_coeffs_to_slots_assign_with_plan`] under the plans of [`Self::ckks_encoding_plan_cache`], created
    /// once by [`Self::ckks_encoding_plans_create`].
    fn ckks_coeffs_to_slots_assign<C>(module: &Module<Self>, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_coeffs_to_slots_assign_derived(module, values)
    }
}
