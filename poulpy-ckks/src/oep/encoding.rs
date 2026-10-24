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
/// `docs/backends.md`. Plans and staging are backend-owned, and these
/// signatures require no host access.
///
/// # Safety
/// Implementations must uphold the backend layout, aliasing, and numeric
/// contracts documented by each operation.
pub unsafe trait CKKSEncodingImpl<F: CKKSEncodingScalar>: Backend {
    /// Opaque backend-specific plan family for precision `F`.
    type Plans: Send + Sync + 'static;

    fn ckks_encoding_plan_cache(module: &Module<Self>) -> &ModulePlanCache;

    /// Builds the transform-plan family for every supported power-of-two
    /// dimension up to `module.max_n()`.
    fn ckks_encoding_plans_create(module: &Module<Self>) -> Result<Self::Plans>;

    /// Hook without an API twin: the backend-native coefficient → plaintext mapping,
    /// without an IFFT, on operands [`Self::ckks_encode_coeffs_into`] validated.
    ///
    /// Follows [`encode_coeffs_into_host`](crate::reference::encoding::encode_coeffs_into_host),
    /// including rounding, one value per coefficient. Rejects non-finite inputs and
    /// rounded values outside the codec's signed integer range before writing
    /// the plaintext. Preserves the input coefficients and plaintext metadata,
    /// except that invariant modules mark the slots [`SlotsKind::Real`](crate::SlotsKind::Real).
    fn ckks_quantize_coeffs_into<P>(
        module: &Module<Self>,
        pt: &mut P,
        coeffs: &CKKSEncodingBufferBackendRef<'_, Self, F>,
    ) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos;

    /// Hook without an API twin: the backend-native plaintext → coefficient mapping,
    /// without an FFT, on operands [`Self::ckks_decode_coeffs_into`] validated.
    ///
    /// Follows [`decode_coeffs_into_host`](crate::reference::encoding::decode_coeffs_into_host)
    /// and preserves the plaintext and its metadata.
    fn ckks_dequantize_coeffs_into<P>(
        module: &Module<Self>,
        pt: &P,
        coeffs: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos;

    /// Hook without an API twin: in-place planar slots → polynomial coefficients
    /// (permutation + IFFT) under the cached `plans`, on a buffer
    /// [`Self::ckks_slots_to_coeffs_assign`] validated.
    /// Invariant modules discard imaginary slots and store the independent
    /// coefficients in the first half of the buffer.
    ///
    /// Uses the ordering and normalization defined by
    /// [`EncodingPermutation`](crate::reference::encoding::EncodingPermutation).
    fn ckks_slots_to_coeffs_assign_with_plan(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;

    /// Hook without an API twin: in-place polynomial coefficients → planar slots
    /// (FFT + permutation) under the cached `plans`, on a buffer
    /// [`Self::ckks_coeffs_to_slots_assign`] validated.
    /// Invariant modules read independent coefficients from the first half
    /// and return zero imaginary parts.
    ///
    /// Inverts the ordering and normalization defined by
    /// [`EncodingPermutation`](crate::reference::encoding::EncodingPermutation).
    fn ckks_coeffs_to_slots_assign_with_plan(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;

    /// Checks the buffer and plaintext shapes, then [`Self::ckks_slots_to_coeffs_assign`]
    /// and [`Self::ckks_encode_coeffs_into`] of the ring's leading coefficients.
    fn ckks_encode_slots_assign_into<P, C>(module: &Module<Self>, pt: &mut P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_encode_slots_assign_into_derived(module, pt, slots)
    }

    /// Checks the buffer and plaintext shapes, then [`Self::ckks_decode_coeffs_into`]
    /// into the ring's leading coefficients and [`Self::ckks_coeffs_to_slots_assign`].
    fn ckks_decode_slots_into<P, C>(module: &Module<Self>, pt: &P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_decode_slots_into_derived(module, pt, slots)
    }

    /// Checks the plaintext rank, degree and coefficient count, then [`Self::ckks_quantize_coeffs_into`].
    fn ckks_encode_coeffs_into<P, C>(module: &Module<Self>, pt: &mut P, coeffs: &C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendRef<Self, F>,
    {
        ckks_encode_coeffs_into_derived(module, pt, coeffs)
    }

    /// Checks the plaintext rank, degree and coefficient count, then [`Self::ckks_dequantize_coeffs_into`].
    fn ckks_decode_coeffs_into<P, C>(module: &Module<Self>, pt: &P, coeffs: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_decode_coeffs_into_derived(module, pt, coeffs)
    }

    /// Checks the buffer length, then [`Self::ckks_slots_to_coeffs_assign_with_plan`] under the
    /// plans of [`Self::ckks_encoding_plan_cache`], created once by [`Self::ckks_encoding_plans_create`].
    fn ckks_slots_to_coeffs_assign<C>(module: &Module<Self>, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_slots_to_coeffs_assign_derived(module, values)
    }

    /// Checks the buffer length, then [`Self::ckks_coeffs_to_slots_assign_with_plan`] under the
    /// plans of [`Self::ckks_encoding_plan_cache`], created once by [`Self::ckks_encoding_plans_create`].
    fn ckks_coeffs_to_slots_assign<C>(module: &Module<Self>, values: &mut C) -> Result<()>
    where
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_coeffs_to_slots_assign_derived(module, values)
    }
}
