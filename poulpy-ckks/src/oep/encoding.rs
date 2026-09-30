use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Module, ModulePlanCache};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef, SetCKKSInfos,
    api::CKKSEncodingScalar,
    layouts::{CKKSEncodingBufferBackendMut, CKKSEncodingBufferBackendRef, CKKSEncodingBufferToBackendMut},
    oep::derived::encoding::{ckks_decode_slots_into, ckks_encode_slots_assign_into},
};

/// Backend extension point for the four primitive CKKS encoding operations
/// and their slot compositions.
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
/// Layouts and scheduling are free. The `encoding_determinism` fixtures of the
/// test suite check the result. See the CKKS encoding section of
/// `docs/backends.md`. Plans and staging are backend-owned, and these
/// signatures require no host access.
///
/// # Safety
/// Implementations must uphold the backend layout, aliasing, and numeric
/// contracts documented by each operation.
pub unsafe trait CKKSEncodingImpl<F: CKKSEncodingScalar>: Backend {
    /// Opaque backend-specific plan family for precision `F`.
    type Plans: Send + Sync + 'static;

    fn ckks_encoding_plan_cache_impl(module: &Module<Self>) -> &ModulePlanCache;

    /// Builds the transform-plan family for every supported power-of-two
    /// dimension up to `module.max_n()`.
    fn ckks_encoding_plans_create_impl(module: &Module<Self>) -> Result<Self::Plans>;

    /// Backend-native coefficient → plaintext mapping, without an IFFT.
    ///
    /// Follows [`encode_coeffs_into_host`](crate::reference::encoding::encode_coeffs_into_host),
    /// including rounding and sparse placement. Rejects non-finite inputs and
    /// rounded values outside the codec's signed integer range before writing
    /// the plaintext. Preserves the input coefficients and plaintext metadata,
    /// except that invariant modules mark the slots [`SlotsKind::Real`](crate::SlotsKind::Real).
    fn ckks_encode_coeffs_into_impl<P>(
        module: &Module<Self>,
        pt: &mut P,
        coeffs: &CKKSEncodingBufferBackendRef<'_, Self, F>,
    ) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos;

    /// Backend-native plaintext → coefficient mapping, without an FFT.
    ///
    /// Follows [`decode_coeffs_into_host`](crate::reference::encoding::decode_coeffs_into_host)
    /// and preserves the plaintext and its metadata.
    fn ckks_decode_coeffs_into_impl<P>(
        module: &Module<Self>,
        pt: &P,
        coeffs: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos;

    /// In-place planar slots → polynomial coefficients (permutation + IFFT).
    /// Invariant modules discard imaginary slots and store the independent
    /// coefficients in the first half of the buffer.
    ///
    /// Uses the ordering and normalization defined by
    /// [`EncodingPermutation`](crate::reference::encoding::EncodingPermutation).
    fn ckks_slots_to_coeffs_assign_impl(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;

    /// In-place polynomial coefficients → planar slots (FFT + permutation).
    /// Invariant modules read independent coefficients from the first half
    /// and return zero imaginary parts.
    ///
    /// Inverts the ordering and normalization defined by
    /// [`EncodingPermutation`](crate::reference::encoding::EncodingPermutation).
    fn ckks_coeffs_to_slots_assign_impl(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;

    /// Destructive planar slots → plaintext: the slot transform, then the
    /// coefficient mapping of the ring's leading coefficients.
    fn ckks_encode_slots_assign_into_impl<P, C>(module: &Module<Self>, pt: &mut P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos + SetCKKSInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_encode_slots_assign_into(module, pt, slots)
    }

    /// Plaintext → planar slots: the coefficient mapping into the ring's
    /// leading coefficients, then the coefficient transform.
    fn ckks_decode_slots_into_impl<P, C>(module: &Module<Self>, pt: &P, slots: &mut C) -> Result<()>
    where
        P: CKKSPlaintextToBackendRef<Self> + IntPolyInfos,
        C: CKKSEncodingBufferToBackendMut<Self, F>,
    {
        ckks_decode_slots_into(module, pt, slots)
    }
}
