use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Module, ModulePlanCache};

use crate::{
    CKKSPlaintextToBackendMut, CKKSPlaintextToBackendRef,
    api::CKKSEncodingScalar,
    layouts::{CKKSEncodingBufferBackendMut, CKKSEncodingBufferBackendRef},
};

/// Backend extension point for the four primitive CKKS encoding operations.
///
/// Every scalar operand is backend-resident. A device implementation can run
/// the permutation, FFT, and quantization kernels directly on arena-carved
/// device memory; host slices appear only in public convenience helpers.
///
/// The canonical circuit is defined by [`crate::reference::encoding`]. An
/// override must produce the same encoding at the selected scalar precision and
/// pass paired tests against a caller-selected validated implementation. Plans
/// and staging are backend-owned; these signatures require no host access.
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
    /// the plaintext. Preserves the input coefficients and plaintext metadata.
    fn ckks_encode_coeffs_into_impl<P>(
        module: &Module<Self>,
        pt: &mut P,
        coeffs: &CKKSEncodingBufferBackendRef<'_, Self, F>,
    ) -> Result<()>
    where
        P: CKKSPlaintextToBackendMut<Self> + IntPolyInfos;

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
    ///
    /// Uses the ordering and normalization defined by
    /// [`EncodingPermutation`](crate::reference::encoding::EncodingPermutation).
    fn ckks_slots_to_coeffs_assign_impl(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;

    /// In-place polynomial coefficients → planar slots (FFT + permutation).
    ///
    /// Inverts the ordering and normalization defined by
    /// [`EncodingPermutation`](crate::reference::encoding::EncodingPermutation).
    fn ckks_coeffs_to_slots_assign_impl(
        module: &Module<Self>,
        plans: &Self::Plans,
        values: &mut CKKSEncodingBufferBackendMut<'_, Self, F>,
    ) -> Result<()>;
}
