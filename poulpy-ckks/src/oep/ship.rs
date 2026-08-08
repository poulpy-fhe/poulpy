use crate::CKKSResult as Result;
use poulpy_core::layouts::{Base2K, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::ShipScalar,
    layouts::{ShipCoeffEncodings, ShipPlan},
    oep::CKKSEncodingImpl,
};

/// Backend extension point for the SHIP coefficient encoding: the bottom
/// ciphertext -> hoisted-plaintext-material transformation consumed by the
/// SHIP pipeline.
///
/// Implementing this trait is the entry requirement for the SHIP operations
/// ([`CKKSShipOps`](crate::api::CKKSShipOps)). The trait imposes no FFT
/// engine, encoder, host codec, or transfer capability; a backend may
/// implement the whole step as one fused native kernel. The complete scheme
/// definition (host ciphertext -> plaintext material) is exposed as
/// [`ship_coeff_encodings_host`](crate::encoding::ship_coeff_encodings_host)
/// so host reference implementations derive from the same math.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by
/// the associated method signatures.
pub unsafe trait CKKSShipCoeffEncodingImpl<BE: Backend>: Backend {
    /// Backend-native arena bytes required by the coefficient encoding.
    fn ckks_ship_coeff_encodings_tmp_bytes_impl<F>(
        module: &Module<BE>,
        plan: &ShipPlan,
        base2k: Base2K,
        complex: bool,
    ) -> Result<usize>
    where
        F: ShipScalar,
        BE: CKKSEncodingImpl<BE, F>;

    /// Builds the input-dependent SHIP plaintext material from the bottom
    /// ciphertext's public coefficients. Implementations must preserve the
    /// plan's degree, radix, scales, and backend ownership, and must reject
    /// malformed ciphertext layouts before reading their limbs.
    fn ckks_ship_coeff_encodings_impl<F, Src>(
        module: &Module<BE>,
        ct: &Src,
        plan: &ShipPlan,
        base2k: Base2K,
        complex: bool,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<ShipCoeffEncodings<BE::OwnedBuf, BE::ZnxWord>>
    where
        F: ShipScalar,
        BE: CKKSEncodingImpl<BE, F>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}
