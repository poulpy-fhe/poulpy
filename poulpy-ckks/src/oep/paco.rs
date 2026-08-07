use crate::CKKSResult as Result;
use poulpy_core::layouts::{Base2K, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::PaCoScalar,
    layouts::{CKKSPlaintext, PaCoPlan},
    oep::CKKSEncodingImpl,
};

/// Backend extension point for the PaCo coefficient encoding: the ciphertext
/// → four-β-plaintexts transformation consumed by PaCo's blind rotation.
///
/// Implementing this trait is the entry requirement for the PaCo
/// bootstrapping operations
/// ([`CKKSPaCoOps`](crate::api::CKKSPaCoOps)): a backend without a ready
/// coefficient-encoding implementation cannot bootstrap. The trait imposes no
/// FFT engine, encoder, host codec, or transfer capability — the encoding
/// method sees only the ciphertext, the validated plan, the radix, and the
/// backend's own precomputed material, so a backend may implement the whole
/// step as one fused native kernel. This crate carries no implementation; the
/// reference implementation lives in `poulpy-cpu-ref`, and the complete
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
pub unsafe trait CKKSPaCoCoeffEncodingImpl<BE: Backend>: Backend {
    /// Backend-native arena bytes required by coefficient embedding and β
    /// packing. This includes any scalar buffers used by native FFT kernels.
    fn ckks_paco_coeff_encodings_tmp_bytes_impl<F>(module: &Module<BE>, plan: &PaCoPlan) -> Result<usize>
    where
        F: PaCoScalar,
        BE: CKKSEncodingImpl<BE, F>;

    /// Encodes the exhausted ciphertext's public coefficients as the four
    /// dense beta plaintexts consumed by PaCo blind rotation.
    ///
    /// Implementations must preserve the plan's degree, radix, scale, budget,
    /// full-slot ordering, and backend ownership, and must reject malformed
    /// ciphertext layouts before reading their active limbs. The β packing
    /// must agree numerically (within CKKS precision) with the `2C`-block
    /// coefficient→evaluation packing used by the σ vectors of
    /// [`PaCoSecretSpec`](crate::layouts::PaCoSecretSpec) at key generation.
    fn ckks_paco_coeff_encodings_impl<F, Src>(
        module: &Module<BE>,
        ct: &Src,
        plan: &PaCoPlan,
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<[CKKSPlaintext<BE::OwnedBuf>; 4]>
    where
        F: PaCoScalar,
        BE: CKKSEncodingImpl<BE, F>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds;
}
