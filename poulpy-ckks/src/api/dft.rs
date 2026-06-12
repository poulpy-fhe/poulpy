//! Public CKKS-facing API for the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs).
//!
//! A thin trait on [`Module`] over the free functions in
//! [`crate::default::dft`], so callers write `module.ckks_coeffs_to_slots(...)`.
//! See [`docs/ckks_dft.md`](https://github.com/poulpy-fhe/poulpy) for the design.

use anyhow::Result;
use poulpy_core::layouts::{
    Base2K, GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef,
    GetGaloisElement, LinearTransformationStrategy, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom},
};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    encoding::reim::Encoder,
    layouts::{CKKSPlaintextVecHostCodec, DFTMatrix, DFTMatrixLiteral},
};

/// Homomorphic DFT operations on a CKKS [`Module`].
///
/// Setup ([`Self::ckks_new_dft_matrix`]) builds the prepared factor operands once;
/// the evaluation methods then apply them. The `*_assign` methods are the
/// `Standard` format; `*_split` returns the real/imaginary parts in two
/// ciphertexts (`SplitRealAndImag`); `*_repack` is the sparse `RepackImagAsReal`
/// path that packs the imaginary part into the right half of a single ciphertext.
pub trait DFTOps<BE: Backend> {
    /// Builds the prepared homomorphic (I)DFT described by `literal` (see
    /// [`crate::default::dft::ckks_new_dft_matrix`]).
    #[allow(clippy::too_many_arguments)]
    fn ckks_new_dft_matrix<E>(
        &self,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        factor_meta: CKKSMeta,
        literal: &DFTMatrixLiteral,
        strategy: LinearTransformationStrategy,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrix<BE>
    where
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>:
            poulpy_hal::api::ModuleNew<HostBytesBackend> + crate::layouts::CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<f64> + NegacyclicFFTNew<f64>,
        crate::layouts::CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>;

    /// Evaluates the homomorphic (I)DFT in place (raw chain, no format wrapper).
    fn ckks_dft_evaluate_assign<Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `CoeffsToSlots`, `Standard` format (in place).
    fn ckks_coeffs_to_slots<Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `SlotsToCoeffs`, `Standard` format (in place).
    fn ckks_slots_to_coeffs<Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `CoeffsToSlots`, `SplitRealAndImag` — real/imag in two ciphertexts.
    #[allow(clippy::too_many_arguments)]
    fn ckks_coeffs_to_slots_split<Dst, Src, H, K>(
        &self,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `SlotsToCoeffs`, `SplitRealAndImag` — combine two ciphertexts then Decode.
    fn ckks_slots_to_coeffs_split<Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `CoeffsToSlots`, sparse `RepackImagAsReal` — imag packed into the right half.
    #[allow(clippy::too_many_arguments)]
    fn ckks_coeffs_to_slots_repack<Dst, Src, H, K>(
        &self,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `SlotsToCoeffs`, sparse `RepackImagAsReal` — inverse of [`Self::ckks_coeffs_to_slots_repack`].
    fn ckks_slots_to_coeffs_repack<Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
