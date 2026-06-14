//! Public CKKS-facing API for the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs).
//!
//! A thin trait on [`Module`] over the free functions in
//! [`crate::default::dft`], so callers write `module.ckks_coeffs_to_slots(...)`.
//! See [`docs/ckks_dft.md`](https://github.com/poulpy-fhe/poulpy) for the design.

use anyhow::Result;
use poulpy_core::layouts::{
    Base2K, GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef,
    GetGaloisElement, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{ModuleNew, NegacyclicFFT, NegacyclicFFTNew},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom},
};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    default::dft::{DftFactor, matrices::DftScalar},
    encoding::reim::Encoder,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, DFTMatrix, DFTMatrixStreamed, DFTPlan},
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
    /// [`crate::default::dft::ckks_new_dft_matrix`]). The BSGS schedule is chosen
    /// cost-optimally per factor matrix.
    #[allow(clippy::too_many_arguments)]
    fn ckks_new_dft_matrix<E, F>(
        &self,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        factor_meta: CKKSMeta,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrix<BE>
    where
        F: CKKSScalar + DftScalar,
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>;

    /// Builds the **streamed** (unprepared-RHS) homomorphic (I)DFT (see
    /// [`crate::default::dft::ckks_new_dft_matrix_streamed`]). For
    /// bandwidth-limited backends: the diagonals are materialized per factor at
    /// eval time instead of kept resident. The host-backed reference build does
    /// not touch `scratch`, but the parameter is kept for backends whose
    /// encode/upload path needs device scratch.
    #[allow(clippy::too_many_arguments)]
    fn ckks_new_dft_matrix_streamed<E, F>(
        &self,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        factor_meta: CKKSMeta,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixStreamed<BE>
    where
        F: CKKSScalar + DftScalar,
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>;

    /// Evaluates the homomorphic (I)DFT in place (raw chain, no format wrapper).
    fn ckks_dft_evaluate_assign<R, Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, R>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: DftFactor<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `CoeffsToSlots`, `Standard` format (in place).
    fn ckks_coeffs_to_slots<R, Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, R>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: DftFactor<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `SlotsToCoeffs`, `Standard` format (in place).
    fn ckks_slots_to_coeffs<R, Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, R>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: DftFactor<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `CoeffsToSlots`, `SplitRealAndImag` — real/imag in two ciphertexts.
    #[allow(clippy::too_many_arguments)]
    fn ckks_coeffs_to_slots_split<R, Dst, Src, H, K>(
        &self,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, R>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: DftFactor<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `SlotsToCoeffs`, `SplitRealAndImag` — combine two ciphertexts then Decode.
    fn ckks_slots_to_coeffs_split<R, Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE, R>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: DftFactor<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `CoeffsToSlots`, sparse `RepackImagAsReal` — imag packed into the right half.
    #[allow(clippy::too_many_arguments)]
    fn ckks_coeffs_to_slots_repack<R, Dst, Src, H, K>(
        &self,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, R>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: DftFactor<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// `SlotsToCoeffs`, sparse `RepackImagAsReal` — inverse of [`Self::ckks_coeffs_to_slots_repack`].
    fn ckks_slots_to_coeffs_repack<R, Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, R>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: DftFactor<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
