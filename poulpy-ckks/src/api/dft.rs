//! Public CKKS-facing API for the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs).
//!
//! A thin trait on [`Module`] over the free functions in
//! [`crate::default::dft`], so callers write `module.ckks_coeffs_to_slots(...)`.
//! See [`docs/ckks_dft.md`](https://github.com/poulpy-fhe/poulpy) for the design.

use anyhow::Result;
use poulpy_core::{
    default::linear_transformation::DiagonalProd,
    layouts::{
        Base2K, GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef,
        GetGaloisElement, LinearTransformation, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleNew, NegacyclicFFT, NegacyclicFFTNew},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom},
};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    default::dft::{DftFactor, matrices::DftScalar},
    encoding::reim::Encoder,
    layouts::{
        CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode,
        DftDirection, DftFormat, Encode, Repack, Split, Standard,
    },
};

/// Homomorphic DFT operations on a CKKS [`Module`].
///
/// Setup ([`Self::ckks_new_dft_matrix`]) builds the factor operands once (as a
/// host, unprepared [`DFTMatrix`]); [`Self::ckks_prepare_dft_matrix`] optionally
/// promotes that to the resident [`DFTMatrixPrepared`] for faster repeated
/// evaluation. The evaluation methods apply either form. The `*_assign` methods
/// are the `Standard` format; `*_split` returns the real/imaginary parts in two
/// ciphertexts (`SplitRealAndImag`); `*_repack` is the sparse `RepackImagAsReal`
/// path that packs the imaginary part into the right half of a single ciphertext.
pub trait DFTOps<BE: Backend> {
    /// Prepares a host, unprepared [`DFTMatrix`] into its resident
    /// convolution-domain form [`DFTMatrixPrepared`] (see
    /// [`crate::default::dft::ckks_prepare_dft_matrix`]): each factor's plaintext
    /// diagonals are prepared into a `CnvPVec` right operand, trading resident
    /// memory for faster repeated evaluation. The plan and output-format variant
    /// are preserved.
    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>;

    /// Builds the (host, unprepared) homomorphic (I)DFT described by `literal`
    /// (see [`crate::default::dft::ckks_new_dft_matrix`]): each factor matrix is
    /// encoded into a CKKS linear transformation with plaintext diagonals,
    /// materialized per factor at eval time. Promote it to the resident form with
    /// [`Self::ckks_prepare_dft_matrix`]. The BSGS schedule is chosen
    /// cost-optimally per factor matrix. The host-backed reference build does not
    /// touch `scratch`, but the parameter is kept for backends whose encode/upload
    /// path needs device scratch.
    #[allow(clippy::too_many_arguments)]
    fn ckks_new_dft_matrix<Dir, Fmt, E, F>(
        &self,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        factor_meta: CKKSMeta,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        F: CKKSScalar + DftScalar,
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>;

    /// Evaluates the homomorphic (I)DFT in place (raw chain, no format wrapper).
    fn ckks_dft_evaluate_assign<Dir, Fmt, R, Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, R>,
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
        dft: &DFTMatrix<BE, Encode, Standard, R>,
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
        dft: &DFTMatrix<BE, Decode, Standard, R>,
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
        dft: &DFTMatrix<BE, Encode, Split, R>,
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
        dft: &DFTMatrix<BE, Decode, Split, R>,
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
        dft: &DFTMatrix<BE, Encode, Repack, R>,
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
        dft: &DFTMatrix<BE, Decode, Repack, R>,
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
