//! Public CKKS-facing API for the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs).
//!
//! A thin trait on [`Module`] over the free functions in
//! [`crate::default::dft`], so callers write `module.ckks_coeffs_to_slots(...)`.
//! The homomorphic DFT is documented as a stage of the bootstrapping pipeline in
//! [`docs/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/bootstrapping.md).

use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    default::linear_transformation::DiagonalProd,
    layouts::{Base2K, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LinearTransformation},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSEncodingScalar, LtDiagonalScale},
    layouts::{DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode, DftDirection, DftFormat, Encode, Repack, Split, Standard},
};

/// Homomorphic DFT operations on a CKKS [`poulpy_hal::layouts::Module`].
///
/// Setup ([`CKKSDFTMatrixOps::ckks_new_dft_matrix`]) builds the factor operands
/// once (as an unprepared [`DFTMatrix`]); [`Self::ckks_prepare_dft_matrix`] optionally
/// promotes that to the resident [`DFTMatrixPrepared`] for faster repeated
/// evaluation. The evaluation methods apply either form. The `*_assign` methods
/// are the `Standard` format; `*_split` returns the real/imaginary parts in two
/// ciphertexts (`SplitRealAndImag`); `*_repack` is the sparse `RepackImagAsReal`
/// path that packs the imaginary part into the right half of a single ciphertext.
pub trait CKKSDFTOps<BE: Backend> {
    /// Prepares an unprepared [`DFTMatrix`] into its resident
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
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>;

    /// Evaluates the homomorphic (I)DFT in place (raw chain, no format wrapper).
    fn ckks_dft_evaluate_assign<Dir, Fmt, P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// `CoeffsToSlots`, `Standard` format (in place).
    fn ckks_coeffs_to_slots<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// `SlotsToCoeffs`, `Standard` format (in place).
    fn ckks_slots_to_coeffs<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    /// `CoeffsToSlots`, `SplitRealAndImag` — real/imag in two ciphertexts.
    #[allow(clippy::too_many_arguments)]
    fn ckks_coeffs_to_slots_split<P, Dst, Src, H>(
        &self,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// `SlotsToCoeffs`, `SplitRealAndImag` — combine two ciphertexts then Decode.
    fn ckks_slots_to_coeffs_split<P, Dst, Src, H>(
        &self,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// `CoeffsToSlots`, sparse `RepackImagAsReal` — imag packed into the right half.
    #[allow(clippy::too_many_arguments)]
    fn ckks_coeffs_to_slots_repack<P, Dst, Src, H>(
        &self,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    /// `SlotsToCoeffs`, sparse `RepackImagAsReal` — inverse of [`Self::ckks_coeffs_to_slots_repack`].
    fn ckks_slots_to_coeffs_repack<P, Dst, Src, H>(
        &self,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;
}

/// Homomorphic DFT matrix generation at scalar precision `F`.
///
/// Split from [`CKKSDFTOps`] so the scalar is a trait parameter and the method
/// stays free of backend bounds: the delegating impl on `Module<BE>` requires
/// the [`DFTMatrixImpl<BE, F>`](crate::oep::DFTMatrixImpl) seam at the impl
/// level, and a backend overrides that seam independently of any bounds the
/// reference implementation carries.
pub trait CKKSDFTMatrixOps<BE: Backend, F: CKKSEncodingScalar> {
    /// Builds the unprepared homomorphic (I)DFT described by `literal` at
    /// scalar precision `F` (the reference chain is
    /// [`crate::default::dft::ckks_new_dft_matrix`]): each factor matrix is
    /// encoded into a CKKS linear transformation with plaintext diagonals,
    /// materialized per factor at eval time. Promote it to the resident form
    /// with [`CKKSDFTOps::ckks_prepare_dft_matrix`]. Each factor's BSGS
    /// giant-step width is taken verbatim from the plan's schedule (`1` is the
    /// direct schedule); no implicit optimum is applied.
    fn ckks_new_dft_matrix<Dir, Fmt>(
        &self,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat;
}
