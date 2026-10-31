//! Public CKKS-facing API for the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs).
//!
//! Calls dispatch through [`crate::oep::CKKSDFTImpl`], including the selected
//! generic DFT operation used by derived format wrappers.
//! The homomorphic DFT is documented as a stage of the bootstrapping pipeline in
//! [`docs/guides/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/guides/bootstrapping.md).

use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWEDiagonalProd,
    layouts::{
        Base2K, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, LinearTransformation,
        LinearTransformationPrepared,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf},
    layouts::{Backend, PrepareHint, ScratchArena},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSEncodingScalar, LtDiagonalMeta},
    layouts::{
        DFTMatrix, DFTMatrixFactors, DFTMatrixPrepared, DFTPlan, Decode, DftDirection, DftFormat, Encode, Repack, Split, Standard,
    },
};

/// Homomorphic DFT operations on a CKKS [`poulpy_hal::layouts::Module`].
///
/// Setup ([`CKKSDFTMatrixOps::ckks_dft_matrix_new`]) builds the factor operands
/// once (as an unprepared [`DFTMatrix`]);
/// [`CKKSDFTMatrixPreparedFactory::ckks_dft_matrix_prepare`] optionally
/// prepares that into the resident [`DFTMatrixPrepared`] for faster repeated
/// evaluation. The evaluation methods apply either form. The `*_assign` methods
/// are the `Standard` format; `*_split` returns the real/imaginary parts in two
/// ciphertexts (`SplitRealAndImag`); `*_repack` is the sparse `RepackImagAsReal`
/// path that packs the imaginary part into the right half of a single ciphertext.
///
/// Size evaluation (including format wrappers) with [`Self::ckks_dft_tmp_bytes`].
pub trait CKKSDFTOps<BE: Backend> {
    /// Selected workspace for raw DFT evaluation and its format wrappers.
    /// `dst` and `src` must bound all destination and source allocations; `key`
    /// must bound every rotation/conjugation key used by the transform.
    fn ckks_dft_tmp_bytes<Dir, Fmt, P, Dst, Src, K>(
        &self,
        dst: &Dst,
        src: &Src,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        key: &K,
    ) -> usize
    where
        P: poulpy_core::layouts::GLWEInfos,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        K: poulpy_core::layouts::GGLWEInfos;

    /// Evaluates the homomorphic (I)DFT in place (raw chain, no format wrapper).
    fn ckks_dft_matrix_eval_assign<Dir, Fmt, P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

    /// `CoeffsToSlots`, `Standard` format (in place).
    fn ckks_coeffs_to_slots_assign<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

    /// `SlotsToCoeffs`, `Standard` format (in place).
    fn ckks_slots_to_coeffs_assign<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

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
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

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
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

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
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

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
        P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;
}

/// Allocates and prepares [`DFTMatrixPrepared`] matrices.
pub trait CKKSDFTMatrixPreparedFactory<BE: Backend>
where
    Self: CnvPVecAlloc<BE> + CnvPVecBytesOf,
{
    /// A prepared matrix of `dft`'s plan and factor schedules, each diagonal at the degree, radix and width of the
    /// factor's first diagonal, which [`Self::ckks_dft_matrix_prepare`] fills. Its diagonals are unspecified until then.
    ///
    /// # Panics
    ///
    /// When a factor of `dft` has no diagonal.
    fn ckks_dft_matrix_prepared_alloc_from_infos<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        Self: Sized,
        P: GLWEInfos,
    {
        let factors = dft
            .factor_operands()
            .iter()
            .map(|factor| {
                let first = factor
                    .first_diagonal_plaintext()
                    .expect("ckks_dft_matrix_prepared_alloc_from_infos: every factor has a diagonal");
                LinearTransformationPrepared::<BE>::alloc_prepared_from_index(self, &factor.index(), first)
            })
            .collect();
        DFTMatrix::from_factors(DFTMatrixFactors::new(dft.plan().clone(), factors))
    }

    /// Bytes of the prepared matrix [`Self::ckks_dft_matrix_prepared_alloc_from_infos`] allocates for `dft`.
    ///
    /// # Panics
    ///
    /// As [`Self::ckks_dft_matrix_prepared_alloc_from_infos`], and when the byte count overflows `usize`.
    fn ckks_dft_matrix_prepared_bytes_of_from_infos<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: LWEInfos,
    {
        dft.factor_operands().iter().fold(0usize, |bytes, factor| {
            let first = factor
                .first_diagonal_plaintext()
                .expect("ckks_dft_matrix_prepared_bytes_of_from_infos: every factor has a diagonal");
            let diagonals: usize = factor.index().index.iter().map(Vec::len).sum();
            diagonals
                .checked_mul(self.cnv_pvec_right_bytes_of(first.n().as_usize(), 1, first.size(), PrepareHint::Reuse))
                .and_then(|factor_bytes| bytes.checked_add(factor_bytes))
                .expect("ckks_dft_matrix_prepared_bytes_of_from_infos: the byte count overflows usize")
        })
    }

    /// Selected workspace for preparing every factor of this matrix.
    fn ckks_dft_matrix_prepare_tmp_bytes<Dir, Fmt, P>(&self, dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>) -> usize
    where
        P: GLWEInfos;

    /// Prepares an unprepared [`DFTMatrix`] into `res`, its resident convolution-domain form [`DFTMatrixPrepared`]
    /// allocated by [`Self::ckks_dft_matrix_prepared_alloc_from_infos`], with no allocation: each factor's plaintext
    /// diagonals are prepared into a `CnvPVec` right operand, trading resident memory for faster repeated evaluation.
    /// `res` then carries `dft`'s plan; preparing again into it refills it. Takes
    /// [`Self::ckks_dft_matrix_prepare_tmp_bytes`] of scratch.
    ///
    /// # Errors
    ///
    /// `InvalidOperand`, before any write to `res`, when `res` was allocated for another factor count, factor schedule
    /// or diagonal degree, radix or width than `dft`'s, or when the diagonals of a factor do not share one scale, width
    /// and slot kind; then `CKKSError::Internal`, before any write, for an arena below
    /// [`Self::ckks_dft_matrix_prepare_tmp_bytes`].
    fn ckks_dft_matrix_prepare<Dir, Fmt, P>(
        &self,
        res: &mut DFTMatrixPrepared<BE, Dir, Fmt>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + GLWEDiagonalProd<BE>;

    /// [`Self::ckks_dft_matrix_prepare`] of `dft` into [`Self::ckks_dft_matrix_prepared_alloc_from_infos`] of `dft`.
    /// The plan and output-format variant are preserved.
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_dft_matrix_prepare`].
    fn ckks_dft_matrix_prepare_alloc<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrixPrepared<BE, Dir, Fmt>>
    where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + GLWEDiagonalProd<BE>;
}

/// Homomorphic DFT matrix generation at scalar precision `F`.
///
/// Split from [`CKKSDFTOps`] so the scalar is a trait parameter and the method
/// stays free of backend bounds: the delegating impl on `Module<BE>` requires
/// the [`CKKSDFTMatrixImpl<F>`](crate::oep::CKKSDFTMatrixImpl) seam at the impl
/// level, and a backend overrides that seam independently of any bounds the
/// reference implementation carries.
pub trait CKKSDFTMatrixOps<BE: Backend, F: CKKSEncodingScalar> {
    /// Scratch bytes of [`Self::ckks_dft_matrix_new`] for `literal`: the factors are encoded one after the other on the
    /// same scratch, so the largest slot encoding of one factor's diagonals.
    ///
    /// # Errors
    ///
    /// The plan-versus-ring error of [`Self::ckks_dft_matrix_new`] for a plan with more slots than the ring supports.
    fn ckks_dft_matrix_new_tmp_bytes<Dir, Fmt>(&self, literal: &DFTPlan) -> Result<usize>
    where
        Dir: DftDirection,
        Fmt: DftFormat;

    /// Builds the unprepared homomorphic (I)DFT described by `literal` at
    /// scalar precision `F` (the reference chain is
    /// [`crate::reference::dft::ckks_dft_matrix_new_reference`]): each factor matrix is
    /// encoded into a CKKS linear transformation with plaintext diagonals,
    /// materialized per factor at eval time. Prepare it into the resident form
    /// with [`CKKSDFTMatrixPreparedFactory::ckks_dft_matrix_prepare`]. Each factor's BSGS
    /// giant-step width is taken verbatim from the plan's schedule (`1` is the
    /// direct schedule); no implicit optimum is applied.
    ///
    /// Size scratch with [`Self::ckks_dft_matrix_new_tmp_bytes`].
    fn ckks_dft_matrix_new<Dir, Fmt>(
        &self,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat;
}
