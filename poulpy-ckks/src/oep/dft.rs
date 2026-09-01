//! Backend extension point for the homomorphic DFT (CoeffsToSlots /
//! SlotsToCoeffs).
//!
//! Mirrors the core `api → oep → delegates ← default` pattern (cf.
//! [`poulpy_core::oep`]). [`DFTImpl`] is the backend hook for evaluation and
//! preparation, [`DFTMatrixImpl`] the hook for matrix generation at a given
//! scalar precision; [`DFTDefault`] / [`DFTMatrixDefault`] are the override
//! surfaces a backend implements (by hand to substitute a fused device
//! kernel, or via [`impl_ckks_dft_defaults`] to inherit the reference chain).
//! The public [`CKKSDFTOps`](crate::api::CKKSDFTOps) trait delegates through here, so a
//! backend can replace the *whole-DFT* evaluation — every factor plus the
//! inter-factor rotations/conjugations and the split/repack glue — with a single
//! kernel.

#![allow(clippy::too_many_arguments)]

use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::Normalized;
use std::borrow::Borrow;

use crate::CKKSResult as Result;
use poulpy_core::{
    default::linear_transformation::DiagonalProd,
    layouts::{Base2K, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LinearTransformation},
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena},
};

use super::CKKSEncodingImpl;
use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSImagOps, CKKSLinearTransformationOps,
        CKKSRotateOps, CKKSSubOps, LtDiagonalScale,
    },
    default::dft::matrices::DftScalar,
    layouts::{
        CKKSModuleAlloc, DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode, DftDirection, DftFormat, Encode, Repack, Split, Standard,
    },
};

/// Backend hook for the homomorphic-DFT family.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
pub unsafe trait DFTImpl<BE: Backend>: Backend {
    fn ckks_prepare_dft_matrix_impl<Dir, Fmt, P>(
        module: &Module<BE>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE, State = Normalized> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>;

    fn ckks_dft_evaluate_assign_impl<Dir, Fmt, P, Dst, H>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    fn ckks_coeffs_to_slots_impl<P, Dst, H>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    fn ckks_slots_to_coeffs_impl<P, Dst, H>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;

    fn ckks_coeffs_to_slots_split_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    fn ckks_slots_to_coeffs_split_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    fn ckks_coeffs_to_slots_repack_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    fn ckks_slots_to_coeffs_repack_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;
}

/// Backend hook for homomorphic-DFT matrix generation at scalar precision `F`.
///
/// Separate from [`DFTImpl`] so the scalar sits at the trait level: the
/// reference chain ([`DFTMatrixDefault`], wired by [`impl_ckks_dft_defaults`])
/// carries its host encoding requirements as impl-level bounds, while a
/// backend overriding this hook (e.g. generating and encoding the factor
/// matrices natively on device) implements it directly with no obligations
/// beyond this signature.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
pub unsafe trait DFTMatrixImpl<BE: Backend, F: CKKSEncodingScalar>: Backend {
    fn ckks_new_dft_matrix_impl<Dir, Fmt>(
        module: &Module<BE>,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat;
}

/// Override surface for homomorphic-DFT matrix generation at scalar precision
/// `F`.
///
/// Abstract: no method bodies, and no bounds beyond the signature — the
/// reference implementation's requirements (host matrix generation via
/// `DftScalar`, backend encoding via `CKKSEncodingImpl`) are impl-level bounds
/// of the [`impl_ckks_dft_defaults`] expansion, invisible to backends that
/// override [`DFTMatrixImpl`] directly.
#[doc(hidden)]
pub trait DFTMatrixDefault<BE: Backend, F: CKKSEncodingScalar> {
    fn ckks_new_dft_matrix_default<Dir, Fmt>(
        &self,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        Self: Borrow<Module<BE>>,
        F: DftScalar,
        BE: CKKSEncodingImpl<BE, F>,
        Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    {
        crate::default::dft::ckks_new_dft_matrix::<Dir, Fmt, BE, F>(self.borrow(), base2k, literal, scratch)
    }
}

unsafe impl<BE, F> DFTMatrixImpl<BE, F> for BE
where
    BE: Backend + CKKSEncodingImpl<BE, F>,
    F: CKKSEncodingScalar + DftScalar,
    Module<BE>: DFTMatrixDefault<BE, F>
        + CnvPVecAlloc<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSEncodingOps<BE, F>,
{
    fn ckks_new_dft_matrix_impl<Dir, Fmt>(
        module: &Module<BE>,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
    {
        module.ckks_new_dft_matrix_default::<Dir, Fmt>(base2k, literal, scratch)
    }
}

/// Override surface for the homomorphic-DFT family.
///
/// Every method has a default body forwarding to the reference algorithms in
/// [`crate::default::dft`] (with per-method bounds — the Default layer may
/// carry bounds; the bound-free override seam is [`DFTImpl`]). A backend opts
/// in with the one-line [`impl_ckks_dft_defaults`] marker; to substitute a
/// fused kernel for one format, override just that method and inherit the rest.
#[doc(hidden)]
pub trait DFTDefault<BE: Backend> {
    fn ckks_prepare_dft_matrix_default<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE, State = Normalized> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    {
        crate::default::dft::ckks_prepare_dft_matrix::<Dir, Fmt, BE, P>(self.borrow(), dft, scratch)
    }

    fn ckks_dft_evaluate_assign_default<Dir, Fmt, P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    {
        crate::default::dft::ckks_dft_evaluate_assign(self.borrow(), ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_default<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    {
        crate::default::dft::ckks_coeffs_to_slots_assign(self.borrow(), ct, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_default<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    {
        crate::default::dft::ckks_slots_to_coeffs_assign(self.borrow(), ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_split_default<P, Dst, Src, H>(
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE>
            + CnvPVecAlloc<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSImagOps<BE>,
    {
        crate::default::dft::ckks_coeffs_to_slots_split(self.borrow(), ct_real, ct_imag, ct_in, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_split_default<P, Dst, Src, H>(
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSAddOps<BE> + CKKSImagOps<BE>,
    {
        crate::default::dft::ckks_slots_to_coeffs_split(self.borrow(), op_out, ct_real, ct_imag, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_repack_default<P, Dst, Src, H>(
        &self,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE>
            + CnvPVecAlloc<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSImagOps<BE>
            + CKKSRotateOps<BE>,
    {
        crate::default::dft::ckks_coeffs_to_slots_repack(self.borrow(), ct_out, ct_in, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_repack_default<P, Dst, Src, H>(
        &self,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
        Self: Borrow<Module<BE>>,
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSCopyOps<BE>,
    {
        crate::default::dft::ckks_slots_to_coeffs_repack(self.borrow(), op_out, ct_in, dft, keys, scratch)
    }
}

unsafe impl<BE> DFTImpl<BE> for BE
where
    BE: Backend,
    Module<BE>: DFTDefault<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSRotateOps<BE>,
{
    fn ckks_prepare_dft_matrix_impl<Dir, Fmt, P>(
        module: &Module<BE>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE, State = Normalized> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        module.ckks_prepare_dft_matrix_default::<Dir, Fmt, P>(dft, scratch)
    }

    fn ckks_dft_evaluate_assign_impl<Dir, Fmt, P, Dst, H>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        module.ckks_dft_evaluate_assign_default(ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_impl<P, Dst, H>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        module.ckks_coeffs_to_slots_default(ct, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_impl<P, Dst, H>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        module.ckks_slots_to_coeffs_default(ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_split_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        module.ckks_coeffs_to_slots_split_default(ct_real, ct_imag, ct_in, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_split_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        module.ckks_slots_to_coeffs_split_default(op_out, ct_real, ct_imag, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_repack_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        module.ckks_coeffs_to_slots_repack_default(ct_out, ct_in, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_repack_impl<P, Dst, Src, H>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        module.ckks_slots_to_coeffs_repack_default(op_out, ct_in, dft, keys, scratch)
    }
}

/// Wires a backend into the reference homomorphic-DFT chain: implements the
/// [`DFTDefault`] and [`DFTMatrixDefault`] marker impls, inheriting every
/// default-bodied method (which forward to [`crate::default::dft`]).
///
/// For partial override (a fused device kernel for one format, defaults for
/// the rest), write the impl block by hand, override just the methods you
/// replace, and inherit the remaining default bodies.
#[macro_export]
macro_rules! impl_ckks_dft_defaults {
    ($be:ty) => {
        impl $crate::oep::DFTDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
        impl<F: $crate::api::CKKSEncodingScalar> $crate::oep::DFTMatrixDefault<$be, F> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_dft_defaults;
