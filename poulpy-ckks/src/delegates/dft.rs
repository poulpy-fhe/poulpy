//! Delegating impl of the public [`CKKSDFTOps`] API onto the [`DFTImpl`] backend
//! hook, completing the `api → oep → delegates ← reference` chain for the
//! homomorphic DFT.

#![allow(clippy::too_many_arguments)]

use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    layouts::{Base2K, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LinearTransformation},
    reference::linear_transformation::DiagonalProd,
};

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingScalar, LtDiagonalScale},
    layouts::{DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode, DftDirection, DftFormat, Encode, Repack, Split, Standard},
    oep::{DFTImpl, DFTMatrixImpl},
};

impl<BE: Backend + DFTImpl> CKKSDFTOps<BE> for Module<BE> {
    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrixPrepared<BE, Dir, Fmt>>
    where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        for factor in &dft.inner().factors {
            for step in &factor.giant_steps {
                for diagonal in &step.diagonals {
                    crate::api::CKKSModuleInfos::ckks_ring(self)
                        .check_plaintext("ckks_prepare_dft_matrix", &diagonal.plaintext)?;
                }
            }
        }
        BE::ckks_prepare_dft_matrix_impl::<Dir, Fmt, P>(self, dft, scratch)
    }

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
        H: GetAutomorphismKey<BE>,
    {
        for factor in &dft.inner().factors {
            crate::reference::linear_transformation::check_linear_transformation(
                "ckks_dft_evaluate_assign",
                crate::api::CKKSModuleInfos::ckks_ring(self),
                factor,
            )?;
        }
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_dft_evaluate_assign", ct)?;
        BE::ckks_dft_evaluate_assign_impl(self, ct, dft, keys, scratch)
    }

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
        H: GetAutomorphismKey<BE>,
    {
        for factor in &dft.inner().factors {
            crate::reference::linear_transformation::check_linear_transformation(
                "ckks_coeffs_to_slots",
                crate::api::CKKSModuleInfos::ckks_ring(self),
                factor,
            )?;
        }
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_coeffs_to_slots", ct)?;
        BE::ckks_coeffs_to_slots_impl(self, ct, dft, keys, scratch)
    }

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
        H: GetAutomorphismKey<BE>,
    {
        for factor in &dft.inner().factors {
            crate::reference::linear_transformation::check_linear_transformation(
                "ckks_slots_to_coeffs",
                crate::api::CKKSModuleInfos::ckks_ring(self),
                factor,
            )?;
        }
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_slots_to_coeffs", ct)?;
        BE::ckks_slots_to_coeffs_impl(self, ct, dft, keys, scratch)
    }

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
        H: GetAutomorphismKey<BE>,
    {
        for factor in &dft.inner().factors {
            crate::reference::linear_transformation::check_linear_transformation(
                "ckks_coeffs_to_slots_split",
                crate::api::CKKSModuleInfos::ckks_ring(self),
                factor,
            )?;
        }
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_coeffs_to_slots_split", ct_real)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_coeffs_to_slots_split", ct_imag)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_coeffs_to_slots_split", ct_in)?;
        BE::ckks_coeffs_to_slots_split_impl(self, ct_real, ct_imag, ct_in, dft, keys, scratch)
    }

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
        H: GetAutomorphismKey<BE>,
    {
        for factor in &dft.inner().factors {
            crate::reference::linear_transformation::check_linear_transformation(
                "ckks_slots_to_coeffs_split",
                crate::api::CKKSModuleInfos::ckks_ring(self),
                factor,
            )?;
        }
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_slots_to_coeffs_split", op_out)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_slots_to_coeffs_split", ct_real)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_slots_to_coeffs_split", ct_imag)?;
        BE::ckks_slots_to_coeffs_split_impl(self, op_out, ct_real, ct_imag, dft, keys, scratch)
    }

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
        H: GetAutomorphismKey<BE>,
    {
        for factor in &dft.inner().factors {
            crate::reference::linear_transformation::check_linear_transformation(
                "ckks_coeffs_to_slots_repack",
                crate::api::CKKSModuleInfos::ckks_ring(self),
                factor,
            )?;
        }
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_coeffs_to_slots_repack", ct_out)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_coeffs_to_slots_repack", ct_in)?;
        BE::ckks_coeffs_to_slots_repack_impl(self, ct_out, ct_in, dft, keys, scratch)
    }

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
        H: GetAutomorphismKey<BE>,
    {
        for factor in &dft.inner().factors {
            crate::reference::linear_transformation::check_linear_transformation(
                "ckks_slots_to_coeffs_repack",
                crate::api::CKKSModuleInfos::ckks_ring(self),
                factor,
            )?;
        }
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_slots_to_coeffs_repack", op_out)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_slots_to_coeffs_repack", ct_in)?;
        BE::ckks_slots_to_coeffs_repack_impl(self, op_out, ct_in, dft, keys, scratch)
    }
}

impl<BE, F> CKKSDFTMatrixOps<BE, F> for Module<BE>
where
    BE: Backend + DFTMatrixImpl<F>,
    F: CKKSEncodingScalar,
{
    fn ckks_new_dft_matrix<Dir, Fmt>(
        &self,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
    {
        crate::ckks_ensure!(
            !crate::api::CKKSModuleInfos::ckks_is_conjugate_invariant(self),
            "homomorphic DFT requires a standard ring"
        );
        BE::ckks_new_dft_matrix_impl::<Dir, Fmt>(self, base2k, literal, scratch)
    }
}
