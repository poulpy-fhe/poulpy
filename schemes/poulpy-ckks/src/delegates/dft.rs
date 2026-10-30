//! Delegating impl of the public [`CKKSDFTOps`] API onto the [`CKKSDFTImpl`] backend
//! hook, completing the `api → oep → delegates ← reference` chain for the
//! homomorphic DFT.

#![allow(clippy::too_many_arguments)]

use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    DiagonalProd,
    layouts::{Base2K, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LinearTransformation},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingScalar, LtDiagonalMeta},
    layouts::{DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode, DftDirection, DftFormat, Encode, Repack, Split, Standard},
    oep::{CKKSDFTImpl, CKKSDFTMatrixImpl},
};

impl<BE: Backend + CKKSDFTImpl> CKKSDFTOps<BE> for Module<BE> {
    fn ckks_prepare_dft_matrix_tmp_bytes<Dir, Fmt, P>(&self, dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>) -> usize
    where
        P: poulpy_core::layouts::GLWEInfos,
    {
        BE::ckks_prepare_dft_matrix_tmp_bytes(self, dft)
    }

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
        K: poulpy_core::layouts::GGLWEInfos,
    {
        BE::ckks_dft_tmp_bytes(self, dst, src, dft, key)
    }

    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        &self,
        res: &mut DFTMatrixPrepared<BE, Dir, Fmt>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        BE::ckks_prepare_dft_matrix::<Dir, Fmt, P>(self, res, dft, scratch)
    }

    fn ckks_prepare_dft_matrix_alloc<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrixPrepared<BE, Dir, Fmt>>
    where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        BE::ckks_prepare_dft_matrix_alloc::<Dir, Fmt, P>(self, dft, scratch)
    }

    fn ckks_dft_evaluate_assign<Dir, Fmt, P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        crate::reference::mul::ensure_full_layout("ckks_dft_evaluate_assign", crate::GLWEPlaintextInfos::slot_layout(ct))?;
        BE::ckks_dft_evaluate_assign(self, ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_coeffs_to_slots(self, ct, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs<P, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_slots_to_coeffs(self, ct, dft, keys, scratch)
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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_coeffs_to_slots_split(self, ct_real, ct_imag, ct_in, dft, keys, scratch)
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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_slots_to_coeffs_split(self, op_out, ct_real, ct_imag, dft, keys, scratch)
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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_coeffs_to_slots_repack(self, ct_out, ct_in, dft, keys, scratch)
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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_slots_to_coeffs_repack(self, op_out, ct_in, dft, keys, scratch)
    }
}

impl<BE, F> CKKSDFTMatrixOps<BE, F> for Module<BE>
where
    BE: Backend + CKKSDFTMatrixImpl<F>,
    F: CKKSEncodingScalar,
{
    fn ckks_new_dft_matrix_tmp_bytes<Dir, Fmt>(&self, literal: &DFTPlan) -> Result<usize>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
    {
        BE::ckks_new_dft_matrix_tmp_bytes::<Dir, Fmt>(self, literal)
    }

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
        BE::ckks_new_dft_matrix::<Dir, Fmt>(self, base2k, literal, scratch)
    }
}
