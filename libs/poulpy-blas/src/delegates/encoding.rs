use poulpy_ckks::{CKKSResult, api::CKKSEncodingScalar, oep::CKKSEncodingImpl};
use poulpy_core::layouts::DiagonalArithmetic;
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    api::{
        BlasContextPreparedFactory, BlasDiagonalCiphertextPreparedFactory, BlasEncodingOps, BlasMatVecOperandPreparedFactory,
        BlasPtOperandPreparedFactory,
    },
    layouts::{
        BlasContext, BlasContextInfos, BlasContextPrepared, BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos,
        BlasDiagonalCiphertextPrepared, BlasMatVecOperand, BlasMatVecOperandInfos, BlasMatVecOperandPrepared, BlasPtOperand,
        BlasPtOperandInfos, BlasPtOperandPrepared, BlasPtSide,
    },
    oep::BlasEncodingImpl,
};

impl<BE, F, Rg: Ring> BlasEncodingOps<BE, F, Rg> for Module<BE>
where
    BE: Backend + BlasEncodingImpl<Rg> + CKKSEncodingImpl<F, Rg>,
    F: CKKSEncodingScalar + DiagonalArithmetic,
{
    fn blas_pt_operand_encode_tmp_bytes<A: BlasPtOperandInfos>(&self, infos: &A) -> usize {
        BE::blas_pt_operand_encode_tmp_bytes::<F, A>(self, infos)
    }

    fn blas_pt_operand_encode<S: BlasPtSide>(
        &self,
        res: &mut BlasPtOperand<BE, S, Rg>,
        entry: impl Fn(usize, usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()> {
        BE::blas_pt_operand_encode(self, res, entry, scratch)
    }

    fn blas_matvec_operand_encode_tmp_bytes<A: BlasMatVecOperandInfos>(&self, infos: &A) -> usize {
        BE::blas_matvec_operand_encode_tmp_bytes::<F, A>(self, infos)
    }

    fn blas_matvec_operand_encode(
        &self,
        res: &mut BlasMatVecOperand<BE, Rg>,
        entry: impl Fn(usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()> {
        BE::blas_matvec_operand_encode(self, res, entry, scratch)
    }

    fn blas_context_encode_tmp_bytes<A: BlasContextInfos>(&self, infos: &A) -> usize {
        BE::blas_context_encode_tmp_bytes::<F, A>(self, infos)
    }

    fn blas_context_encode(&self, res: &mut BlasContext<BE, Rg>, scratch: &mut ScratchArena<'_, BE>) -> CKKSResult<()> {
        BE::blas_context_encode::<F>(self, res, scratch)
    }
}

impl<BE: Backend + BlasEncodingImpl<Rg>, Rg: Ring> BlasPtOperandPreparedFactory<BE, Rg> for Module<BE> {
    fn blas_pt_operand_prepare_tmp_bytes<A: BlasPtOperandInfos>(&self, infos: &A) -> usize {
        BE::blas_pt_operand_prepare_tmp_bytes(self, infos)
    }

    fn blas_pt_operand_prepare<S: BlasPtSide>(
        &self,
        res: &mut BlasPtOperandPrepared<BE, S, Rg>,
        other: &BlasPtOperand<BE, S, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::blas_pt_operand_prepare(self, res, other, scratch)
    }
}

impl<BE: Backend + BlasEncodingImpl<Rg>, Rg: Ring> BlasMatVecOperandPreparedFactory<BE, Rg> for Module<BE> {
    fn blas_matvec_operand_prepare_tmp_bytes<A: BlasMatVecOperandInfos>(&self, infos: &A) -> usize {
        BE::blas_matvec_operand_prepare_tmp_bytes(self, infos)
    }

    fn blas_matvec_operand_prepare(
        &self,
        res: &mut BlasMatVecOperandPrepared<BE, Rg>,
        other: &BlasMatVecOperand<BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::blas_matvec_operand_prepare(self, res, other, scratch)
    }
}

impl<BE: Backend + BlasEncodingImpl<Rg>, Rg: Ring> BlasDiagonalCiphertextPreparedFactory<BE, Rg> for Module<BE> {
    fn blas_diagonal_ciphertext_prepare_tmp_bytes<A: BlasDiagonalCiphertextInfos>(&self, infos: &A) -> usize {
        BE::blas_diagonal_ciphertext_prepare_tmp_bytes(self, infos)
    }

    fn blas_diagonal_ciphertext_prepare(
        &self,
        res: &mut BlasDiagonalCiphertextPrepared<BE, Rg>,
        other: &BlasDiagonalCiphertext<BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::blas_diagonal_ciphertext_prepare(self, res, other, scratch)
    }
}

impl<BE: Backend + BlasEncodingImpl<Rg>, Rg: Ring> BlasContextPreparedFactory<BE, Rg> for Module<BE> {
    fn blas_context_prepare_tmp_bytes<A: BlasContextInfos>(&self, infos: &A) -> usize {
        BE::blas_context_prepare_tmp_bytes(self, infos)
    }

    fn blas_context_prepare(
        &self,
        res: &mut BlasContextPrepared<BE, Rg>,
        other: &BlasContext<BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::blas_context_prepare(self, res, other, scratch)
    }
}
