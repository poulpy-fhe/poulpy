//! Delegating impl of the public [`DFTOps`] API onto the [`DFTImpl`] backend
//! hook, completing the `api → oep → delegates ← default` chain for the
//! homomorphic DFT.

#![allow(clippy::too_many_arguments)]

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
    api::DFTOps,
    default::dft::{DftFactor, matrices::DftScalar},
    encoding::reim::Encoder,
    layouts::{
        CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode,
        DftDirection, DftFormat, Encode, Repack, Split, Standard,
    },
    oep::DFTImpl,
};

impl<BE: Backend + DFTImpl<BE>> DFTOps<BE> for Module<BE> {
    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        BE::ckks_prepare_dft_matrix::<Dir, Fmt, P>(self, dft, scratch)
    }

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
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
    {
        BE::ckks_new_dft_matrix::<Dir, Fmt, E, F>(self, host_module, encoder, base2k, factor_meta, literal, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::ckks_dft_evaluate_assign(self, ct, dft, keys, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::ckks_coeffs_to_slots(self, ct, dft, keys, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::ckks_slots_to_coeffs(self, ct, dft, keys, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::ckks_coeffs_to_slots_split(self, ct_real, ct_imag, ct_in, dft, keys, conj_key, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::ckks_slots_to_coeffs_split(self, op_out, ct_real, ct_imag, dft, keys, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::ckks_coeffs_to_slots_repack(self, ct_out, ct_in, dft, keys, conj_key, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::ckks_slots_to_coeffs_repack(self, op_out, ct_in, dft, keys, scratch)
    }
}
