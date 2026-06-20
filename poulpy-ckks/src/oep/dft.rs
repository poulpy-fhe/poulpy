//! Backend extension point for the homomorphic DFT (CoeffsToSlots /
//! SlotsToCoeffs).
//!
//! Mirrors the core `api → oep → delegates ← default` pattern (cf.
//! [`poulpy_core::oep`]). [`DFTImpl`] is the backend hook; [`DFTDefault`] is the
//! override surface a backend implements (by hand to substitute a fused device
//! kernel, or via [`impl_ckks_dft_defaults`] to inherit the reference chain).
//! The public [`DFTOps`](crate::api::DFTOps) trait delegates through here, so a
//! backend can replace the *whole-DFT* evaluation — every factor plus the
//! inter-factor rotations/conjugations and the split/repack glue — with a single
//! kernel.

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
    CKKSCtBounds, SetCKKSInfos,
    api::LtDiagonalScale,
    default::dft::matrices::DftScalar,
    encoding::reim::Encoder,
    layouts::{
        CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode,
        DftDirection, DftFormat, Encode, Repack, Split, Standard,
    },
};

/// Backend hook for the homomorphic-DFT family.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
pub unsafe trait DFTImpl<BE: Backend>: Backend {
    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        module: &Module<BE>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>;

    fn ckks_new_dft_matrix<Dir, Fmt, E, F>(
        module: &Module<BE>,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> ::anyhow::Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        F: CKKSScalar + DftScalar,
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>;

    fn ckks_dft_evaluate_assign<Dir, Fmt, P, Dst, H, K>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_coeffs_to_slots<P, Dst, H, K>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_slots_to_coeffs<P, Dst, H, K>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_coeffs_to_slots_split<P, Dst, Src, H, K>(
        module: &Module<BE>,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_slots_to_coeffs_split<P, Dst, Src, H, K>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_coeffs_to_slots_repack<P, Dst, Src, H, K>(
        module: &Module<BE>,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_slots_to_coeffs_repack<P, Dst, Src, H, K>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

/// Override surface for the homomorphic-DFT family.
///
/// Abstract: no method bodies. See [`crate::default::dft`] for the reference
/// algorithms a backend may forward to (the [`impl_ckks_dft_defaults`] macro
/// wires every method to them). To substitute a fused whole-DFT kernel for one
/// format, write the impl block by hand and forward only the methods you keep.
#[doc(hidden)]
pub trait DFTDefault<BE: Backend> {
    fn ckks_prepare_dft_matrix_default<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>;

    fn ckks_new_dft_matrix_default<Dir, Fmt, E, F>(
        &self,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> ::anyhow::Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        F: CKKSScalar + DftScalar,
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>;

    fn ckks_dft_evaluate_assign_default<Dir, Fmt, P, Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_coeffs_to_slots_default<P, Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_slots_to_coeffs_default<P, Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_coeffs_to_slots_split_default<P, Dst, Src, H, K>(
        &self,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_slots_to_coeffs_split_default<P, Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_coeffs_to_slots_repack_default<P, Dst, Src, H, K>(
        &self,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn ckks_slots_to_coeffs_repack_default<P, Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

unsafe impl<BE> DFTImpl<BE> for BE
where
    BE: Backend,
    Module<BE>: DFTDefault<BE>,
{
    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        module: &Module<BE>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        module.ckks_prepare_dft_matrix_default::<Dir, Fmt, P>(dft, scratch)
    }

    fn ckks_new_dft_matrix<Dir, Fmt, E, F>(
        module: &Module<BE>,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> ::anyhow::Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        F: CKKSScalar + DftScalar,
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
    {
        module.ckks_new_dft_matrix_default::<Dir, Fmt, E, F>(host_module, encoder, base2k, literal, scratch)
    }

    fn ckks_dft_evaluate_assign<Dir, Fmt, P, Dst, H, K>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.ckks_dft_evaluate_assign_default(ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots<P, Dst, H, K>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.ckks_coeffs_to_slots_default(ct, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs<P, Dst, H, K>(
        module: &Module<BE>,
        ct: &mut Dst,
        dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.ckks_slots_to_coeffs_default(ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_split<P, Dst, Src, H, K>(
        module: &Module<BE>,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.ckks_coeffs_to_slots_split_default(ct_real, ct_imag, ct_in, dft, keys, conj_key, scratch)
    }

    fn ckks_slots_to_coeffs_split<P, Dst, Src, H, K>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.ckks_slots_to_coeffs_split_default(op_out, ct_real, ct_imag, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_repack<P, Dst, Src, H, K>(
        module: &Module<BE>,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.ckks_coeffs_to_slots_repack_default(ct_out, ct_in, dft, keys, conj_key, scratch)
    }

    fn ckks_slots_to_coeffs_repack<P, Dst, Src, H, K>(
        module: &Module<BE>,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: DiagonalProd<BE> + LtDiagonalScale,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        module.ckks_slots_to_coeffs_repack_default(op_out, ct_in, dft, keys, scratch)
    }
}

/// Implements [`DFTDefault`] for `Module<$be>` by forwarding every method to the
/// corresponding `crate::default::dft` reference function.
///
/// For partial override (a fused device kernel for one format, defaults for the
/// rest), write the impl block by hand and forward only the methods you keep.
#[macro_export]
macro_rules! impl_ckks_dft_defaults {
    ($be:ty) => {
        impl $crate::oep::DFTDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn ckks_prepare_dft_matrix_default<Dir, Fmt, P>(
                &self,
                dft: &$crate::layouts::DFTMatrix<$be, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::layouts::DFTMatrixPrepared<$be, Dir, Fmt>
            where
                P: ::poulpy_core::layouts::GLWEToBackendRef<$be>
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::default::linear_transformation::DiagonalProd<$be>,
            {
                $crate::default::dft::ckks_prepare_dft_matrix::<Dir, Fmt, $be, P>(self, dft, scratch)
            }

            fn ckks_new_dft_matrix_default<Dir, Fmt, E, F>(
                &self,
                host_module: &::poulpy_hal::layouts::Module<::poulpy_hal::layouts::HostBytesBackend>,
                encoder: &$crate::encoding::reim::Encoder<E>,
                base2k: ::poulpy_core::layouts::Base2K,
                literal: &$crate::layouts::DFTPlan,
                _scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<$crate::layouts::DFTMatrix<$be, Dir, Fmt>>
            where
                Dir: $crate::layouts::DftDirection,
                Fmt: $crate::layouts::DftFormat,
                F: $crate::layouts::CKKSScalar + $crate::default::dft::matrices::DftScalar,
                $be: ::poulpy_hal::layouts::TransferFrom<::poulpy_hal::layouts::HostBytesBackend>,
                ::poulpy_hal::layouts::Module<::poulpy_hal::layouts::HostBytesBackend>:
                    ::poulpy_hal::api::ModuleNew<::poulpy_hal::layouts::HostBytesBackend>
                        + $crate::layouts::CKKSModuleAlloc<::poulpy_hal::layouts::HostBytesBackend>,
                E: ::poulpy_hal::api::NegacyclicFFT<F> + ::poulpy_hal::api::NegacyclicFFTNew<F>,
                $crate::layouts::CKKSPlaintext<Vec<u8>>: $crate::layouts::CKKSPlaintextVecHostCodec<f64>,
            {
                $crate::default::dft::ckks_new_dft_matrix::<Dir, Fmt, $be, E, F>(self, host_module, encoder, base2k, literal)
            }

            fn ckks_dft_evaluate_assign_default<Dir, Fmt, P, Dst, H, K>(
                &self,
                ct: &mut Dst,
                dft: &$crate::layouts::DFTMatrix<$be, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<()>
            where
                P: ::poulpy_core::default::linear_transformation::DiagonalProd<$be> + $crate::api::LtDiagonalScale,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                K: ::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GetGaloisElement
                    + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::dft::ckks_dft_evaluate_assign(self, ct, dft, keys, scratch)
            }

            fn ckks_coeffs_to_slots_default<P, Dst, H, K>(
                &self,
                ct: &mut Dst,
                dft: &$crate::layouts::DFTMatrix<$be, $crate::layouts::Encode, $crate::layouts::Standard, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<()>
            where
                P: ::poulpy_core::default::linear_transformation::DiagonalProd<$be> + $crate::api::LtDiagonalScale,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                K: ::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GetGaloisElement
                    + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::dft::ckks_coeffs_to_slots_assign(self, ct, dft, keys, scratch)
            }

            fn ckks_slots_to_coeffs_default<P, Dst, H, K>(
                &self,
                ct: &mut Dst,
                dft: &$crate::layouts::DFTMatrix<$be, $crate::layouts::Decode, $crate::layouts::Standard, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<()>
            where
                P: ::poulpy_core::default::linear_transformation::DiagonalProd<$be> + $crate::api::LtDiagonalScale,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                K: ::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GetGaloisElement
                    + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::dft::ckks_slots_to_coeffs_assign(self, ct, dft, keys, scratch)
            }

            fn ckks_coeffs_to_slots_split_default<P, Dst, Src, H, K>(
                &self,
                ct_real: &mut Dst,
                ct_imag: &mut Dst,
                ct_in: &Src,
                dft: &$crate::layouts::DFTMatrix<$be, $crate::layouts::Encode, $crate::layouts::Split, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                conj_key: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<()>
            where
                P: ::poulpy_core::default::linear_transformation::DiagonalProd<$be> + $crate::api::LtDiagonalScale,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GetGaloisElement
                    + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::dft::ckks_coeffs_to_slots_split(self, ct_real, ct_imag, ct_in, dft, keys, conj_key, scratch)
            }

            fn ckks_slots_to_coeffs_split_default<P, Dst, Src, H, K>(
                &self,
                op_out: &mut Dst,
                ct_real: &Src,
                ct_imag: &Src,
                dft: &$crate::layouts::DFTMatrix<$be, $crate::layouts::Decode, $crate::layouts::Split, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<()>
            where
                P: ::poulpy_core::default::linear_transformation::DiagonalProd<$be> + $crate::api::LtDiagonalScale,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GetGaloisElement
                    + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::dft::ckks_slots_to_coeffs_split(self, op_out, ct_real, ct_imag, dft, keys, scratch)
            }

            fn ckks_coeffs_to_slots_repack_default<P, Dst, Src, H, K>(
                &self,
                ct_out: &mut Dst,
                ct_in: &Src,
                dft: &$crate::layouts::DFTMatrix<$be, $crate::layouts::Encode, $crate::layouts::Repack, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                conj_key: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<()>
            where
                P: ::poulpy_core::default::linear_transformation::DiagonalProd<$be> + $crate::api::LtDiagonalScale,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GetGaloisElement
                    + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::dft::ckks_coeffs_to_slots_repack(self, ct_out, ct_in, dft, keys, conj_key, scratch)
            }

            fn ckks_slots_to_coeffs_repack_default<P, Dst, Src, H, K>(
                &self,
                op_out: &mut Dst,
                ct_in: &Src,
                dft: &$crate::layouts::DFTMatrix<$be, $crate::layouts::Decode, $crate::layouts::Repack, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) -> $crate::__macro_reexports::anyhow::Result<()>
            where
                P: ::poulpy_core::default::linear_transformation::DiagonalProd<$be> + $crate::api::LtDiagonalScale,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::layouts::GetGaloisElement
                    + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            {
                $crate::default::dft::ckks_slots_to_coeffs_repack(self, op_out, ct_in, dft, keys, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_dft_defaults;
