//! Decryption of ciphertexts using secret keys.
//!
//! This module provides traits and implementations for decrypting
//! lattice-based ciphertexts back into plaintexts:
//!
//! - [`GLWEDecrypt`]: decrypts a GLWE ciphertext using a prepared GLWE secret key.
//! - [`GLWETensorDecrypt`](crate::api::GLWETensorDecrypt): decrypts a GLWE tensor ciphertext using both a
//!   standard GLWE secret key and a tensor secret key.
//! - [`LWEDecrypt`](crate::api::LWEDecrypt): decrypts an LWE ciphertext using an LWE secret key.
//!
//! Each trait exposes a scratch-bytes query method and a decryption method.
//! Scratch space must be pre-allocated by the caller using the corresponding
//! `*_tmp_bytes` function.

pub mod glwe;
pub mod glwe_tensor;
pub mod lwe;
pub mod lwe_matrix;

pub(crate) use glwe::glwe_decrypt_backend_inner;
pub use glwe::*;

use crate::layouts::{
    GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef, SetBase2k,
    prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait DecryptionReference<BE: Backend> {
    fn glwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt_reference<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    fn lwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt_reference<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos;

    fn lwe_matrix_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: LWEMatrixInfos;

    fn lwe_matrix_decrypt_reference<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendRef<BE> + LWEMatrixInfos,
        P: GLWEToBackendMut<BE> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos;

    fn glwe_tensor_decrypt_reference<R: Data, P: Data, S0: Data, S1: Data>(
        &self,
        res: &GLWETensor<R, BE::ZnxWord>,
        pt: &mut GLWEPlaintext<P, BE::ZnxWord>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R, BE::ZnxWord>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWEPlaintext<P, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
        GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos;

    fn glwe_tensor_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;
}

impl<BE: Backend> DecryptionReference<BE> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: poulpy_hal::api::ModuleN
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::SvpPPolBytesOf
        + crate::layouts::prepared::GLWESecretPreparedFactory<BE>
        + poulpy_hal::api::VecZnxBigFromSmall<BE>
        + poulpy_hal::api::VecZnxDftApply<BE>
        + poulpy_hal::api::SvpApplyDftToDftAssign<BE>
        + poulpy_hal::api::VecZnxIdftApplyTmpA<BE>
        + poulpy_hal::api::VecZnxBigAddAssign<BE>
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::SvpPPolCopy<BE>
        + poulpy_hal::api::VecZnxScalarProduct<BE>
        + poulpy_hal::api::VecZnxBigInnerSum<BE>
        + poulpy_hal::api::VecZnxBigAddSmallAssign<BE>
        + poulpy_hal::api::VecZnxZero<BE>
        + poulpy_hal::api::VecZnxBigColWeightedSum<BE>
        + poulpy_hal::api::VecZnxCopy<BE>,
{
    fn glwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe::glwe_decrypt_tmp_bytes_reference::<Self, BE, _>(self, infos)
    }

    fn glwe_decrypt_reference<R, P, S>(
        &self,
        res: &R,
        pt: &mut P,
        sk: &S,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
        P: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos + crate::layouts::SetBase2k,
        S: crate::layouts::prepared::GLWESecretPreparedToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe::glwe_decrypt_reference::<Self, BE, _, _, _>(self, res, pt, sk, scratch)
    }

    fn lwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: crate::layouts::LWEInfos,
    {
        crate::reference::decryption::lwe::lwe_decrypt_tmp_bytes_reference::<Self, BE, _>(self, infos)
    }

    fn lwe_decrypt_reference<R, P, S>(
        &self,
        res: &R,
        pt: &mut P,
        sk: &S,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::LWEToBackendRef<BE> + crate::layouts::LWEInfos,
        P: crate::layouts::LWEPlaintextToBackendMut<BE> + crate::layouts::SetBase2k + crate::layouts::LWEInfos,
        S: crate::layouts::LWESecretToBackendRef<BE> + crate::layouts::LWEInfos,
    {
        crate::reference::decryption::lwe::lwe_decrypt_reference::<Self, BE, _, _, _>(self, res, pt, sk, scratch)
    }

    fn lwe_matrix_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: crate::layouts::LWEMatrixInfos,
    {
        crate::reference::decryption::lwe_matrix::lwe_matrix_decrypt_tmp_bytes_reference::<BE, _>(self, infos)
    }

    fn lwe_matrix_decrypt_reference<R, P, S>(
        &self,
        res: &R,
        pt: &mut P,
        sk: &S,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::LWEMatrixToBackendRef<BE> + crate::layouts::LWEMatrixInfos,
        P: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::SetBase2k + crate::layouts::GLWEInfos,
        S: crate::layouts::LWESecretToBackendRef<BE> + crate::layouts::LWEInfos,
    {
        crate::reference::decryption::lwe_matrix::lwe_matrix_decrypt_reference::<BE, _, _, _>(self, res, pt, sk, scratch)
    }

    fn glwe_tensor_decrypt_reference<
        R: ::poulpy_hal::layouts::Data,
        P: ::poulpy_hal::layouts::Data,
        S0: ::poulpy_hal::layouts::Data,
        S1: ::poulpy_hal::layouts::Data,
    >(
        &self,
        res: &crate::layouts::GLWETensor<R, <BE as poulpy_hal::layouts::Backend>::ZnxWord>,
        pt: &mut crate::layouts::GLWEPlaintext<P, <BE as poulpy_hal::layouts::Backend>::ZnxWord>,
        sk: &crate::layouts::GLWESecretPrepared<S0, BE>,
        sk_tensor: &crate::layouts::GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        crate::layouts::GLWETensor<R, <BE as poulpy_hal::layouts::Backend>::ZnxWord>:
            crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
        crate::layouts::GLWEPlaintext<P, <BE as poulpy_hal::layouts::Backend>::ZnxWord>:
            crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos + crate::layouts::SetBase2k,
        crate::layouts::GLWESecretPrepared<S0, BE>:
            crate::layouts::prepared::GLWESecretPreparedToBackendRef<BE> + crate::layouts::GLWEInfos,
        crate::layouts::GLWESecretTensorPrepared<S1, BE>:
            crate::layouts::prepared::GLWESecretTensorPreparedToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_reference::<Self, BE, R, P, S0, S1>(
            self, res, pt, sk, sk_tensor, scratch,
        )
    }

    fn glwe_tensor_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_tmp_bytes_reference::<Self, BE, _>(self, infos)
    }
}
