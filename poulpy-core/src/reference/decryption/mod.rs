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
    GLWEInfos, GLWEMaskToBackendRef, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWEPlaintextToBackendMut, LWESecretToBackendRef,
    LWEToBackendRef, SetBase2k,
    prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, Data, Module, Ring, ScratchArena, Standard};

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait DecryptionReference<BE: Backend, Rg: Ring = Standard> {
    fn glwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt_reference<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        P: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_mask_inner_product_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_mask_inner_product_reference<R, A, S>(&self, res: &mut R, mask: &A, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
        A: GLWEMaskToBackendRef<BE, Rg> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos;

    fn lwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt_reference<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendRef<BE, Rg> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE, Rg> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<BE, Rg> + LWEInfos;

    fn lwe_matrix_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: LWEMatrixInfos;

    fn lwe_matrix_decrypt_reference<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendRef<BE, Rg> + LWEMatrixInfos,
        P: GLWEToBackendMut<BE, Rg> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<BE, Rg> + LWEInfos;

    fn glwe_tensor_decrypt_reference<R: Data, P: Data, S0: Data, S1: Data>(
        &self,
        res: &GLWETensor<R, BE::ZnxWord, Rg>,
        pt: &mut GLWEPlaintext<P, BE::ZnxWord, Rg>,
        sk: &GLWESecretPrepared<S0, BE, Rg>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R, BE::ZnxWord, Rg>: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        GLWEPlaintext<P, BE::ZnxWord, Rg>: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
        GLWESecretPrepared<S0, BE, Rg>: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE, Rg>: GLWESecretTensorPreparedToBackendRef<BE, Rg> + GLWEInfos;

    fn glwe_tensor_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;
}

impl<BE: Backend, Rg: Ring> DecryptionReference<BE, Rg> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: poulpy_hal::api::ModuleN
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::SvpPPolBytesOf
        + crate::api::GLWESecretPreparedFactory<BE, Rg>
        + poulpy_hal::api::VecZnxBigFromSmall<BE, Rg>
        + poulpy_hal::api::VecZnxDftApply<BE, Rg>
        + poulpy_hal::api::SvpApplyDftToDftAssign<BE, Rg>
        + poulpy_hal::api::VecZnxIdftApply<BE, Rg>
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxBigAddAssign<BE, Rg>
        + poulpy_hal::api::VecZnxBigNormalize<BE, Rg>
        + poulpy_hal::api::SvpPPolCopy<BE, Rg>
        + poulpy_hal::api::VecZnxBigScalarProduct<BE, Rg>
        + poulpy_hal::api::VecZnxBigInnerSum<BE, Rg>
        + poulpy_hal::api::VecZnxBigAddSmallAssign<BE, Rg>
        + poulpy_hal::api::VecZnxZero<BE, Rg>
        + poulpy_hal::api::VecZnxBigColWeightedSum<BE, Rg>
        + poulpy_hal::api::VecZnxCopy<BE, Rg>
        + poulpy_hal::api::VecZnxNormalize<BE, Rg>
        + poulpy_hal::api::VecZnxNormalizeTmpBytes
        + crate::api::GLWENormalize<BE, Rg>,
{
    fn glwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe::glwe_decrypt_tmp_bytes_reference::<Self, BE, _, Rg>(self, infos)
    }

    fn glwe_decrypt_reference<R, P, S>(
        &self,
        res: &R,
        pt: &mut P,
        sk: &S,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
        P: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos + crate::layouts::SetBase2k,
        S: crate::layouts::prepared::GLWESecretPreparedToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe::glwe_decrypt_reference::<Self, BE, _, _, _, Rg>(self, res, pt, sk, scratch)
    }

    fn glwe_mask_inner_product_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe::glwe_mask_inner_product_tmp_bytes_reference::<Self, _>(self, infos)
    }

    fn glwe_mask_inner_product_reference<R, A, S>(
        &self,
        res: &mut R,
        mask: &A,
        sk: &S,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos + crate::layouts::SetBase2k,
        A: crate::layouts::GLWEMaskToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
        S: crate::layouts::prepared::GLWESecretPreparedToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe::glwe_mask_inner_product_reference::<Self, BE, _, _, _, Rg>(
            self, res, mask, sk, scratch,
        )
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
        R: crate::layouts::LWEToBackendRef<BE, Rg> + crate::layouts::LWEInfos,
        P: crate::layouts::LWEPlaintextToBackendMut<BE, Rg> + crate::layouts::SetBase2k + crate::layouts::LWEInfos,
        S: crate::layouts::LWESecretToBackendRef<BE, Rg> + crate::layouts::LWEInfos,
    {
        crate::reference::decryption::lwe::lwe_decrypt_reference::<Self, BE, _, _, _, Rg>(self, res, pt, sk, scratch)
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
        R: crate::layouts::LWEMatrixToBackendRef<BE, Rg> + crate::layouts::LWEMatrixInfos,
        P: crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::SetBase2k + crate::layouts::GLWEInfos,
        S: crate::layouts::LWESecretToBackendRef<BE, Rg> + crate::layouts::LWEInfos,
    {
        crate::reference::decryption::lwe_matrix::lwe_matrix_decrypt_reference::<BE, _, _, _, Rg>(self, res, pt, sk, scratch)
    }

    fn glwe_tensor_decrypt_reference<
        R: ::poulpy_hal::layouts::Data,
        P: ::poulpy_hal::layouts::Data,
        S0: ::poulpy_hal::layouts::Data,
        S1: ::poulpy_hal::layouts::Data,
    >(
        &self,
        res: &crate::layouts::GLWETensor<R, <BE as poulpy_hal::layouts::Backend>::ZnxWord, Rg>,
        pt: &mut crate::layouts::GLWEPlaintext<P, <BE as poulpy_hal::layouts::Backend>::ZnxWord, Rg>,
        sk: &crate::layouts::GLWESecretPrepared<S0, BE, Rg>,
        sk_tensor: &crate::layouts::GLWESecretTensorPrepared<S1, BE, Rg>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        crate::layouts::GLWETensor<R, <BE as poulpy_hal::layouts::Backend>::ZnxWord, Rg>:
            crate::layouts::GLWEToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
        crate::layouts::GLWEPlaintext<P, <BE as poulpy_hal::layouts::Backend>::ZnxWord, Rg>:
            crate::layouts::GLWEToBackendMut<BE, Rg> + crate::layouts::GLWEInfos + crate::layouts::SetBase2k,
        crate::layouts::GLWESecretPrepared<S0, BE, Rg>:
            crate::layouts::prepared::GLWESecretPreparedToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
        crate::layouts::GLWESecretTensorPrepared<S1, BE, Rg>:
            crate::layouts::prepared::GLWESecretTensorPreparedToBackendRef<BE, Rg> + crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_reference::<Self, BE, R, P, S0, S1, Rg>(
            self, res, pt, sk, sk_tensor, scratch,
        )
    }

    fn glwe_tensor_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: crate::layouts::GLWEInfos,
    {
        crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_tmp_bytes_reference::<Self, BE, _, Rg>(self, infos)
    }
}
