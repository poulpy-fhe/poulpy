use poulpy_hal::layouts::{Backend, Data, Module, Ring, ScratchArena};

use crate::{
    api::{GLWEDecrypt, GLWEMaskInnerProduct, GLWETensorDecrypt, LWEDecrypt, LWEMatrixDecrypt},
    layouts::{
        GLWEInfos, GLWEMaskToBackendRef, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWEPlaintextToBackendMut,
        LWESecretToBackendRef, LWEToBackendRef, SetBase2k,
        prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
    },
    oep::DecryptionImpl,
};

macro_rules! impl_decryption_delegate {
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE, Rg: Ring> $trait for Module<BE>
        where
            BE: Backend + DecryptionImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl_decryption_delegate!(
    GLWEDecrypt<BE, Rg>,
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_decrypt_tmp_bytes(self, infos)
    },
    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        P: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_decrypt(self, res, pt, sk, scratch)
    }
);

impl_decryption_delegate!(
    GLWEMaskInnerProduct<BE, Rg>,
    fn glwe_mask_inner_product_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_mask_inner_product_tmp_bytes(self, infos)
    },
    fn glwe_mask_inner_product<R, A, S>(&self, res: &mut R, mask: &A, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
        A: GLWEMaskToBackendRef<BE, Rg> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_mask_inner_product(self, res, mask, sk, scratch)
    }
);

impl_decryption_delegate!(
    LWEDecrypt<BE, Rg>,
    fn lwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendRef<BE, Rg> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE, Rg> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<BE, Rg> + LWEInfos,
    {
        BE::lwe_decrypt(self, res, pt, sk, scratch)
    },
    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        BE::lwe_decrypt_tmp_bytes(self, infos)
    }
);

impl_decryption_delegate!(
    LWEMatrixDecrypt<BE, Rg>,
    fn lwe_matrix_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEMatrixInfos,
    {
        BE::lwe_matrix_decrypt_tmp_bytes(self, infos)
    },
    fn lwe_matrix_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendRef<BE, Rg> + LWEMatrixInfos,
        P: GLWEToBackendMut<BE, Rg> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<BE, Rg> + LWEInfos,
    {
        BE::lwe_matrix_decrypt(self, res, pt, sk, scratch)
    }
);

impl_decryption_delegate!(
    GLWETensorDecrypt<BE, Rg>,
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_tensor_decrypt_tmp_bytes(self, infos)
    },
    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
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
        GLWESecretTensorPrepared<S1, BE, Rg>: GLWESecretTensorPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_tensor_decrypt(self, res, pt, sk, sk_tensor, scratch)
    }
);
