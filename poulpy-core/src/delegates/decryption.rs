use poulpy_hal::layouts::{Backend, Data, HostBackend, Module, ScratchArena};

use crate::{
    api::{GLWEDecrypt, GLWETensorDecrypt, LWEDecrypt},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef, SetLWEInfos,
        prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
    },
    oep::DecryptionImpl,
};

macro_rules! impl_decryption_delegate {
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + HostBackend + DecryptionImpl<BE>,
        {
            $($body)+
        }
    };
}

impl_decryption_delegate!(
    GLWEDecrypt<BE>,
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_decrypt_tmp_bytes(self, infos)
    },
    fn glwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
    {
        BE::glwe_decrypt(self, res, pt, sk, scratch)
    }
);

impl_decryption_delegate!(
    LWEDecrypt<BE>,
    fn lwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
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
    GLWETensorDecrypt<BE>,
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_tensor_decrypt_tmp_bytes(self, infos)
    },
    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWEPlaintext<P>: GLWEToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_tensor_decrypt(self, res, pt, sk, sk_tensor, scratch)
    }
);
