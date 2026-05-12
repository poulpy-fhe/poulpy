use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::layouts::{
    GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef, SetLWEInfos,
    prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
};

pub trait GLWEDecrypt<BE: Backend> {
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>;
}

pub trait LWEDecrypt<BE: Backend> {
    fn lwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>;

    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;
}

pub trait GLWETensorDecrypt<BE: Backend> {
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

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
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos;
}
