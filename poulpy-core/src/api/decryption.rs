use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::layouts::{
    GLWEInfos, GLWEMaskToBackendRef, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWEPlaintextToBackendMut, LWESecretToBackendRef,
    LWEToBackendRef, SetBase2k,
    prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
};

pub trait GLWEDecrypt<BE: Backend> {
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Decrypts a GLWE ciphertext into the destination plaintext.
    ///
    /// # Panics
    /// Panics if the ciphertext, plaintext and prepared secret key degrees differ
    /// or exceed the module degree, including in release builds.
    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;
}

/// Inner product of a GLWE mask with a secret, `Σ_j a_j·s_j`: the phase of a
/// ciphertext without its body.
pub trait GLWEMaskInnerProduct<BE: Backend> {
    fn glwe_mask_inner_product_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes into `res` the inner product of `mask`, a [`GLWEMask`](crate::layouts::GLWEMask)
    /// or the mask of a [`GLWE`](crate::layouts::GLWE), with `sk`, normalized at the
    /// precision of `res`.
    ///
    /// # Panics
    /// Panics if the mask is not canonical, if the output, mask and secret degrees
    /// differ or exceed the module degree, or if the mask rank is zero or differs
    /// from the secret's.
    fn glwe_mask_inner_product<R, A, S>(&self, res: &mut R, mask: &A, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
        A: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;
}

pub trait LWEDecrypt<BE: Backend> {
    fn lwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos;

    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;
}

pub trait LWEMatrixDecrypt<BE: Backend> {
    fn lwe_matrix_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEMatrixInfos;

    fn lwe_matrix_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendRef<BE> + LWEMatrixInfos,
        P: GLWEToBackendMut<BE> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos;
}

pub trait GLWETensorDecrypt<BE: Backend> {
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
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
}
