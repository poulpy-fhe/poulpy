use poulpy_hal::layouts::{Backend, Data, Ring, ScratchArena, Standard};

use crate::layouts::{
    GLWEInfos, GLWEMaskToBackendRef, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWEPlaintextToBackendMut, LWESecretToBackendRef,
    LWEToBackendRef, SetBase2k,
    prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
};

pub trait GLWEDecrypt<BE: Backend, Rg: Ring = Standard> {
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Decrypts a GLWE ciphertext into the destination plaintext.
    ///
    /// # Panics
    ///
    /// If the ciphertext, plaintext and prepared secret key degrees differ
    /// or exceed the module degree, including in release builds.
    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        P: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos;
}

/// Inner product of a GLWE mask with a secret, `Σ_j a_j·s_j`: the phase of a
/// ciphertext without its body.
pub trait GLWEMaskInnerProduct<BE: Backend, Rg: Ring = Standard> {
    fn glwe_mask_inner_product_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Writes into `res` the inner product of `mask`, a [`GLWEMask`](crate::layouts::GLWEMask)
    /// or the mask of a [`GLWE`](crate::layouts::GLWE), with `sk`, normalized at the
    /// precision of `res`.
    ///
    /// # Panics
    ///
    /// If the mask is not canonical, if the output, mask and secret degrees
    /// differ or exceed the module degree, or if the mask rank is zero or differs
    /// from the secret's.
    fn glwe_mask_inner_product<R, A, S>(&self, res: &mut R, mask: &A, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos + SetBase2k,
        A: GLWEMaskToBackendRef<BE, Rg> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos;
}

pub trait LWEDecrypt<BE: Backend, Rg: Ring = Standard> {
    fn lwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendRef<BE, Rg> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE, Rg> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<BE, Rg> + LWEInfos;

    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;
}

pub trait LWEMatrixDecrypt<BE: Backend, Rg: Ring = Standard> {
    fn lwe_matrix_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEMatrixInfos;

    fn lwe_matrix_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendRef<BE, Rg> + LWEMatrixInfos,
        P: GLWEToBackendMut<BE, Rg> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<BE, Rg> + LWEInfos;
}

pub trait GLWETensorDecrypt<BE: Backend, Rg: Ring = Standard> {
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

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
        GLWESecretTensorPrepared<S1, BE, Rg>: GLWESecretTensorPreparedToBackendRef<BE, Rg> + GLWEInfos;
}
