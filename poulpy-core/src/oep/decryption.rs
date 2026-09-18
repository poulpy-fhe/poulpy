use poulpy_hal::{
    layouts::{Backend, Data, Module, ScratchArena},
    oep::{HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl},
};

use crate::layouts::{
    GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef, SetBase2k,
    prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
};

/// Backend-provided decryption operations.
///
/// # Safety
/// Implementations must interpret ciphertexts, plaintexts, and secrets according to their layout
/// metadata, avoid out-of-bounds or aliased writes, and only use scratch space within the
/// advertised temporary-size contracts.
pub unsafe trait DecryptionImpl: Backend {
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendRef<Self> + GLWEInfos,
        P: GLWEToBackendMut<Self> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt<R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEToBackendRef<Self> + LWEInfos,
        P: LWEPlaintextToBackendMut<Self> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<Self> + LWEInfos;

    fn lwe_matrix_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: LWEMatrixInfos;

    fn lwe_matrix_decrypt<R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEMatrixToBackendRef<Self> + LWEMatrixInfos,
        P: GLWEToBackendMut<Self> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<Self> + LWEInfos;

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<Self>,
        res: &GLWETensor<R, Self::ZnxWord>,
        pt: &mut GLWEPlaintext<P, Self::ZnxWord>,
        sk: &GLWESecretPrepared<S0, Self>,
        sk_tensor: &GLWESecretTensorPrepared<S1, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        GLWETensor<R, Self::ZnxWord>: GLWEToBackendRef<Self> + GLWEInfos,
        GLWEPlaintext<P, Self::ZnxWord>: GLWEToBackendMut<Self> + GLWEInfos + SetBase2k,
        GLWESecretPrepared<S0, Self>: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        GLWESecretTensorPrepared<S1, Self>: GLWESecretTensorPreparedToBackendRef<Self> + GLWEInfos;

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;
}

/// Override surface for the decryption family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`crate::reference::decryption`]
/// for reference algorithms a backend may forward to.
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

/// Implements [`DecryptionReference`] for `Module<$be>` by forwarding every method to
/// the corresponding free function in [`crate::reference::decryption`].
#[macro_export]
macro_rules! impl_decryption_reference_full {
    ($be:ty) => {
        impl $crate::oep::DecryptionReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::decryption::glwe::glwe_decrypt_tmp_bytes_reference::<Self, $be, _>(self, infos)
            }

            fn glwe_decrypt_reference<R, P, S>(
                &self,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                P: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos + $crate::layouts::SetBase2k,
                S: $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::decryption::glwe::glwe_decrypt_reference::<Self, $be, _, _, _>(self, res, pt, sk, scratch)
            }

            fn lwe_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::LWEInfos,
            {
                $crate::reference::decryption::lwe::lwe_decrypt_tmp_bytes_reference::<Self, $be, _>(self, infos)
            }

            fn lwe_decrypt_reference<R, P, S>(
                &self,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
                P: $crate::layouts::LWEPlaintextToBackendMut<$be> + $crate::layouts::SetBase2k + $crate::layouts::LWEInfos,
                S: $crate::layouts::LWESecretToBackendRef<$be> + $crate::layouts::LWEInfos,
            {
                $crate::reference::decryption::lwe::lwe_decrypt_reference::<Self, $be, _, _, _>(self, res, pt, sk, scratch)
            }

            fn lwe_matrix_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::LWEMatrixInfos,
            {
                $crate::reference::decryption::lwe_matrix::lwe_matrix_decrypt_tmp_bytes_reference::<$be, _>(self, infos)
            }

            fn lwe_matrix_decrypt_reference<R, P, S>(
                &self,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEMatrixToBackendRef<$be> + $crate::layouts::LWEMatrixInfos,
                P: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::SetBase2k + $crate::layouts::GLWEInfos,
                S: $crate::layouts::LWESecretToBackendRef<$be> + $crate::layouts::LWEInfos,
            {
                $crate::reference::decryption::lwe_matrix::lwe_matrix_decrypt_reference::<$be, _, _, _>(
                    self, res, pt, sk, scratch,
                )
            }

            fn glwe_tensor_decrypt_reference<
                R: ::poulpy_hal::layouts::Data,
                P: ::poulpy_hal::layouts::Data,
                S0: ::poulpy_hal::layouts::Data,
                S1: ::poulpy_hal::layouts::Data,
            >(
                &self,
                res: &$crate::layouts::GLWETensor<R, <$be as poulpy_hal::layouts::Backend>::ZnxWord>,
                pt: &mut $crate::layouts::GLWEPlaintext<P, <$be as poulpy_hal::layouts::Backend>::ZnxWord>,
                sk: &$crate::layouts::GLWESecretPrepared<S0, $be>,
                sk_tensor: &$crate::layouts::GLWESecretTensorPrepared<S1, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                $crate::layouts::GLWETensor<R, <$be as poulpy_hal::layouts::Backend>::ZnxWord>:
                    $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                $crate::layouts::GLWEPlaintext<P, <$be as poulpy_hal::layouts::Backend>::ZnxWord>:
                    $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos + $crate::layouts::SetBase2k,
                $crate::layouts::GLWESecretPrepared<S0, $be>:
                    $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos,
                $crate::layouts::GLWESecretTensorPrepared<S1, $be>:
                    $crate::layouts::prepared::GLWESecretTensorPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_reference::<Self, $be, R, P, S0, S1>(
                    self, res, pt, sk, sk_tensor, scratch,
                )
            }

            fn glwe_tensor_decrypt_tmp_bytes_reference<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_tmp_bytes_reference::<Self, $be, _>(self, infos)
            }
        }
    };
}

unsafe impl<BE: Backend + HalVecZnxImpl + HalVecZnxBigImpl + HalVecZnxDftImpl + HalSvpImpl> DecryptionImpl for BE
where
    Module<BE>: DecryptionReference<BE>,
{
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::glwe_decrypt_tmp_bytes_reference(module, infos)
    }

    fn glwe_decrypt<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::glwe_decrypt_reference(module, res, pt, sk, scratch)
    }

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::lwe_decrypt_tmp_bytes_reference(module, infos)
    }

    fn lwe_decrypt<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::lwe_decrypt_reference(module, res, pt, sk, scratch)
    }

    fn lwe_matrix_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEMatrixInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::lwe_matrix_decrypt_tmp_bytes_reference(module, infos)
    }

    fn lwe_matrix_decrypt<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendRef<BE> + LWEMatrixInfos,
        P: GLWEToBackendMut<BE> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::lwe_matrix_decrypt_reference(module, res, pt, sk, scratch)
    }

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<BE>,
        res: &GLWETensor<R, BE::ZnxWord>,
        pt: &mut GLWEPlaintext<P, BE::ZnxWord>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R, BE::ZnxWord>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWEPlaintext<P, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
        GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::glwe_tensor_decrypt_reference(module, res, pt, sk, sk_tensor, scratch)
    }

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as DecryptionReference<BE>>::glwe_tensor_decrypt_tmp_bytes_reference(module, infos)
    }
}
