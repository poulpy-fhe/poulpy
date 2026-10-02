use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

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

/// Selects the portable HAL algorithms for this backend operation family.
#[macro_export]
macro_rules! impl_decryption_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::DecryptionImpl for $be {
            fn glwe_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::decryption::glwe::glwe_decrypt_tmp_bytes_reference::<::poulpy_hal::layouts::Module<$be>, $be, _>(module, infos)
            }

            fn glwe_decrypt<R, P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                P: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos + $crate::layouts::SetBase2k,
                S: $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::decryption::glwe::glwe_decrypt_reference::<::poulpy_hal::layouts::Module<$be>, $be, _, _, _>(module, res, pt, sk, scratch)
            }

            fn lwe_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::LWEInfos,
            {
                $crate::reference::decryption::lwe::lwe_decrypt_tmp_bytes_reference::<::poulpy_hal::layouts::Module<$be>, $be, _>(module, infos)
            }

            fn lwe_decrypt<R, P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
                P: $crate::layouts::LWEPlaintextToBackendMut<$be> + $crate::layouts::SetBase2k + $crate::layouts::LWEInfos,
                S: $crate::layouts::LWESecretToBackendRef<$be> + $crate::layouts::LWEInfos,
            {
                $crate::reference::decryption::lwe::lwe_decrypt_reference::<::poulpy_hal::layouts::Module<$be>, $be, _, _, _>(module, res, pt, sk, scratch)
            }

            fn lwe_matrix_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::LWEMatrixInfos,
            {
                $crate::reference::decryption::lwe_matrix::lwe_matrix_decrypt_tmp_bytes_reference::<$be, _>(module, infos)
            }

            fn lwe_matrix_decrypt<R, P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
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
                    module, res, pt, sk, scratch,
                )
            }

            fn glwe_tensor_decrypt<
                R: ::poulpy_hal::layouts::Data,
                P: ::poulpy_hal::layouts::Data,
                S0: ::poulpy_hal::layouts::Data,
                S1: ::poulpy_hal::layouts::Data,
            >(
                module: &::poulpy_hal::layouts::Module<$be>,
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
                $crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_reference::<::poulpy_hal::layouts::Module<$be>, $be, R, P, S0, S1>(
                    module, res, pt, sk, sk_tensor, scratch,
                )
            }

            fn glwe_tensor_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_tmp_bytes_reference::<::poulpy_hal::layouts::Module<$be>, $be, _>(module, infos)
            }
        }
    };
}

// Reference helpers remain available through OEP for source compatibility.
pub use crate::reference::decryption::DecryptionReference;
