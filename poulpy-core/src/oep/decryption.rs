use poulpy_hal::layouts::{Backend, Data, Module, Ring, ScratchArena, Standard};

use crate::layouts::{
    GLWEInfos, GLWEMaskToBackendRef, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWEPlaintextToBackendMut, LWESecretToBackendRef,
    LWEToBackendRef, SetBase2k,
    prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
};

/// Backend-provided decryption operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method taking scratch wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait DecryptionImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::glwe_decrypt`]; may report less than the reference, never less than the body takes.
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces [`glwe_decrypt_reference`](crate::reference::decryption::glwe::glwe_decrypt_reference) bit for bit
    /// under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within [`Self::glwe_decrypt_tmp_bytes`]; free in
    /// kernels and scratch use before the wipe.
    fn glwe_decrypt<R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendRef<Self, Rg> + GLWEInfos,
        P: GLWEToBackendMut<Self, Rg> + GLWEInfos + SetBase2k,
        S: GLWESecretPreparedToBackendRef<Self, Rg> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_mask_inner_product`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_mask_inner_product_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces
    /// [`glwe_mask_inner_product_reference`](crate::reference::decryption::glwe::glwe_mask_inner_product_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mask_inner_product_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_mask_inner_product<R, A, S>(
        module: &Module<Self>,
        res: &mut R,
        mask: &A,
        sk: &S,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos + SetBase2k,
        A: GLWEMaskToBackendRef<Self, Rg> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self, Rg> + GLWEInfos;

    /// Scratch bound of [`Self::lwe_decrypt`]; may report less than the reference, never less than the body takes.
    fn lwe_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: LWEInfos;

    /// Reproduces [`lwe_decrypt_reference`](crate::reference::decryption::lwe::lwe_decrypt_reference) bit for bit,
    /// within [`Self::lwe_decrypt_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn lwe_decrypt<R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEToBackendRef<Self, Rg> + LWEInfos,
        P: LWEPlaintextToBackendMut<Self, Rg> + SetBase2k + LWEInfos,
        S: LWESecretToBackendRef<Self, Rg> + LWEInfos;

    /// Scratch bound of [`Self::lwe_matrix_decrypt`]; may report less than the reference, never less than the body
    /// takes.
    fn lwe_matrix_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: LWEMatrixInfos;

    /// Reproduces
    /// [`lwe_matrix_decrypt_reference`](crate::reference::decryption::lwe_matrix::lwe_matrix_decrypt_reference) bit
    /// for bit, within [`Self::lwe_matrix_decrypt_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn lwe_matrix_decrypt<R, P, S>(module: &Module<Self>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEMatrixToBackendRef<Self, Rg> + LWEMatrixInfos,
        P: GLWEToBackendMut<Self, Rg> + SetBase2k + GLWEInfos,
        S: LWESecretToBackendRef<Self, Rg> + LWEInfos;

    /// Reproduces
    /// [`glwe_tensor_decrypt_reference`](crate::reference::decryption::glwe_tensor::glwe_tensor_decrypt_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_tensor_decrypt_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<Self>,
        res: &GLWETensor<R, Self::ZnxWord, Rg>,
        pt: &mut GLWEPlaintext<P, Self::ZnxWord, Rg>,
        sk: &GLWESecretPrepared<S0, Self, Rg>,
        sk_tensor: &GLWESecretTensorPrepared<S1, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        GLWETensor<R, Self::ZnxWord, Rg>: GLWEToBackendRef<Self, Rg> + GLWEInfos,
        GLWEPlaintext<P, Self::ZnxWord, Rg>: GLWEToBackendMut<Self, Rg> + GLWEInfos + SetBase2k,
        GLWESecretPrepared<S0, Self, Rg>: GLWESecretPreparedToBackendRef<Self, Rg> + GLWEInfos,
        GLWESecretTensorPrepared<S1, Self, Rg>: GLWESecretTensorPreparedToBackendRef<Self, Rg> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_tensor_decrypt`]; may report less than the reference, never less than the body
    /// takes.
    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;
}

/// Selects the portable HAL algorithms for this backend operation family.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_decryption_reference_full {
    ($be:ty) => {
        $crate::impl_decryption_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_decryption_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::DecryptionImpl<$ring> for $be {
            fn glwe_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::glwe_decrypt_tmp_bytes_reference(module, infos)
            }

            fn glwe_decrypt<R, P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
                P: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos + $crate::layouts::SetBase2k,
                S: $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::glwe_decrypt_reference(module, res, pt, sk, scratch)
            }

            fn glwe_mask_inner_product_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::glwe_mask_inner_product_tmp_bytes_reference(module, infos)
            }

            fn glwe_mask_inner_product<R, A, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                mask: &A,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos + $crate::layouts::SetBase2k,
                A: $crate::layouts::GLWEMaskToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
                S: $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::glwe_mask_inner_product_reference(module, res, mask, sk, scratch)
            }

            fn lwe_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::LWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::lwe_decrypt_tmp_bytes_reference(module, infos)
            }

            fn lwe_decrypt<R, P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendRef<$be, $ring> + $crate::layouts::LWEInfos,
                P: $crate::layouts::LWEPlaintextToBackendMut<$be, $ring> + $crate::layouts::SetBase2k + $crate::layouts::LWEInfos,
                S: $crate::layouts::LWESecretToBackendRef<$be, $ring> + $crate::layouts::LWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::lwe_decrypt_reference(module, res, pt, sk, scratch)
            }

            fn lwe_matrix_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::LWEMatrixInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::lwe_matrix_decrypt_tmp_bytes_reference(module, infos)
            }

            fn lwe_matrix_decrypt<R, P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEMatrixToBackendRef<$be, $ring> + $crate::layouts::LWEMatrixInfos,
                P: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::SetBase2k + $crate::layouts::GLWEInfos,
                S: $crate::layouts::LWESecretToBackendRef<$be, $ring> + $crate::layouts::LWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::lwe_matrix_decrypt_reference(
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
                res: &$crate::layouts::GLWETensor<R, <$be as poulpy_hal::layouts::Backend>::ZnxWord, $ring>,
                pt: &mut $crate::layouts::GLWEPlaintext<P, <$be as poulpy_hal::layouts::Backend>::ZnxWord, $ring>,
                sk: &$crate::layouts::GLWESecretPrepared<S0, $be, $ring>,
                sk_tensor: &$crate::layouts::GLWESecretTensorPrepared<S1, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                $crate::layouts::GLWETensor<R, <$be as poulpy_hal::layouts::Backend>::ZnxWord, $ring>:
                    $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
                $crate::layouts::GLWEPlaintext<P, <$be as poulpy_hal::layouts::Backend>::ZnxWord, $ring>:
                    $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos + $crate::layouts::SetBase2k,
                $crate::layouts::GLWESecretPrepared<S0, $be, $ring>:
                    $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
                $crate::layouts::GLWESecretTensorPrepared<S1, $be, $ring>:
                    $crate::layouts::prepared::GLWESecretTensorPreparedToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::glwe_tensor_decrypt_reference(
                    module, res, pt, sk, sk_tensor, scratch,
                )
            }

            fn glwe_tensor_decrypt_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decryption::DecryptionReference<$be, $ring>>::glwe_tensor_decrypt_tmp_bytes_reference(module, infos)
            }
        }
    };
}
