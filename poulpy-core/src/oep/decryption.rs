use poulpy_hal::{
    layouts::{Backend, Data, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena},
    oep::{HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef, SetLWEInfos,
        prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
    },
};

/// Backend-provided decryption operations.
///
/// # Safety
/// Implementations must interpret ciphertexts, plaintexts, and secrets according to their layout
/// metadata, avoid out-of-bounds or aliased writes, and only use scratch space within the
/// advertised temporary-size contracts.
pub unsafe trait DecryptionImpl<BE: Backend>: Backend {
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos;

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<BE>,
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

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;
}

/// Override surface for the decryption family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`decryption_defaults`]
/// for reference algorithms a backend may forward to.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait DecryptionDefault<BE: Backend>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt_default<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    fn lwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt_default<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
        BE: HostBackend,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE::BufRef<'a>: HostDataRef;

    fn glwe_tensor_decrypt_default<R: Data, P: Data, S0: Data, S1: Data>(
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

    fn glwe_tensor_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;
}

/// Implements [`DecryptionDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`decryption_defaults`] free function.
#[macro_export]
macro_rules! impl_decryption_defaults_full {
    ($be:ty) => {
        impl $crate::oep::DecryptionDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                $crate::default::decryption::glwe::glwe_decrypt_tmp_bytes_default::<Self, $be, _>(self, infos)
            }

            fn glwe_decrypt_default<R, P, S>(
                &self,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                P: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos + $crate::layouts::SetLWEInfos,
                S: $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::decryption::glwe::glwe_decrypt_default::<Self, $be, _, _, _>(self, res, pt, sk, scratch)
            }

            fn lwe_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::LWEInfos,
            {
                $crate::default::decryption::lwe::lwe_decrypt_tmp_bytes_default::<Self, $be, _>(self, infos)
            }

            fn lwe_decrypt_default<R, P, S>(
                &self,
                res: &R,
                pt: &mut P,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
                P: $crate::layouts::LWEPlaintextToBackendMut<$be> + $crate::layouts::SetLWEInfos + $crate::layouts::LWEInfos,
                S: $crate::layouts::LWESecretToBackendRef<$be> + $crate::layouts::LWEInfos,
                $be: ::poulpy_hal::layouts::HostBackend,
            {
                $crate::default::decryption::lwe::lwe_decrypt_default::<Self, $be, _, _, _>(self, res, pt, sk, scratch)
            }

            fn glwe_tensor_decrypt_default<
                R: ::poulpy_hal::layouts::Data,
                P: ::poulpy_hal::layouts::Data,
                S0: ::poulpy_hal::layouts::Data,
                S1: ::poulpy_hal::layouts::Data,
            >(
                &self,
                res: &$crate::layouts::GLWETensor<R>,
                pt: &mut $crate::layouts::GLWEPlaintext<P>,
                sk: &$crate::layouts::GLWESecretPrepared<S0, $be>,
                sk_tensor: &$crate::layouts::GLWESecretTensorPrepared<S1, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                $crate::layouts::GLWETensor<R>: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                $crate::layouts::GLWEPlaintext<P>:
                    $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos + $crate::layouts::SetLWEInfos,
                $crate::layouts::GLWESecretPrepared<S0, $be>:
                    $crate::layouts::prepared::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos,
                $crate::layouts::GLWESecretTensorPrepared<S1, $be>:
                    $crate::layouts::prepared::GLWESecretTensorPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::decryption::glwe_tensor::glwe_tensor_decrypt_default::<Self, $be, R, P, S0, S1>(
                    self, res, pt, sk, sk_tensor, scratch,
                )
            }

            fn glwe_tensor_decrypt_tmp_bytes_default<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                $crate::default::decryption::glwe_tensor::glwe_tensor_decrypt_tmp_bytes_default::<Self, $be, _>(self, infos)
            }
        }
    };
}

#[allow(private_bounds)]
unsafe impl<BE: Backend + HostBackend + HalVecZnxImpl<BE> + HalVecZnxBigImpl<BE> + HalVecZnxDftImpl<BE> + HalSvpImpl<BE>>
    DecryptionImpl<BE> for BE
where
    Module<BE>: DecryptionDefault<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as DecryptionDefault<BE>>::glwe_decrypt_tmp_bytes_default(module, infos)
    }

    fn glwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        <Module<BE> as DecryptionDefault<BE>>::glwe_decrypt_default(module, res, pt, sk, scratch)
    }

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        <Module<BE> as DecryptionDefault<BE>>::lwe_decrypt_tmp_bytes_default(module, infos)
    }

    fn lwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
    {
        <Module<BE> as DecryptionDefault<BE>>::lwe_decrypt_default(module, res, pt, sk, scratch)
    }

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<BE>,
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
        <Module<BE> as DecryptionDefault<BE>>::glwe_tensor_decrypt_default(module, res, pt, sk, sk_tensor, scratch)
    }

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as DecryptionDefault<BE>>::glwe_tensor_decrypt_tmp_bytes_default(module, infos)
    }
}
