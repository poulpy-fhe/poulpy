use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWEPreparedToBackendRef, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GGLWEPatOwned;

/// # Safety
/// Reproduce the reference share within the queried scratch budget.
pub unsafe trait GLWETensorKeyShareImpl: Backend {
    fn glwe_tensor_key_share_tmp_bytes<A, B>(module: &Module<Self>, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_key_share<S, K, E>(
        module: &Module<Self>,
        res: &mut GGLWEPatOwned<Self>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretToBackendRef<Self> + GLWEInfos,
        K: GLWEPreparedToBackendRef<Self> + GetDistribution + GLWEInfos,
        E: EncryptionInfos;
}

/// Selects the reference tensor key share.
#[macro_export]
macro_rules! impl_mhe_tensor_key_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWETensorKeyShareImpl for $be {
            fn glwe_tensor_key_share_tmp_bytes<A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &A,
                pk_infos: &B,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWETensorKeyShareReference<$be>>::glwe_tensor_key_share_tmp_bytes_reference(module, res_infos, pk_infos)
            }

            fn glwe_tensor_key_share<S, K, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GGLWEPatOwned<$be>,
                sk: &S,
                pk: &K,
                enc_infos: &E,
                source_xu: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::GetDistribution
                    + ::poulpy_core::layouts::GLWEInfos,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWETensorKeyShareReference<$be>>::glwe_tensor_key_share_reference(module, res, sk, pk, enc_infos, source_xu, source_xe, scratch)
            }
        }
    };
}
