use poulpy_core::{
    EncryptionInfos,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    source::Source,
};

use super::derived::tensor_key as derived;
use crate::{layouts::GLWETensorKeyShareOwned, oep::GGLWEPatImpl};

/// # Safety
/// Reproduce the reference share within the queried scratch budget.
pub unsafe trait GLWETensorKeyMHEProtocolImpl: GGLWEPatImpl {
    fn mhe_glwe_tensor_key_share_gen_tmp_bytes<A, B>(module: &Module<Self>, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_tensor_key_share_gen<S, K, E>(
        module: &Module<Self>,
        res: &mut GLWETensorKeyShareOwned<Self>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos;

    fn mhe_glwe_tensor_key_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWETensorKeyShareOwned<Self>,
        a: &GLWETensorKeyShareOwned<Self>,
    ) {
        derived::mhe_glwe_tensor_key_share_aggregate_derived(module, res, a)
    }

    fn mhe_glwe_tensor_key_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_finalize_tmp_bytes(module)
    }

    fn mhe_glwe_tensor_key_share_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWETensorKeyShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
    {
        derived::mhe_glwe_tensor_key_share_finalize_derived(module, res, share, scratch)
    }
}

/// Selects the reference share generation; aggregation and finalization keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_mhe_tensor_key_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWETensorKeyMHEProtocolImpl for $be {
            fn mhe_glwe_tensor_key_share_gen_tmp_bytes<A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &A,
                pk_infos: &B,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWETensorKeyMHEProtocolReference<$be>>::mhe_glwe_tensor_key_share_gen_tmp_bytes_reference(module, res_infos, pk_infos)
            }

            fn mhe_glwe_tensor_key_share_gen<S, K, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWETensorKeyShareOwned<$be>,
                sk: &S,
                pk: &K,
                enc_infos: &E,
                source_xu: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GLWEPublicKeyPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWETensorKeyMHEProtocolReference<$be>>::mhe_glwe_tensor_key_share_gen_reference(module, res, sk, pk, enc_infos, source_xu, source_xe, scratch)
            }
        }
    };
}
