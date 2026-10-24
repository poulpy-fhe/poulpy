use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretToBackendRef};
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    source::Source,
};

use super::derived::tensor_key as derived;
use crate::{layouts::GLWETensorKeyShareOwned, oep::GGLWEPatImpl};

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns. The
/// synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md`
/// apply.
pub unsafe trait GLWETensorKeyMHEProtocolImpl: GGLWEPatImpl {
    /// Upper bound on the scratch of `mhe_glwe_tensor_key_share_gen`, as `GLWETensorKeyMHEProtocolReference::mhe_glwe_tensor_key_share_gen_tmp_bytes_reference`; panics unless `res_infos` has the module's degree.
    fn mhe_glwe_tensor_key_share_gen_tmp_bytes<A, B>(module: &Module<Self>, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos;

    /// Reproduces `GLWETensorKeyMHEProtocolReference::mhe_glwe_tensor_key_share_gen_reference`: public-key encryptions of the products of `sk` entries under `pk`, ephemerals from `source_xu`, errors from `source_xe`.
    /// Panics unless `res`, `sk` and `pk` have the module's degree, the key layout matches its rank, `pk` has the key's rank and radix, is at least as precise and has a samplable distribution consistent with its provenance.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_tensor_key_share_gen<S, K>(
        module: &Module<Self>,
        res: &mut GLWETensorKeyShareOwned<Self>,
        sk: &S,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos;

    /// Default body `mhe_glwe_tensor_key_share_aggregate_derived`: the `GGLWEPat` aggregation of the inner shares.
    fn mhe_glwe_tensor_key_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWETensorKeyShareOwned<Self>,
        a: &GLWETensorKeyShareOwned<Self>,
    ) {
        derived::mhe_glwe_tensor_key_share_aggregate_derived(module, res, a)
    }

    /// Default body: the scratch of the `GGLWEPat` finalization.
    fn mhe_glwe_tensor_key_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_finalize_tmp_bytes(module)
    }

    /// Default body `mhe_glwe_tensor_key_share_finalize_derived`: the `GGLWEPat` finalization of the inner share.
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
macro_rules! impl_glwe_tensor_key_mhe_protocol_reference_full {
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

            fn mhe_glwe_tensor_key_share_gen<S, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWETensorKeyShareOwned<$be>,
                sk: &S,
                pk: &K,
                source_xu: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GLWEPublicKeyPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWETensorKeyMHEProtocolReference<$be>>::mhe_glwe_tensor_key_share_gen_reference(module, res, sk, pk, source_xu, source_xe, scratch)
            }
        }
    };
}
