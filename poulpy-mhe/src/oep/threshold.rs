use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{
        GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWESecretToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    layouts::{
        GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned, GLWEShamirLayout, GLWEShamirPolynomialOwned,
        GLWEShamirShareOwned, GLWEWideSecretOwned, GLWEWideSecretPreparedOwned,
    },
    oep::{GLWEKeyswitchMHEProtocolImpl, GLWEPublicKeyswitchMHEProtocolImpl},
};

/// # Safety
/// Reproduce the reference polynomial, shares, aggregation and additive share
/// within the queried scratch budgets.
pub unsafe trait GLWEShamirMHEProtocolImpl: Backend {
    fn mhe_glwe_shamir_polynomial_gen_tmp_bytes(module: &Module<Self>, layout: &GLWEShamirLayout) -> usize;

    fn mhe_glwe_shamir_polynomial_gen<S>(
        module: &Module<Self>,
        res: &mut GLWEShamirPolynomialOwned<Self>,
        sk: &S,
        source_xm: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretToBackendRef<Self> + GLWEInfos;

    fn mhe_glwe_shamir_share_gen_tmp_bytes(module: &Module<Self>, layout: &GLWEShamirLayout) -> usize;

    fn mhe_glwe_shamir_share_gen(
        module: &Module<Self>,
        res: &mut GLWEShamirShareOwned<Self>,
        poly: &GLWEShamirPolynomialOwned<Self>,
        recipient: u32,
        scratch: &mut ScratchArena<'_, Self>,
    );

    fn mhe_glwe_shamir_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEShamirShareOwned<Self>,
        a: &GLWEShamirShareOwned<Self>,
    );

    fn mhe_glwe_shamir_share_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn mhe_glwe_shamir_share_finalize(
        module: &Module<Self>,
        res: &mut GLWEWideSecretOwned<Self>,
        share: &GLWEShamirShareOwned<Self>,
        own: u32,
        actives: &[u32],
        scratch: &mut ScratchArena<'_, Self>,
    );
}

/// # Safety
/// Reproduce the reference digit secrets within the queried scratch budget.
pub unsafe trait GLWEWideSecretPrepareImpl: Backend {
    fn mhe_glwe_wide_secret_prepare_tmp_bytes(module: &Module<Self>, layout: &GLWEShamirLayout) -> usize;

    fn mhe_glwe_wide_secret_prepare(
        module: &Module<Self>,
        res: &mut GLWEWideSecretPreparedOwned<Self>,
        sk: &GLWEWideSecretOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

/// # Safety
/// Reproduce the reference share within the queried scratch budget.
pub unsafe trait GLWEThresholdKeyswitchMHEProtocolImpl: GLWEKeyswitchMHEProtocolImpl {
    fn mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_threshold_keyswitch_share_gen<C, S, E>(
        module: &Module<Self>,
        res: &mut GLWEKeyswitchShareOwned<Self>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<Self>,
        sk_out: &S,
        flood: &E,
        failure_bits: usize,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        C: GLWEToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        E: SmudgingInfos;

    fn mhe_glwe_threshold_keyswitch_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEKeyswitchShareOwned<Self>,
        a: &GLWEKeyswitchShareOwned<Self>,
    ) {
        Self::mhe_glwe_keyswitch_share_aggregate(module, res, a)
    }

    fn mhe_glwe_threshold_keyswitch_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::mhe_glwe_keyswitch_share_finalize_tmp_bytes(module)
    }

    fn mhe_glwe_threshold_keyswitch_share_finalize<R, C>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos,
    {
        Self::mhe_glwe_keyswitch_share_finalize(module, res, ct, share, scratch)
    }
}

/// # Safety
/// Reproduce the reference share within the queried scratch budget.
pub unsafe trait GLWEThresholdPublicKeyswitchMHEProtocolImpl: GLWEPublicKeyswitchMHEProtocolImpl {
    fn mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes<A, B, P>(
        module: &Module<Self>,
        ct_infos: &A,
        res_infos: &B,
        pk_infos: &P,
    ) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_threshold_public_keyswitch_share_gen<C, K, E1, E2>(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyswitchShareOwned<Self>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<Self>,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        failure_bits: usize,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        C: GLWEToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;

    fn mhe_glwe_threshold_public_keyswitch_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyswitchShareOwned<Self>,
        a: &GLWEPublicKeyswitchShareOwned<Self>,
    ) {
        Self::mhe_glwe_public_keyswitch_share_aggregate(module, res, a)
    }

    fn mhe_glwe_threshold_public_keyswitch_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(module)
    }

    fn mhe_glwe_threshold_public_keyswitch_share_finalize<R, C>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &GLWEPublicKeyswitchShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos,
    {
        Self::mhe_glwe_public_keyswitch_share_finalize(module, res, ct, share, scratch)
    }
}

/// Selects the reference Shamir thresholdization, digit preparation and
/// threshold key switching shares. The finalization and the preparation read
/// and write limbs on the host, so `$be` must have host-readable buffers.
#[macro_export]
macro_rules! impl_mhe_threshold_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEShamirMHEProtocolImpl for $be {
            fn mhe_glwe_shamir_polynomial_gen_tmp_bytes(
                module: &::poulpy_hal::layouts::Module<$be>,
                layout: &$crate::layouts::GLWEShamirLayout,
            ) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShamirMHEProtocolReference<$be>>::mhe_glwe_shamir_polynomial_gen_tmp_bytes_reference(module, layout)
            }

            fn mhe_glwe_shamir_polynomial_gen<S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEShamirPolynomialOwned<$be>,
                sk: &S,
                source_xm: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShamirMHEProtocolReference<$be>>::mhe_glwe_shamir_polynomial_gen_reference(module, res, sk, source_xm, scratch)
            }

            fn mhe_glwe_shamir_share_gen_tmp_bytes(
                module: &::poulpy_hal::layouts::Module<$be>,
                layout: &$crate::layouts::GLWEShamirLayout,
            ) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShamirMHEProtocolReference<$be>>::mhe_glwe_shamir_share_gen_tmp_bytes_reference(module, layout)
            }

            fn mhe_glwe_shamir_share_gen(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEShamirShareOwned<$be>,
                poly: &$crate::layouts::GLWEShamirPolynomialOwned<$be>,
                recipient: u32,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShamirMHEProtocolReference<$be>>::mhe_glwe_shamir_share_gen_reference(module, res, poly, recipient, scratch)
            }

            fn mhe_glwe_shamir_share_aggregate(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEShamirShareOwned<$be>,
                a: &$crate::layouts::GLWEShamirShareOwned<$be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShamirMHEProtocolReference<$be>>::mhe_glwe_shamir_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_shamir_share_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShamirMHEProtocolReference<$be>>::mhe_glwe_shamir_share_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_shamir_share_finalize(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEWideSecretOwned<$be>,
                share: &$crate::layouts::GLWEShamirShareOwned<$be>,
                own: u32,
                actives: &[u32],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShamirMHEProtocolReference<$be>>::mhe_glwe_shamir_share_finalize_reference(module, res, share, own, actives, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEWideSecretPrepareImpl for $be {
            fn mhe_glwe_wide_secret_prepare_tmp_bytes(
                module: &::poulpy_hal::layouts::Module<$be>,
                layout: &$crate::layouts::GLWEShamirLayout,
            ) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEWideSecretPrepareReference<$be>>::mhe_glwe_wide_secret_prepare_tmp_bytes_reference(module, layout)
            }

            fn mhe_glwe_wide_secret_prepare(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEWideSecretPreparedOwned<$be>,
                sk: &$crate::layouts::GLWEWideSecretOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEWideSecretPrepareReference<$be>>::mhe_glwe_wide_secret_prepare_reference(module, res, sk, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEThresholdKeyswitchMHEProtocolImpl for $be {
            fn mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEThresholdKeyswitchMHEProtocolReference<$be>>::mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_glwe_threshold_keyswitch_share_gen<C, S, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEKeyswitchShareOwned<$be>,
                ct: &C,
                sk_in: &$crate::layouts::GLWEWideSecretPreparedOwned<$be>,
                sk_out: &S,
                flood: &E,
                failure_bits: usize,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                E: ::poulpy_core::SmudgingInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEThresholdKeyswitchMHEProtocolReference<$be>>::mhe_glwe_threshold_keyswitch_share_gen_reference(module, res, ct, sk_in, sk_out, flood, failure_bits, source_xe, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEThresholdPublicKeyswitchMHEProtocolImpl for $be {
            fn mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes<A, B, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_infos: &A,
                res_infos: &B,
                pk_infos: &P,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEThresholdPublicKeyswitchMHEProtocolReference<$be>>::mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes_reference(module, ct_infos, res_infos, pk_infos)
            }

            fn mhe_glwe_threshold_public_keyswitch_share_gen<C, K, E1, E2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEPublicKeyswitchShareOwned<$be>,
                ct: &C,
                sk_in: &$crate::layouts::GLWEWideSecretPreparedOwned<$be>,
                pk_out: &K,
                flood: &E1,
                enc_infos: &E2,
                failure_bits: usize,
                source_xu: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GLWEPublicKeyPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                E1: ::poulpy_core::SmudgingInfos,
                E2: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEThresholdPublicKeyswitchMHEProtocolReference<$be>>::mhe_glwe_threshold_public_keyswitch_share_gen_reference(module, res, ct, sk_in, pk_out, flood, enc_infos, failure_bits, source_xu, source_xe, scratch)
            }
        }
    };
}
