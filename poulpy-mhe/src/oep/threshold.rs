use poulpy_core::layouts::{GLWEInfos, GLWESecretToBackendRef};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEShamirLayout, GLWEShamirPolynomialOwned, GLWEShamirShareOwned, GLWEWideSecretOwned};

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

/// Selects the reference Shamir thresholdization. The finalization reads and
/// writes limbs on the host, so `$be` must have host-readable buffers.
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
    };
}
