use poulpy_core::layouts::{GLWEInfos, GLWESecretToBackendRef};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::GLWEShamirMHEProtocol,
    layouts::{GLWEShamirLayout, GLWEShamirPolynomialOwned, GLWEShamirShareOwned, GLWEWideSecretOwned},
    oep::GLWEShamirMHEProtocolImpl,
};

impl<BE: Backend + GLWEShamirMHEProtocolImpl> GLWEShamirMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_shamir_polynomial_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize {
        BE::mhe_glwe_shamir_polynomial_gen_tmp_bytes(self, layout)
    }

    fn mhe_glwe_shamir_polynomial_gen<S>(
        &self,
        res: &mut GLWEShamirPolynomialOwned<BE>,
        sk: &S,
        source_xm: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_shamir_polynomial_gen(self, res, sk, source_xm, scratch)
    }

    fn mhe_glwe_shamir_share_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize {
        BE::mhe_glwe_shamir_share_gen_tmp_bytes(self, layout)
    }

    fn mhe_glwe_shamir_share_gen(
        &self,
        res: &mut GLWEShamirShareOwned<BE>,
        poly: &GLWEShamirPolynomialOwned<BE>,
        recipient: u32,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::mhe_glwe_shamir_share_gen(self, res, poly, recipient, scratch)
    }

    fn mhe_glwe_shamir_share_aggregate(&self, res: &mut GLWEShamirShareOwned<BE>, a: &GLWEShamirShareOwned<BE>) {
        BE::mhe_glwe_shamir_share_aggregate(self, res, a)
    }

    fn mhe_glwe_shamir_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_shamir_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_shamir_share_finalize(
        &self,
        res: &mut GLWEWideSecretOwned<BE>,
        share: &GLWEShamirShareOwned<BE>,
        own: u32,
        actives: &[u32],
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::mhe_glwe_shamir_share_finalize(self, res, share, own, actives, scratch)
    }
}
