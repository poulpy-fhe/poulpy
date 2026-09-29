use poulpy_core::{
    EncryptionInfos, SmudgingNoise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::GLWERefreshMHEProtocol, layouts::GLWERefreshShareOwned, oep::GLWERefreshMHEProtocolImpl};

impl<BE: Backend + GLWERefreshMHEProtocolImpl> GLWERefreshMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_refresh_share_gen_tmp_bytes<A, B>(&self, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::mhe_glwe_refresh_share_gen_tmp_bytes(self, ct_infos, res_infos)
    }

    fn mhe_glwe_refresh_share_gen<C, S, E>(
        &self,
        res: &mut GLWERefreshShareOwned<BE>,
        mask: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: SmudgingNoise,
        enc_infos: &E,
        source_xm: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        BE::mhe_glwe_refresh_share_gen(
            self,
            res,
            mask,
            sk,
            log_bound,
            seed,
            flood,
            enc_infos,
            source_xm,
            source_xe,
            source_smudge,
            scratch,
        )
    }

    fn mhe_glwe_refresh_share_aggregate(&self, res: &mut GLWERefreshShareOwned<BE>, a: &GLWERefreshShareOwned<BE>) {
        BE::mhe_glwe_refresh_share_aggregate(self, res, a)
    }

    fn mhe_glwe_refresh_share_finalize_tmp_bytes<A>(&self, res_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::mhe_glwe_refresh_share_finalize_tmp_bytes(self, res_infos)
    }

    fn mhe_glwe_refresh_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWERefreshShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_refresh_share_finalize(self, res, ct, share, scratch)
    }
}
