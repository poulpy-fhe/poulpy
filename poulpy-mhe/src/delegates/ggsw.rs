use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos,
        prepared::{GGLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena},
    source::Source,
};

use crate::{api::GGSWMHEProtocol, layouts::GGSWShareOwned, oep::GGSWMHEProtocolImpl};

impl<BE: Backend + GGSWMHEProtocolImpl> GGSWMHEProtocol<BE> for Module<BE> {
    fn mhe_ggsw_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::mhe_ggsw_share_gen_tmp_bytes(self, infos)
    }

    fn mhe_ggsw_share_gen<P, S, U, E>(
        &self,
        res: &mut GGSWShareOwned<BE>,
        pt: &P,
        sk: &S,
        u: &U,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        U: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        BE::mhe_ggsw_share_gen(self, res, pt, sk, u, seed, enc_infos, source_xe, scratch)
    }

    fn mhe_ggsw_share_aggregate(&self, res: &mut GGSWShareOwned<BE>, a: &GGSWShareOwned<BE>) {
        BE::mhe_ggsw_share_aggregate(self, res, a)
    }

    fn mhe_ggsw_share_finalize_tmp_bytes<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GGLWEInfos,
    {
        BE::mhe_ggsw_share_finalize_tmp_bytes(self, res_infos, key_infos)
    }

    fn mhe_ggsw_share_finalize<R, K>(&self, res: &mut R, share: &GGSWShareOwned<BE>, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        BE::mhe_ggsw_share_finalize(self, res, share, key, scratch)
    }
}
