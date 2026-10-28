use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    layouts::{Backend, Data, ScratchArena, ZnxWord},
    source::Source,
};

use poulpy_core::{
    GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey};

pub use crate::api::BlindRotationKeyEncryptSk;

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BlindRotationKey<D, BRA, W> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, S0, S1, BE>(
        &mut self,
        module: &M,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        module.blind_rotation_key_encrypt_sk(self, sk_lwe, sk_glwe, source_xe, source_xa, scratch);
    }
}

impl<BRA: BlindRotationAlgo> BlindRotationKey<AlignedBuf, BRA, i64> {
    pub fn encrypt_sk_tmp_bytes<A, M, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: BlindRotationKeyEncryptSk<BRA, BE>,
    {
        module.blind_rotation_key_encrypt_sk_tmp_bytes(infos)
    }
}
