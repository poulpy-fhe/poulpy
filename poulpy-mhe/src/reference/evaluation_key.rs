use poulpy_core::{
    EncryptionInfos, GLWESwitchingKeyCompressedEncryptSk, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GLWESwitchingKeyPatCompressedOwned;

pub trait GLWESwitchingKeyShareReference<BE: Backend> {
    fn glwe_switching_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_share_reference<S1, S2, E>(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
        sk_in: &S1,
        sk_out: &S2,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos;
}

impl<BE: Backend> GLWESwitchingKeyShareReference<BE> for Module<BE>
where
    Self: GLWESwitchingKeyCompressedEncryptSk<BE>,
{
    fn glwe_switching_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn glwe_switching_key_share_reference<S1, S2, E>(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
        sk_in: &S1,
        sk_out: &S2,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos,
    {
        self.glwe_switching_key_compressed_encrypt_sk(res, sk_in, sk_out, seed, enc_infos, source_xe, scratch);
        res.set_canonical(true);
    }
}
