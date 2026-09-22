//! Public module methods dispatch to the selected backend.
use crate::api::*;
use crate::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyInfos, BlindRotationKeyPrepared,
    LookUpTableRotationDirection, LookupTable,
};
use crate::oep::*;
use poulpy_core::{EncryptionInfos, GetDistribution, layouts::*};
use poulpy_hal::{layouts::*, source::Source};

impl<BRA: BlindRotationAlgo, BE: BlindRotationExecuteImpl<BRA>> BlindRotationExecute<BRA, BE> for Module<BE> {
    fn blind_rotation_execute_tmp_bytes<G, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &G,
        brk_infos: &B,
    ) -> usize
    where
        G: GLWEInfos,
        B: BlindRotationKeyInfos,
    {
        BE::blind_rotation_execute_tmp_bytes(self, block_size, extension_factor, glwe_infos, brk_infos)
    }
    fn blind_rotation_execute<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
        brk: &BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        BE::blind_rotation_execute(self, res, lwe, lut, brk, scratch)
    }
}

impl<BRA: BlindRotationAlgo, BE: BlindRotationKeyEncryptSkImpl<BRA>> BlindRotationKeyEncryptSk<BRA, BE> for Module<BE> {
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A: GGSWInfos>(&self, infos: &A) -> usize {
        BE::blind_rotation_key_encrypt_sk_tmp_bytes(self, infos)
    }
    fn blind_rotation_key_encrypt_sk<S0, S1, E>(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
    {
        BE::blind_rotation_key_encrypt_sk(self, res, sk_glwe, sk_lwe, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BRA: BlindRotationAlgo, BE: BlindRotationKeyCompressedEncryptSkImpl<BRA>> BlindRotationKeyCompressedEncryptSk<BE, BRA>
    for Module<BE>
{
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::blind_rotation_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    }
    fn blind_rotation_key_compressed_encrypt_sk<S0, S1, E>(
        &self,
        res: &mut BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
    {
        BE::blind_rotation_key_compressed_encrypt_sk(self, res, sk_glwe, sk_lwe, seed_xa, enc_infos, source_xe, scratch)
    }
}

impl<BRA: BlindRotationAlgo, BE: BlindRotationKeyPreparedImpl<BRA>> BlindRotationKeyPreparedFactory<BRA, BE> for Module<BE> {
    fn blind_rotation_key_prepared_alloc<A>(&self, infos: &A) -> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: BlindRotationKeyInfos,
    {
        BE::blind_rotation_key_prepared_alloc(self, infos)
    }
    fn blind_rotation_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
    {
        BE::blind_rotation_key_prepare_tmp_bytes(self, infos)
    }
    fn prepare_blind_rotation_key(
        &self,
        res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::prepare_blind_rotation_key(self, res, other, scratch)
    }
}

impl<BE: BlindRotationModSwitchImpl> BlindRotationModSwitch<BE> for Module<BE> {
    fn blind_rotation_mod_switch<L: LWEToBackendRef<BE> + LWEInfos>(
        &self,
        modulus: usize,
        res: &mut [i64],
        lwe: &L,
        direction: LookUpTableRotationDirection,
    ) {
        BE::blind_rotation_mod_switch(self, modulus, res, lwe, direction)
    }
}
impl<BRA: BlindRotationAlgo, BE: BlindRotationKeyDecompressImpl<BRA>> BlindRotationKeyDecompress<BRA, BE> for Module<BE> {
    fn blind_rotation_key_decompress_tmp_bytes<A: BlindRotationKeyInfos>(&self, infos: &A) -> usize {
        BE::blind_rotation_key_decompress_tmp_bytes(self, infos)
    }
    fn blind_rotation_key_decompress(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        src: &BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::blind_rotation_key_decompress(self, res, src, scratch)
    }
}

mod blind_rotation_key_compressed_factory;
mod lookup_table_factory;
