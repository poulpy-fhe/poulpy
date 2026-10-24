//! Public module methods dispatch to the selected backend.
use crate::api::*;
use crate::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyInfos, BlindRotationKeyPrepared,
    LookUpTableRotationDirection, LookupTable,
};
use crate::oep::*;
use poulpy_core::{GetDistribution, layouts::*};
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
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::blind_rotation_key_encrypt_sk_tmp_bytes(self, infos)
    }
    fn blind_rotation_key_encrypt_sk<S0, S1>(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::blind_rotation_key_encrypt_sk(self, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch)
    }
}

impl<BRA: BlindRotationAlgo, BE: BlindRotationKeyCompressedEncryptSkImpl<BRA>> BlindRotationKeyCompressedEncryptSk<BRA, BE>
    for Module<BE>
{
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::blind_rotation_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    }
    fn blind_rotation_key_compressed_encrypt_sk<S0, S1>(
        &self,
        res: &mut BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::blind_rotation_key_compressed_encrypt_sk(self, res, sk_lwe, sk_glwe, seed_xa, source_xe, scratch)
    }
}

impl<BRA: BlindRotationAlgo, BE: BlindRotationKeyPreparedImpl<BRA>> BlindRotationKeyPreparedFactory<BRA, BE> for Module<BE> {
    fn blind_rotation_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
    {
        BE::blind_rotation_key_prepare_tmp_bytes(self, infos)
    }
    fn blind_rotation_key_prepare(
        &self,
        res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::blind_rotation_key_prepare(self, res, other, scratch)
    }
}

impl<BE: BlindRotationModSwitchImpl> BlindRotationModSwitch<BE> for Module<BE> {
    fn blind_rotation_mod_switch<L: LWEToBackendRef<BE> + LWEInfos>(
        &self,
        modulus: usize,
        direction: LookUpTableRotationDirection,
        res: &mut [i64],
        lwe: &L,
    ) {
        BE::blind_rotation_mod_switch(self, modulus, direction, res, lwe)
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

mod lookup_table_factory;
