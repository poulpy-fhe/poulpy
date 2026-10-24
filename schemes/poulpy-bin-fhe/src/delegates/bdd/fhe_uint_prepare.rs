use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
impl<BRA: BlindRotationAlgo, BE: Backend> FheUintPrepare<BRA, BE> for Module<BE>
where
    BE: crate::oep::FheUintPrepareImpl<BRA>,
{
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare_tmp_bytes<R, A, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        bits_infos: &A,
        bdd_infos: &B,
    ) -> usize
    where
        R: GGSWInfos,
        A: GLWEInfos,
        B: BDDKeyInfos,
    {
        BE::fhe_uint_prepare_tmp_bytes::<R, A, B>(self, block_size, extension_factor, res_infos, bits_infos, bdd_infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare<K, T: UnsignedInteger>(
        &self,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    {
        BE::fhe_uint_prepare::<K, T>(self, res, bits, key, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare_custom<K, T: UnsignedInteger>(
        &self,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    {
        BE::fhe_uint_prepare_custom::<K, T>(self, res, bits, bit_start, bit_count, key, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare_custom_multi_thread<K, T: UnsignedInteger>(
        &self,
        threads: usize,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    {
        BE::fhe_uint_prepare_custom_multi_thread::<K, T>(self, threads, res, bits, bit_start, bit_count, key, scratch)
    }
}
