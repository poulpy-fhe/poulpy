use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_hal::layouts::*;
#[allow(clippy::too_many_arguments)]
pub(crate) fn fhe_uint_prepare_derived<BRA: BlindRotationAlgo, BE: Backend, K, T: UnsignedInteger>(
    module: &Module<BE>,
    res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
    bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    Module<BE>: FheUintPrepare<BRA, BE>,
{
    module.fhe_uint_prepare_custom(res, bits, 0, T::BITS as usize, key, scratch);
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn fhe_uint_prepare_custom_derived<BRA: BlindRotationAlgo, BE: Backend, K, T: UnsignedInteger>(
    module: &Module<BE>,
    res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
    bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
    bit_start: usize,
    bit_count: usize,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    Module<BE>: FheUintPrepare<BRA, BE>,
{
    module.fhe_uint_prepare_custom_multi_thread(1, res, bits, bit_start, bit_count, key, scratch)
}
