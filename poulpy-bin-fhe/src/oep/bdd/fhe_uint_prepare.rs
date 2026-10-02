use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`FheUintPrepare`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait FheUintPrepareImpl<BRA: BlindRotationAlgo>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare_tmp_bytes<R, A, B>(
        module: &Module<Self>,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        bits_infos: &A,
        bdd_infos: &B,
    ) -> usize
    where
        R: GGSWInfos,
        A: GLWEInfos,
        B: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare<K, T: UnsignedInteger>(
        module: &Module<Self>,
        res: &mut FheUintPrepared<Self::OwnedBuf, T, Self>,
        bits: &FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
        key: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        K: BDDKeyHelper<Self::OwnedBuf, BRA, Self> + BDDKeyInfos,
    {
        crate::oep::derived::bdd::fhe_uint_prepare_derived::<BRA, Self, _, T>(module, res, bits, key, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare_custom<K, T: UnsignedInteger>(
        module: &Module<Self>,
        res: &mut FheUintPrepared<Self::OwnedBuf, T, Self>,
        bits: &FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        K: BDDKeyHelper<Self::OwnedBuf, BRA, Self> + BDDKeyInfos,
    {
        crate::oep::derived::bdd::fhe_uint_prepare_custom_derived::<BRA, Self, _, T>(
            module, res, bits, bit_start, bit_count, key, scratch,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare_custom_multi_thread<K, T: UnsignedInteger>(
        module: &Module<Self>,
        threads: usize,
        res: &mut FheUintPrepared<Self::OwnedBuf, T, Self>,
        bits: &FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        K: BDDKeyHelper<Self::OwnedBuf, BRA, Self> + BDDKeyInfos;
}
