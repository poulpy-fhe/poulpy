use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Bootstraps packed integer bits into prepared GGSW ciphertexts.
///
/// Full, partial-range and parallel execution use the selected backend contract.
pub trait FheUintPrepare<BRA: BlindRotationAlgo, BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Scratch for one worker, already rounded to the backend's worker alignment.
    /// Sequential calls use this size. For `fhe_uint_prepare_custom_multi_thread`,
    /// multiply it by `worker_count::<BE::TaskExecutor>(threads, bit_count)`.
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
        B: BDDKeyInfos;
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare<K, T: UnsignedInteger>(
        &self,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos;
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
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos;
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
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos;
}
