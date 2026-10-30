use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`FheUintPrepare`](crate::api::FheUintPrepare), on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait FheUintPrepareImpl<BRA: BlindRotationAlgo>: Backend<Ring = Standard> {
    /// Scratch bound of one worker of [`Self::fhe_uint_prepare_custom_multi_thread`] and of the serial forms, rounded for
    /// the worker alignment; may report less than the reference, never less than the body takes.
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
    /// Default body: `fhe_uint_prepare_derived`, the selected [`Self::fhe_uint_prepare_custom`] over the bits
    /// `0..T::BITS`.
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
    /// Default body: `fhe_uint_prepare_custom_derived`, the selected [`Self::fhe_uint_prepare_custom_multi_thread`] with
    /// `threads = 1`.
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
    /// Reproduces
    /// [`fhe_uint_prepare_custom_multi_thread_reference`](crate::reference::bdd::fhe_uint_prepare_custom_multi_thread_reference)
    /// bit for bit, within [`Self::fhe_uint_prepare_tmp_bytes`]; free in kernels, scheduling and scratch use within one
    /// worker bound per active worker, output bit-identical for every `threads` and executor.
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
