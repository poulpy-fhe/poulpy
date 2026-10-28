use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`GLWEBlindRotation`](crate::api::GLWEBlindRotation), on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEBlindRotationImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::glwe_blind_rotation`] and [`Self::glwe_blind_rotation_assign`]; may report less than the
    /// reference, never less than the body takes.
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;
    /// Reproduces [`glwe_blind_rotation_assign_reference`](crate::reference::bdd::glwe_blind_rotation_assign_reference)
    /// bit for bit, within [`Self::glwe_blind_rotation_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_assign<R, K>(
        module: &Module<Self>,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        value: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64>;
    /// Reproduces [`glwe_blind_rotation_reference`](crate::reference::bdd::glwe_blind_rotation_reference) bit for bit,
    /// within [`Self::glwe_blind_rotation_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation<R, A, K>(
        module: &Module<Self>,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self>,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64>;
}
