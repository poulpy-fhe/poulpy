use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`GGSWBlindRotation`](crate::api::GGSWBlindRotation), on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGSWBlindRotationImpl<T: UnsignedInteger>: Backend + crate::oep::GLWEBlindRotationImpl {
    /// Default body: `ggsw_blind_rotation_tmp_bytes_derived`, the selected `glwe_blind_rotation_tmp_bytes` at the
    /// same infos.
    ///
    /// Scratch bound of [`Self::ggsw_blind_rotation`] and [`Self::ggsw_blind_rotation_assign`]; may report less than the
    /// reference, never less than the body takes.
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos,
    {
        crate::oep::derived::bdd::ggsw_blind_rotation_tmp_bytes_derived::<Self, _, _, _>(module, res_infos, a_infos, k_infos)
    }
    /// Default body: `ggsw_blind_rotation_assign_derived`, the selected `glwe_blind_rotation_assign` on every `(row,
    /// col)` view of `res`, within [`Self::ggsw_blind_rotation_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation_assign<R, K>(
        module: &Module<Self>,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64>,
    {
        crate::oep::derived::bdd::ggsw_blind_rotation_assign_derived::<Self, _, _>(
            module, sign, bit_rsh, bit_mask, bit_lsh, res, fhe_uint, scratch,
        )
    }
    /// Default body: `ggsw_blind_rotation_derived`, after asserting `res.dnum() <= a.dnum()` and equal `k_digit`, the
    /// selected `glwe_blind_rotation` on every `(row, col)` view, within [`Self::ggsw_blind_rotation_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation<R, A, K>(
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
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWAtViewRef<Self> + GGSWInfos,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64>,
    {
        crate::oep::derived::bdd::ggsw_blind_rotation_derived::<Self, _, _, _>(
            module, sign, bit_rsh, bit_mask, bit_lsh, res, a, fhe_uint, scratch,
        )
    }
    /// Scratch bound of [`Self::scalar_to_ggsw_blind_rotation`]; may report less than the reference, never less than the
    /// body takes.
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;
    /// Reproduces
    /// [`scalar_to_ggsw_blind_rotation_reference`](crate::reference::bdd::scalar_to_ggsw_blind_rotation_reference) bit
    /// for bit, within [`Self::scalar_to_ggsw_blind_rotation_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, A, K>(
        module: &Module<Self>,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        test_vector: &A,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: ScalarZnxToBackendRef<Self>,
        K: GetGGSWBit<Self>,
        Self: Backend<ZnxWord = i64>;
}
