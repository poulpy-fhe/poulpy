use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`Cmux`](crate::api::Cmux).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CmuxImpl: Backend {
    /// Scratch bound of [`Self::cmux`], [`Self::cmux_assign`] and [`Self::cmux_assign_neg`], one shared bound; may report
    /// less than the reference, never less than the body takes.
    #[allow(clippy::too_many_arguments)]
    fn cmux_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;
    /// Reproduces [`cmux_reference`](crate::reference::bdd::cmux_reference) bit for bit, within [`Self::cmux_tmp_bytes`];
    /// free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn cmux<R, T, F>(
        module: &Module<Self>,
        res: &mut R,
        t: &T,
        f: &F,
        s: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        T: GLWEToBackendRef<Self>,
        F: GLWEToBackendRef<Self>;
    /// Reproduces [`cmux_assign_neg_reference`](crate::reference::bdd::cmux_assign_neg_reference) bit for bit, within
    /// [`Self::cmux_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn cmux_assign_neg<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        s: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self>;
    /// Reproduces [`cmux_assign_reference`](crate::reference::bdd::cmux_assign_reference) bit for bit, within
    /// [`Self::cmux_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn cmux_assign<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        s: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self>;
}
