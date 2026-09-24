use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Backend implementation contract for [`Cmux`](crate::api::Cmux).
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait CmuxImpl: Backend {
    #[allow(clippy::too_many_arguments)]
    fn cmux_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos;
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
