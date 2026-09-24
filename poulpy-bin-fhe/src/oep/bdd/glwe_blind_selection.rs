use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
use std::collections::HashMap;
/// Backend implementation contract for [`GLWEBlindSelection`].
///
/// # Safety
/// Implementations must preserve the canonical circuit, buffer bounds, metadata,
/// and the paired scratch query contract.
pub unsafe trait GLWEBlindSelectionImpl<T: UnsignedInteger>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, input_infos: &[A], k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        fhe_uint: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        K: GetGGSWBit<Self>;
}
