use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
use std::collections::HashMap;
/// Backend implementation contract for [`GLWEBlindSelection`](crate::api::GLWEBlindSelection).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEBlindSelectionImpl<T: UnsignedInteger>: Backend {
    /// Scratch bound of [`Self::glwe_blind_selection`]; may report less than the reference, never less than the body
    /// takes. `input_infos` describes every input at its actual capacity, and the bound covers every ordering and sparse
    /// arrangement of the map.
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, input_infos: &[A], k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;
    /// Reproduces [`glwe_blind_selection_reference`](crate::reference::bdd::glwe_blind_selection_reference) bit for bit,
    /// within [`Self::glwe_blind_selection_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection<R, A, K>(
        module: &Module<Self>,
        bit_rsh: usize,
        bit_mask: usize,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + GLWEInfos,
        K: GetGGSWBit<Self>;
}
