use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
use std::collections::HashMap;
/// Oblivious selection of one GLWE ciphertext from an encrypted-indexed map.
///
/// Given a `HashMap` of GLWE ciphertexts keyed by integer index and a set of
/// GGSW ciphertexts encoding a selection index `k`, selects:
///
/// ```text
/// res = a[(k >> bit_rsh) % 2^bit_mask]
/// ```
///
/// The selection is performed via a binary-tree reduction of CMux gates over the
/// `bit_mask` most-significant bits of the selected index sub-field, traversing
/// from MSB to LSB.  Indices absent from the map are treated as encryptions of
/// zero.
pub trait GLWEBlindSelection<T: UnsignedInteger, BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns scratch-space bytes for [`glwe_blind_selection`][Self::glwe_blind_selection].
    ///
    /// `input_infos` must describe every ciphertext present in the selection map,
    /// including its allocated capacity. Inputs may have different precisions or
    /// capacities, but must share the output's degree, rank, and radix. The query
    /// covers every ordering and sparse arrangement of these inputs.
    fn glwe_blind_selection_tmp_bytes<R, A, K>(&self, res_infos: &R, input_infos: &[A], k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection<R, A, K>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        fhe_uint: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>;
}
