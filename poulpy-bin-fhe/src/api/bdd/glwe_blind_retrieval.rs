use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
/// Oblivious in-place sorting / retrieval of a GLWE vector by an encrypted index.
///
/// Where `GLWEBlindSelection` extracts one element from a map given an encrypted
/// key, `GLWEBlindRetrieval` operates on an ordered `Vec<R>` and performs a
/// sorting-network-style rearrangement: after
/// [`glwe_blind_retrieval_statefull`][Self::glwe_blind_retrieval_statefull],
/// element `0` of the vector encrypts the input element whose index equals the
/// encrypted selector.
///
/// The rearrangement uses conditional-swap ([`Cswap`]) operations, one per bit
/// of the selector sub-field.  The `_rev` variant applies the operations in
/// reverse, useful for undoing the permutation.
pub trait GLWEBlindRetrieval<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Returns the minimum scratch-space size in bytes required by
    /// [`glwe_blind_retrieval_statefull`][Self::glwe_blind_retrieval_statefull].
    fn glwe_blind_retrieval_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;
    #[allow(clippy::too_many_arguments)]
    /// Rearranges `res` in-place so that `res[0]` encrypts the element at the
    /// encrypted index `(bits >> bit_rsh) % 2^bit_mask`.
    ///
    /// Uses a butterfly network of [`Cswap`] gates, iterating from the
    /// most-significant to the least-significant bit of the selector sub-field.
    fn glwe_blind_retrieval_statefull<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>;
    #[allow(clippy::too_many_arguments)]
    /// Reverses the permutation applied by
    /// [`glwe_blind_retrieval_statefull`][Self::glwe_blind_retrieval_statefull].
    ///
    /// Applies the same butterfly network in reverse order, restoring the original
    /// element ordering after an oblivious retrieval.
    fn glwe_blind_retrieval_statefull_rev<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>;
}
