use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
#[allow(clippy::too_many_arguments)]
pub(crate) fn glwe_blind_retrieval_tmp_bytes_derived<BE: Backend, R, K>(module: &Module<BE>, res_infos: &R, k_infos: &K) -> usize
where
    R: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: Cswap<BE>,
{
    module.cswap_tmp_bytes(res_infos, res_infos, k_infos)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn glwe_blind_retrieval_statefull_derived<BE: Backend, R, K>(
    module: &Module<BE>,
    res: &mut [R],
    bits: &K,
    bit_rsh: usize,
    bit_mask: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGGSWBit<BE> + 'static,
    Module<BE>: Cswap<BE>,
{
    for i in 0..bit_mask {
        let t: usize = 1 << (bit_mask - i - 1);
        let bit = bits.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal
        for j in 0..t {
            if j + t < res.len() {
                let (lo, hi) = res.split_at_mut(j + t);
                module.cswap(&mut lo[j], &mut hi[0], &bit.to_backend_ref(), &mut scratch.borrow());
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn glwe_blind_retrieval_statefull_rev_derived<BE: Backend, R, K>(
    module: &Module<BE>,
    res: &mut [R],
    bits: &K,
    bit_rsh: usize,
    bit_mask: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGGSWBit<BE> + 'static,
    Module<BE>: Cswap<BE>,
{
    for i in (0..bit_mask).rev() {
        let t: usize = 1 << (bit_mask - i - 1);
        let bit = bits.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal
        for j in 0..t {
            if j < res.len() && j + t < res.len() {
                let (lo, hi) = res.split_at_mut(j + t);
                module.cswap(&mut lo[j], &mut hi[0], &bit.to_backend_ref(), &mut scratch.borrow());
            }
        }
    }
}
