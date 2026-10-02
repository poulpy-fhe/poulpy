use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::layouts::*;
use std::collections::{HashMap, HashSet};
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GLWEBlindSelection::glwe_blind_selection_tmp_bytes`].
pub fn glwe_blind_selection_tmp_bytes_reference<T: UnsignedInteger, BE: Backend<ZnxWord = i64>, R, A, K>(
    module: &Module<BE>,
    res_infos: &R,
    input_infos: &[A],
    k_infos: &K,
) -> usize
where
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: GLWECopy<BE> + Cmux<BE> + GLWEZero<BE>,
{
    // Scratch holds a compact zero branch; the caller's output may have a
    // larger allocation. Preserve the actual capacities of the map entries.
    let zero = res_infos.glwe_layout();
    let retained = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&zero));
    let mut seen = HashSet::new();
    let inputs: Vec<_> = input_infos
        .iter()
        .filter(|input| {
            seen.insert((
                input.n().as_u32(),
                input.base2k().as_u32(),
                input.k().as_u32(),
                input.rank().as_u32(),
                input.max_size(),
                input.size(),
            ))
        })
        .collect();
    let mut work = 0;
    for dst in &inputs {
        // A missing low branch writes into zero, then back into the high
        // branch. Any surviving map entry can be copied into the final output.
        work = work
            .max(module.cmux_tmp_bytes(&zero, dst, k_infos))
            .max(module.cmux_tmp_bytes(dst, &zero, k_infos))
            .max(module.glwe_copy_tmp_bytes(dst, &zero))
            .max(module.glwe_copy_tmp_bytes(res_infos, dst));
        for src in &inputs {
            work = work.max(module.cmux_tmp_bytes(dst, src, k_infos));
        }
    }
    // Deduplicating before the ordered pair scan keeps uniform maps linear.
    // Queries may be non-monotonic in either layout or capacity, so replacing
    // the distinct inputs with a single maximum-width layout is not valid.
    retained + work
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GLWEBlindSelection::glwe_blind_selection`].
pub fn glwe_blind_selection_reference<T: UnsignedInteger, BE: Backend<ZnxWord = i64>, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    mut a: HashMap<usize, &mut A>,
    fhe_uint: &K,
    bit_rsh: usize,
    bit_mask: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGGSWBit<BE>,
    Module<BE>: GLWECopy<BE> + Cmux<BE> + GLWEZero<BE>,
{
    assert!(bit_rsh + bit_mask <= T::BITS as usize);
    let (mut zero, mut scratch) = scratch.borrow().take_glwe_scratch(res);

    for i in 0..bit_mask {
        let t: usize = 1 << (bit_mask - i - 1);

        let bit = fhe_uint.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal

        for j in 0..t {
            let hi: Option<&mut A> = a.remove(&j);
            let lo: Option<&mut A> = a.remove(&(j + t));

            match (lo, hi) {
                (Some(lo), Some(hi)) => {
                    module.cmux_assign(lo, hi, &bit.to_backend_ref(), &mut scratch.borrow());
                    a.insert(j, lo);
                }

                (Some(lo), None) => {
                    module.glwe_zero(&mut zero);
                    module.cmux_assign(lo, &zero, &bit.to_backend_ref(), &mut scratch.borrow());
                    a.insert(j, lo);
                }

                (None, Some(hi)) => {
                    module.glwe_zero(&mut zero);
                    module.cmux_assign(&mut zero, hi, &bit.to_backend_ref(), &mut scratch.borrow());
                    module.glwe_copy(hi, &zero, &mut scratch);
                    a.insert(j, hi);
                }

                (None, None) => {
                    // No low or high branch — nothing to insert
                    // leave empty; future iterations will combine actual ciphertexts
                }
            }
        }
    }

    let out: Option<&mut A> = a.remove(&0);

    if let Some(out) = out {
        module.glwe_copy(res, out, &mut scratch);
    } else {
        module.glwe_zero(res);
    }
}
