use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::layouts::*;
/// Workspace for copying the actual source and then invoking the selected in-place rotation.
pub fn glwe_blind_rotation_tmp_bytes_reference<BE: Backend, R, A, K>(
    module: &Module<BE>,
    res_infos: &R,
    a_infos: &A,
    k_infos: &K,
) -> usize
where
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: GLWECopy<BE> + GLWEBlindRotation<BE>,
{
    module
        .glwe_copy_tmp_bytes(res_infos, a_infos)
        .max(module.glwe_blind_rotation_assign_tmp_bytes(res_infos, k_infos))
}

/// Workspace for the canonical in-place rotation, including the final copy.
pub fn glwe_blind_rotation_assign_tmp_bytes_reference<BE: Backend, R, K>(module: &Module<BE>, res_infos: &R, k_infos: &K) -> usize
where
    R: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: GLWEBytesOf<BE> + GLWECopy<BE> + Cmux<BE>,
{
    // Scratch allocation uses active precision, while the destination may
    // retain a larger capacity. Query both alternating CMux directions and
    // the final compact-to-destination copy with their actual layouts.
    let temporary = res_infos.glwe_layout();
    module
        .cmux_tmp_bytes(&temporary, res_infos, k_infos)
        .max(module.cmux_tmp_bytes(res_infos, &temporary, k_infos))
        .max(module.glwe_copy_tmp_bytes(res_infos, &temporary))
        + module.glwe_bytes_of_from_infos(&temporary)
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GLWEBlindRotation::glwe_blind_rotation_assign`].
pub fn glwe_blind_rotation_assign_reference<BE, R, K>(
    module: &Module<BE>,
    res: &mut R,
    value: &K,
    sign: bool,
    bit_rsh: usize,
    bit_mask: usize,
    bit_lsh: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEBytesOf<BE> + GLWECopy<BE> + GLWERotate<BE> + Cmux<BE>,
{
    let (mut tmp_res, mut scratch_1) = scratch.borrow().take_glwe_scratch(res);
    let mut res_is_cur = true;

    for i in 0..bit_mask {
        if res_is_cur {
            match sign {
                true => module.glwe_rotate(1 << (i + bit_lsh), &mut tmp_res, res),
                false => module.glwe_rotate(-1 << (i + bit_lsh), &mut tmp_res, res),
            }

            let bit = value.get_bit(i + bit_rsh);
            module.cmux_assign(&mut tmp_res, res, &bit.to_backend_ref(), &mut scratch_1.borrow());
        } else {
            match sign {
                true => module.glwe_rotate(1 << (i + bit_lsh), res, &tmp_res),
                false => module.glwe_rotate(-1 << (i + bit_lsh), res, &tmp_res),
            }

            let bit = value.get_bit(i + bit_rsh);
            module.cmux_assign(res, &tmp_res, &bit.to_backend_ref(), &mut scratch_1.borrow());
        }

        res_is_cur = !res_is_cur;
    }

    if !res_is_cur {
        module.glwe_copy(res, &tmp_res, &mut scratch_1);
    }
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GLWEBlindRotation::glwe_blind_rotation`].
pub fn glwe_blind_rotation_reference<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    fhe_uint: &K,
    sign: bool,
    bit_rsh: usize,
    bit_mask: usize,
    bit_lsh: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE>,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWECopy<BE> + GLWEBlindRotation<BE>,
{
    module.glwe_copy(res, a, scratch);
    module.glwe_blind_rotation_assign(res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
}
