use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::layouts::*;
/// Workspace for the out-of-place rotation and, with the destination as the source, the in-place one.
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
    Module<BE>: GLWEBytesOf<BE> + GLWECopy<BE> + GLWECmux<BE>,
{
    module
        .glwe_copy_tmp_bytes(res_infos, a_infos)
        .max(glwe_blind_rotation_assign_scratch(module, res_infos, k_infos))
}

/// Workspace for the canonical in-place rotation, including the final copy.
fn glwe_blind_rotation_assign_scratch<BE: Backend, R, K>(module: &Module<BE>, res_infos: &R, k_infos: &K) -> usize
where
    R: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: GLWEBytesOf<BE> + GLWECopy<BE> + GLWECmux<BE>,
{
    // Scratch allocation uses active precision, while the destination may
    // retain a larger capacity. Query both alternating CMux directions and
    // the final compact-to-destination copy with their actual layouts.
    let temporary = res_infos.glwe_layout();
    module
        .glwe_cmux_tmp_bytes(&temporary, res_infos, k_infos)
        .max(module.glwe_cmux_tmp_bytes(res_infos, &temporary, k_infos))
        .max(module.glwe_copy_tmp_bytes(res_infos, &temporary))
        + module.glwe_bytes_of_from_infos(&temporary)
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GLWEBlindRotation::glwe_blind_rotation_assign`].
pub fn glwe_blind_rotation_assign_reference<BE, R, K>(
    module: &Module<BE>,
    sign: bool,
    bit_rsh: usize,
    bit_mask: usize,
    bit_lsh: usize,
    res: &mut R,
    value: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEBytesOf<BE> + GLWECopy<BE> + GLWERotate<BE> + GLWECmux<BE>,
{
    assert!(
        scratch.available() >= glwe_blind_rotation_tmp_bytes_reference(module, res, res, value.get_bit(0)),
        "glwe_blind_rotation_assign: insufficient scratch"
    );
    let (mut tmp_res, mut scratch_1) = scratch.borrow().take_glwe_scratch(res);
    let mut res_is_cur = true;

    for i in 0..bit_mask {
        if res_is_cur {
            match sign {
                true => module.glwe_rotate(1 << (i + bit_lsh), &mut tmp_res, res),
                false => module.glwe_rotate(-1 << (i + bit_lsh), &mut tmp_res, res),
            }

            let bit = value.get_bit(i + bit_rsh);
            module.glwe_cmux_assign(&mut tmp_res, res, &bit.to_backend_ref(), &mut scratch_1.borrow());
        } else {
            match sign {
                true => module.glwe_rotate(1 << (i + bit_lsh), res, &tmp_res),
                false => module.glwe_rotate(-1 << (i + bit_lsh), res, &tmp_res),
            }

            let bit = value.get_bit(i + bit_rsh);
            module.glwe_cmux_assign(res, &tmp_res, &bit.to_backend_ref(), &mut scratch_1.borrow());
        }

        res_is_cur = !res_is_cur;
    }

    if !res_is_cur {
        module.glwe_copy(res, &tmp_res, &mut scratch_1);
    }

    res.set_noise(None);
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GLWEBlindRotation::glwe_blind_rotation`].
pub fn glwe_blind_rotation_reference<BE, R, A, K>(
    module: &Module<BE>,
    sign: bool,
    bit_rsh: usize,
    bit_mask: usize,
    bit_lsh: usize,
    res: &mut R,
    a: &A,
    fhe_uint: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE>,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWECopy<BE> + GLWEBlindRotation<BE>,
{
    assert!(
        scratch.available() >= module.glwe_blind_rotation_tmp_bytes(res, &a.to_backend_ref(), fhe_uint.get_bit(0)),
        "glwe_blind_rotation: insufficient scratch"
    );
    module.glwe_copy(res, a, scratch);
    module.glwe_blind_rotation_assign(sign, bit_rsh, bit_mask, bit_lsh, res, fhe_uint, scratch);
}

/// Reference bodies of the [`GLWEBlindRotationImpl`](crate::oep::GLWEBlindRotationImpl) methods, which
/// [`impl_glwe_blind_rotation_reference_full!`](crate::impl_glwe_blind_rotation_reference_full) forwards to.
pub trait GLWEBlindRotationReference<BE: Backend> {
    fn glwe_blind_rotation_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_assign_reference<R, K>(
        &self,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        value: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGGSWBit<BE>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_reference<R, A, K>(
        &self,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        K: GetGGSWBit<BE>;
}

impl<BE: Backend<ZnxWord = i64>> GLWEBlindRotationReference<BE> for Module<BE>
where
    Module<BE>: GLWEBytesOf<BE> + GLWECopy<BE> + GLWERotate<BE> + GLWECmux<BE> + GLWEBlindRotation<BE>,
{
    fn glwe_blind_rotation_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGSWInfos,
    {
        glwe_blind_rotation_tmp_bytes_reference(self, res_infos, a_infos, k_infos)
    }

    fn glwe_blind_rotation_assign_reference<R, K>(
        &self,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        value: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
    {
        glwe_blind_rotation_assign_reference(self, sign, bit_rsh, bit_mask, bit_lsh, res, value, scratch)
    }

    fn glwe_blind_rotation_reference<R, A, K>(
        &self,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        K: GetGGSWBit<BE>,
    {
        glwe_blind_rotation_reference(self, sign, bit_rsh, bit_mask, bit_lsh, res, a, fhe_uint, scratch)
    }
}
