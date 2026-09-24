use crate::bdd_arithmetic::*;
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
#[allow(clippy::too_many_arguments)]
pub(crate) fn ggsw_to_ggsw_blind_rotation_tmp_bytes_derived<BE: Backend, R, A, K>(
    module: &Module<BE>,
    res_infos: &R,
    a_infos: &A,
    k_infos: &K,
) -> usize
where
    R: GLWEInfos,
    A: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: GLWEBlindRotation<BE>,
{
    module.glwe_blind_rotation_tmp_bytes(res_infos, a_infos, k_infos)
}
pub(crate) fn ggsw_blind_rotation_assign_tmp_bytes_derived<BE: Backend, R: GLWEInfos, K: GGSWInfos>(
    module: &Module<BE>,
    res_infos: &R,
    k_infos: &K,
) -> usize
where
    Module<BE>: GLWEBlindRotation<BE>,
{
    module.glwe_blind_rotation_assign_tmp_bytes(res_infos, k_infos)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn ggsw_blind_rotation_assign_derived<BE, R, K>(
    module: &Module<BE>,
    res: &mut R,
    fhe_uint: &K,
    sign: bool,
    bit_rsh: usize,
    bit_mask: usize,
    bit_lsh: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEBlindRotation<BE>,
{
    for col in 0..(res.rank() + 1).into() {
        for row in 0..res.dnum().into() {
            module.glwe_blind_rotation_assign(
                &mut res.at_view_mut(row, col),
                fhe_uint,
                sign,
                bit_rsh,
                bit_mask,
                bit_lsh,
                scratch,
            );
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn ggsw_blind_rotation_derived<BE, R, A, K>(
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
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEBlindRotation<BE>,
{
    assert!(res.dnum() <= a.dnum());
    assert_eq!(res.dsize(), a.dsize());

    for col in 0..(res.rank() + 1).into() {
        for row in 0..res.dnum().into() {
            module.glwe_blind_rotation(
                &mut res.at_view_mut(row, col),
                &a.at_view(row, col),
                fhe_uint,
                sign,
                bit_rsh,
                bit_mask,
                bit_lsh,
                scratch,
            );
        }
    }
}
