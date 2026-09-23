use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GGSWBlindRotation::scalar_to_ggsw_blind_rotation_tmp_bytes`].
pub fn scalar_to_ggsw_blind_rotation_tmp_bytes_reference<T: UnsignedInteger, BE: Backend<ZnxWord = i64>, R, K>(
    module: &Module<BE>,
    res_infos: &R,
    k_infos: &K,
) -> usize
where
    R: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: GLWEBytesOf<BE> + GLWEBlindRotation<BE> + GLWEZero<BE> + VecZnxAddScalarAssign<BE> + VecZnxNormalizeAssign<BE>,
{
    let temporary = res_infos.glwe_layout();
    module.glwe_blind_rotation_tmp_bytes(res_infos, &temporary, k_infos) + module.glwe_bytes_of_from_infos(&temporary)
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GGSWBlindRotation::scalar_to_ggsw_blind_rotation`].
pub fn scalar_to_ggsw_blind_rotation_reference<T: UnsignedInteger, BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    test_vector: &A,
    fhe_uint: &K,
    sign: bool,
    bit_rsh: usize,
    bit_mask: usize,
    bit_lsh: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    A: ScalarZnxToBackendRef<BE>,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEBytesOf<BE> + GLWEBlindRotation<BE> + GLWEZero<BE> + VecZnxAddScalarAssign<BE> + VecZnxNormalizeAssign<BE>,
{
    let base2k: usize = res.base2k().into();
    let dsize: usize = res.dsize().into();
    let (mut tmp_glwe, mut scratch_1) = scratch.borrow().take_glwe_scratch(res);
    let tmp_glwe_k = tmp_glwe.k().as_usize();
    let test_vector = test_vector.to_backend_ref();

    for col in 0..(res.rank() + 1).into() {
        for row in 0..res.dnum().into() {
            module.glwe_zero(&mut tmp_glwe);
            {
                let mut tmp_glwe_inner = tmp_glwe.data_mut();
                let mut tmp_glwe_data = VecZnxToBackendMut::<BE>::to_backend_mut(&mut tmp_glwe_inner);
                module.vec_znx_add_scalar_assign(&mut tmp_glwe_data, col, (dsize - 1) + row * dsize, &test_vector, 0);
                module.vec_znx_normalize_assign(base2k, tmp_glwe_k, 0, &mut tmp_glwe_data, col, &mut scratch_1.borrow());
            }

            module.glwe_blind_rotation(
                &mut res.at_view_mut(row, col),
                &tmp_glwe,
                fhe_uint,
                sign,
                bit_rsh,
                bit_mask,
                bit_lsh,
                &mut scratch_1.borrow(),
            );
        }
    }
}
