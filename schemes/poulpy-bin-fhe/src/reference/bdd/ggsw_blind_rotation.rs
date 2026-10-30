use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GGSWBlindRotation::ggsw_blind_rotation_from_scalar_tmp_bytes`].
pub fn ggsw_blind_rotation_from_scalar_tmp_bytes_reference<T: UnsignedInteger, BE: Backend<ZnxWord = i64>, R, K>(
    module: &Module<BE>,
    res_infos: &R,
    k_infos: &K,
) -> usize
where
    R: GLWEInfos,
    K: GGSWInfos,
    Module<BE>: GLWEBytesOf<BE> + GLWEBlindRotation<BE> + GLWEZero<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
{
    let temporary = res_infos.glwe_layout();
    module
        .glwe_blind_rotation_tmp_bytes(res_infos, &temporary, k_infos)
        .max(module.vec_znx_normalize_tmp_bytes())
        + module.glwe_bytes_of_from_infos(&temporary)
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`GGSWBlindRotation::ggsw_blind_rotation_from_scalar`].
pub fn ggsw_blind_rotation_from_scalar_reference<T: UnsignedInteger, BE, R, A, K>(
    module: &Module<BE>,
    sign: bool,
    bit_rsh: usize,
    bit_mask: usize,
    bit_lsh: usize,
    res: &mut R,
    test_vector: &A,
    fhe_uint: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    A: ScalarZnxToBackendRef<BE>,
    K: GetGGSWBit<BE>,
    BE: Backend<ZnxWord = i64> + poulpy_hal::oep::HalVecZnxImpl,
    Module<BE>: GLWEBytesOf<BE> + GLWEBlindRotation<BE> + GLWEZero<BE> + VecZnxNormalizeTmpBytes,
{
    assert!(
        scratch.available()
            >= ggsw_blind_rotation_from_scalar_tmp_bytes_reference::<T, BE, _, _>(module, res, fhe_uint.get_bit(0)),
        "ggsw_blind_rotation_from_scalar: insufficient scratch"
    );
    let base2k: usize = res.base2k().into();
    let k_digit: usize = res.k_digit().as_usize();
    let (mut tmp_glwe, mut scratch_1) = scratch.borrow().take_glwe_scratch(res);
    let tmp_glwe_k = tmp_glwe.k().as_usize();
    let test_vector = test_vector.to_backend_ref();

    for col in 0..(res.rank() + 1).into() {
        for row in 0..res.dnum().into() {
            module.glwe_zero(&mut tmp_glwe);
            {
                let mut tmp_glwe_inner = tmp_glwe.data_mut();
                let mut tmp_glwe_data = VecZnxToBackendMut::<BE>::to_backend_mut(&mut tmp_glwe_inner);
                poulpy_core::gglwe_gadget_row_plaintext(
                    module,
                    &mut tmp_glwe_data,
                    col,
                    base2k,
                    tmp_glwe_k,
                    row,
                    k_digit,
                    0,
                    &test_vector,
                    0,
                    &mut scratch_1.borrow(),
                );
            }

            module.glwe_blind_rotation(
                sign,
                bit_rsh,
                bit_mask,
                bit_lsh,
                &mut res.at_view_mut(row, col),
                &tmp_glwe,
                fhe_uint,
                &mut scratch_1.borrow(),
            );
        }
    }
}

/// Reference bodies of the required [`GGSWBlindRotationImpl`](crate::oep::GGSWBlindRotationImpl) methods, which
/// [`impl_ggsw_blind_rotation_reference_full!`](crate::impl_ggsw_blind_rotation_reference_full) forwards to.
pub trait GGSWBlindRotationReference<T: UnsignedInteger, BE: Backend> {
    fn ggsw_blind_rotation_from_scalar_tmp_bytes_reference<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos;

    #[allow(clippy::too_many_arguments)]
    fn ggsw_blind_rotation_from_scalar_reference<R, A, K>(
        &self,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        test_vector: &A,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: ScalarZnxToBackendRef<BE>,
        K: GetGGSWBit<BE>;
}

impl<T: UnsignedInteger, BE> GGSWBlindRotationReference<T, BE> for Module<BE>
where
    BE: Backend<ZnxWord = i64> + poulpy_hal::oep::HalVecZnxImpl,
    Module<BE>: GLWEBytesOf<BE> + GLWEBlindRotation<BE> + GLWEZero<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
{
    fn ggsw_blind_rotation_from_scalar_tmp_bytes_reference<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        ggsw_blind_rotation_from_scalar_tmp_bytes_reference::<T, BE, R, K>(self, res_infos, k_infos)
    }

    fn ggsw_blind_rotation_from_scalar_reference<R, A, K>(
        &self,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        res: &mut R,
        test_vector: &A,
        fhe_uint: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: ScalarZnxToBackendRef<BE>,
        K: GetGGSWBit<BE>,
    {
        ggsw_blind_rotation_from_scalar_reference::<T, BE, R, A, K>(
            self,
            sign,
            bit_rsh,
            bit_mask,
            bit_lsh,
            res,
            test_vector,
            fhe_uint,
            scratch,
        )
    }
}
