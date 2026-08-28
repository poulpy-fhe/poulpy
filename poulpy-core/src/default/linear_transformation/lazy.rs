//! Lazy DFT-domain giant rotations and final normalization.
//!
//! Implements the ROT and Finalize pieces of docs/linear_transformation.md. The PROD
//! result rides through giant rotations in `VecZnxDft`: mask columns are still
//! normalized through scratch BIG/SMALL before key-switching because gadget
//! decomposition requires limb-aligned input, but the body add, automorphism,
//! and cross-giant accumulation remain in DFT. The sub-limb `cnv_offset_lo`
//! shift is folded into the single final normalize at Phase C.

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero,
        VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, ScratchArena, VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxBigToBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendRef, VecZnxToBackendRef,
    },
};

use crate::{
    default::keyswitching::{GGLWEProductDefault, GLWEKeyswitchInternal},
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetGaloisElement, prepared::GGLWEPreparedToBackendRef},
};

pub(super) fn glwe_lazy_giant_automorphism_tmp_bytes<BE, M, R, K>(module: &M, a_infos: &R, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchInternal<BE> + VecZnxBigAutomorphismAssignTmpBytes + VecZnxDftBytesOf + VecZnxIdftApplyTmpBytes,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = a_infos.rank().as_usize() + 1;
    let key_size = key_infos.size();
    let lvl_0 = module.bytes_of_vec_znx_dft(cols, key_size);
    let lvl_1 = module
        .glwe_keyswitch_internal_tmp_bytes(key_infos, a_infos, key_infos)
        .max(module.vec_znx_idft_apply_tmp_bytes())
        .max(module.vec_znx_big_automorphism_assign_tmp_bytes());

    lvl_0 + lvl_1
}

/// Scratch bytes for [`glwe_lazy_giant_automorphism_from_dft`].
pub(super) fn glwe_lazy_giant_automorphism_from_dft_tmp_bytes<BE, M, K>(
    module: &M,
    rank: usize,
    prod_size: usize,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    M: ModuleN + GGLWEProductDefault<BE> + VecZnxBigBytesOf + VecZnxDftBytesOf + VecZnxIdftApplyTmpBytes,
    K: GGLWEInfos,
{
    let cols = rank + 1;
    let key_size = key_infos.size();
    let mask_small_size = prod_size.min(key_size);
    let mask_big = module.bytes_of_vec_znx_big(1, prod_size);
    let mask_dft = module.bytes_of_vec_znx_dft(rank, mask_small_size);
    let mask_small = mask_small_size * core::mem::size_of::<i64>() * module.n();
    let ks_dft = module.bytes_of_vec_znx_dft(cols, key_size);
    let inner = module.gglwe_product_dft_tmp_bytes_default(key_size, mask_small_size, key_infos);

    mask_dft + mask_big + mask_small + module.vec_znx_idft_apply_tmp_bytes() + ks_dft + inner
}

/// ROT for a giant step whose PROD result is still in `VecZnxDft`.
///
/// Each mask column is IDFT'd into one-column BIG scratch before key-switching,
/// the body column is added in DFT with `vec_znx_dft_add_assign`, and the giant
/// automorphism is applied in DFT before the caller folds the contribution into
/// the accumulator.
#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_lazy_giant_automorphism_from_dft<BE, M, K>(
    module: &M,
    res_dft: &mut VecZnxDftBackendMut<'_, BE>,
    prod_dft: &VecZnxDftBackendRef<'_, BE>,
    prod_base2k: usize,
    key: &K,
    output_size: usize,
    term_count: usize,
    accumulate: bool,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let cols = res_dft.cols();
    let rank = cols - 1;
    let key_base2k = key.base2k().as_usize();
    assert_eq!(prod_base2k, key_base2k, "lazy DFT path requires prod_base2k == key.base2k()");
    assert_eq!(prod_dft.cols(), cols);
    let output_size = output_size.min(key.size());
    assert!(res_dft.size() >= output_size);
    let mask_small_size = prod_dft.size().min(output_size);

    let scratch = scratch.borrow();
    let (mut a_dft, mut scratch_1) = scratch.take_vec_znx_dft_scratch(module, rank, mask_small_size);
    {
        let (mut mask_big, scratch_2) = scratch_1.borrow().take_vec_znx_big_scratch(module, 1, prod_dft.size());
        let (mut col_small, mut scratch_3) = scratch_2.take_vec_znx_scratch(module.n(), 1, mask_small_size);
        for c in 0..rank {
            module.vec_znx_idft_apply(&mut mask_big, 0, prod_dft, c + 1, &mut scratch_3.borrow());
            let mask_big_ref = mask_big.to_backend_ref();
            module.vec_znx_big_normalize(
                &mut col_small,
                key_base2k,
                0,
                0,
                &mask_big_ref,
                prod_base2k,
                0,
                &mut scratch_3.borrow(),
            );
            module.vec_znx_dft_apply(1, 0, &mut a_dft, c, &col_small.to_backend_ref(), 0);
        }
    }

    let (mut ks_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, cols, output_size);
    let key_ref = key.to_backend_ref();
    module.gglwe_product_dft_default(
        &mut ks_dft,
        &a_dft.to_backend_ref(),
        &key_ref,
        term_count,
        &mut scratch_2.borrow(),
    );

    // Carry the body in DFT. `vec_znx_dft_add_assign` truncates to `output_size`,
    // matching the existing BIG lazy path's rotated contribution size.
    module.vec_znx_dft_add_assign(&mut ks_dft, 0, prod_dft, 0);

    let plan = module.vec_znx_dft_automorphism_plan(key.p());
    let ks_dft_ref = ks_dft.to_backend_ref();
    for col in 0..cols {
        if accumulate {
            module.vec_znx_dft_automorphism_add_with_plan(&plan, res_dft, col, &ks_dft_ref, col);
        } else {
            module.vec_znx_dft_automorphism_with_plan(&plan, res_dft, col, &ks_dft_ref, col);
        }
    }
}

/// Adds every column of `a` (DFT) into `res` (DFT).
pub(super) fn glwe_dft_add_dft_assign<BE, M>(module: &M, res: &mut VecZnxDftBackendMut<'_, BE>, a: &VecZnxDftBackendRef<'_, BE>)
where
    BE: Backend,
    M: VecZnxDftAddAssign<BE>,
{
    let cols = res.cols();
    assert_eq!(a.cols(), cols);
    for col in 0..cols {
        module.vec_znx_dft_add_assign(res, col, a, col);
    }
}

/// Copies every column of `a` (DFT) into `res` (DFT), zeroing limbs outside
/// the source active size.
pub(super) fn glwe_dft_copy_dft<BE, M>(module: &M, res: &mut VecZnxDftBackendMut<'_, BE>, a: &VecZnxDftBackendRef<'_, BE>)
where
    BE: Backend,
    M: VecZnxDftCopy<BE>,
{
    let cols = res.cols();
    assert_eq!(a.cols(), cols);
    for col in 0..cols {
        module.vec_znx_dft_copy(1, 0, res, col, a, col);
    }
}

/// IDFTs all DFT columns into a same-shaped BIG buffer.
pub(super) fn glwe_idft_dft_into_big<BE, M>(
    module: &M,
    res_big: &mut VecZnxBigBackendMut<'_, BE>,
    a: &mut VecZnxDftBackendMut<'_, BE>,
) where
    BE: Backend,
    M: VecZnxIdftApplyTmpA<BE>,
{
    let cols = res_big.cols();
    assert_eq!(a.cols(), cols);
    assert_eq!(res_big.size(), a.size());
    for col in 0..cols {
        module.vec_znx_idft_apply_tmpa(res_big, col, a, col);
    }
}

/// Final BIG → SMALL normalize with sub-limb offset; this is the single
/// rounding allowed by docs/linear_transformation.md. `cnv_offset_lo` is the
/// fractional limb shift PROD never applied, so it lands here at the end.
#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_normalize_big_into<BE, M, R>(
    module: &M,
    res: &mut R,
    a: &VecZnxBigBackendRef<'_, BE>,
    a_base2k: usize,
    cnv_offset_lo: i64,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: VecZnxBigNormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k().as_usize();
    let mut res_ref = res.to_backend_mut();
    for col in 0..cols {
        module.vec_znx_big_normalize(
            &mut res_ref.data,
            res_base2k,
            cnv_offset_lo,
            col,
            a,
            a_base2k,
            col,
            &mut scratch.borrow(),
        );
    }
}
