//! Lazy DFT-domain giant rotations and final normalization.
//!
//! Implements the ROT and Finalize pieces of docs/guides/linear_transformation.md. The PROD
//! result rides through giant rotations in `VecZnxDft`: mask columns are still
//! normalized through scratch BIG/SMALL before key-switching because gadget
//! decomposition requires limb-aligned input, but the body add, automorphism,
//! and cross-giant accumulation remain in DFT. The sub-limb `cnv_offset_lo`
//! shift is folded into the single final normalize at Phase C.

use poulpy_hal::layouts::vec_znx_dft_backend_ref_from_mut;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismAddWithPlanTmpBytes, VecZnxDftBytesOf,
        VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, ScratchArena, VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendRef, VecZnxToBackendRef,
    },
};

use crate::{
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, LWEInfos, TorusPrecision, prepared::GGLWEPreparedBackendRef},
    reference::keyswitching::{
        GGLWEProductReference, GLWEKeyswitchInternal,
        glwe::{gadget_dft_apply, gadget_dft_apply_tmp_bytes, gadget_input_size},
    },
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
    let lvl_0 = module.bytes_of_vec_znx_dft(module.n(), cols, key_size);
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
    M: ModuleN
        + GGLWEProductReference<BE>
        + VecZnxBigBytesOf
        + VecZnxDftAutomorphismAddWithPlanTmpBytes
        + VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxIdftNormalizeTmpBytes
        + poulpy_hal::api::VecZnxGadgetDecomposeTmpBytes,
    K: GGLWEInfos,
{
    let cols = rank + 1;
    let key_size = key_infos.size();
    let mask_small_size = prod_size.min(key_size);
    // Each mask column is the gadget operand, at most `mask_small_size` full limbs.
    let (base2k, k_digit, dnum) = (key_infos.base2k(), key_infos.k_digit(), key_infos.dnum());
    let mask_k = TorusPrecision((mask_small_size * base2k.as_usize()) as u32);
    let operand_size = gadget_input_size(base2k, k_digit, dnum, mask_k, mask_small_size);
    let mask_dft = module.bytes_of_vec_znx_dft(module.n(), rank, operand_size);
    let mask_small = mask_small_size * core::mem::size_of::<i64>() * module.n();
    let ks_dft = module.bytes_of_vec_znx_dft(module.n(), cols, key_size);
    let inner = module
        .gglwe_product_dft_tmp_bytes_reference(key_size, operand_size, key_infos)
        .max(module.vec_znx_dft_automorphism_add_with_plan_tmp_bytes(key_size, key_size));
    let digits = gadget_dft_apply_tmp_bytes::<BE, _>(module, base2k, k_digit, dnum, mask_k, mask_small_size);

    let mask_work = module
        .vec_znx_idft_normalize_tmp_bytes(mask_small_size, prod_size)
        .max(digits);
    BE::scratch_aligned(mask_dft) + BE::scratch_aligned(mask_small) + mask_work + BE::scratch_aligned(ks_dft) + inner
}

/// ROT for a giant step whose PROD result is still in `VecZnxDft`.
///
/// Each mask column is normalized from its inverse transform before key-switching
/// ([`glwe_lazy_giant_operand`]), the body column is added in DFT with
/// `vec_znx_dft_add_assign`, and the giant automorphism is applied in DFT before
/// the caller folds the contribution into the accumulator. The mask columns of
/// `prod_dft` are unspecified afterwards.
#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_lazy_giant_automorphism_from_dft<BE, M>(
    module: &M,
    res_dft: &mut VecZnxDftBackendMut<'_, BE>,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    prod_base2k: usize,
    p: i64,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    output_size: usize,
    term_count: usize,
    accumulate: bool,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + GGLWEProductReference<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + poulpy_hal::api::VecZnxIdftNormalize<BE>
        + poulpy_hal::api::VecZnxGadgetDecompose<BE>,
{
    let cols = res_dft.cols();
    assert_eq!(prod_dft.cols(), cols);
    let output_size = output_size.min(key.size());
    assert!(res_dft.size() >= output_size);

    let scratch = scratch.borrow();
    let operand_size = glwe_lazy_giant_operand_size(key, prod_dft.size(), output_size);
    let (mut a_dft, mut scratch_1) = scratch.take_vec_znx_dft_scratch(res_dft.n(), cols - 1, operand_size);
    glwe_lazy_giant_operand(module, &mut a_dft, prod_dft, prod_base2k, key, output_size, &mut scratch_1);

    let (mut ks_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(res_dft.n(), cols, output_size);
    crate::reference::keyswitching::glwe::assert_no_row_offset("glwe_eval_linear_transformation", &key);
    module.gglwe_product_dft_reference(&mut ks_dft, &a_dft.to_backend_ref(), key, term_count, &mut scratch_2.borrow());

    // Carry the body in DFT. `vec_znx_dft_add_assign` truncates to `output_size`,
    // matching the existing BIG lazy path's rotated contribution size.
    module.vec_znx_dft_add_assign(&mut ks_dft, 0, &vec_znx_dft_backend_ref_from_mut(prod_dft), 0);

    let plan = module.vec_znx_dft_automorphism_plan(ks_dft.n(), p);
    let ks_dft_ref = ks_dft.to_backend_ref();
    for col in 0..cols {
        if accumulate {
            module.vec_znx_dft_automorphism_add_with_plan(&plan, res_dft, col, &ks_dft_ref, col, &mut scratch_2);
        } else {
            module.vec_znx_dft_automorphism_with_plan(&plan, res_dft, col, &ks_dft_ref, col);
        }
    }
}

/// Limbs of the gadget operand [`glwe_lazy_giant_operand`] lays out for `key` from a product of
/// `prod_size` limbs rotated over `output_size`, at most `key.size()`.
pub(super) fn glwe_lazy_giant_operand_size<BE: Backend>(
    key: &GGLWEPreparedBackendRef<'_, BE>,
    prod_size: usize,
    output_size: usize,
) -> usize {
    let mask_small_size = prod_size.min(output_size);
    let mask_k = TorusPrecision((mask_small_size * key.base2k().as_usize()) as u32);
    gadget_input_size(key.base2k(), key.k_digit(), key.dnum(), mask_k, mask_small_size)
}

/// The gadget operand of the mask columns of `prod_dft` for `key`, into `a_dft`: each column
/// normalized from its inverse transform over `prod_dft.size().min(output_size)` limbs, then
/// laid out in digits. `output_size` is at most `key.size()`. The mask columns of `prod_dft`
/// are unspecified afterwards, its body column unchanged.
pub(super) fn glwe_lazy_giant_operand<BE, M>(
    module: &M,
    a_dft: &mut VecZnxDftBackendMut<'_, BE>,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    prod_base2k: usize,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    output_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN + VecZnxDftApply<BE> + poulpy_hal::api::VecZnxIdftNormalize<BE> + poulpy_hal::api::VecZnxGadgetDecompose<BE>,
{
    let rank = prod_dft.cols() - 1;
    let key_base2k = key.base2k().as_usize();
    assert_eq!(prod_base2k, key_base2k, "lazy DFT path requires prod_base2k == key.base2k()");
    let mask_small_size = prod_dft.size().min(output_size);
    let (k_digit, dnum) = (key.k_digit(), key.dnum());
    let mask_k = TorusPrecision((mask_small_size * key_base2k) as u32);
    let (mut col_small, mut scratch_1) = scratch.borrow().take_vec_znx_scratch(prod_dft.n(), 1, mask_small_size);
    for c in 0..rank {
        module.vec_znx_idft_normalize(
            &mut col_small,
            key_base2k,
            mask_small_size * key_base2k,
            0,
            0,
            prod_dft,
            c + 1,
            prod_base2k,
            None,
            &mut scratch_1.borrow(),
        );
        gadget_dft_apply(
            module,
            a_dft,
            c,
            &col_small.to_backend_ref(),
            0,
            key.base2k(),
            k_digit,
            dnum,
            mask_k,
            &mut scratch_1.borrow(),
        );
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
    a: &VecZnxDftBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: VecZnxIdftApply<BE>,
{
    let cols = res_big.cols();
    assert_eq!(a.cols(), cols);
    assert_eq!(res_big.size(), a.size());
    for col in 0..cols {
        module.vec_znx_idft_apply(res_big, col, a, col, scratch);
    }
}

/// Final BIG → SMALL normalize with sub-limb offset. `cnv_offset_lo` is the
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
    let res_k = res.k().as_usize();
    let mut res_ref = res.to_backend_mut();
    for col in 0..cols {
        module.vec_znx_big_normalize(
            &mut res_ref.data,
            res_base2k,
            res_k,
            cnv_offset_lo,
            col,
            a,
            a_base2k,
            col,
            &mut scratch.borrow(),
        );
    }
}
