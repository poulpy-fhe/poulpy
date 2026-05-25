//! Lazy BIG-domain giant rotations and final normalization.
//!
//! Implements the ROT and Finalize pieces of docs/lt_bsgs.md §6.3-§6.4. The PROD
//! result rides through giant rotations in `VecZnxBig`: only the mask columns
//! are dropped to SMALL because gadget decomposition requires limb-aligned
//! input. The body lives in BIG the whole way; the sub-limb `cnv_offset_lo`
//! shift is folded into the single final normalize at Phase C.

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAddAssign, VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, ScratchArena, VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxDftToBackendRef, VecZnxToBackendRef, ZnxInfos,
    },
};

use crate::{
    default::keyswitching::{GGLWEProductDefault, GLWEKeyswitchInternal},
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetGaloisElement, prepared::GGLWEPreparedToBackendRef},
};

pub(super) fn glwe_lazy_giant_automorphism_tmp_bytes<BE, M, R, K>(
    module: &M,
    a_infos: &R,
    key_infos: &K,
    key_size: usize,
) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchInternal<BE> + VecZnxBigAutomorphismAssignTmpBytes + VecZnxDftBytesOf + VecZnxIdftApplyTmpBytes,
    R: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = a_infos.rank().as_usize() + 1;
    let key_size = key_size.min(key_infos.size());
    let lvl_0 = module.bytes_of_vec_znx_dft(cols, key_size);
    let lvl_1 = module
        .glwe_keyswitch_internal_tmp_bytes(key_infos, a_infos, key_infos)
        .max(module.vec_znx_idft_apply_tmp_bytes())
        .max(module.vec_znx_big_automorphism_assign_tmp_bytes());

    lvl_0 + lvl_1
}

/// Scratch bytes for [`glwe_lazy_giant_automorphism_from_big`].
pub(super) fn glwe_lazy_giant_automorphism_from_big_tmp_bytes<BE, M, K>(
    module: &M,
    rank: usize,
    prod_size: usize,
    key_infos: &K,
    key_size: usize,
) -> usize
where
    BE: Backend,
    M: ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes,
    K: GGLWEInfos,
{
    let cols = rank + 1;
    let key_size = key_size.min(key_infos.size());
    let mask_small_size = prod_size.min(key_size);

    // Mask DFT (rank cols, mask_small_size limbs) lives alongside a one-column
    // SMALL workspace used to normalize each mask column before its DFT.
    let mask_dft = module.bytes_of_vec_znx_dft(rank, mask_small_size);
    let mask_small = mask_small_size * core::mem::size_of::<i64>() * module.n();
    // The KS result occupies a (rank+1)-column DFT at key_size, then is IDFT'd
    // and the automorphism runs on the resulting BIG columns.
    let res_dft = module.bytes_of_vec_znx_dft(cols, key_size);
    let inner = module
        .gglwe_product_dft_tmp_bytes_default(key_size, mask_small_size, key_infos)
        .max(module.vec_znx_idft_apply_tmp_bytes())
        .max(module.vec_znx_big_automorphism_assign_tmp_bytes());

    mask_dft + mask_small + res_dft + inner
}

/// ROT for a giant step whose PROD result is already in `VecZnxBig`.
///
/// `prod_big` is the `(r+1)`-column BIG output of PROD (un-normalized, in
/// `prod_base2k`). The mask columns are dropped to SMALL only because the
/// gadget decomposition needs limb-aligned input; the body rides BIG straight
/// into `res_big`, exactly as docs/lt_bsgs.md §6.3 prescribes.
#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_lazy_giant_automorphism_from_big<BE, M, K>(
    module: &M,
    res_big: &mut VecZnxBigBackendMut<'_, BE>,
    prod_big: &VecZnxBigBackendRef<'_, BE>,
    prod_base2k: usize,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let cols = res_big.cols();
    let rank = cols - 1;
    let key_base2k = key.base2k().as_usize();
    assert_eq!(prod_base2k, key_base2k, "lazy BIG path requires prod_base2k == key.base2k()");
    assert_eq!(prod_big.cols(), cols);
    let key_size = key_size.min(key.size());
    assert_eq!(res_big.size(), key_size);
    let mask_small_size = prod_big.size().min(key_size);

    let scratch = scratch.borrow();

    // Stage the mask columns of `prod_big` as `rank` DFT columns. The body
    // column is NOT normalized — it stays in BIG, exactly the spec's lt_bsgs.md
    // §6.3 savings #4 ("body never normalized between steps").
    let (mut a_dft, mut scratch_1) = scratch.take_vec_znx_dft_scratch(module, rank, mask_small_size);
    {
        let (mut col_small, mut scratch_2) = scratch_1.borrow().take_vec_znx_scratch(module.n(), 1, mask_small_size);
        for c in 0..rank {
            module.vec_znx_big_normalize(
                &mut col_small,
                key_base2k,
                0,
                0,
                prod_big,
                prod_base2k,
                c + 1,
                &mut scratch_2.borrow(),
            );
            module.vec_znx_dft_apply(1, 0, &mut a_dft, c, &col_small.to_backend_ref(), 0);
        }
    }

    let (mut res_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, cols, key_size);
    for col in 0..res_dft.cols() {
        module.vec_znx_dft_zero(&mut res_dft, col);
    }
    let key_ref = key.to_backend_ref();
    module.gglwe_product_dft_default(&mut res_dft, &a_dft.to_backend_ref(), &key_ref, &mut scratch_2.borrow());

    {
        let res_dft_ref = res_dft.to_backend_ref();
        for col in 0..cols {
            module.vec_znx_idft_apply(res_big, col, &res_dft_ref, col, &mut scratch_2.borrow());
        }
    }

    // Carry the BIG body and apply the automorphism on every BIG column.
    module.vec_znx_big_add_assign(res_big, 0, prod_big, 0);
    for col in 0..cols {
        module.vec_znx_big_automorphism_assign(key.p(), res_big, col, &mut scratch_2.borrow());
    }
}

/// Adds every column of `a` (BIG) into `res` (BIG). Used both to fold a
/// rotated giant contribution into the lazy accumulator and to absorb a
/// `j == 0` PROD straight into it without normalizing — see docs/lt_bsgs.md
/// §8 savings #4 and #6.
pub(super) fn glwe_big_add_big_assign<BE, M>(module: &M, res: &mut VecZnxBigBackendMut<'_, BE>, a: &VecZnxBigBackendRef<'_, BE>)
where
    BE: Backend,
    M: VecZnxBigAddAssign<BE>,
{
    let cols = res.cols();
    for col in 0..cols {
        module.vec_znx_big_add_assign(res, col, a, col);
    }
}

/// Final BIG → SMALL normalize with sub-limb offset; this is the single
/// rounding allowed by docs/lt_bsgs.md §6.4 / saving #4. `cnv_offset_lo` is the
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
