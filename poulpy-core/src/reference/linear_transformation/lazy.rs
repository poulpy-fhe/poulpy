//! Lazy DFT-domain giant rotations and final normalization.
//!
//! Implements the ROT and Finalize pieces of docs/guides/linear_transformation.md. The PROD
//! result rides through giant rotations in `VecZnxDft`: mask columns are still
//! normalized through scratch BIG/SMALL before key-switching because gadget
//! decomposition requires limb-aligned input, but the body add, automorphism,
//! and cross-giant accumulation remain in DFT. The sub-limb `cnv_offset_lo`
//! shift is folded into the single final normalize at Phase C.

use poulpy_hal::{
    api::{
        ModuleN, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigNormalize, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxDftCopy,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, ScratchArena, VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef},
};

use crate::{
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, LWEInfos},
    reference::keyswitching::GLWEKeyswitchInternal,
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
