//! Conversions expressed by composing core operations.
use crate::ScratchArenaTakeCore;
use crate::{
    api::{GGSWExpandRows, GLWEBytesOf, GLWECopy, GLWEKeyswitch, GLWERotate, LWESampleExtract},
    layouts::{
        GGLWEInfos, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWELayout, GLWEToBackendRef, LWEInfos,
        LWEToBackendMut, Rank, glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena},
};
pub(crate) fn ggsw_from_gglwe_tmp_bytes_derived<BE, M, R, A, T>(module: &M, res_infos: &R, a_infos: &A, tsk_infos: &T) -> usize
where
    BE: Backend,
    M: GGSWExpandRows<BE> + GLWECopy<BE>,
    R: GGSWInfos,
    A: GGLWEInfos,
    T: GGLWEInfos,
{
    module
        .ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos)
        .max(module.glwe_copy_tmp_bytes(res_infos, a_infos))
}

pub(crate) fn ggsw_from_gglwe_derived<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGSWExpandRows<BE> + ModuleN + GLWECopy<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGLWEToBackendRef<BE> + GGLWEInfos,
{
    let mut res_backend = res.to_backend_mut();
    let a_backend = a.to_backend_ref();

    assert_eq!(res_backend.rank(), a_backend.rank_out());
    assert_eq!(res_backend.dnum(), a_backend.dnum());
    assert_eq!(res_backend.n(), module.n() as u32);
    assert_eq!(a_backend.n(), module.n() as u32);
    assert_eq!(tsk.n(), module.n() as u32);
    assert_eq!(res_backend.base2k(), a_backend.base2k());
    assert!(
        scratch.available() >= ggsw_from_gglwe_tmp_bytes_derived::<BE, _, _, _, _>(module, &res_backend, &a_backend, tsk),
        "scratch.available(): {} < GGSWFromGGLWE::ggsw_from_gglwe_tmp_bytes: {}",
        scratch.available(),
        ggsw_from_gglwe_tmp_bytes_derived::<BE, _, _, _, _>(module, &res_backend, &a_backend, tsk)
    );

    for row in 0..res_backend.dnum().into() {
        let mut res_at = res_backend.at_view_mut(row, 0);
        let a_at = a_backend.at_view(row, 0);
        module.glwe_copy(&mut res_at, &a_at, scratch);
    }

    module.ggsw_expand_row(&mut res_backend, tsk, scratch)
}

pub(crate) fn lwe_from_glwe_tmp_bytes_derived<BE, M, R, A, K>(module: &M, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + ModuleN + GLWEKeyswitch<BE> + GLWERotate<BE>,
    R: LWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, glwe_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let res_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: lwe_infos.base2k(),
        k: lwe_infos.k(),
        rank: Rank(1),
    };

    let lvl_0: usize = module.glwe_bytes_of(module.n().into(), lwe_infos.base2k(), lwe_infos.k(), 1u32.into());
    let lvl_1: usize = module.glwe_keyswitch_tmp_bytes(&res_infos, glwe_infos, key_infos);
    let lvl_2: usize = module.glwe_rotate_tmp_bytes();

    BE::scratch_aligned(lvl_0) + lvl_1.max(lvl_2)
}

pub(crate) fn lwe_from_glwe_derived<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    a_idx: usize,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE> + LWESampleExtract<BE> + ModuleN + GLWEKeyswitch<BE> + GLWERotate<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let a_backend = a.to_backend_ref();

    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);
    assert!(res.n() <= module.n() as u32);
    assert!(
        scratch.available() >= lwe_from_glwe_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, key),
        "scratch.available(): {} < LWEFromGLWE::lwe_from_glwe_tmp_bytes: {}",
        scratch.available(),
        lwe_from_glwe_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, key)
    );

    let glwe_layout: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: res.base2k(),
        k: res.k(),
        rank: Rank(1),
    };

    let scratch = scratch.borrow();
    let (mut tmp_glwe_rank_1, mut scratch_1) = scratch.take_glwe_scratch(&glwe_layout);

    let a_backend_view = &a_backend;
    module.glwe_keyswitch(&mut tmp_glwe_rank_1, &a_backend_view, &key.to_backend_ref(), &mut scratch_1);
    if a_idx != 0 {
        module.glwe_rotate_assign(-(a_idx as i64), &mut tmp_glwe_rank_1, &mut scratch_1);
    }

    let tmp_glwe_rank_1_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe_rank_1);
    module.lwe_sample_extract(res, &&tmp_glwe_rank_1_ref);
}
