//! Reference implementations of the [`GGSWKeyswitchReference`] methods.
//!
//! Re-exported publicly through `crate::oep::ggsw_keyswitch_reference`.

use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena},
};

use crate::{
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, LWEInfos,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
    },
    oep::{ConversionReference, GGSWKeyswitchReference, GLWEKeyswitchReference},
};

pub fn ggsw_keyswitch_tmp_bytes_reference<BE, M, R, A, K, T>(
    module: &M,
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
    tsk_infos: &T,
) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchReference<BE> + ConversionReference<BE>,
    R: GGSWInfos,
    A: GGSWInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    assert_eq!(key_infos.rank_in(), key_infos.rank_out());
    assert_eq!(tsk_infos.rank_in(), tsk_infos.rank_out());
    assert_eq!(key_infos.rank_in(), tsk_infos.rank_in());
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());
    assert_eq!(module.n() as u32, tsk_infos.n());

    module
        .glwe_keyswitch_tmp_bytes_reference(res_infos, a_infos, key_infos)
        .max(module.ggsw_expand_rows_tmp_bytes_reference(res_infos, tsk_infos))
}

#[allow(clippy::too_many_arguments)]
pub fn ggsw_keyswitch_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGSWKeyswitchReference<BE> + ModuleN + GLWEKeyswitchReference<BE> + ConversionReference<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE> + GGSWInfos,
{
    let mut res_backend = res.to_backend_mut();
    let a_backend = a.to_backend_ref();

    assert!(res_backend.dnum() <= a_backend.dnum());
    assert_eq!(res_backend.dsize(), a_backend.dsize());
    assert_eq!(res_backend.base2k(), a_backend.base2k());
    assert!(
        scratch.available() >= module.ggsw_keyswitch_tmp_bytes_reference(&res_backend, &a_backend, key, tsk),
        "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_keyswitch_tmp_bytes_reference(&res_backend, &a_backend, key, tsk)
    );

    for row in 0..a_backend.dnum().into() {
        let mut res_at = res_backend.at_view_mut(row, 0);
        let a_at = a_backend.at_view(row, 0);
        module.glwe_keyswitch_reference(&mut res_at, &a_at, &key.to_backend_ref(), &mut scratch.borrow());
    }

    module.ggsw_expand_row_reference(&mut res_backend, tsk, scratch)
}

pub fn ggsw_keyswitch_assign_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGSWKeyswitchReference<BE> + ModuleN + GLWEKeyswitchReference<BE> + ConversionReference<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
{
    let mut res_backend = res.to_backend_mut();

    assert!(
        scratch.available() >= module.ggsw_keyswitch_tmp_bytes_reference(&res_backend, &res_backend, key, tsk),
        "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_keyswitch_tmp_bytes_reference(&res_backend, &res_backend, key, tsk)
    );

    for row in 0..res_backend.dnum().into() {
        let mut res_at = res_backend.at_view_mut(row, 0);
        module.glwe_keyswitch_assign_reference(&mut res_at, &key.to_backend_ref(), &mut scratch.borrow());
    }

    module.ggsw_expand_row_reference(&mut res_backend, tsk, scratch)
}
