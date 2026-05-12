//! Reference implementations of the [`GGSWKeyswitchDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::ggsw_keyswitch_defaults`.

#![allow(private_bounds)]

use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, LWEInfos,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
    oep::{ConversionDefault, GGSWKeyswitchDefault, GLWEKeyswitchDefault},
};

pub fn ggsw_keyswitch_tmp_bytes_default<BE, M, R, A, K, T>(
    module: &M,
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
    tsk_infos: &T,
) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchDefault<BE> + ConversionDefault<BE>,
    R: GGSWInfos,
    A: GGSWInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert_eq!(key_infos.rank_in(), key_infos.rank_out());
    assert_eq!(tsk_infos.rank_in(), tsk_infos.rank_out());
    assert_eq!(key_infos.rank_in(), tsk_infos.rank_in());
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());
    assert_eq!(module.n() as u32, tsk_infos.n());

    module
        .glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos)
        .max(module.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
}

#[allow(clippy::too_many_arguments)]
pub fn ggsw_keyswitch_default<'s, BE, M, R, A, K, T>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &K,
    key_size: usize,
    tsk: &T,
    tsk_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGSWKeyswitchDefault<BE> + ModuleN + GLWEKeyswitchDefault<BE> + ConversionDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE> + GGSWInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let mut res_backend = res.to_backend_mut();
    let a_backend = a.to_backend_ref();

    assert!(res_backend.dnum() <= a_backend.dnum());
    assert_eq!(res_backend.dsize(), a_backend.dsize());
    assert_eq!(res_backend.base2k(), a_backend.base2k());
    assert!(
        scratch.available() >= module.ggsw_keyswitch_tmp_bytes(&res_backend, &a_backend, key, tsk),
        "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_keyswitch_tmp_bytes(&res_backend, &a_backend, key, tsk)
    );

    for row in 0..a_backend.dnum().into() {
        let mut res_at = res_backend.at_view_mut(row, 0);
        let a_at = a_backend.at_view(row, 0);
        module.glwe_keyswitch(&mut res_at, &a_at, key, key_size, &mut scratch.borrow());
    }

    module.ggsw_expand_row(&mut res_backend, tsk, tsk_size, scratch)
}

pub fn ggsw_keyswitch_assign_default<'s, BE, M, R, K, T>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    tsk: &T,
    tsk_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGSWKeyswitchDefault<BE> + ModuleN + GLWEKeyswitchDefault<BE> + ConversionDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let mut res_backend = res.to_backend_mut();

    assert!(
        scratch.available() >= module.ggsw_keyswitch_tmp_bytes(&res_backend, &res_backend, key, tsk),
        "scratch.available(): {} < GGSWKeyswitch::ggsw_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_keyswitch_tmp_bytes(&res_backend, &res_backend, key, tsk)
    );

    for row in 0..res_backend.dnum().into() {
        let mut res_at = res_backend.at_view_mut(row, 0);
        module.glwe_keyswitch_assign(&mut res_at, key, key_size, &mut scratch.borrow());
    }

    module.ggsw_expand_row(&mut res_backend, tsk, tsk_size, scratch)
}
