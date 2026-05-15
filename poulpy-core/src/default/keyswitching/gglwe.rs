//! Reference implementations of the [`GGLWEKeyswitchDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::gglwe_keyswitch_defaults`.

#![allow(private_bounds)]

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, prepared::GGLWEPreparedToBackendRef},
    oep::{GGLWEKeyswitchDefault, GLWEKeyswitchDefault},
};

pub fn gglwe_keyswitch_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEKeyswitchDefault<BE>,
    R: GGLWEInfos,
    A: GGLWEInfos,
    K: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    module.glwe_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos)
}

pub fn gglwe_keyswitch_default<'s, BE, M, R, A, B>(
    module: &M,
    res: &mut R,
    a: &A,
    b: &B,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGLWEKeyswitchDefault<BE> + GLWEKeyswitchDefault<BE>,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    A: GGLWEToBackendRef<BE> + GGLWEInfos,
    B: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(
        res.rank_in(),
        a.rank_in(),
        "res input rank: {} != a input rank: {}",
        res.rank_in(),
        a.rank_in()
    );
    assert_eq!(
        a.rank_out(),
        b.rank_in(),
        "res output rank: {} != b input rank: {}",
        a.rank_out(),
        b.rank_in()
    );
    assert_eq!(
        res.rank_out(),
        b.rank_out(),
        "res output rank: {} != b output rank: {}",
        res.rank_out(),
        b.rank_out()
    );
    assert!(res.dnum() <= a.dnum(), "res.dnum()={} > a.dnum()={}", res.dnum(), a.dnum());
    assert_eq!(res.dsize(), a.dsize(), "res dsize: {} != a dsize: {}", res.dsize(), a.dsize());
    assert_eq!(res.base2k(), a.base2k());
    assert!(
        scratch.available() >= module.gglwe_keyswitch_tmp_bytes_default(res, a, b),
        "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.gglwe_keyswitch_tmp_bytes_default(res, a, b)
    );

    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    for row in 0..res.dnum().into() {
        for col in 0..res.rank_in().into() {
            let mut res_at = res.at_view_mut(row, col);
            let a_at = a.at_view(row, col);
            module.glwe_keyswitch_default(&mut res_at, &a_at, b, key_size, &mut scratch.borrow());
        }
    }
}

pub fn gglwe_keyswitch_assign_default<'s, BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGLWEKeyswitchDefault<BE> + GLWEKeyswitchDefault<BE>,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    A: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let mut res = res.to_backend_mut();

    assert_eq!(
        res.rank_out(),
        a.rank_out(),
        "res output rank: {} != a output rank: {}",
        res.rank_out(),
        a.rank_out()
    );
    assert!(
        scratch.available() >= module.gglwe_keyswitch_tmp_bytes_default(&res, &res, a),
        "scratch.available(): {} < GGLWEKeyswitch::gglwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.gglwe_keyswitch_tmp_bytes_default(&res, &res, a)
    );

    for row in 0..res.dnum().into() {
        for col in 0..res.rank_in().into() {
            let mut res_at = res.at_view_mut(row, col);
            module.glwe_keyswitch_assign_default(&mut res_at, a, key_size, &mut scratch.borrow());
        }
    }
}
