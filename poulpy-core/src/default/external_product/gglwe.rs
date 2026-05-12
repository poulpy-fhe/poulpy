//! Reference implementations of the [`GGLWEExternalProductDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::gglwe_external_product_defaults`.

#![allow(private_bounds)]

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    default::operations::GLWEZeroDefault,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GLWEViewMut, prepared::GGSWPreparedToBackendRef},
    oep::{GGLWEExternalProductDefault, GLWEExternalProductDefault},
};

pub fn gglwe_external_product_tmp_bytes_default<BE, M, R, A, B>(module: &M, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
where
    BE: Backend,
    M: GLWEExternalProductDefault<BE>,
    R: GGLWEInfos,
    A: GGLWEInfos,
    B: GGSWInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    module.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
}

pub fn gglwe_external_product_default<'s, BE, M, R, A, B>(
    module: &M,
    res: &mut R,
    a: &A,
    b: &B,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGLWEExternalProductDefault<BE> + GLWEExternalProductDefault<BE> + GLWEZeroDefault<BE>,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    A: GGLWEToBackendRef<BE> + GGLWEInfos,
    B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(
        res.rank_in(),
        a.rank_in(),
        "res input rank_in: {} != a input rank_in: {}",
        res.rank_in(),
        a.rank_in()
    );
    assert_eq!(
        a.rank_out(),
        b.rank(),
        "a output rank_out: {} != b rank: {}",
        a.rank_out(),
        b.rank()
    );
    assert_eq!(
        res.rank_out(),
        b.rank(),
        "res output rank_out: {} != b rank: {}",
        res.rank_out(),
        b.rank()
    );
    assert_eq!(res.base2k(), a.base2k());
    assert!(
        scratch.available() >= module.gglwe_external_product_tmp_bytes(res, a, b),
        "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
        scratch.available(),
        module.gglwe_external_product_tmp_bytes(res, a, b)
    );

    let min_dnum: usize = res.dnum().min(a.dnum()).into();
    let res_dnum: usize = res.dnum().into();
    let res_rank_in: usize = res.rank_in().into();
    {
        let mut res = res.to_backend_mut();
        let a = a.to_backend_ref();
        for row in 0..min_dnum {
            for col in 0..res_rank_in {
                let mut res_at = res.at_view_mut(row, col);
                let a_at = a.at_view(row, col);
                module.glwe_external_product(&mut res_at, &a_at, b, key_size, &mut scratch.borrow());
            }
        }
    }

    if min_dnum < res_dnum {
        let mut res = res.to_backend_mut();
        for row in min_dnum..res_dnum {
            for col in 0..res_rank_in {
                let mut ct: GLWEViewMut<'_, BE> = res.at_view_mut(row, col);
                module.glwe_zero_default(&mut ct);
            }
        }
    }
}

pub fn gglwe_external_product_assign_default<'s, BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGLWEExternalProductDefault<BE> + GLWEExternalProductDefault<BE>,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(
        res.rank_out(),
        a.rank(),
        "res output rank: {} != a rank: {}",
        res.rank_out(),
        a.rank()
    );
    assert!(
        scratch.available() >= module.gglwe_external_product_tmp_bytes(res, res, a),
        "scratch.available(): {} < GGLWEExternalProduct::gglwe_external_product_tmp_bytes: {}",
        scratch.available(),
        module.gglwe_external_product_tmp_bytes(res, res, a)
    );

    let res_dnum: usize = res.dnum().into();
    let res_rank_in: usize = res.rank_in().into();
    let mut res = res.to_backend_mut();
    for row in 0..res_dnum {
        for col in 0..res_rank_in {
            let mut res_at = res.at_view_mut(row, col);
            module.glwe_external_product_assign(&mut res_at, a, key_size, &mut scratch.borrow());
        }
    }
}
