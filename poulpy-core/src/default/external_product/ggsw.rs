//! Reference implementations of the [`GGSWExternalProductDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::ggsw_external_product_defaults`.

#![allow(private_bounds)]

use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    default::operations::GLWEZeroDefault,
    layouts::{GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, prepared::GGSWPreparedToBackendRef},
    oep::{GGSWExternalProductDefault, GLWEExternalProductDefault},
};

pub fn ggsw_external_product_tmp_bytes_default<BE, M, R, A, B>(module: &M, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
where
    BE: Backend,
    M: GLWEExternalProductDefault<BE>,
    R: GGSWInfos,
    A: GGSWInfos,
    B: GGSWInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    module.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
}

pub fn ggsw_external_product_default<'s, BE, M, R, A, B>(
    module: &M,
    res: &mut R,
    a: &A,
    b: &B,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGSWExternalProductDefault<BE> + GLWEExternalProductDefault<BE> + GLWEZeroDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
    B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
    assert_eq!(res.rank(), b.rank(), "res rank: {} != b rank: {}", res.rank(), b.rank());
    assert_eq!(res.base2k(), a.base2k());
    assert!(
        scratch.available() >= module.ggsw_external_product_tmp_bytes(res, a, b),
        "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_external_product_tmp_bytes(res, a, b)
    );

    let min_dnum: usize = res.dnum().min(a.dnum()).into();
    let res_dnum: usize = res.dnum().into();
    let res_rank: usize = (res.rank() + 1).into();
    for row in 0..min_dnum {
        for col in 0..res_rank {
            let mut res_at = res.at_view_mut(row, col);
            let a_at = a.at_view(row, col);
            module.glwe_external_product(&mut res_at, &a_at, b, key_size, &mut scratch.borrow());
        }
    }

    if min_dnum < res_dnum {
        for row in min_dnum..res_dnum {
            for col in 0..res_rank {
                module.glwe_zero_default(&mut res.at_view_mut(row, col));
            }
        }
    }
}

pub fn ggsw_external_product_assign_default<'s, BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGSWExternalProductDefault<BE> + GLWEExternalProductDefault<BE> + ModuleN,
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
    assert!(
        scratch.available() >= module.ggsw_external_product_tmp_bytes(res, res, a),
        "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_external_product_tmp_bytes(res, res, a)
    );

    let res_dnum: usize = res.dnum().into();
    let res_rank: usize = (res.rank() + 1).into();
    for row in 0..res_dnum {
        for col in 0..res_rank {
            module.glwe_external_product_assign(&mut res.at_view_mut(row, col), a, key_size, &mut scratch.borrow());
        }
    }
}
