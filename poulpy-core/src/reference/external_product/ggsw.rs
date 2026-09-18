//! Reference implementations of the [`GGSWExternalProductReference`] methods.
//!
//! Re-exported publicly through `crate::oep::ggsw_external_product_reference`.

use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ScratchArena},
};

use crate::{
    layouts::{
        GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, LWEInfos,
        prepared::GGSWPreparedBackendRef,
    },
    oep::{GGSWExternalProductReference, GLWEExternalProductReference},
    reference::operations::GLWEZeroReference,
};

pub fn ggsw_external_product_tmp_bytes_reference<BE, M, R, A, B>(module: &M, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
where
    BE: Backend,
    M: GLWEExternalProductReference<BE>,
    R: GGSWInfos,
    A: GGSWInfos,
    B: GGSWInfos,
{
    module.glwe_external_product_tmp_bytes_reference(res_infos, a_infos, b_infos)
}

pub fn ggsw_external_product_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    b: &GGSWPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGSWExternalProductReference<BE> + GLWEExternalProductReference<BE> + GLWEZeroReference<BE>,
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
{
    assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
    assert_eq!(res.rank(), b.rank(), "res rank: {} != b rank: {}", res.rank(), b.rank());
    assert_eq!(res.base2k(), a.base2k());
    assert!(
        scratch.available() >= module.ggsw_external_product_tmp_bytes_reference(res, a, b),
        "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_external_product_tmp_bytes_reference(res, a, b)
    );

    let min_dnum: usize = res.dnum().min(a.dnum()).into();
    let res_dnum: usize = res.dnum().into();
    let res_rank: usize = (res.rank() + 1).into();
    for row in 0..min_dnum {
        for col in 0..res_rank {
            let mut res_at = res.at_view_mut(row, col);
            let a_at = a.at_view(row, col);
            module.glwe_external_product_reference(&mut res_at, &a_at, b, &mut scratch.borrow());
        }
    }

    if min_dnum < res_dnum {
        for row in min_dnum..res_dnum {
            for col in 0..res_rank {
                module.glwe_zero_reference(&mut res.at_view_mut(row, col));
            }
        }
    }
}

pub fn ggsw_external_product_assign_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    a: &GGSWPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGSWExternalProductReference<BE> + GLWEExternalProductReference<BE> + ModuleN,
    R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
{
    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(res.rank(), a.rank(), "res rank: {} != a rank: {}", res.rank(), a.rank());
    assert!(
        scratch.available() >= module.ggsw_external_product_tmp_bytes_reference(res, res, a),
        "scratch.available(): {} < GGSWExternalProduct::ggsw_external_product_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_external_product_tmp_bytes_reference(res, res, a)
    );

    let res_dnum: usize = res.dnum().into();
    let res_rank: usize = (res.rank() + 1).into();
    for row in 0..res_dnum {
        for col in 0..res_rank {
            module.glwe_external_product_assign_reference(&mut res.at_view_mut(row, col), a, &mut scratch.borrow());
        }
    }
}
