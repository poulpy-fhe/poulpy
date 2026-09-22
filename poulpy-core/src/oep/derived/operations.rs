//! Core-derived arithmetic built from backend-selected core operations.

use crate::layouts::{
    GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
};
use poulpy_hal::layouts::{Module, ScratchArena};

pub(crate) fn ggsw_rotate_tmp_bytes_derived<BE: crate::oep::GLWERotateImpl>(module: &Module<BE>) -> usize {
    BE::glwe_rotate_tmp_bytes(module)
}

pub(crate) fn ggsw_rotate_derived<BE: crate::oep::GLWERotateImpl, R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
where
    R: crate::layouts::GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    A: crate::layouts::GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
{
    assert!(res.dnum() <= a.dnum());
    assert_eq!(res.dsize(), a.dsize());
    assert_eq!(res.rank(), a.rank());
    let rows: usize = res.dnum().into();
    let cols: usize = (res.rank() + 1).into();

    for row in 0..rows {
        for col in 0..cols {
            let mut res_at = res.at_view_mut(row, col);
            let a_at = a.at_view(row, col);
            BE::glwe_rotate(module, k, &mut res_at, &a_at);
        }
    }
}

pub(crate) fn ggsw_rotate_assign_derived<BE: crate::oep::GLWERotateImpl, R>(
    module: &Module<BE>,
    k: i64,
    res: &mut R,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGSWToBackendMut<BE> + GGSWInfos,
{
    let mut res = res.to_backend_mut();
    assert!(
        scratch.available() >= BE::glwe_rotate_tmp_bytes(module),
        "scratch.available(): {} < GGSWRotate::ggsw_rotate_tmp_bytes: {}",
        scratch.available(),
        BE::glwe_rotate_tmp_bytes(module)
    );

    let rows: usize = res.dnum().into();
    let cols: usize = (res.rank() + 1).into();

    for row in 0..rows {
        for col in 0..cols {
            let mut scratch_iter = scratch.borrow();
            let mut res_at = res.at_view_mut(row, col);
            BE::glwe_rotate_assign(module, k, &mut res_at, &mut scratch_iter);
        }
    }
}

/// `res = a - res`, with a rank-zero operand affecting the body only.
/// The old ciphertext's mask is negated as part of negating the whole GLWE.
pub(crate) fn glwe_sub_negate_assign_derived<BE, R, A>(module: &Module<BE>, res: &mut R, a: &A)
where
    BE: crate::oep::GLWESubImpl,
    R: GLWEToBackendMut<BE>,
    A: GLWEToBackendRef<BE>,
{
    {
        let res = res.to_backend_ref();
        let a = a.to_backend_ref();
        assert_eq!(res.n(), module.n() as u32);
        assert_eq!(a.n(), module.n() as u32);
        assert_eq!(res.base2k(), a.base2k());
        assert!(res.rank() == a.rank() || a.rank() == 0);
    }
    BE::glwe_negate_assign(module, res);
    BE::glwe_add_assign(module, res, a);
}
