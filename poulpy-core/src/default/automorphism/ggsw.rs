//! Reference implementations of the [`GGSWAutomorphismDefault`] methods.
//!
//! Each free function carries the HAL bounds it actually needs in its own `where` clause.
//!
//! Re-exported publicly through `crate::oep::ggsw_automorphism_defaults`.

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef,
        prepared::{GGLWEToGGSWKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedBackendRef},
    },
    oep::{ConversionDefault, GGSWAutomorphismDefault, GLWEAutomorphismDefault},
};

pub fn ggsw_automorphism_tmp_bytes_default<BE, M, R, A, K, T>(
    module: &M,
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
    tsk_infos: &T,
) -> usize
where
    BE: Backend,
    M: GLWEAutomorphismDefault<BE> + ConversionDefault<BE>,
    R: GGSWInfos,
    A: GGSWInfos,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    module
        .glwe_automorphism_tmp_bytes_default(res_infos, a_infos, key_infos)
        .max(module.ggsw_expand_rows_tmp_bytes_default(res_infos, tsk_infos))
}

#[allow(clippy::too_many_arguments)]
pub fn ggsw_automorphism_default<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGSWAutomorphismDefault<BE> + GLWEAutomorphismDefault<BE> + ConversionDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE> + GGSWInfos,
{
    {
        let a_backend = a.to_backend_ref();
        let rows = res.dnum().as_usize();
        let mut res_backend = res.to_backend_mut();
        for row in 0..rows {
            let mut res_at = res_backend.at_view_mut(row, 0);
            let a_at = a_backend.at_view(row, 0);
            module.glwe_automorphism_default(&mut res_at, &a_at, key, scratch);
        }
    }
    module.ggsw_expand_row_default(&mut res.to_backend_mut(), tsk, scratch);
}

pub fn ggsw_automorphism_assign_default<BE, M, R>(
    module: &M,
    res: &mut R,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
    tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GGSWAutomorphismDefault<BE> + GLWEAutomorphismDefault<BE> + ConversionDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
{
    {
        let rows = res.dnum().as_usize();
        let mut res_backend = res.to_backend_mut();
        for row in 0..rows {
            let mut res_at = res_backend.at_view_mut(row, 0);
            module.glwe_automorphism_assign_default(&mut res_at, key, &mut scratch.borrow());
        }
    }
    module.ggsw_expand_row_default(&mut res.to_backend_mut(), tsk, scratch);
}
