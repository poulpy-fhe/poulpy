//! Reference implementations of the [`GGSWAutomorphismDefault`] methods.
//!
//! Each free function carries the HAL bounds it actually needs in its own `where` clause.
//!
//! Re-exported publicly through `crate::oep::ggsw_automorphism_defaults`.

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GetGaloisElement,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
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
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    module
        .glwe_automorphism_tmp_bytes(res_infos, a_infos, key_infos)
        .max(module.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos))
}

#[allow(clippy::too_many_arguments)]
pub fn ggsw_automorphism_default<'s, BE, M, R, A, K, T>(
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
    M: GGSWAutomorphismDefault<BE> + GLWEAutomorphismDefault<BE> + ConversionDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGSWToBackendRef<BE> + GGSWInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    {
        let a_backend = a.to_backend_ref();
        let rows = res.dnum().as_usize();
        let mut res_backend = res.to_backend_mut();
        for row in 0..rows {
            let mut res_at = res_backend.at_view_mut(row, 0);
            let a_at = a_backend.at_view(row, 0);
            module.glwe_automorphism(&mut res_at, &a_at, key, key_size, scratch);
        }
    }
    module.ggsw_expand_row(&mut res.to_backend_mut(), tsk, tsk_size, scratch);
}

pub fn ggsw_automorphism_assign_default<'s, BE, M, R, K, T>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    tsk: &T,
    tsk_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGSWAutomorphismDefault<BE> + GLWEAutomorphismDefault<BE> + ConversionDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    {
        let rows = res.dnum().as_usize();
        let mut res_backend = res.to_backend_mut();
        for row in 0..rows {
            let mut res_at = res_backend.at_view_mut(row, 0);
            module.glwe_automorphism_assign(&mut res_at, key, key_size, &mut scratch.borrow());
        }
    }
    module.ggsw_expand_row(&mut res.to_backend_mut(), tsk, tsk_size, scratch);
}
