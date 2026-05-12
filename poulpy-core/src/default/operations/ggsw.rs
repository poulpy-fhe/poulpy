use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    default::operations::GLWERotateDefault,
    layouts::{GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut},
};

#[doc(hidden)]
pub trait GGSWRotateDefault<BE: Backend> {
    fn ggsw_rotate_tmp_bytes_default(&self) -> usize;

    fn ggsw_rotate_default<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: crate::layouts::GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: crate::layouts::GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos;

    fn ggsw_rotate_assign_default<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchAvailable;
}

impl<BE: Backend> GGSWRotateDefault<BE> for Module<BE>
where
    Module<BE>: GLWERotateDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn ggsw_rotate_tmp_bytes_default(&self) -> usize {
        self.glwe_rotate_tmp_bytes_default()
    }

    fn ggsw_rotate_default<R, A>(&self, k: i64, res: &mut R, a: &A)
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
                self.glwe_rotate_default(k, &mut res_at, &a_at);
            }
        }
    }

    fn ggsw_rotate_assign_default<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchAvailable,
    {
        assert!(
            scratch.available() >= <Self as GGSWRotateDefault<BE>>::ggsw_rotate_tmp_bytes_default(self),
            "scratch.available(): {} < GGSWRotate::ggsw_rotate_tmp_bytes: {}",
            scratch.available(),
            <Self as GGSWRotateDefault<BE>>::ggsw_rotate_tmp_bytes_default(self)
        );

        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                let mut scratch_iter = scratch.borrow();
                let mut res_at = res.at_view_mut(row, col);
                self.glwe_rotate_assign_default(k, &mut res_at, &mut scratch_iter);
            }
        }
    }
}
