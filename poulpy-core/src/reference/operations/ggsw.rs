use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    layouts::{GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut},
    reference::operations::GLWERotateReference,
};

#[doc(hidden)]
pub trait GGSWRotateReference<BE: Backend> {
    fn ggsw_rotate_tmp_bytes_reference(&self) -> usize;

    fn ggsw_rotate_reference<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: crate::layouts::GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: crate::layouts::GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos;

    fn ggsw_rotate_assign_reference<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos;
}

impl<BE: Backend> GGSWRotateReference<BE> for Module<BE>
where
    Module<BE>: GLWERotateReference<BE>,
{
    fn ggsw_rotate_tmp_bytes_reference(&self) -> usize {
        self.glwe_rotate_tmp_bytes_reference()
    }

    fn ggsw_rotate_reference<R, A>(&self, k: i64, res: &mut R, a: &A)
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
                self.glwe_rotate_reference(k, &mut res_at, &a_at);
            }
        }
    }

    fn ggsw_rotate_assign_reference<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    {
        assert!(
            scratch.available() >= Self::ggsw_rotate_tmp_bytes_reference(self),
            "scratch.available(): {} < GGSWRotate::ggsw_rotate_tmp_bytes: {}",
            scratch.available(),
            Self::ggsw_rotate_tmp_bytes_reference(self)
        );

        let rows: usize = res.dnum().into();
        let cols: usize = (res.rank() + 1).into();

        for row in 0..rows {
            for col in 0..cols {
                let mut scratch_iter = scratch.borrow();
                let mut res_at = res.at_view_mut(row, col);
                self.glwe_rotate_assign_reference(k, &mut res_at, &mut scratch_iter);
            }
        }
    }
}
