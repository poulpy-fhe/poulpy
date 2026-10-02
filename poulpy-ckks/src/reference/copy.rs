use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos};

pub trait CKKSCopyReference<BE: Backend> {
    fn ckks_copy_tmp_bytes_reference<Dst, Src>(&self, dst: &Dst, src: &Src) -> usize
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: poulpy_core::layouts::GLWEInfos,
        Src: poulpy_core::layouts::GLWEInfos,
    {
        self.glwe_shift_tmp_bytes(dst.max_size())
            .max(self.glwe_copy_tmp_bytes(dst, src))
    }

    fn ckks_copy_reference<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSInfos,
    {
        crate::ckks_copy_stamp_unary(self, "copy", dst, src, scratch)
    }
}

impl<BE: Backend> CKKSCopyReference<BE> for poulpy_hal::layouts::Module<BE> {}
