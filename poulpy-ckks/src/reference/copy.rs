use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, ckks_offset_unary};

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
        let offset = ckks_offset_unary(dst, src);
        if offset == 0 {
            // Use the same destination layout as the scratch query. This copy
            // is exact, including radix conversion, so its result is already
            // canonical at the source precision stamped afterward.
            self.glwe_copy(dst, src, scratch);
            dst.set_meta(src.meta());
            dst.set_log_budget(src.log_budget());
        } else {
            crate::ckks_shift_stamp_unary(self, "copy", dst, src, 0, 0, 0, scratch)?;
        }
        Ok(())
    }
}

impl<BE: Backend> CKKSCopyReference<BE> for poulpy_hal::layouts::Module<BE> {}
