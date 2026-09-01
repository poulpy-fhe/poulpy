use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::{Backend, FitsIn, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, ckks_offset_unary};

pub trait CKKSCopyDefault<BE: Backend> {
    fn ckks_copy_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_copy_default<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSInfos,
        <Src as GLWEToBackendRef<BE>>::State: FitsIn<<Dst as GLWEToBackendRef<BE>>::State>,
    {
        let offset = ckks_offset_unary(dst, src);
        if offset == 0 {
            self.glwe_copy(dst, src);
            dst.set_meta(src.meta());
            // `set_meta` no longer carries the budget (it lives in the GLWE `k`),
            // so propagate `src`'s width explicitly.
            dst.set_log_budget(src.log_budget());
        } else {
            crate::ckks_shift_stamp_unary(self, "copy", dst, src, 0, 0, scratch)?;
        }
        Ok(())
    }
}
