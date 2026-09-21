use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, ckks_offset_unary};

// Predict the metadata stamped before the core copy while retaining the
// destination's physical allocation, which a selected core query may inspect.
struct CopyDestinationInfos<'a, D> {
    dst: &'a D,
    k: poulpy_core::layouts::TorusPrecision,
}
impl<D: poulpy_core::layouts::GLWEInfos> poulpy_core::layouts::LWEInfos for CopyDestinationInfos<'_, D> {
    fn n(&self) -> poulpy_core::layouts::Degree {
        self.dst.n()
    }
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.dst.base2k()
    }
    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.k
    }
    fn max_size(&self) -> usize {
        self.dst.max_size()
    }
}
impl<D: poulpy_core::layouts::GLWEInfos> poulpy_core::layouts::GLWEInfos for CopyDestinationInfos<'_, D> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.dst.rank()
    }
}

pub trait CKKSCopyReference<BE: Backend> {
    fn ckks_copy_tmp_bytes_reference<Dst, Src>(&self, dst: &Dst, src: &Src) -> usize
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: poulpy_core::layouts::GLWEInfos,
        Src: poulpy_core::layouts::GLWEInfos,
    {
        let copied_layout = CopyDestinationInfos {
            dst,
            k: dst.k().min(src.k()),
        };
        self.glwe_shift_tmp_bytes(dst.max_size())
            .max(self.glwe_copy_tmp_bytes(&copied_layout, src))
    }

    fn ckks_copy_reference<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSInfos,
    {
        let offset = ckks_offset_unary(dst, src);
        if offset == 0 {
            dst.set_meta(src.meta());
            // `set_meta` no longer carries the budget (it lives in the GLWE `k`),
            // so propagate `src`'s width explicitly. Stamped before the write,
            // like every unary op, so the label matches the data.
            dst.set_log_budget(src.log_budget());
            self.glwe_copy(dst, src, scratch);
        } else {
            crate::ckks_shift_stamp_unary(self, "copy", dst, src, 0, 0, 0, scratch)?;
        }
        Ok(())
    }
}

impl<BE: Backend> CKKSCopyReference<BE> for poulpy_hal::layouts::Module<BE> {}
