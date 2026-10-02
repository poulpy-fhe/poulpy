use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWEAutomorphismKeyPreparedBackendRef},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub trait CKKSRotateReference<BE: Backend> {
    fn ckks_rotate_tmp_bytes_reference<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
            .max(self.glwe_shift_tmp_bytes(ct_infos.max_size()))
    }

    fn ckks_rotate_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        let offset = ckks_offset_unary(dst, src);
        // Validate before mutating: on error `dst` must remain untouched.
        let log_budget = checked_log_budget_sub("rotate", src.log_budget(), offset)?;

        // Stamp before the key switch, not after. The key switch normalizes its
        // output at `dst.k()`, so a stale, wider `k` leaves the result carrying
        // key-switch noise below the width it is about to be labelled with, and
        // a convolution consumer rescales by `2^k`, which lifts that noise back
        // to full magnitude. `offset` is computed against the pre-stamp `dst`,
        // so moving the stamp does not change it.
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);

        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_automorphism_assign(dst, key, scratch);
        } else {
            self.glwe_automorphism(dst, src, key, scratch);
        }

        Ok(())
    }

    fn ckks_rotate_assign_reference<Dst>(
        &self,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    {
        self.glwe_automorphism_assign(dst, key, scratch);
        Ok(())
    }
}

impl<BE: Backend> CKKSRotateReference<BE> for poulpy_hal::layouts::Module<BE> {}
