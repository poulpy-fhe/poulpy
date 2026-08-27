use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub trait CKKSConjugateDefault<BE: Backend> {
    fn ckks_conjugate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_conjugate_into_default<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        p: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        let offset = ckks_offset_unary(dst, src);
        // Validate before mutating: on error `dst` must remain untouched.
        let log_budget = checked_log_budget_sub("conjugate", src.log_budget(), offset)?;
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_automorphism_assign(dst, p, keys, scratch);
        } else {
            self.glwe_automorphism(dst, src, p, keys, scratch);
        }

        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        Ok(())
    }

    fn ckks_conjugate_assign_default<Dst, H>(
        &self,
        dst: &mut Dst,
        p: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        self.glwe_automorphism_assign(dst, p, keys, scratch);
        Ok(())
    }
}
