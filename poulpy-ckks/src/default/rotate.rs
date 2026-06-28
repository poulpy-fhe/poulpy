use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GGLWEPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_unary};

pub trait CKKSRotateDefault<BE: Backend> {
    fn ckks_rotate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_rotate_into_default<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let offset = ckks_offset_unary(dst, src);

        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_automorphism_assign(dst, key, dst.max_size() + key.dsize().as_usize(), scratch);
        } else {
            self.glwe_automorphism(dst, src, key, src.max_size() + key.dsize().as_usize(), scratch);
        }

        dst.set_meta(src.meta());
        dst.set_log_budget(checked_log_budget_sub("rotate", src.log_budget(), offset)?);
        Ok(())
    }

    fn ckks_rotate_assign_default<Dst, K>(&self, dst: &mut Dst, key: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + LWEInfos + CKKSInfos + SetCKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        self.glwe_automorphism_assign(dst, key, dst.max_size() + key.dsize().as_usize(), scratch);
        Ok(())
    }
}
