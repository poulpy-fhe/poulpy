use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWEAutomorphismKeyPreparedBackendRef},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    CKKSInfos, SetCKKSInfos,
    api::CKKSRoundOps,
    checked_log_budget_sub, ckks_offset_unary,
    reference::rotate::{automorphism_assign, automorphism_into, automorphism_tmp_bytes},
};

pub trait CKKSConjugateReference<BE: Backend> {
    fn ckks_conjugate_with_key_tmp_bytes_reference<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: crate::CKKSCtBounds,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE>,
    {
        automorphism_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_with_key_into_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        let offset = ckks_offset_unary(dst, src);
        // Validate before mutating: on error `dst` must remain untouched.
        let log_budget = checked_log_budget_sub("ckks_conjugate_into", src.log_budget(), offset)?;

        // Stamp before the key switch, as in `ckks_rotate_into`: the key switch
        // normalizes at `dst.k()`, so a stale, wider `k` would leave the result
        // carrying key-switch noise below the width it is then labelled with.
        // `offset` is computed against the pre-stamp `dst`.
        dst.set_meta(src.meta());
        dst.set_log_budget(log_budget);
        automorphism_into(self, dst, src, key, offset, scratch)
    }

    fn ckks_conjugate_with_key_assign_reference<Dst>(
        &self,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
    {
        automorphism_assign(self, dst, key, scratch)
    }
}

impl<BE: Backend> CKKSConjugateReference<BE> for poulpy_hal::layouts::Module<BE> {}
