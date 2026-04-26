use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub(crate) trait CKKSConjugateDefault<BE: Backend> {
    fn ckks_conjugate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_conjugate_into_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(src);
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_automorphism_assign(dst, key, scratch);
        } else {
            self.glwe_automorphism(dst, src, key, scratch);
        }

        dst.meta = src.meta();
        dst.meta.log_budget = checked_log_budget_sub("conjugate", dst.log_budget(), offset)?;
        Ok(())
    }

    fn ckks_conjugate_assign_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_automorphism_assign(dst, key, scratch);
        Ok(())
    }
}

impl<BE: Backend> CKKSConjugateDefault<BE> for Module<BE> {}
