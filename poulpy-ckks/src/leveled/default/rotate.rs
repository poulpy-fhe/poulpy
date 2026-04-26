use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GLWEInfos, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    CKKSCompositionError, CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub(crate) trait CKKSRotateDefault<BE: Backend> {
    fn ckks_rotate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_rotate_into_default<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate",
                rotation: k,
            })?;

        let offset = dst.offset_unary(src);

        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_automorphism_assign(dst, key, scratch);
        } else {
            self.glwe_automorphism(dst, src, key, scratch);
        }

        dst.meta = src.meta();
        dst.meta.log_budget = checked_log_budget_sub("rotate", dst.log_budget(), offset)?;
        Ok(())
    }

    fn ckks_rotate_assign_default<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate_assign",
                rotation: k,
            })?;
        self.glwe_automorphism_assign(dst, key, scratch);
        Ok(())
    }
}

impl<BE: Backend> CKKSRotateDefault<BE> for Module<BE> {}
