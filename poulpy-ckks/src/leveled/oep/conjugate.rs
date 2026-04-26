use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub(crate) trait CKKSConjugateOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>;

    fn ckks_conjugate_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_conjugate_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSConjugateOep<BE> for Module<BE> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        BE::ckks_conjugate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_into(
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
        BE::ckks_conjugate_into(self, dst, src, key, scratch)
    }

    fn ckks_conjugate_assign(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        BE::ckks_conjugate_assign(self, dst, key, scratch)
    }
}
