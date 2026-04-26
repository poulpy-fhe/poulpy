use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Scratch};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub trait CKKSConjugateOps<BE: Backend + CKKSImpl<BE>> {
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
