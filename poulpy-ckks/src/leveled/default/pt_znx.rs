use anyhow::Result;
use poulpy_core::{
    ScratchTakeCore,
    layouts::{GLWEPlaintext, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, ensure_base2k_match, ensure_plaintext_alignment,
    layouts::{CKKSCiphertext, CKKSPlaintextVecZnx},
};

pub(crate) trait CKKSPlaintextZnxDefault<BE: Backend> {
    fn ckks_add_pt_vec_znx_into_default(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        pt: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshAddInto<BE>,
    {
        ensure_base2k_match("ckks_add_pt_vec_znx_into", ct.base2k().as_usize(), pt.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_add_pt_vec_znx_into",
            ct.log_budget(),
            pt.log_delta(),
            pt.max_k().as_usize(),
        )?;
        self.vec_znx_rsh_add_into(ct.base2k().as_usize(), offset, ct.data_mut(), 0, pt.data(), 0, scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_into_default(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxRshSub<BE>,
    {
        ensure_base2k_match("ckks_sub_pt_vec_znx_into", ct.base2k().as_usize(), pt_znx.base2k().as_usize())?;
        let offset = ensure_plaintext_alignment(
            "ckks_sub_pt_vec_znx_into",
            ct.log_budget(),
            pt_znx.log_delta(),
            pt_znx.max_k().as_usize(),
        )?;
        self.vec_znx_rsh_sub(ct.base2k().as_usize(), offset, ct.data_mut(), 0, pt_znx.data(), 0, scratch);
        Ok(())
    }

    fn ckks_extract_pt_znx_tmp_bytes_default(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes())
    }

    fn ckks_extract_pt_znx_default<S>(
        &self,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &GLWEPlaintext<impl DataRef>,
        src_meta: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE> + VecZnxRsh<BE>,
    {
        ensure_base2k_match("ckks_extract_pt_znx", src.base2k().as_usize(), dst.base2k().as_usize())?;
        let available = src_meta.log_budget() + dst.log_delta();
        if available < dst.effective_k() {
            return Err(crate::CKKSCompositionError::PlaintextAlignmentImpossible {
                op: "ckks_extract_pt_znx",
                ct_log_budget: src_meta.log_budget(),
                pt_log_delta: dst.log_delta(),
                pt_max_k: dst.max_k().as_usize(),
            }
            .into());
        }
        let dst_k = dst.max_k().as_usize();
        if available < dst_k {
            self.vec_znx_rsh(
                dst.base2k().into(),
                dst_k - available,
                dst.data_mut(),
                0,
                src.data(),
                0,
                scratch,
            );
        } else if available > dst_k {
            self.vec_znx_lsh(
                dst.base2k().into(),
                available - dst_k,
                dst.data_mut(),
                0,
                src.data(),
                0,
                scratch,
            );
        } else {
            self.vec_znx_rsh(dst.base2k().into(), 0, dst.data_mut(), 0, src.data(), 0, scratch);
        }
        Ok(())
    }
}

impl<BE: Backend> CKKSPlaintextZnxDefault<BE> for Module<BE> {}
