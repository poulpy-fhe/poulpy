use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos};
use poulpy_hal::{
    api::{
        VecZnxLshAddCoeffToCoeffBackend, VecZnxLshAddIntoBackend, VecZnxLshBackend, VecZnxLshSubBackend,
        VecZnxLshSubCoeffToCoeffBackend, VecZnxLshTmpBytes, VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend,
        VecZnxRshBackend, VecZnxRshSubBackend, VecZnxRshSubCoeffIntoBackend, VecZnxRshTmpBytes,
    },
    layouts::{Backend, ScratchArena},
};

use crate::GLWEToBackendRef;

use crate::{
    CKKSInfos, CKKSMeta, SetCKKSInfos, ensure_base2k_match, ensure_plaintext_alignment, ensure_plaintext_coeff_in_range,
    ensure_plaintext_degree_match,
};

#[derive(Clone, Copy)]
enum PlaintextShift {
    Rsh(usize),
    Lsh(usize),
}

fn plaintext_shift<P: CKKSInfos + LWEInfos>(ct_log_budget: usize, pt: &P) -> PlaintextShift {
    let available = ct_log_budget + pt.log_delta();
    let encoded_k = pt.max_k().as_usize();
    if available >= encoded_k {
        PlaintextShift::Rsh(available - encoded_k)
    } else {
        PlaintextShift::Lsh(encoded_k - available)
    }
}

pub trait CKKSPlaintextDefault<BE: Backend> {
    fn ckks_add_pt_vec_into_default<Dst, A>(&self, ct: &mut Dst, pt: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        const OP: &str = "ckks_add_pt_vec";
        ensure_base2k_match(OP, ct.base2k().as_usize(), pt.base2k().as_usize())?;
        ensure_plaintext_degree_match(OP, ct.n().as_usize(), pt.n().as_usize())?;
        ensure_plaintext_alignment(OP, ct.log_budget(), pt.log_delta(), pt.effective_k())?;
        let shift = plaintext_shift(ct.log_budget(), pt);
        let base2k = ct.base2k().as_usize();
        let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
        let pt_ref = GLWEToBackendRef::to_backend_ref(pt);
        match shift {
            PlaintextShift::Rsh(k) => {
                self.vec_znx_rsh_add_into_backend(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch)
            }
            PlaintextShift::Lsh(k) => {
                self.vec_znx_lsh_add_into_backend(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch)
            }
        }
        Ok(())
    }

    fn ckks_add_pt_const_into_default<Dst, A>(
        &self,
        ct: &mut Dst,
        coeff_ct: usize,
        pt: &A,
        coeff_pt: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddCoeffIntoBackend<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        const OP: &str = "ckks_add_pt_const";
        ensure_base2k_match(OP, ct.base2k().as_usize(), pt.base2k().as_usize())?;
        ensure_plaintext_coeff_in_range(OP, "ciphertext", coeff_ct, ct.n().as_usize())?;
        ensure_plaintext_coeff_in_range(OP, "plaintext", coeff_pt, pt.n().as_usize())?;
        ensure_plaintext_alignment(OP, ct.log_budget(), pt.log_delta(), pt.effective_k())?;
        let shift = plaintext_shift(ct.log_budget(), pt);
        let base2k = ct.base2k().as_usize();
        let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
        let pt_ref = GLWEToBackendRef::to_backend_ref(pt);
        match shift {
            PlaintextShift::Rsh(k) => self.vec_znx_rsh_add_coeff_into_backend(
                base2k,
                k,
                ct_ref.data_mut(),
                0,
                pt_ref.data(),
                0,
                coeff_pt,
                coeff_ct,
                scratch,
            ),
            PlaintextShift::Lsh(k) => self.vec_znx_lsh_add_coeff_to_coeff_backend(
                base2k,
                k,
                ct_ref.data_mut(),
                0,
                pt_ref.data(),
                0,
                coeff_pt,
                coeff_ct,
                scratch,
            ),
        }

        Ok(())
    }

    fn ckks_sub_pt_const_into_default<Dst, A>(
        &self,
        ct: &mut Dst,
        coeff_ct: usize,
        pt: &A,
        coeff_pt: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubCoeffIntoBackend<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        const OP: &str = "ckks_sub_pt_const";
        ensure_base2k_match(OP, ct.base2k().as_usize(), pt.base2k().as_usize())?;
        ensure_plaintext_coeff_in_range(OP, "ciphertext", coeff_ct, ct.n().as_usize())?;
        ensure_plaintext_coeff_in_range(OP, "plaintext", coeff_pt, pt.n().as_usize())?;
        ensure_plaintext_alignment(OP, ct.log_budget(), pt.log_delta(), pt.effective_k())?;
        let shift = plaintext_shift(ct.log_budget(), pt);
        let base2k = ct.base2k().as_usize();
        let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
        let pt_ref = GLWEToBackendRef::to_backend_ref(pt);
        match shift {
            PlaintextShift::Rsh(k) => self.vec_znx_rsh_sub_coeff_into_backend(
                base2k,
                k,
                ct_ref.data_mut(),
                0,
                pt_ref.data(),
                0,
                coeff_pt,
                coeff_ct,
                scratch,
            ),
            PlaintextShift::Lsh(k) => self.vec_znx_lsh_sub_coeff_to_coeff_backend(
                base2k,
                k,
                ct_ref.data_mut(),
                0,
                pt_ref.data(),
                0,
                coeff_pt,
                coeff_ct,
                scratch,
            ),
        }

        Ok(())
    }

    fn ckks_sub_pt_vec_into_default<Dst, A>(&self, ct: &mut Dst, pt: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        const OP: &str = "ckks_sub_pt_vec";
        ensure_base2k_match(OP, ct.base2k().as_usize(), pt.base2k().as_usize())?;
        ensure_plaintext_degree_match(OP, ct.n().as_usize(), pt.n().as_usize())?;
        ensure_plaintext_alignment(OP, ct.log_budget(), pt.log_delta(), pt.effective_k())?;
        let shift = plaintext_shift(ct.log_budget(), pt);
        let base2k = ct.base2k().as_usize();
        let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
        let pt_ref = GLWEToBackendRef::to_backend_ref(pt);
        match shift {
            PlaintextShift::Rsh(k) => self.vec_znx_rsh_sub_backend(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch),
            PlaintextShift::Lsh(k) => self.vec_znx_lsh_sub_backend(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch),
        }
        Ok(())
    }

    fn ckks_extract_pt_tmp_bytes_default(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        self.vec_znx_rsh_tmp_bytes().max(self.vec_znx_lsh_tmp_bytes())
    }

    fn ckks_extract_pt_default<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
    {
        self.ckks_extract_pt_with_meta_default(dst, src, src.meta(), scratch)
    }

    fn ckks_extract_pt_with_meta_default<D, S>(
        &self,
        dst: &mut D,
        src: &S,
        src_meta: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos,
        Self: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
    {
        ensure_base2k_match("ckks_extract_pt", src.base2k().as_usize(), dst.base2k().as_usize())?;
        let available = src_meta.log_budget() + dst.log_delta();
        if available < dst.effective_k() {
            return Err(crate::CKKSCompositionError::PlaintextAlignmentImpossible {
                op: "ckks_extract_pt",
                ct_log_budget: src_meta.log_budget(),
                pt_log_delta: dst.log_delta(),
                pt_k: dst.effective_k(),
            }
            .into());
        }
        let dst_k = dst.max_k().as_usize();
        let dst_base2k: usize = dst.base2k().into();
        let mut dst_ref = GLWEToBackendMut::to_backend_mut(dst);
        let src_ref = GLWEToBackendRef::to_backend_ref(src);

        if available < dst_k {
            self.vec_znx_rsh_backend(
                dst_base2k,
                dst_k - available,
                dst_ref.data_mut(),
                0,
                src_ref.data(),
                0,
                scratch,
            );
        } else if available > dst_k {
            self.vec_znx_lsh_backend(
                dst_base2k,
                available - dst_k,
                dst_ref.data_mut(),
                0,
                src_ref.data(),
                0,
                scratch,
            );
        } else {
            self.vec_znx_rsh_backend(dst_base2k, 0, dst_ref.data_mut(), 0, src_ref.data(), 0, scratch);
        }
        Ok(())
    }
}
