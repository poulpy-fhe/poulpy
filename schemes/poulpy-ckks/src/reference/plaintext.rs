use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos};
use poulpy_hal::{
    api::{
        ModuleN, VecZnxLsh, VecZnxLshAdd, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAdd, VecZnxRshSub,
        VecZnxRshTmpBytes,
    },
    layouts::{Backend, Ring, ScratchArena, Standard},
};

use crate::GLWEToBackendRef;

use crate::{CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos, ckks_log_delta, ensure_base2k_match, ensure_plaintext_operand};

#[derive(Clone, Copy)]
enum PlaintextShift {
    Rsh(usize),
    Lsh(usize),
}

// A plaintext is encoded across its whole allocation, so its `max_k` is its `encoded_k`.
fn plaintext_shift<P: CKKSInfos>(ct_log_budget: usize, pt: &P) -> PlaintextShift {
    let available = ct_log_budget + pt.log_delta();
    let encoded_k = pt.max_k().as_usize();
    if available >= encoded_k {
        PlaintextShift::Rsh(available - encoded_k)
    } else {
        PlaintextShift::Lsh(encoded_k - available)
    }
}

/// `ct += pt`, `pt` aligned to the budget of `ct`, without normalization.
pub(crate) fn add_pt_raw<BE, M, Dst, A, Rg: Ring>(
    module: &M,
    op: &'static str,
    ct: &mut Dst,
    pt: &A,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: VecZnxRshAdd<BE, Rg> + VecZnxLshAdd<BE, Rg> + ?Sized,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos,
{
    ensure_plaintext_operand(op, ct.base2k().as_usize(), ct.n().as_usize(), ct.log_budget(), pt)?;
    let shift = plaintext_shift(ct.log_budget(), pt);
    let base2k = ct.base2k().as_usize();
    ct.set_canonical(false);
    let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
    let pt_ref = GLWEToBackendRef::to_backend_ref(pt);
    match shift {
        PlaintextShift::Rsh(k) => module.vec_znx_rsh_add(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch),
        PlaintextShift::Lsh(k) => module.vec_znx_lsh_add(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch),
    }
    Ok(())
}

/// `ct -= pt`, `pt` aligned to the budget of `ct`, without normalization.
pub(crate) fn sub_pt_raw<BE, M, Dst, A, Rg: Ring>(
    module: &M,
    op: &'static str,
    ct: &mut Dst,
    pt: &A,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: VecZnxRshSub<BE, Rg> + VecZnxLshSub<BE, Rg> + ?Sized,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos,
{
    ensure_plaintext_operand(op, ct.base2k().as_usize(), ct.n().as_usize(), ct.log_budget(), pt)?;
    let shift = plaintext_shift(ct.log_budget(), pt);
    let base2k = ct.base2k().as_usize();
    ct.set_canonical(false);
    let mut ct_ref = GLWEToBackendMut::to_backend_mut(ct);
    let pt_ref = GLWEToBackendRef::to_backend_ref(pt);
    match shift {
        PlaintextShift::Rsh(k) => module.vec_znx_rsh_sub(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch),
        PlaintextShift::Lsh(k) => module.vec_znx_lsh_sub(base2k, k, ct_ref.data_mut(), 0, pt_ref.data(), 0, scratch),
    }
    Ok(())
}

pub trait CKKSPlaintextVecReference<BE: Backend, Rg: Ring = Standard>: ModuleN {
    fn ckks_extract_pt_tmp_bytes_reference(&self, res_size: usize) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        self.vec_znx_rsh_tmp_bytes(res_size).max(self.vec_znx_lsh_tmp_bytes(res_size))
    }

    fn ckks_extract_pt_reference<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE, Rg> + CKKSInfos + IntPolyInfos + SetCKKSInfos,
        S: GLWEToBackendRef<BE, Rg> + GLWEInfos + CKKSInfos,
        Self: VecZnxLsh<BE, Rg> + VecZnxRsh<BE, Rg>,
    {
        self.ckks_extract_pt_with_meta_reference(dst, src, src.meta(), scratch)
    }

    fn ckks_extract_pt_with_meta_reference<D, S>(
        &self,
        dst: &mut D,
        src: &S,
        src_meta: GLWEPlaintextMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        D: GLWEToBackendMut<BE, Rg> + CKKSInfos + IntPolyInfos + SetCKKSInfos,
        S: GLWEToBackendRef<BE, Rg> + GLWEInfos + LWEInfos,
        Self: VecZnxLsh<BE, Rg> + VecZnxRsh<BE, Rg>,
    {
        crate::ensure_operand_degrees("ckks_extract_pt", self, [src.n()])?;
        ensure_base2k_match("ckks_extract_pt", src.base2k().as_usize(), dst.base2k().as_usize())?;
        if dst.n() != src.n() {
            return Err(crate::CKKSCompositionError::PlaintextDegreeMismatch {
                op: "ckks_extract_pt",
                ct_n: src.n().as_usize(),
                pt_n: dst.n().as_usize(),
                rule: "the ciphertext's",
            }
            .into());
        }
        // The source budget is derived from the decrypted plaintext's torus width
        // `k` (which spans the source ciphertext) and the source scale `log_delta`.
        let src_log_budget = src.k().as_usize().saturating_sub(ckks_log_delta(&src_meta));
        let available = src_log_budget + dst.log_delta();
        // Validity is checked against the meaningful (effective) precision; the
        // physical `max_k` integer-poly width is truncated by the shift below.
        let dst_effective_k = dst.log_delta() + dst.log_budget();
        if available < dst_effective_k {
            return Err(crate::CKKSCompositionError::PlaintextAlignmentImpossible {
                op: "ckks_extract_pt",
                ct_log_budget: src_log_budget,
                pt_log_delta: dst.log_delta(),
                pt_k: dst_effective_k,
            }
            .into());
        }
        let dst_k = dst.encoded_k().as_usize();
        let dst_base2k: usize = dst.base2k().into();
        let mut dst_ref = GLWEToBackendMut::to_backend_mut(dst);
        let src_ref = GLWEToBackendRef::to_backend_ref(src);

        if available < dst_k {
            self.vec_znx_rsh(
                dst_base2k,
                dst_k - available,
                dst_ref.data_mut(),
                0,
                src_ref.data(),
                0,
                scratch,
            );
        } else if available > dst_k {
            self.vec_znx_lsh(
                dst_base2k,
                available - dst_k,
                dst_ref.data_mut(),
                0,
                src_ref.data(),
                0,
                scratch,
            );
        } else {
            self.vec_znx_rsh(dst_base2k, 0, dst_ref.data_mut(), 0, src_ref.data(), 0, scratch);
        }
        Ok(())
    }
}

impl<BE: Backend, Rg: Ring> CKKSPlaintextVecReference<BE, Rg> for poulpy_hal::layouts::Module<BE> {}
