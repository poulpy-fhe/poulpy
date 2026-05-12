use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift, ScratchArenaTakeCore,
    layouts::{Base2K, GLWEPlaintext, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena, VecZnx},
};

use crate::{
    CKKSInfos, CKKSMeta, SetCKKSInfos, checked_log_budget_sub, ckks_offset_binary, ckks_offset_unary,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec},
    leveled::default::CKKSPlaintextDefault,
};

pub(crate) fn ckks_one_pt<BE>(base2k: Base2K) -> Result<CKKSPlaintext<BE::OwnedBuf>>
where
    BE: Backend,
{
    let meta = CKKSMeta {
        log_delta: 1,
        log_budget: 0,
    };

    let mut host_pt = CKKSPlaintext::from_inner(
        GLWEPlaintext::alloc_with_meta(1usize.into(), base2k, meta.min_k(base2k)),
        meta,
    );
    host_pt.encode_host_floats(&[1.0f64])?;

    let shape = host_pt.inner.data.shape();
    let backend_inner = GLWEPlaintext {
        data: VecZnx::from_data_with_max_size(
            BE::from_host_bytes(host_pt.inner.data.data.as_ref()),
            shape.n(),
            shape.cols(),
            shape.size(),
            shape.max_size(),
        ),
        base2k,
    };
    Ok(CKKSPlaintext::from_inner(backend_inner, meta))
}

pub trait CKKSAddDefault<BE: Backend> {
    fn ckks_add_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
    {
        self.glwe_shift_tmp_bytes().max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_add_pt_vec_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_shift_tmp_bytes()
            .max(self.vec_znx_rsh_tmp_bytes())
            .max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_add_pt_const_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_shift_tmp_bytes()
            .max(self.glwe_normalize_tmp_bytes())
            .max(self.vec_znx_rsh_tmp_bytes())
    }

    fn ckks_add_into_default<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + SetCKKSInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_into_unsafe_default(dst, a, b, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_into_unsafe_default<Dst, A, B>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + SetCKKSInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_binary(dst, a, b);

        if offset == 0 && a.log_budget() == b.log_budget() {
            self.glwe_add_into(dst, a, b);
        } else if a.log_budget() <= b.log_budget() {
            self.glwe_lsh(dst, a, offset, scratch);
            self.glwe_lsh_add(dst, b, b.log_budget() - a.log_budget() + offset, scratch);
        } else {
            self.glwe_lsh(dst, b, offset, scratch);
            self.glwe_lsh_add(dst, a, a.log_budget() - b.log_budget() + offset, scratch);
        }

        let log_budget = checked_log_budget_sub("add", a.log_budget().min(b.log_budget()), offset)?;
        dst.set_log_delta(a.log_delta().min(b.log_delta()));
        dst.set_log_budget(log_budget);
        Ok(())
    }

    fn ckks_add_assign_default<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_assign_unsafe_default(dst, a, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_assign_unsafe_default<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let dst_log_budget = dst.log_budget();

        if dst_log_budget < a.log_budget() {
            self.glwe_lsh_add(dst, a, a.log_budget() - dst_log_budget, scratch);
        } else if dst_log_budget > a.log_budget() {
            self.glwe_lsh_assign(dst, dst_log_budget - a.log_budget(), scratch);
            self.glwe_add_assign(dst, a);
        } else {
            self.glwe_add_assign(dst, a);
        }

        dst.set_log_budget(dst_log_budget.min(a.log_budget()));
        dst.set_log_delta(dst.log_delta().min(a.log_delta()));
        Ok(())
    }

    fn ckks_add_one_assign_default<Dst>(&self, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let one = ckks_one_pt::<BE>(dst.base2k())?;
        self.ckks_add_pt_const_assign_default(dst, 0, &one, 0, scratch)
    }

    fn ckks_add_pt_vec_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_vec_into_unsafe_default(dst, a, pt, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_vec_into_unsafe_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.set_meta(a.meta());
        dst.set_log_budget(checked_log_budget_sub("add_pt_vec", a.log_budget(), offset)?);
        self.ckks_add_pt_vec_assign_unsafe_default(dst, pt, scratch)?;
        Ok(())
    }

    fn ckks_add_pt_vec_assign_default<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_vec_assign_unsafe_default(dst, pt, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_vec_assign_unsafe_default<Dst, P>(
        &self,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPlaintextDefault::ckks_add_pt_vec_into_default(self, dst, pt, scratch)?;
        Ok(())
    }

    fn ckks_add_pt_const_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        cst: &P,
        const_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_const_into_unsafe_default(dst, a, dst_coeff, cst, const_coeff, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_const_into_unsafe_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        cst: &P,
        const_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.set_meta(a.meta());
        dst.set_log_budget(checked_log_budget_sub("add_const", a.log_budget(), offset)?);
        self.ckks_add_pt_const_assign_unsafe_default(dst, dst_coeff, cst, const_coeff, scratch)
    }

    fn ckks_add_pt_const_assign_default<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        cst: &P,
        const_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_const_assign_unsafe_default(dst, dst_coeff, cst, const_coeff, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_const_assign_unsafe_default<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        cst: &P,
        const_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPlaintextDefault::ckks_add_pt_const_into_default(self, dst, dst_coeff, cst, const_coeff, scratch)?;
        Ok(())
    }
}

impl<BE: Backend> CKKSAddDefault<BE> for Module<BE> {}
