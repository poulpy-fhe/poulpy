use anyhow::{Result, bail, ensure};
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETensoring, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddAssign, VecZnxRshAddInto},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, CKKSMeta, checked_log_budget_sub, checked_mul_ct_log_budget,
    layouts::{
        CKKSCiphertext,
        ciphertext::CKKSOffset,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    leveled::api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAddOpsUnsafe, CKKSDotProductOps, CKKSMulAddOps, CKKSMulManyOps, CKKSMulOps,
        CKKSMulSubOps, CKKSRescaleOps, CKKSSubOps,
    },
    oep::CKKSImpl,
};

fn take_mul_tmp<'a, BE: Backend, D: DataMut>(
    dst: &CKKSCiphertext<D>,
    scratch: &'a mut Scratch<BE>,
) -> (CKKSCiphertext<&'a mut [u8]>, &'a mut Scratch<BE>)
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let layout = dst.glwe_layout();
    let (tmp, scratch_r) = scratch.take_glwe(&layout);
    (CKKSCiphertext::from_inner(tmp, CKKSMeta::default()), scratch_r)
}

fn ensure_accumulation_fits<D: Data>(op: &'static str, dst: &CKKSCiphertext<D>, n: usize) -> Result<()> {
    let base2k: usize = dst.base2k().as_usize();
    ensure!(base2k < 64, "{op}: unsupported base2k={base2k}");
    ensure!(
        n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
    Ok(())
}

// --- CKKSAddManyOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSAddManyOps<BE> for Module<BE> {
    fn ckks_add_many_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + CKKSAddOps<BE>,
    {
        self.ckks_add_tmp_bytes()
    }

    fn ckks_add_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsUnsafe<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        match inputs.len() {
            0 => bail!("ckks_add_many: inputs must contain at least one ciphertext"),
            1 => {
                let offset = dst.offset_unary(inputs[0]);
                self.glwe_lsh(dst, inputs[0], offset, scratch);
                dst.meta = inputs[0].meta();
                dst.meta.log_budget = checked_log_budget_sub("ckks_add_many", inputs[0].log_budget(), offset)?;
            }
            _ => {
                ensure_accumulation_fits("ckks_add_many", dst, inputs.len())?;
                unsafe {
                    self.ckks_add_into_unsafe(dst, inputs[0], inputs[1], scratch)?;
                    for ct in &inputs[2..] {
                        self.ckks_add_assign_unsafe(dst, ct, scratch)?;
                    }
                }
                self.glwe_normalize_assign(dst, scratch);
            }
        }
        Ok(())
    }
}

// --- CKKSMulManyOps ---

fn ceil_log2(n: usize) -> usize {
    debug_assert!(n >= 1);
    if n <= 1 { 0 } else { (n - 1).ilog2() as usize + 1 }
}

fn mul_many_rec<BE, D, M>(
    module: &M,
    dst: &mut CKKSCiphertext<impl DataMut>,
    inputs: &[&CKKSCiphertext<D>],
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) -> Result<()>
where
    BE: Backend + CKKSImpl<BE>,
    D: DataRef,
    M: GLWEShift<BE> + GLWETensoring<BE> + CKKSMulOps<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    match inputs.len() {
        1 => {
            let offset = dst.offset_unary(inputs[0]);
            module.glwe_lsh(dst, inputs[0], offset, scratch);
            dst.meta = inputs[0].meta();
            dst.meta.log_budget = checked_log_budget_sub("ckks_mul_many", inputs[0].log_budget(), offset)?;
            Ok(())
        }
        2 => module.ckks_mul_into(dst, inputs[0], inputs[1], tsk, scratch),
        _ => {
            let mid: usize = inputs.len() / 2;
            let (left_slice, right_slice) = inputs.split_at(mid);

            let log_delta: usize = inputs[0].log_delta();
            let left_min_eff_k: usize = left_slice.iter().map(|c| c.effective_k()).min().unwrap();
            let right_min_eff_k: usize = right_slice.iter().map(|c| c.effective_k()).min().unwrap();
            let left_max_k: usize = left_min_eff_k.saturating_sub(ceil_log2(left_slice.len()) * log_delta);
            let right_max_k: usize = right_min_eff_k.saturating_sub(ceil_log2(right_slice.len()) * log_delta);

            let left_layout = GLWELayout {
                n: dst.n(),
                base2k: dst.base2k(),
                k: TorusPrecision(left_max_k as u32),
                rank: dst.rank(),
            };
            let right_layout = GLWELayout {
                n: dst.n(),
                base2k: dst.base2k(),
                k: TorusPrecision(right_max_k as u32),
                rank: dst.rank(),
            };

            let (left_glwe, scratch_a) = scratch.take_glwe(&left_layout);
            let (right_glwe, scratch_b) = scratch_a.take_glwe(&right_layout);
            let mut left: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(left_glwe, CKKSMeta::default());
            let mut right: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(right_glwe, CKKSMeta::default());

            mul_many_rec(module, &mut left, left_slice, tsk, scratch_b)?;
            mul_many_rec(module, &mut right, right_slice, tsk, scratch_b)?;

            module.ckks_mul_into(dst, &left, &right, tsk, scratch_b)
        }
    }
}

impl<BE: Backend + CKKSImpl<BE>> CKKSMulManyOps<BE> for Module<BE> {
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE> + CKKSMulOps<BE>,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 2 {
            return mul_scratch;
        }
        let depth: usize = ceil_log2(n);
        2 * depth * GLWE::<Vec<u8>>::bytes_of_from_infos(res) + mul_scratch
    }

    fn ckks_mul_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if inputs.is_empty() {
            bail!("ckks_mul_many: inputs must contain at least one ciphertext");
        }
        mul_many_rec(self, dst, inputs, tsk, scratch)
    }
}

// --- CKKSMulAddOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSMulAddOps<BE> for Module<BE> {
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_ct(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_into(&mut tmp, a, b, tsk, scratch_r)?;
        self.ckks_add_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_add_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_vec_znx_into(&mut tmp, a, pt_znx, scratch_r)?;
        self.ckks_add_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_add_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEAdd + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_vec_rnx_into(&mut tmp, a, pt_rnx, prec, scratch_r)?;
        self.ckks_add_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_add_pt_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if cst_znx.re().is_none() && cst_znx.im().is_none() {
            return Ok(());
        }

        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_const_znx_into(&mut tmp, a, cst_znx, scratch_r)?;
        self.ckks_add_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_add_pt_const_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            return Ok(());
        }

        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_const_rnx_into(&mut tmp, a, cst_rnx, prec, scratch_r)?;
        self.ckks_add_assign(dst, &tmp, scratch_r)
    }
}

// --- CKKSMulSubOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSMulSubOps<BE> for Module<BE> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_ct(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_into(&mut tmp, a, b, tsk, scratch_r)?;
        self.ckks_sub_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_vec_znx_into(&mut tmp, a, pt_znx, scratch_r)?;
        self.ckks_sub_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWESub + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_vec_rnx_into(&mut tmp, a, pt_rnx, prec, scratch_r)?;
        self.ckks_sub_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_pt_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if cst_znx.re().is_none() && cst_znx.im().is_none() {
            return Ok(());
        }

        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_const_znx_into(&mut tmp, a, cst_znx, scratch_r)?;
        self.ckks_sub_assign(dst, &tmp, scratch_r)
    }

    fn ckks_mul_sub_pt_const_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWESub + GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            return Ok(());
        }

        let (mut tmp, scratch_r) = take_mul_tmp(dst, scratch);
        self.ckks_mul_pt_const_rnx_into(&mut tmp, a, cst_rnx, prec, scratch_r)?;
        self.ckks_sub_assign(dst, &tmp, scratch_r)
    }
}

// --- CKKSDotProductOps ---

fn check_lengths(op: &'static str, a_len: usize, b_len: usize) -> Result<()> {
    if a_len == 0 {
        bail!("{op}: inputs must contain at least one pair");
    }
    if a_len != b_len {
        bail!("{op}: length mismatch between ct vector ({a_len}) and weight vector ({b_len})");
    }
    Ok(())
}

fn accumulate_unnormalized<BE, D, F>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D>,
    n: usize,
    scratch: &mut Scratch<BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend + CKKSImpl<BE>,
    D: DataMut,
    Module<BE>: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsUnsafe<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    F: FnMut(&mut CKKSCiphertext<&mut [u8]>, usize, &mut Scratch<BE>) -> Result<()>,
{
    if n <= 1 {
        return Ok(());
    }
    let layout = dst.glwe_layout();
    let (tmp_glwe, scratch_r) = scratch.take_glwe(&layout);
    let mut tmp: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(tmp_glwe, CKKSMeta::default());
    for i in 1..n {
        mul_term_into_tmp(&mut tmp, i, scratch_r)?;
        unsafe {
            module.ckks_add_assign_unsafe(dst, &tmp, scratch_r)?;
        }
    }
    module.glwe_normalize_assign(dst, scratch_r);
    Ok(())
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDotProductOps<BE> for Module<BE> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 1 {
            return mul_scratch;
        }
        let ct_bytes: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(res);
        let fallback: usize = ct_bytes + mul_scratch.max(self.ckks_add_tmp_bytes());
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };
        let tensor_bytes: usize = GLWETensor::bytes_of_from_infos(&tensor_layout);
        let inner: usize = self
            .ckks_rescale_tmp_bytes()
            .max(self.glwe_tensor_apply_tmp_bytes(&tensor_layout, res, res))
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
        let fast: usize = 2 * n * ct_bytes + tensor_bytes + inner;
        fallback.max(fast)
    }

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_ct<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETensoring<BE>
            + VecZnxAddAssign
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_ct", dst, n)?;

        if n == 1 {
            return self.ckks_mul_into(dst, a[0], b[0], tsk, scratch);
        }

        let a_min_lhr: usize = a.iter().map(|c| c.log_budget()).min().unwrap();
        let b_min_lhr: usize = b.iter().map(|c| c.log_budget()).min().unwrap();
        let a_aligned: bool = a
            .iter()
            .all(|c| c.log_budget() == a_min_lhr && c.log_delta() == a[0].log_delta());
        let b_aligned: bool = b
            .iter()
            .all(|c| c.log_budget() == b_min_lhr && c.log_delta() == b[0].log_delta());
        let uniform_ld =
            a.iter().all(|c| c.log_delta() == a[0].log_delta()) && b.iter().all(|c| c.log_delta() == b[0].log_delta());

        if !uniform_ld {
            self.ckks_mul_into(dst, a[0], b[0], tsk, scratch)?;
            return accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| self.ckks_mul_into(tmp, a[i], b[i], tsk, s));
        }

        let a_ld: usize = a[0].log_delta();
        let b_ld: usize = b[0].log_delta();
        let a_target_eff_k: usize = a_min_lhr + a_ld;
        let b_target_eff_k: usize = b_min_lhr + b_ld;

        let a_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: TorusPrecision(a_target_eff_k as u32),
            rank: dst.rank(),
        };
        let b_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: TorusPrecision(b_target_eff_k as u32),
            rank: dst.rank(),
        };

        let (a_buf_raw, scratch_aa) = if a_aligned {
            (Vec::new(), &mut *scratch)
        } else {
            scratch.take_glwe_slice(n, &a_layout)
        };
        let (b_buf_raw, scratch_ab) = if b_aligned {
            (Vec::new(), scratch_aa)
        } else {
            scratch_aa.take_glwe_slice(n, &b_layout)
        };

        let mut a_buf: Vec<CKKSCiphertext<&mut [u8]>> = a_buf_raw
            .into_iter()
            .map(|g| CKKSCiphertext::from_inner(g, CKKSMeta::default()))
            .collect();
        let mut b_buf: Vec<CKKSCiphertext<&mut [u8]>> = b_buf_raw
            .into_iter()
            .map(|g| CKKSCiphertext::from_inner(g, CKKSMeta::default()))
            .collect();

        if !a_aligned {
            for (i, ai) in a.iter().enumerate() {
                let shift = ai.log_budget() - a_min_lhr;
                self.ckks_rescale_into(&mut a_buf[i], shift, *ai, scratch_ab)?;
            }
        }
        if !b_aligned {
            for (i, bi) in b.iter().enumerate() {
                let shift = bi.log_budget() - b_min_lhr;
                self.ckks_rescale_into(&mut b_buf[i], shift, *bi, scratch_ab)?;
            }
        }

        let dst_max_k: usize = dst.max_k().as_usize();
        let res_log_delta: usize = a_ld.min(b_ld);
        let lhr0: usize = checked_mul_ct_log_budget("dot_product_ct", a_min_lhr, b_min_lhr, a_ld, b_ld)?;
        let res_offset: usize = (lhr0 + res_log_delta).saturating_sub(dst_max_k);
        let res_log_budget: usize = checked_log_budget_sub("dot_product_ct", lhr0, res_offset)?;
        let cnv_offset: usize = a_target_eff_k.max(b_target_eff_k) + res_offset;

        let tensor_max_k: usize = a_target_eff_k.max(b_target_eff_k);
        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: TorusPrecision(tensor_max_k as u32),
            rank: dst.rank(),
        };

        let (mut acc_tensor, scratch_t2) = scratch_ab.take_glwe_tensor(&tensor_layout);

        let a0_ref = if a_aligned { a[0].to_ref() } else { a_buf[0].to_ref() };
        let b0_ref = if b_aligned { b[0].to_ref() } else { b_buf[0].to_ref() };
        self.glwe_tensor_apply(
            cnv_offset,
            &mut acc_tensor,
            &a0_ref,
            a_target_eff_k,
            &b0_ref,
            b_target_eff_k,
            scratch_t2,
        );

        for i in 1..n {
            let ai_ref = if a_aligned { a[i].to_ref() } else { a_buf[i].to_ref() };
            let bi_ref = if b_aligned { b[i].to_ref() } else { b_buf[i].to_ref() };
            self.glwe_tensor_apply_add_assign(
                cnv_offset,
                &mut acc_tensor,
                &ai_ref,
                a_target_eff_k,
                &bi_ref,
                b_target_eff_k,
                scratch_t2,
            );
        }

        self.glwe_tensor_relinearize(&mut dst.to_mut(), &acc_tensor, tsk, tsk.size(), scratch_t2);
        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
        Ok(())
    }

    fn ckks_dot_product_pt_vec_znx<D: DataRef, E: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecZnx<E>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        check_lengths("ckks_dot_product_pt_vec_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec_znx", dst, n)?;
        self.ckks_mul_pt_vec_znx_into(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_znx_into(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_vec_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddInto<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        check_lengths("ckks_dot_product_pt_vec_rnx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec_rnx", dst, n)?;
        self.ckks_mul_pt_vec_rnx_into(dst, a[0], b[0], prec, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_rnx_into(tmp, a[i], b[i], prec, s)
        })
    }

    fn ckks_dot_product_pt_const_znx<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstZnx],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        check_lengths("ckks_dot_product_pt_const_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_const_znx", dst, n)?;
        self.ckks_mul_pt_const_znx_into(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_znx_into(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_const_rnx<D: DataRef, F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstRnx<F>],
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        check_lengths("ckks_dot_product_pt_const_rnx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_const_rnx", dst, n)?;
        self.ckks_mul_pt_const_rnx_into(dst, a[0], b[0], prec, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_rnx_into(tmp, a[i], b[i], prec, s)
        })
    }
}
