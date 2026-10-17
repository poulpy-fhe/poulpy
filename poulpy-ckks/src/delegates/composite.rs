use crate::reference::composite::ckks_add_many_reference;
use crate::reference::mul::{check_prepared_layout, get_mul_ct_params, mul_one_pass};
use crate::{CKKSCompositionError, CKKSResult as Result, SlotsKind, ckks_bail};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWEAdd, GLWEMulRelinearizeSum, GLWENormalize, GLWEPreparedTensoring, GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorViewMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision,
    },
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::CKKSCopyOps,
    api::{CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSSubOps},
    layouts::{CKKSCiphertext, CKKSCiphertextViewMut, CKKSPreparedLeft, CKKSPreparedRight, ScratchArenaTakeCKKS},
};
use poulpy_core::GLWEBytesOf;

/// Shared body of the fused multiply-then-accumulate composites
/// (`ckks_mul_{add,sub}_*_into`): carve a `dst`-shaped temporary
/// inside a scratch scope, run the variant's multiply into it, then fold it
/// into `dst` with the variant's carry verb.
fn mul_then_combine<BE, Dst, MulF, CombineF>(
    dst: &mut Dst,
    scratch: &mut ScratchArena<'_, BE>,
    mul: MulF,
    combine: CombineF,
) -> Result<()>
where
    BE: Backend,
    Dst: GLWEInfos + CKKSInfos,
    MulF: for<'t> FnOnce(&mut CKKSCiphertextViewMut<'t, BE>, &mut ScratchArena<'t, BE>) -> Result<()>,
    CombineF: for<'t> FnOnce(&mut Dst, &CKKSCiphertextViewMut<'t, BE>, &mut ScratchArena<'t, BE>) -> Result<()>,
{
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        mul(&mut tmp, &mut scratch_local)?;
        combine(dst, &tmp, &mut scratch_local)
    })
}

// --- CKKSAddManyOps ---

impl<BE: Backend> CKKSAddManyOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE>,
{
    fn ckks_add_many_tmp_bytes(&self, res_size: usize) -> usize {
        self.ckks_add_tmp_bytes(res_size)
    }

    fn ckks_add_many<Dst, Src>(&self, dst: &mut Dst, inputs: &[&Src], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        ckks_add_many_reference(self, dst, inputs, scratch)
    }
}

// --- CKKSMulAddOps ---

impl<BE: Backend> CKKSMulAddOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE>,
{
    fn ckks_mul_add_ct_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_tmp_bytes(res, a, b, tsk)
                .max(self.ckks_add_tmp_bytes(res.size()))
    }

    fn ckks_mul_add_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_pt_vec_tmp_bytes(res, a, b)
                .max(self.ckks_add_tmp_bytes(res.size()))
    }

    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_pt_const_tmp_bytes(res, a, b)
                .max(self.ckks_add_tmp_bytes(res.size()))
    }

    fn ckks_mul_add_ct_into<Dst, A, B, H>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_into(tmp, a, b, tsk, s),
            |dst, tmp, s| self.ckks_add_assign(dst, tmp, s),
        )
    }

    fn ckks_mul_add_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_pt_vec_into(tmp, a, pt, s),
            |dst, tmp, s| self.ckks_add_assign(dst, tmp, s),
        )
    }

    fn ckks_mul_add_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_pt_const_into(tmp, a, pt, pt_coeff, s),
            |dst, tmp, s| self.ckks_add_assign(dst, tmp, s),
        )
    }
}

// --- CKKSAffineOps ---

impl<BE: Backend> CKKSAffineOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE>,
{
    fn ckks_affine_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, affine_const: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.ckks_mul_pt_const_tmp_bytes(res, a, affine_const)
            .max(self.ckks_add_pt_const_tmp_bytes(res.size()))
    }

    fn ckks_affine_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        affine_const: &P,
        offset_coeff: usize,
        scale_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        self.ckks_mul_pt_const_into(dst, a, affine_const, scale_coeff, scratch)?;
        self.ckks_add_pt_const_assign(dst, 0, affine_const, offset_coeff, scratch)
    }

    fn ckks_affine_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        affine_const: &P,
        offset_coeff: usize,
        scale_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        self.ckks_mul_pt_const_assign(dst, affine_const, scale_coeff, scratch)?;
        self.ckks_add_pt_const_assign(dst, 0, affine_const, offset_coeff, scratch)
    }

    fn ckks_affine_pt_vec_tmp_bytes<R, A, S>(&self, res: &R, a: &A, scale: &S) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        S: CKKSInfos,
    {
        self.ckks_mul_pt_vec_tmp_bytes(res, a, scale)
            .max(self.ckks_add_pt_vec_tmp_bytes(res.size()))
    }

    fn ckks_affine_pt_vec_into<Dst, A, S, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        scale: &S,
        offset: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        self.ckks_mul_pt_vec_into(dst, a, scale, scratch)?;
        self.ckks_add_pt_vec_assign(dst, offset, scratch)
    }

    fn ckks_affine_pt_vec_assign<Dst, S, P>(
        &self,
        dst: &mut Dst,
        scale: &S,
        offset: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        self.ckks_mul_pt_vec_assign(dst, scale, scratch)?;
        self.ckks_add_pt_vec_assign(dst, offset, scratch)
    }
}

// --- CKKSMulSubOps ---

impl<BE: Backend> CKKSMulSubOps<BE> for Module<BE>
where
    Module<BE>: CKKSMulOps<BE> + CKKSSubOps<BE>,
{
    fn ckks_mul_sub_ct_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_tmp_bytes(res, a, b, tsk)
                .max(self.ckks_sub_tmp_bytes(res.size()))
    }

    fn ckks_mul_sub_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_pt_vec_tmp_bytes(res, a, b)
                .max(self.ckks_sub_tmp_bytes(res.size()))
    }

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_pt_const_tmp_bytes(res, a, b)
                .max(self.ckks_sub_tmp_bytes(res.size()))
    }

    fn ckks_mul_sub_ct_into<Dst, A, B, H>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_into(tmp, a, b, tsk, s),
            |dst, tmp, s| self.ckks_sub_assign(dst, tmp, s),
        )
    }

    fn ckks_mul_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_pt_vec_into(tmp, a, pt, s),
            |dst, tmp, s| self.ckks_sub_assign(dst, tmp, s),
        )
    }

    fn ckks_mul_sub_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_pt_const_into(tmp, a, pt, pt_coeff, s),
            |dst, tmp, s| self.ckks_sub_assign(dst, tmp, s),
        )
    }
}

// --- CKKSDotProductOps ---

fn check_lengths(op: &'static str, a_len: usize, b_len: usize) -> Result<()> {
    if a_len == 0 {
        ckks_bail!("{op}: inputs must contain at least one pair");
    }
    if a_len != b_len {
        ckks_bail!("{op}: length mismatch between ct vector ({a_len}) and weight vector ({b_len})");
    }
    Ok(())
}

/// Whether every ciphertext shares the first one's layout and scale, the inputs of the ct × ct rule.
fn share_layout<C: CKKSCtBounds>(cts: &[&C]) -> bool {
    cts.iter()
        .all(|ct| (ct.glwe_layout(), ct.log_delta()) == (cts[0].glwe_layout(), cts[0].log_delta()))
}

/// `dst = Σ a[i] · b[i]` with one relinearization, for pairs that share one ct × ct rule: the body of
/// `ckks_dot_product_ct`'s lazy path and of `ckks_dot_product_prepared`, which form the products of their pairs with
/// `one_pass` and `tensor`.
///
/// Checks the tensor key and the rule before writing `dst`. Where `ckks_mul_into` takes its one-pass product, the
/// sum is `one_pass(dst, cnv_offset, res_k)`, which adds the products in the transform domain: it writes `dst` at the
/// stamped width `res_k`, then `dst` is stamped, as in `ckks_mul_into`. Otherwise `tensor(t, i, cnv_offset)` writes
/// pair `i`'s tensor product into `t`, canonical at the tensor width, and the terms are added without normalizing, as
/// other lazy additions: the sum is exact, and it is normalized once, right before the relinearization reads it
/// through a DFT. One pair is no sum: one tensor, not normalized, as in `ckks_mul_into`. `dst` is stamped before the
/// relinearization, which normalizes at the stamped width, as in `ckks_mul_into`.
#[allow(clippy::too_many_arguments)]
fn dot_product_lazy<BE, Dst, A, B, H>(
    module: &Module<BE>,
    op: &'static str,
    dst: &mut Dst,
    a: &[&A],
    b: &[&B],
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
    one_pass: impl FnOnce(&mut Dst, usize, TorusPrecision, &mut ScratchArena<'_, BE>),
    mut tensor: impl for<'t> FnMut(&mut GLWETensorViewMut<'t, BE>, usize, usize, &mut ScratchArena<'t, BE>),
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWETensoring<BE> + GLWEAdd<BE> + GLWENormalize<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: CKKSCtBounds,
    B: CKKSCtBounds,
    H: GetTensorKey<BE>,
{
    // `ckks_mul_into`'s tensor width, and so its key width, whatever `dst`'s width.
    let tensor_layout = GLWELayout {
        n: dst.n(),
        base2k: dst.base2k(),
        k: a[0].k().max(b[0].k()),
        rank: dst.rank(),
    };
    tsk.get_tensor_key(tensor_layout.k)
        .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
            op,
            k: tensor_layout.k.into(),
        })?;
    let (log_budget, log_delta, cnv_offset) = get_mul_ct_params(dst, a[0], b[0])?;
    // The eager sum's label and slot kind: the smallest label and the join over every operand.
    let metas = || a.iter().map(|ct| ct.meta()).chain(b.iter().map(|ct| ct.meta()));
    let log_sparsity = metas().map(|meta| meta.log_sparsity).min().unwrap();
    let slots = metas().map(|meta| meta.slots).reduce(SlotsKind::join).unwrap();

    if mul_one_pass(dst, &[a[0].base2k(), b[0].base2k()], tensor_layout.k, cnv_offset, tsk) {
        one_pass(dst, cnv_offset, TorusPrecision((log_budget + log_delta) as u32), scratch);
        dst.set_log_budget(log_budget);
        dst.set_log_delta(log_delta);
        dst.set_log_sparsity(log_sparsity);
        dst.set_slots(slots);
        // Stamping lowers `k`, which clears the flag of a sum canonical at its new width.
        dst.set_canonical(true);
        return Ok(());
    }

    let scratch = scratch.borrow();
    let (mut acc, scratch) = scratch.take_glwe_tensor_scratch(&tensor_layout);
    let (mut term, mut scratch) = match a.len() {
        1 => (None, scratch),
        _ => {
            let (term, scratch) = scratch.take_glwe_tensor_scratch(&tensor_layout);
            (Some(term), scratch)
        }
    };
    tensor(&mut acc, 0, cnv_offset, &mut scratch);
    if let Some(term) = term.as_mut() {
        for i in 1..a.len() {
            tensor(term, i, cnv_offset, &mut scratch);
            module.glwe_add_assign(&mut acc, &*term);
        }
        // A tensor view reports canonical, so it must hold canonical digits before the relinearization's DFT.
        module.glwe_normalize_assign(&mut acc, &mut scratch);
    }
    dst.set_log_budget(log_budget);
    dst.set_log_delta(log_delta);
    dst.set_log_sparsity(log_sparsity);
    dst.set_slots(slots);
    module.glwe_tensor_relinearize(dst, &acc, tsk, &mut scratch);
    Ok(())
}

fn accumulate<BE, Dst, F>(
    module: &Module<BE>,
    dst: &mut Dst,
    n: usize,
    scratch: &mut ScratchArena<'_, BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    F: for<'a> FnMut(&mut CKKSCiphertextViewMut<'a, BE>, usize, &mut ScratchArena<'a, BE>) -> Result<()>,
{
    if n <= 1 {
        return Ok(());
    }
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        for i in 1..n {
            mul_term_into_tmp(&mut tmp, i, &mut scratch_local)?;
            module.ckks_add_assign(dst, &tmp, &mut scratch_local)?;
        }
        Ok(())
    })
}

impl<BE: Backend> CKKSDotProductOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSMulOps<BE>
        + GLWETensoring<BE>
        + GLWEMulRelinearizeSum<BE>
        + GLWEPreparedTensoring<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
{
    fn ckks_dot_product_ct_tmp_bytes<R, A, B, T>(&self, n: usize, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // `a`/`b` describe the widest input pair; the internal tensor
        // intermediate and the apply scratch scale with the operand widths, not
        // only with `res` (which may legitimately be narrower).
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, a, b, tsk);
        // The path that relinearizes once, over ciphertexts or prepared operands: an accumulator at this layout,
        // and a term for a sum.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: a.k().max(b.k()),
            rank: res.rank(),
        };
        let tensor_bytes: usize = BE::scratch_aligned(self.glwe_tensor_bytes_of_from_infos(&tensor_layout));
        let apply: usize = self
            .glwe_tensor_apply_tmp_bytes(&tensor_layout, a, b)
            .max(self.glwe_tensor_apply_prepared_tmp_bytes(&tensor_layout, a.size(), b.size(), a.base2k()));
        let relinearize: usize = self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk);
        let size: usize = tensor_layout.k.as_usize().div_ceil(res.base2k().as_usize());
        if n <= 1 {
            // `ckks_mul_into`, or one prepared pair: in one pass, or one tensor, not normalized.
            let one_pass: usize = self.glwe_mul_relinearize_tmp_bytes(res, size, size, tensor_layout.k, tsk);
            return mul_scratch.max(one_pass).max(tensor_bytes + apply.max(relinearize));
        }
        let ct_bytes: usize = self.glwe_bytes_of_from_infos(res);
        let fallback: usize = ct_bytes + mul_scratch.max(self.ckks_add_tmp_bytes(res.size()));
        let fast: usize = 2 * tensor_bytes + apply.max(self.glwe_normalize_tmp_bytes()).max(relinearize);
        // Its one-pass form, whatever the number of pairs.
        let one_pass: usize = self
            .glwe_mul_relinearize_sum_tmp_bytes(res, size, size, tensor_layout.k, tsk)
            .max(self.glwe_mul_prepared_relinearize_sum_tmp_bytes(res, size, size, tensor_layout.k, tsk));
        fallback.max(fast).max(one_pass)
    }

    fn ckks_dot_product_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_pt_vec_tmp_bytes(res, a, b)
                .max(self.ckks_add_tmp_bytes(res.size()))
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res)
            + self
                .ckks_mul_pt_const_tmp_bytes(res, a, b)
                .max(self.ckks_add_tmp_bytes(res.size()))
    }

    fn ckks_dot_product_ct<Dst, A, B, H>(
        &self,
        dst: &mut Dst,
        a: &[&A],
        b: &[&B],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        // Past base2k 62, two canonical tensor digits can overflow an `i64`, and no backend models a
        // product there (`Module::max_base2k`): such pairs keep the eager path.
        if n > 1 && dst.base2k().as_usize() <= 62 && share_layout(a) && share_layout(b) {
            return dot_product_lazy(
                self,
                "ckks_dot_product_ct",
                dst,
                a,
                b,
                tsk,
                scratch,
                |dst, cnv_offset, res_k, s| {
                    let pairs: Vec<(&A, &B)> = a.iter().copied().zip(b.iter().copied()).collect();
                    self.glwe_mul_relinearize_sum(cnv_offset, dst, res_k, &pairs, tsk, s)
                },
                |term, i, cnv_offset, s| self.glwe_tensor_apply(cnv_offset, term, a[i], b[i], s),
            );
        }
        self.ckks_mul_into(dst, a[0], b[0], tsk, scratch)?;
        accumulate(self, dst, n, scratch, |tmp, i, s| self.ckks_mul_into(tmp, a[i], b[i], tsk, s))
    }

    fn ckks_dot_product_prepared<Dst, H>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSPreparedLeft<BE>],
        b: &[&CKKSPreparedRight<BE>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        const OP: &str = "ckks_dot_product_prepared";
        check_lengths(OP, a.len(), b.len())?;
        for operand in a {
            check_prepared_layout(OP, dst, *operand)?;
        }
        for operand in b {
            check_prepared_layout(OP, dst, *operand)?;
        }
        let invalid = |reason: String| -> Result<()> { Err(CKKSCompositionError::InvalidOperand { op: OP, reason }.into()) };
        if !share_layout(a) || !share_layout(b) {
            return invalid("every a[i] must share one layout and scale, and every b[i] one".into());
        }
        // Past base2k 62, where `ckks_dot_product_ct` keeps the eager path, which prepared pairs do not have.
        if dst.base2k().as_usize() > 62 {
            return invalid(format!(
                "base2k {} > 62: two canonical tensor digits can overflow an i64",
                dst.base2k().as_usize()
            ));
        }
        dot_product_lazy(
            self,
            OP,
            dst,
            a,
            b,
            tsk,
            scratch,
            |dst, cnv_offset, res_k, s| {
                let pairs: Vec<_> = a.iter().zip(b).map(|(ai, bi)| (ai.cnv(), bi.cnv())).collect();
                let (a, b) = (a[0], b[0]);
                self.glwe_mul_prepared_relinearize_sum(cnv_offset, dst, res_k, &pairs, a.size(), a.k(), b.size(), b.k(), tsk, s)
            },
            |term, i, cnv_offset, s| {
                let (a, b) = (a[i], b[i]);
                self.glwe_tensor_apply_prepared(cnv_offset, term, a.cnv(), a.size(), b.cnv(), b.size(), a.base2k(), s)
            },
        )
    }

    fn ckks_dot_product_pt_vec<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord, BE::Ring>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        check_lengths("ckks_dot_product_pt_vec", a.len(), b.len())?;
        let n: usize = a.len();
        self.ckks_mul_pt_vec_into(dst, a[0], b[0], scratch)?;
        accumulate(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_into(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_const<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord, BE::Ring>],
        b: &[&E],
        pt_coeffs: &[usize],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        check_lengths("ckks_dot_product_pt_const", a.len(), b.len())?;
        check_lengths("ckks_dot_product_pt_const coeffs", a.len(), pt_coeffs.len())?;
        let n: usize = a.len();
        self.ckks_mul_pt_const_into(dst, a[0], b[0], pt_coeffs[0], scratch)?;
        accumulate(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_into(tmp, a[i], b[i], pt_coeffs[i], s)
        })
    }
}
