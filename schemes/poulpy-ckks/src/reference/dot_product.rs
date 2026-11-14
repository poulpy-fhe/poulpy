//! Dot products of ciphertexts, or of prepared operands, relinearized once where the pairs share one ct × ct rule.
use poulpy_core::{
    GLWEAdd, GLWEBytesOf, GLWEMulPreparedPlainSum, GLWEMulRelinearizeSum, GLWENormalize, GLWEPreparedTensoring,
    GLWETensorBytesOf, GLWETensoring, ScratchArenaTakeGLWETensor,
    gadget::cnv_offset_to_limb_offset,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorViewMut, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes},
    layouts::{Backend, Module, Ring, ScratchArena, Standard, VecZnxDftBackendMut},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSResult as Result, GLWEPlaintextInfos, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{CKKSAddOps, CKKSMulOps, CKKSRoundOps},
    ckks_ensure, ckks_ensure_operand, ensure_plaintext_degree_embeds, get_mul_ct_params,
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
    mul_pt_params_raw,
    oep::derived::dot_product::{accumulate, check_lengths},
    reference::mul::{
        MulStamp, StampOrder, check_prepared_layout, is_replicated, mul_one_pass, product_key_k, round_into, take_wide,
        tensor_mul_core, with_shaped_bytes,
    },
};

/// The reference algorithms of [`CKKSDotProductImpl`](crate::oep::CKKSDotProductImpl), which
/// `impl_ckks_dot_product_reference_full!` forwards to and an override can call for the methods it keeps.
pub trait CKKSDotProductReference<BE: Backend, Rg: Ring = Standard>: Sized + ModuleN {
    fn ckks_dot_product_ct_tmp_bytes_reference<R, A, B, T>(&self, n: usize, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        Self: CKKSMulOps<BE, Rg>
            + CKKSAddOps<BE, Rg>
            + CKKSRoundOps<BE, Rg>
            + GLWEBytesOf<BE>
            + GLWETensorBytesOf<BE>
            + GLWETensoring<BE, Rg>
            + GLWEMulRelinearizeSum<BE, Rg>
            + GLWEPreparedTensoring<BE, Rg>
            + GLWENormalize<BE, Rg>,
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
            // `ckks_mul_into`, or one prepared pair: in one pass, which fits a single product's scratch, or one
            // tensor, not normalized.
            let one_pass: usize = self.glwe_mul_relinearize_tmp_bytes(res, size, size, tensor_layout.k, tsk);
            return mul_scratch.max(one_pass).max(tensor_bytes + apply.max(relinearize));
        }
        let ct_bytes: usize = self.glwe_bytes_of_from_infos(res);
        let fallback: usize = ct_bytes + mul_scratch.max(self.ckks_add_tmp_bytes(res.size()));
        // The lazy sum into `res`, or on a replicated layout into a wider ciphertext rounded into `res`.
        let lazy = |res: &GLWELayout, _: TorusPrecision| {
            let relinearize: usize = self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk);
            let fast: usize = 2 * tensor_bytes + apply.max(self.glwe_normalize_tmp_bytes()).max(relinearize);
            // Its one-pass form, whatever the number of pairs.
            let one_pass: usize = self
                .glwe_mul_relinearize_sum_tmp_bytes(res, size, size, tensor_layout.k, tsk)
                .max(self.glwe_mul_prepared_relinearize_sum_tmp_bytes(res, size, size, tensor_layout.k, tsk));
            fast.max(one_pass)
        };
        let replicated = is_replicated(a) || is_replicated(b);
        fallback.max(with_shaped_bytes(self, res, tensor_layout.k, replicated, lazy))
    }

    fn ckks_dot_product_ct_reference<Dst, A, B, H>(
        &self,
        dst: &mut Dst,
        a: &[&A],
        b: &[&B],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSMulOps<BE, Rg>
            + CKKSAddOps<BE, Rg>
            + CKKSRoundOps<BE, Rg>
            + GLWETensoring<BE, Rg>
            + GLWEMulRelinearizeSum<BE, Rg>
            + GLWEAdd<BE, Rg>
            + GLWENormalize<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetTensorKey<BE, Rg>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let terms = a.iter().map(|t| t.n()).chain(b.iter().map(|t| t.n()));
        crate::ensure_operand_degrees("ckks_dot_product_ct", self, std::iter::once(dst.n()).chain(terms))?;
        let n: usize = a.len();
        // Past base2k 62, two canonical tensor digits can overflow an `i64`, and no backend models a
        // product there (`Module::max_base2k`): such pairs keep the eager path.
        if n > 1 && dst.base2k().as_usize() <= 62 && share_layout(a) && share_layout(b) {
            let rule = lazy_rule("ckks_dot_product_ct", dst, a, b, tsk)?;
            macro_rules! sum_into {
                ($out:expr, $stamp:expr, $scratch:expr) => {
                    dot_product_lazy(
                        self,
                        $out,
                        a,
                        b,
                        tsk,
                        &rule,
                        $stamp,
                        $scratch,
                        |out, cnv_offset, res_k, s| self.glwe_mul_relinearize_sum(cnv_offset, out, res_k, a, b, tsk, s),
                        |term, i, cnv_offset, s| self.glwe_tensor_apply(cnv_offset, term, a[i], b[i], s),
                    )
                };
            }
            return match rule.lambda {
                None => sum_into!(dst, rule.stamp, scratch),
                Some(lambda) => {
                    let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
                    sum_into!(&mut wide, rule.stamp.widen(lambda), &mut scratch)?;
                    round_into(self, dst, &wide, &rule.stamp, &mut scratch)
                }
            };
        }
        self.ckks_mul_into(dst, a[0], b[0], tsk, scratch)?;
        accumulate(self, dst, n, scratch, |tmp, i, s| self.ckks_mul_into(tmp, a[i], b[i], tsk, s))
    }

    fn ckks_dot_product_prepared_reference<Dst, H>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEPreparedTensoring<BE, Rg>
            + GLWETensoring<BE, Rg>
            + GLWEAdd<BE, Rg>
            + GLWENormalize<BE, Rg>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>,
    {
        const OP: &str = "ckks_dot_product_prepared";
        crate::ensure_operand_degrees(OP, self, [dst.n()])?;
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
        check_lazy_base2k(OP, dst)?;
        let rule = lazy_rule(OP, dst, a, b, tsk)?;
        macro_rules! sum_into {
            ($out:expr, $stamp:expr, $scratch:expr) => {
                dot_product_lazy(
                    self,
                    $out,
                    a,
                    b,
                    tsk,
                    &rule,
                    $stamp,
                    $scratch,
                    |out, cnv_offset, res_k, s| {
                        self.glwe_mul_prepared_relinearize_sum(
                            cnv_offset,
                            out,
                            res_k,
                            a.iter().map(|a| a.cnv()),
                            a[0].size(),
                            a[0].k(),
                            b.iter().map(|b| b.cnv()),
                            b[0].size(),
                            b[0].k(),
                            tsk,
                            s,
                        )
                    },
                    |term, i, cnv_offset, s| {
                        let (a, b) = (a[i], b[i]);
                        self.glwe_tensor_apply_prepared(cnv_offset, term, a.cnv(), a.size(), b.cnv(), b.size(), a.base2k(), s)
                    },
                )
            };
        }
        match rule.lambda {
            None => sum_into!(dst, rule.stamp, scratch),
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
                sum_into!(&mut wide, rule.stamp.widen(lambda), &mut scratch)?;
                round_into(self, dst, &wide, &rule.stamp, &mut scratch)
            }
        }
    }

    /// Core's summed product's scratch, whatever the number of terms.
    fn ckks_dot_product_prepared_pt_vec_tmp_bytes_reference<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        self.glwe_mul_prepared_plain_sum_tmp_bytes(res, a.size(), b.size())
    }

    /// Core's `glwe_mul_prepared_plain_sum` at the rule of `ckks_mul_pt_vec_into`, after the checks of
    /// `ckks_dot_product_prepared` on the terms, the plaintexts' and the Galois elements', before any write.
    fn ckks_dot_product_prepared_pt_vec_reference<Dst>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    {
        let (log_budget, log_delta, cnv_offset) =
            prepared_pt_rule(self, "ckks_dot_product_prepared_pt_vec", &*dst, a, b, galois)?;
        let (a0, b0, base2k) = (a[0], b[0], dst.base2k());
        let log_sparsity = a
            .iter()
            .zip(b)
            .map(|(ct, pt)| ct.log_sparsity().min(ct.log_n() - pt.log_n()))
            .min()
            .unwrap_or(0);
        let slots = a
            .iter()
            .zip(b)
            .fold(a0.slots(), |slots, (ct, pt)| slots.join(ct.slots()).join(pt.slots()));
        dst.set_log_budget(log_budget);
        dst.set_log_delta(log_delta);
        dst.set_log_sparsity(log_sparsity);
        dst.set_slots(slots);
        dst.set_slot_layout(SlotLayout::Full);
        self.glwe_mul_prepared_plain_sum(cnv_offset, dst, a, a0.size(), b, b0.size(), galois, base2k, scratch);
        Ok(())
    }

    /// Core's `glwe_mul_prepared_plain_sum_add` query, whatever the number of terms.
    fn ckks_dot_product_prepared_pt_vec_add_tmp_bytes_reference<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        self.glwe_mul_prepared_plain_sum_add_tmp_bytes(res, a.size(), b.size())
    }

    /// Core's `glwe_mul_prepared_plain_sum_add` at the rule of `ckks_mul_pt_vec_into`, after the checks of
    /// [`Self::ckks_dot_product_prepared_pt_vec_reference`], that the rule leaves `res`'s scale and budget, and its scratch,
    /// before any write.
    fn ckks_dot_product_prepared_pt_vec_add_reference<R>(
        &self,
        res: &mut R,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    {
        const OP: &str = "ckks_dot_product_prepared_pt_vec_add";
        crate::layouts::validation::ensure_full_layout(OP, crate::GLWEPlaintextInfos::slot_layout(&*res))?;
        let (log_budget, log_delta, cnv_offset) = prepared_pt_rule(self, OP, &*res, a, b, galois)?;
        ckks_ensure_operand!(
            OP,
            (log_budget, log_delta) == (res.log_budget(), res.log_delta()),
            "res must hold the product's scale and budget"
        );
        let (a0, b0, base2k) = (a[0], b[0], res.base2k());
        ckks_ensure!(
            scratch.available() >= self.ckks_dot_product_prepared_pt_vec_add_tmp_bytes_reference(&*res, a0, b0),
            "{OP}: scratch below its query"
        );
        let log_sparsity = a
            .iter()
            .zip(b)
            .map(|(ct, pt)| ct.log_sparsity().min(ct.log_n() - pt.log_n()))
            .fold(res.log_sparsity(), usize::min);
        let slots = a
            .iter()
            .zip(b)
            .fold(res.slots(), |slots, (ct, pt)| slots.join(ct.slots()).join(pt.slots()));
        res.set_log_sparsity(log_sparsity);
        res.set_slots(slots);
        res.set_noise(None);
        self.glwe_mul_prepared_plain_sum_add(cnv_offset, res, a, a0.size(), b, b0.size(), galois, base2k, scratch);
        Ok(())
    }

    /// Core's `glwe_mul_prepared_plain_sum_dft` query on an accumulator of `a.size() + b.size()` limbs.
    fn ckks_dot_product_prepared_pt_vec_dft_tmp_bytes_reference<R, A, P>(&self, _res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        self.glwe_mul_prepared_plain_sum_dft_tmp_bytes(a.size() + b.size(), a.size(), b.size())
    }

    /// Core's `glwe_mul_prepared_plain_sum_dft` at the rule of `ckks_mul_pt_vec_into` for `res`, after the checks of
    /// [`Self::ckks_dot_product_prepared_pt_vec_reference`] with `res` as the destination, then those of `acc`.
    fn ckks_dot_product_prepared_pt_vec_dft_reference<R>(
        &self,
        acc: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        R: CKKSCtBounds,
    {
        const OP: &str = "ckks_dot_product_prepared_pt_vec_dft";
        let (_, _, cnv_offset) = prepared_pt_rule(self, OP, res, a, b, galois)?;
        let (a_size, b_size) = (a[0].size(), b[0].size());
        check_accumulator(OP, acc, res, a_size + b_size)?;
        self.glwe_mul_prepared_plain_sum_dft(cnv_offset, acc, a, a_size, b, b_size, galois, res.base2k(), scratch);
        Ok(())
    }

    /// Core's `glwe_mul_prepared_plain_sum_dft_add` query on an accumulator of `a.size() + b.size()` limbs.
    fn ckks_dot_product_prepared_pt_vec_dft_add_tmp_bytes_reference<R, A, P>(&self, _res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        self.glwe_mul_prepared_plain_sum_dft_add_tmp_bytes(a.size() + b.size(), a.size(), b.size())
    }

    /// Core's `glwe_mul_prepared_plain_sum_dft_add`, after the checks of
    /// [`Self::ckks_dot_product_prepared_pt_vec_dft_reference`].
    fn ckks_dot_product_prepared_pt_vec_dft_add_reference<R>(
        &self,
        acc: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPreparedPlainSum<BE, Rg>,
        R: CKKSCtBounds,
    {
        const OP: &str = "ckks_dot_product_prepared_pt_vec_dft_add";
        let (_, _, cnv_offset) = prepared_pt_rule(self, OP, res, a, b, galois)?;
        let (a_size, b_size) = (a[0].size(), b[0].size());
        check_accumulator(OP, acc, res, a_size + b_size)?;
        self.glwe_mul_prepared_plain_sum_dft_add(cnv_offset, acc, a, a_size, b, b_size, galois, res.base2k(), scratch);
        Ok(())
    }

    /// One `vec_znx_idft_normalize` per column from an accumulator of `a.size() + b.size()` limbs.
    fn ckks_dot_product_prepared_pt_vec_from_dft_tmp_bytes_reference<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: VecZnxIdftNormalizeTmpBytes,
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        self.vec_znx_idft_normalize_tmp_bytes(res.size(), a.size() + b.size())
    }

    /// The stamp of [`Self::ckks_dot_product_prepared_pt_vec_reference`] for the one pair `(a, b)`, after its checks and
    /// those of `acc`, then one `vec_znx_idft_normalize` per column at `res.k()` and the offset's sub-limb part.
    fn ckks_dot_product_prepared_pt_vec_from_dft_reference<Dst>(
        &self,
        res: &mut Dst,
        acc: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &CKKSLeftOperandPrepared<BE, Rg>,
        b: &CKKSRightOperandPrepared<BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxIdftNormalize<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    {
        const OP: &str = "ckks_dot_product_prepared_pt_vec_from_dft";
        let (log_budget, log_delta, cnv_offset) = prepared_pt_rule(self, OP, &*res, &[a], &[b], &[1])?;
        check_accumulator(OP, acc, &*res, a.size() + b.size())?;
        res.set_log_budget(log_budget);
        res.set_log_delta(log_delta);
        res.set_log_sparsity(a.log_sparsity().min(a.log_n() - b.log_n()));
        res.set_slots(a.slots().join(b.slots()));
        res.set_slot_layout(SlotLayout::Full);
        let (base2k, res_k, cols) = (res.base2k().as_usize(), res.k().as_usize(), res.rank().as_usize() + 1);
        let (_, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, base2k);
        {
            let mut res_backend = res.to_backend_mut();
            for col in 0..cols {
                self.vec_znx_idft_normalize(
                    res_backend.data_mut(),
                    base2k,
                    res_k,
                    cnv_offset_lo,
                    col,
                    acc,
                    col,
                    base2k,
                    None,
                    &mut scratch.borrow(),
                );
            }
        }
        res.set_canonical(true);
        Ok(())
    }
}

/// The checks of the summed products by prepared plaintexts into a destination of `res`'s layout and metadata, at most
/// the degree of `module`, in their order and before any write, then the rule of `ckks_mul_pt_vec_into`: `(log_budget, log_delta, cnv_offset)`.
fn prepared_pt_rule<BE: Backend, R: CKKSCtBounds, Rg: Ring>(
    module: &impl ModuleN,
    op: &'static str,
    res: &R,
    a: &[&CKKSLeftOperandPrepared<BE, Rg>],
    b: &[&CKKSRightOperandPrepared<BE, Rg>],
    galois: &[i64],
) -> Result<(usize, usize, usize)> {
    crate::ensure_operand_degrees(op, module, [res.n()])?;
    check_lengths(op, a.len(), b.len())?;
    let invalid = |reason: &str| -> Result<(usize, usize, usize)> {
        Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: reason.into(),
        }
        .into())
    };
    if galois.len() != a.len() {
        return invalid("one Galois element per term");
    }
    for operand in a {
        check_prepared_layout(op, res, *operand)?;
    }
    if galois.iter().any(|&g| g & 1 == 0) {
        return invalid("a Galois element is even");
    }
    if a.iter().any(|ct| ct.slot_layout() != SlotLayout::Full) || b.iter().any(|pt| pt.slot_layout() != SlotLayout::Full) {
        return invalid("prepared dot products round plainly, which a replicated slot layout forbids");
    }
    if b.iter().any(|pt| pt.rank().as_usize() != 0 || pt.base2k() != res.base2k()) {
        return invalid("every b[t] must be a plaintext at the base2k of dst");
    }
    if !share_layout(a) || !share_layout(b) {
        return invalid("every a[t] must share one layout and scale, and every b[t] one");
    }
    let (a0, b0) = (a[0], b[0]);
    ensure_plaintext_degree_embeds(op, res.n().as_usize(), b0.n().as_usize(), BE::MIN_DEGREE)?;
    mul_pt_params_raw(
        op,
        res.k().as_usize(),
        a0.log_delta(),
        a0.log_budget(),
        b0.log_delta(),
        b0.log_budget(),
        b0.max_k().as_usize(),
    )
}

/// `InvalidOperand` unless `acc` holds the degree and the `rank + 1` columns of `res` on `size` limbs.
fn check_accumulator<BE: Backend, R: CKKSCtBounds, Rg: Ring>(
    op: &'static str,
    acc: &VecZnxDftBackendMut<'_, BE, Rg>,
    res: &R,
    size: usize,
) -> Result<()> {
    let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
    ckks_ensure_operand!(
        op,
        (acc.n(), acc.cols(), acc.size()) == (n, cols, size),
        "acc must hold the degree and rank + 1 columns of res on a.size() + b.size() limbs"
    );
    Ok(())
}

impl<BE: Backend, Rg: Ring> CKKSDotProductReference<BE, Rg> for Module<BE> {}

/// `InvalidOperand` past base2k 62, where `ckks_dot_product_ct` keeps the eager path, which prepared pairs do not have.
pub(crate) fn check_lazy_base2k<D: GLWEInfos>(op: &'static str, dst: &D) -> Result<()> {
    ckks_ensure_operand!(
        op,
        dst.base2k().as_usize() <= 62,
        "base2k {} > 62: two canonical tensor digits can overflow an i64",
        dst.base2k().as_usize()
    );
    Ok(())
}

/// Whether every ciphertext shares the first one's layout and scale, the inputs of the ct × ct rule.
fn share_layout<C: CKKSCtBounds>(cts: &[&C]) -> bool {
    cts.iter()
        .all(|ct| (ct.glwe_layout(), ct.log_delta()) == (cts[0].glwe_layout(), cts[0].log_delta()))
}

/// The rule of [`dot_product_lazy`] into `dst`: the eager sum's stamp, the tensor width, the offset of the products,
/// and on a replicated layout the bits `lambda` under the scale at which the sum is taken, as `ckks_mul_into` takes a
/// product, before one shaped rounding. Checks the tensor key at the tensor width, the rule, the slot layouts and on a
/// replicated layout the key of the wider intermediate, in that order and before any write.
struct LazyRule {
    stamp: MulStamp,
    tensor_k: TorusPrecision,
    cnv_offset: usize,
    lambda: Option<usize>,
}

fn lazy_rule<BE, Dst, A, B, H, Rg: Ring>(op: &'static str, dst: &Dst, a: &[&A], b: &[&B], tsk: &H) -> Result<LazyRule>
where
    BE: Backend,
    Dst: CKKSCtBounds,
    A: CKKSCtBounds,
    B: CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
{
    // `ckks_mul_into`'s tensor width, and so its key width, whatever `dst`'s width.
    let tensor_k = a[0].k().max(b[0].k());
    crate::reference::mul::tensor_key_at(op, tsk, tensor_k)?;
    let (log_budget, log_delta, cnv_offset) = get_mul_ct_params(op, dst, a[0], b[0])?;
    // The eager sum's label and slot kind: the smallest label and the join over every operand.
    let metas = || a.iter().map(|ct| ct.meta()).chain(b.iter().map(|ct| ct.meta()));
    let slot_layout = a[0].slot_layout();
    if metas().any(|meta| meta.slot_layout != slot_layout) {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: "the operands' slot layouts differ".into(),
        }
        .into());
    }
    // A replicated layout's wider intermediate, after the rule.
    let lambda = crate::shaping::product_depth(slot_layout, log_delta);
    let key_k = product_key_k(
        dst,
        &[a[0].base2k(), b[0].base2k()],
        tensor_k,
        cnv_offset,
        log_budget + log_delta,
        lambda,
        tsk,
    );
    if lambda.is_some() {
        crate::reference::mul::tensor_key_at(op, tsk, key_k)?;
    }
    Ok(LazyRule {
        stamp: MulStamp {
            log_budget,
            log_delta,
            log_sparsity: metas().map(|meta| meta.log_sparsity).min(),
            slots: metas().map(|meta| meta.slots).reduce(SlotsKind::join),
            slot_layout: Some(slot_layout),
        },
        tensor_k,
        cnv_offset,
        lambda,
    })
}

/// `out = Σ a[i] · b[i]` with one relinearization at `stamp`, for pairs that share one ct × ct rule `rule`:
/// `ckks_mul_into`'s [`tensor_mul_core`], with `one_pass` and `tensor` forming the products of every pair.
///
/// Where `ckks_mul_into` takes its one-pass product, the sum is `one_pass(out, cnv_offset, res_k)`, which adds the
/// products in the transform domain. Otherwise `tensor(t, i, cnv_offset)` writes pair `i`'s tensor product into `t`,
/// canonical at the tensor width, and the terms are added without normalizing, as other lazy additions: the sum is
/// exact, and it is normalized once, right before the relinearization reads it through a DFT. One pair is no sum: one
/// tensor, not normalized, as in `ckks_mul_into`. On a replicated layout, `out` is `rule.lambda` bits wider than the
/// destination, `stamp` that many bits finer, and the caller rounds it with shaping.
#[expect(
    clippy::too_many_arguments,
    reason = "the two product forms of the pairs, the operation's operands and its rule"
)]
fn dot_product_lazy<BE, M, Out, A, B, H, Rg: Ring>(
    module: &M,
    out: &mut Out,
    a: &[&A],
    b: &[&B],
    tsk: &H,
    rule: &LazyRule,
    stamp: MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
    one_pass: impl FnOnce(&mut Out, usize, TorusPrecision, &mut ScratchArena<'_, BE>),
    mut tensor: impl for<'t> FnMut(&mut GLWETensorViewMut<'t, BE, Rg>, usize, usize, &mut ScratchArena<'t, BE>),
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE, Rg> + GLWEAdd<BE, Rg> + GLWENormalize<BE, Rg>,
    Out: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    A: CKKSCtBounds,
    B: CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
{
    let cnv_offset = rule.cnv_offset;
    let tensor_layout = GLWELayout {
        n: out.n(),
        base2k: out.base2k(),
        k: rule.tensor_k,
        rank: out.rank(),
    };
    let one_pass = mul_one_pass(out, &[a[0].base2k(), b[0].base2k()], tensor_layout.k, cnv_offset, tsk)
        .then_some(|out: &mut Out, res_k, s: &mut ScratchArena<'_, BE>| one_pass(out, cnv_offset, res_k, s));
    tensor_mul_core(
        module,
        out,
        tsk,
        tensor_layout.k,
        stamp,
        StampOrder::BeforeApply,
        scratch,
        one_pass,
        |acc, _dst, scratch| {
            tensor(acc, 0, cnv_offset, scratch);
            if a.len() > 1 {
                let (mut term, mut scratch) = scratch.borrow().take_glwe_tensor_scratch(&tensor_layout);
                for i in 1..a.len() {
                    tensor(&mut term, i, cnv_offset, &mut scratch);
                    module.glwe_add_assign(acc, &term);
                }
                // A tensor view reports canonical, so it must hold canonical digits before the relinearization's DFT.
                module.glwe_normalize_assign(acc, &mut scratch);
            }
        },
    )
}
