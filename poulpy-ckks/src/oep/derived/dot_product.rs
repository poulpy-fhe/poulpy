//! Dot products over plaintexts compose the ct × pt products with the additions.
use poulpy_core::{
    GLWEBytesOf, GLWEMulPreparedPlainSum,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSResult as Result, GLWEPlaintextInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps},
    ckks_bail, ensure_plaintext_degree_embeds,
    layouts::{CKKSCiphertext, CKKSCiphertextViewMut, CKKSPreparedLeft, CKKSPreparedRight, ScratchArenaTakeCKKS},
    oep::CKKSDotProductImpl,
    reference::{
        dot_product::share_layout,
        mul::{check_prepared_layout, mul_pt_params_raw},
    },
};

pub(crate) fn check_lengths(op: &'static str, a_len: usize, b_len: usize) -> Result<()> {
    if a_len == 0 {
        ckks_bail!("{op}: inputs must contain at least one pair");
    }
    if a_len != b_len {
        ckks_bail!("{op}: length mismatch between ct vector ({a_len}) and weight vector ({b_len})");
    }
    Ok(())
}

/// Adds the products `1..n`, each formed into a `dst`-shaped temporary, to `dst`.
pub(crate) fn accumulate<BE, M, Dst, F>(
    module: &M,
    dst: &mut Dst,
    n: usize,
    scratch: &mut ScratchArena<'_, BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend,
    M: CKKSAddOps<BE>,
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

pub(crate) fn ckks_dot_product_pt_vec_tmp_bytes<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSDotProductImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_vec_tmp_bytes(res, a, b)
            .max(module.ckks_add_tmp_bytes(res.size()))
}

pub(crate) fn ckks_dot_product_pt_const_tmp_bytes<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: CKKSDotProductImpl,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSInfos,
{
    module.glwe_bytes_of_from_infos(res)
        + module
            .ckks_mul_pt_const_tmp_bytes(res, a, b)
            .max(module.ckks_add_tmp_bytes(res.size()))
}

pub(crate) fn ckks_dot_product_pt_vec<BE, Dst: Data, D: Data, E>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>,
    a: &[&CKKSCiphertext<D, BE::ZnxWord, BE::Ring>],
    b: &[&E],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSDotProductImpl,
    CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + GLWEInfos,
    E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
{
    check_lengths("ckks_dot_product_pt_vec", a.len(), b.len())?;
    module.ckks_mul_pt_vec_into(dst, a[0], b[0], scratch)?;
    accumulate(module, dst, a.len(), scratch, |tmp, i, s| {
        module.ckks_mul_pt_vec_into(tmp, a[i], b[i], s)
    })
}

pub(crate) fn ckks_dot_product_pt_const<BE, Dst: Data, D: Data, E>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>,
    a: &[&CKKSCiphertext<D, BE::ZnxWord, BE::Ring>],
    b: &[&E],
    pt_coeffs: &[usize],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSDotProductImpl,
    CKKSCiphertext<Dst, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + GLWEInfos,
    E: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
{
    check_lengths("ckks_dot_product_pt_const", a.len(), b.len())?;
    check_lengths("ckks_dot_product_pt_const coeffs", a.len(), pt_coeffs.len())?;
    module.ckks_mul_pt_const_into(dst, a[0], b[0], pt_coeffs[0], scratch)?;
    accumulate(module, dst, a.len(), scratch, |tmp, i, s| {
        module.ckks_mul_pt_const_into(tmp, a[i], b[i], pt_coeffs[i], s)
    })
}

pub(crate) fn ckks_dot_product_prepared_pt_tmp_bytes<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
where
    BE: Backend,
    Module<BE>: GLWEMulPreparedPlainSum<BE>,
    R: CKKSCtBounds,
    A: GLWEInfos,
    P: GLWEInfos,
{
    module.glwe_mul_prepared_plain_sum_tmp_bytes(res, a.size(), b.size())
}

/// Core's `glwe_mul_prepared_plain_sum` at the rule of `ckks_mul_pt_vec_into`, after the checks of
/// `ckks_dot_product_prepared` on the terms, the plaintexts' and the Galois elements', before any write.
pub(crate) fn ckks_dot_product_prepared_pt<BE, Dst>(
    module: &Module<BE>,
    dst: &mut Dst,
    a: &[&CKKSPreparedLeft<BE>],
    b: &[&CKKSPreparedRight<BE>],
    galois: &[i64],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWEMulPreparedPlainSum<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    const OP: &str = "ckks_dot_product_prepared_pt";
    check_lengths(OP, a.len(), b.len())?;
    let invalid = |reason: &str| -> Result<()> {
        Err(CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: reason.into(),
        }
        .into())
    };
    if galois.len() != a.len() {
        return invalid("one Galois element per term");
    }
    for operand in a {
        check_prepared_layout(OP, &*dst, *operand)?;
    }
    if galois.iter().any(|&g| g & 1 == 0) {
        return invalid("a Galois element is even");
    }
    if b.iter().any(|pt| pt.rank().as_usize() != 0 || pt.base2k() != dst.base2k()) {
        return invalid("every b[t] must be a plaintext at the base2k of dst");
    }
    if !share_layout(a) || !share_layout(b) {
        return invalid("every a[t] must share one layout and scale, and every b[t] one");
    }
    let (a0, b0, base2k) = (a[0], b[0], dst.base2k());
    ensure_plaintext_degree_embeds(OP, dst.n().as_usize(), b0.n().as_usize(), BE::MIN_DEGREE)?;
    let (log_budget, log_delta, cnv_offset) = mul_pt_params_raw(
        dst.k().as_usize(),
        a0.log_delta(),
        a0.log_budget(),
        b0.log_delta(),
        b0.log_budget(),
        b0.max_k().as_usize(),
    )?;
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
    module.glwe_mul_prepared_plain_sum(cnv_offset, dst, a, a0.size(), b, b0.size(), galois, base2k, scratch);
    Ok(())
}
