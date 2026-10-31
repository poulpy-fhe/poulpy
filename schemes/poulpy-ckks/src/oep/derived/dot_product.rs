//! Dot products over plaintexts compose the ct × pt products with the additions.
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps},
    ckks_ensure_operand,
    layouts::{CKKSCiphertext, CKKSCiphertextViewMut, ScratchArenaTakeCKKSCiphertext},
    oep::CKKSDotProductImpl,
};

pub(crate) fn check_lengths(op: &'static str, a_len: usize, b_len: usize) -> Result<()> {
    check_slice_lengths(op, "weight vector", a_len, b_len)
}

/// `InvalidOperand` for an empty `a` or a second slice, named `other`, of another length.
pub(crate) fn check_slice_lengths(op: &'static str, other: &str, a_len: usize, b_len: usize) -> Result<()> {
    ckks_ensure_operand!(op, a_len != 0, "inputs must contain at least one pair");
    ckks_ensure_operand!(
        op,
        a_len == b_len,
        "length mismatch between ct vector ({a_len}) and {other} ({b_len})"
    );
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

pub(crate) fn ckks_dot_product_pt_vec_tmp_bytes_derived<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
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

pub(crate) fn ckks_dot_product_pt_const_tmp_bytes_derived<BE, R, A, P>(module: &Module<BE>, res: &R, a: &A, b: &P) -> usize
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

pub(crate) fn ckks_dot_product_pt_vec_derived<BE, Dst: Data, D: Data, E>(
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

pub(crate) fn ckks_dot_product_pt_const_derived<BE, Dst: Data, D: Data, E>(
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
    check_slice_lengths("ckks_dot_product_pt_const", "coefficient vector", a.len(), pt_coeffs.len())?;
    module.ckks_mul_pt_const_into(dst, a[0], b[0], pt_coeffs[0], scratch)?;
    accumulate(module, dst, a.len(), scratch, |tmp, i, s| {
        module.ckks_mul_pt_const_into(tmp, a[i], b[i], pt_coeffs[i], s)
    })
}
