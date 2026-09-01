use crate::{CKKSResult as Result, ckks_bail, ckks_ensure};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWENormalize, GLWETensoring,
    layouts::{GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision},
};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::Unnormalized;
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::CKKSCopyOps,
    api::{CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSSubOps},
    layouts::{
        CKKSCiphertext, CKKSCiphertextViewMut, ScratchArenaTakeCKKS, UnnormalizedCKKSCiphertext,
        ciphertext::UnnormalizedCKKSCiphertextRefMut,
    },
    oep::CKKSAddImpl,
};
use poulpy_core::GLWEBytesOf;

/// Guards `n` un-normalized accumulations against worst-case `i64` overflow.
///
/// Signed limb digits lie in `[−2^(base2k−1), 2^(base2k−1))`.  In the worst
/// case (all summands aligned in sign) the digit magnitude after `n` additions
/// is `n · 2^(base2k−1)`, which overflows `i64` once `n ≥ 2^(64 − base2k)`.
/// The bound enforced here, `n ≤ 2^(63 − base2k)`, provides one extra bit of
/// headroom below that threshold.
///
/// In the typical case (sign-balanced CKKS inputs) digit growth follows an
/// Irwin–Hall distribution with std dev `O(sqrt(n) · 2^(base2k−1) / sqrt(3))`,
/// so the practical limit is much higher than this conservative bound.
fn ensure_accumulation_fits<C: LWEInfos + ?Sized>(op: &'static str, dst: &C, n: usize) -> Result<()> {
    let base2k: usize = dst.base2k().as_usize();
    ckks_ensure!(base2k < 64, "{op}: unsupported base2k={base2k}");
    ckks_ensure!(
        n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
    Ok(())
}

/// Shared body of the fused multiply-then-accumulate composites
/// (`ckks_mul_{add,sub}_*_into[_unnormalized]`): carve a `dst`-shaped temporary
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
    fn ckks_add_many_tmp_bytes(&self) -> usize {
        self.ckks_add_tmp_bytes()
    }

    fn ckks_add_many<Dst, Src>(&self, dst: &mut Dst, inputs: &[&Src], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    {
        match inputs.len() {
            0 => ckks_bail!("ckks_add_many: inputs must contain at least one ciphertext"),
            1 => {
                self.ckks_copy(dst, inputs[0], scratch)?;
            }
            _ => {
                ensure_accumulation_fits("ckks_add_many", dst, inputs.len())?;
                self.ckks_add_into(dst, inputs[0], inputs[1], scratch)?;
                for ct in &inputs[2..] {
                    self.ckks_add_assign(dst, *ct, scratch)?;
                }
            }
        }
        Ok(())
    }
}

// --- CKKSMulAddOps ---

impl<BE: Backend + CKKSAddImpl<BE>> CKKSMulAddOps<BE> for Module<BE>
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
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, a, b, tsk).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_pt_vec_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_pt_const_into(tmp, a, pt, pt_coeff, s),
            |dst, tmp, s| self.ckks_add_assign(dst, tmp, s),
        )
    }

    fn ckks_mul_add_pt_const_into_unnormalized<Dst: Data, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_pt_const_into(tmp, a, pt, pt_coeff, s),
            |dst, tmp, s| self.ckks_add_assign_unnormalized(dst, tmp, s),
        )
    }

    fn ckks_mul_add_pt_vec_into_unnormalized<Dst: Data, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        GLWE<Dst, BE::ZnxWord, Unnormalized>: GLWEToBackendMut<BE, State = Unnormalized>,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        mul_then_combine(
            dst,
            scratch,
            |tmp, s| self.ckks_mul_pt_vec_into(tmp, a, pt, s),
            |dst, tmp, s| self.ckks_add_assign_unnormalized(dst, tmp, s),
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
            .max(self.ckks_add_pt_const_tmp_bytes())
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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
            .max(self.ckks_add_pt_vec_tmp_bytes())
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        S: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        S: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, a, b, tsk).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_pt_vec_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
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

fn accumulate_unnormalized<BE, D, F>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D, BE::ZnxWord>,
    n: usize,
    scratch: &mut ScratchArena<'_, BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend,
    D: Data,
    BE: CKKSAddImpl<BE>,
    Module<BE>: GLWENormalize<BE>,
    CKKSCiphertext<D, BE::ZnxWord>: GLWEToBackendMut<BE, State = Normalized>,
    F: for<'a> FnMut(&mut CKKSCiphertextViewMut<'a, BE>, usize, &mut ScratchArena<'a, BE>) -> Result<()>,
{
    if n <= 1 {
        module.glwe_normalize_assign(dst, scratch);
        return Ok(());
    }
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        let mut acc = UnnormalizedCKKSCiphertextRefMut::new(dst);
        for i in 1..n {
            mul_term_into_tmp(&mut tmp, i, &mut scratch_local)?;
            BE::ckks_add_assign_unnormalized_ref_impl(module, &mut acc, &tmp, &mut scratch_local)?;
        }
        acc.normalize(module, &mut scratch_local);
        Ok(())
    })
}

impl<BE: Backend + CKKSAddImpl<BE>> CKKSDotProductOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + GLWENormalize<BE> + GLWETensoring<BE>,
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
        if n <= 1 {
            return mul_scratch.max(self.glwe_normalize_tmp_bytes());
        }
        let ct_bytes: usize = self.glwe_bytes_of_from_infos(res);
        let fallback: usize = ct_bytes + mul_scratch.max(self.ckks_add_tmp_bytes());
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.k().max(a.k()).max(b.k()).as_u32()),
            rank: res.rank(),
        };
        let tensor_bytes: usize = self.glwe_tensor_bytes_of_from_infos(&tensor_layout);
        let inner: usize = self
            .glwe_tensor_apply_tmp_bytes(&tensor_layout, a, b)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
        let fast: usize = 2 * n * ct_bytes + tensor_bytes + inner;
        fallback.max(fast)
    }

    fn ckks_dot_product_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_pt_vec_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        self.glwe_bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_ct<Dst: Data, D: Data, E: Data, H>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord>],
        b: &[&CKKSCiphertext<E, BE::ZnxWord>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord>: GLWEToBackendMut<BE, State = Normalized>,
        CKKSCiphertext<D, BE::ZnxWord>: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
        CKKSCiphertext<E, BE::ZnxWord>: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
        H: GetTensorKey<BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_ct", dst, n)?;
        self.ckks_mul_into(dst, a[0], b[0], tsk, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| self.ckks_mul_into(tmp, a[i], b[i], tsk, s))
    }

    fn ckks_dot_product_pt_vec<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord>: GLWEToBackendMut<BE, State = Normalized>,
        CKKSCiphertext<D, BE::ZnxWord>: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
        E: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        check_lengths("ckks_dot_product_pt_vec", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec", dst, n)?;
        self.ckks_mul_pt_vec_into(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_into(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_const<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord>],
        b: &[&E],
        pt_coeffs: &[usize],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord>: GLWEToBackendMut<BE, State = Normalized>,
        CKKSCiphertext<D, BE::ZnxWord>: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
        E: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
    {
        check_lengths("ckks_dot_product_pt_const", a.len(), b.len())?;
        check_lengths("ckks_dot_product_pt_const coeffs", a.len(), pt_coeffs.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_const", dst, n)?;
        self.ckks_mul_pt_const_into(dst, a[0], b[0], pt_coeffs[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_into(tmp, a[i], b[i], pt_coeffs[i], s)
        })
    }
}
