use anyhow::{Result, bail, ensure};
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift, GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos,
    layouts::{CKKSCiphertext, CKKSCiphertextViewMut, ScratchArenaTakeCKKS, UnnormalizedCKKSCiphertext, ciphertext::CKKSOffset},
    leveled::api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAddOpsUnnormalized, CKKSAffineOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulOps,
        CKKSMulSubOps, CKKSRescaleOps, CKKSSubOps,
    },
    leveled::default::CKKSAddDefault,
};

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

impl<BE: Backend> CKKSAddManyOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSRescaleOps<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_add_many_tmp_bytes(&self) -> usize {
        self.ckks_add_tmp_bytes()
    }

    fn ckks_add_many<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
    {
        match inputs.len() {
            0 => bail!("ckks_add_many: inputs must contain at least one ciphertext"),
            1 => {
                self.ckks_rescale_into(dst, dst.offset_unary(inputs[0]), inputs[0], scratch)?;
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

impl<BE: Backend> CKKSMulAddOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSAddOpsUnnormalized<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            self.ckks_mul_into(&mut tmp, a, b, tsk, &mut scratch_local)?;
            self.ckks_add_assign(dst, &tmp, &mut scratch_local)
        })
    }

    fn ckks_mul_add_pt_vec_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            self.ckks_mul_pt_vec_into(&mut tmp, a, pt, &mut scratch_local)?;
            self.ckks_add_assign(dst, &tmp, &mut scratch_local)
        })
    }

    fn ckks_mul_add_pt_const_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            self.ckks_mul_pt_const_into(&mut tmp, a, pt, pt_coeff, &mut scratch_local)?;
            self.ckks_add_assign(dst, &tmp, &mut scratch_local)
        })
    }

    fn ckks_mul_add_pt_const_into_unnormalized<Dst: Data, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&dst.inner);
            self.ckks_mul_pt_const_into(&mut tmp, a, pt, pt_coeff, &mut scratch_local)?;
            self.ckks_add_assign_unnormalized(dst, &tmp, &mut scratch_local)
        })
    }

    fn ckks_mul_add_pt_vec_into_unnormalized<Dst: Data, A, P>(
        &self,
        dst: &mut UnnormalizedCKKSCiphertext<Dst>,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&dst.inner);
            self.ckks_mul_pt_vec_into(&mut tmp, a, pt, &mut scratch_local)?;
            self.ckks_add_assign_unnormalized(dst, &tmp, &mut scratch_local)
        })
    }
}

// --- CKKSAffineOps ---

impl<BE: Backend> CKKSAffineOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
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
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
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
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
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
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
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
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + crate::SetCKKSInfos,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.ckks_mul_pt_vec_assign(dst, scale, scratch)?;
        self.ckks_add_pt_vec_assign(dst, offset, scratch)
    }
}

// --- CKKSMulSubOps ---

impl<BE: Backend> CKKSMulSubOps<BE> for Module<BE>
where
    Module<BE>: CKKSMulOps<BE> + CKKSSubOps<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            self.ckks_mul_into(&mut tmp, a, b, tsk, &mut scratch_local)?;
            self.ckks_sub_assign(dst, &tmp, &mut scratch_local)
        })
    }

    fn ckks_mul_sub_pt_vec_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            self.ckks_mul_pt_vec_into(&mut tmp, a, pt, &mut scratch_local)?;
            self.ckks_sub_assign(dst, &tmp, &mut scratch_local)
        })
    }

    fn ckks_mul_sub_pt_const_into<Dst: Data, A: Data, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            self.ckks_mul_pt_const_into(&mut tmp, a, pt, pt_coeff, &mut scratch_local)?;
            self.ckks_sub_assign(dst, &tmp, &mut scratch_local)
        })
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
    scratch: &mut ScratchArena<'_, BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend,
    D: Data,
    Module<BE>: CKKSAddDefault<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
    CKKSCiphertext<D>: GLWEToBackendMut<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    F: for<'a> FnMut(&mut CKKSCiphertextViewMut<'a, BE>, usize, &mut ScratchArena<'a, BE>) -> Result<()>,
{
    if n <= 1 {
        module.glwe_normalize_assign(dst, scratch);
        return Ok(());
    }
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
        for i in 1..n {
            mul_term_into_tmp(&mut tmp, i, &mut scratch_local)?;
            <Module<BE> as CKKSAddDefault<BE>>::ckks_add_assign_unsafe_default(module, dst, &tmp, &mut scratch_local)?;
        }
        module.glwe_normalize_assign(dst, &mut scratch_local);
        Ok(())
    })
}

impl<BE: Backend> CKKSDotProductOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSAddDefault<BE>
        + CKKSMulOps<BE>
        + CKKSRescaleOps<BE>
        + GLWEAdd<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + GLWETensoring<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 1 {
            return mul_scratch.max(self.glwe_normalize_tmp_bytes());
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

    fn ckks_dot_product_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_ct<Dst: Data, D: Data, E: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<E>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_ct", dst, n)?;
        self.ckks_mul_into(dst, a[0], b[0], tsk, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| self.ckks_mul_into(tmp, a[i], b[i], tsk, s))
    }

    fn ckks_dot_product_pt_vec<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds,
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
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&E],
        pt_coeffs: &[usize],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        E: GLWEToBackendRef<BE> + CKKSCtBounds,
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
