use crate::CKKSResult as Result;
use crate::reference::composite::{ckks_add_many_reference, ckks_add_many_tmp_bytes_reference};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::CKKSCopyOps,
    api::{CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSSubOps},
    layouts::{CKKSCiphertext, CKKSCiphertextViewMut, CKKSPreparedLeft, CKKSPreparedRight, ScratchArenaTakeCKKS},
    oep::CKKSDotProductImpl,
};
use poulpy_core::{GLWEBytesOf, GLWEMulPreparedPlainSum, GLWENormalize};

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
    Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWENormalize<BE>,
{
    fn ckks_add_many_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
    {
        ckks_add_many_tmp_bytes_reference(self, res, a)
    }

    fn ckks_add_many<R, A>(&self, res: &mut R, a: &[&A], normalize_after: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        ckks_add_many_reference(self, res, a, normalize_after, scratch)
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
        crate::reference::mul::ensure_full_layout("ckks_affine_pt_const_into", crate::GLWEPlaintextInfos::slot_layout(a))?;
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
        crate::reference::mul::ensure_full_layout("ckks_affine_pt_const_assign", crate::GLWEPlaintextInfos::slot_layout(dst))?;
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
        crate::reference::mul::product_layout(
            "ckks_affine_pt_vec_into",
            &[a.slot_layout(), scale.slot_layout(), offset.slot_layout()],
        )?;
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
        crate::reference::mul::product_layout(
            "ckks_affine_pt_vec_assign",
            &[dst.slot_layout(), scale.slot_layout(), offset.slot_layout()],
        )?;
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

impl<BE: Backend + CKKSDotProductImpl> CKKSDotProductOps<BE> for Module<BE> {
    fn ckks_dot_product_ct_tmp_bytes<R, A, B, T>(&self, n: usize, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_dot_product_ct_tmp_bytes_impl(self, n, res, a, b, tsk)
    }

    fn ckks_dot_product_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_dot_product_pt_vec_tmp_bytes_impl(self, res, a, b)
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_dot_product_pt_const_tmp_bytes_impl(self, res, a, b)
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
        BE::ckks_dot_product_ct_impl(self, dst, a, b, tsk, scratch)
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
        BE::ckks_dot_product_prepared_impl(self, dst, a, b, tsk, scratch)
    }

    fn ckks_dot_product_prepared_pt_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEMulPreparedPlainSum<BE>,
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        BE::ckks_dot_product_prepared_pt_tmp_bytes_impl(self, res, a, b)
    }

    fn ckks_dot_product_prepared_pt<Dst>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSPreparedLeft<BE>],
        b: &[&CKKSPreparedRight<BE>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPreparedPlainSum<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_dot_product_prepared_pt_impl(self, dst, a, b, galois, scratch)
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
        BE::ckks_dot_product_pt_vec_impl(self, dst, a, b, scratch)
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
        BE::ckks_dot_product_pt_const_impl(self, dst, a, b, pt_coeffs, scratch)
    }
}
