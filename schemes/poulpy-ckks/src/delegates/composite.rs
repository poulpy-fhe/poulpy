use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Data, Module, Ring, ScratchArena, VecZnxDftBackendMut};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSAddManyOps, CKKSAffineOps, CKKSDotProductOps, CKKSMulAddOps, CKKSMulSubOps},
    layouts::{CKKSCiphertext, CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
    oep::{CKKSAddManyImpl, CKKSAffineImpl, CKKSDotProductImpl, CKKSMulAddImpl, CKKSMulSubImpl},
};

impl<BE: Backend + CKKSAddManyImpl<Rg>, Rg: Ring> CKKSAddManyOps<BE, Rg> for Module<BE> {
    fn ckks_add_many_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
    {
        BE::ckks_add_many_tmp_bytes(self, res, a)
    }

    fn ckks_add_many<R, A>(&self, normalize_after: usize, res: &mut R, a: &[&A], scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    {
        BE::ckks_add_many(self, normalize_after, res, a, scratch)
    }
}

impl<BE: Backend + CKKSMulAddImpl<Rg>, Rg: Ring> CKKSMulAddOps<BE, Rg> for Module<BE> {
    fn ckks_mul_add_ct_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_mul_add_ct_tmp_bytes(self, res, a, b, tsk)
    }

    fn ckks_mul_add_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_add_pt_vec_tmp_bytes(self, res, a, b)
    }

    fn ckks_mul_add_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_add_pt_const_tmp_bytes(self, res, a, b)
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
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetTensorKey<BE, Rg>,
    {
        BE::ckks_mul_add_ct_into(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_add_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_add_pt_vec_into(self, dst, a, pt, scratch)
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
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_add_pt_const_into(self, dst, a, pt, pt_coeff, scratch)
    }
}

impl<BE: Backend + CKKSAffineImpl<Rg>, Rg: Ring> CKKSAffineOps<BE, Rg> for Module<BE> {
    fn ckks_affine_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, affine_const: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_affine_pt_const_tmp_bytes(self, res, a, affine_const)
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
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_affine_pt_const_into(self, dst, a, affine_const, offset_coeff, scale_coeff, scratch)
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
        Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_affine_pt_const_assign(self, dst, affine_const, offset_coeff, scale_coeff, scratch)
    }

    fn ckks_affine_pt_vec_tmp_bytes<R, A, S>(&self, res: &R, a: &A, scale: &S) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        S: CKKSInfos,
    {
        BE::ckks_affine_pt_vec_tmp_bytes(self, res, a, scale)
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
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_affine_pt_vec_into(self, dst, a, scale, offset, scratch)
    }

    fn ckks_affine_pt_vec_assign<Dst, S, P>(
        &self,
        dst: &mut Dst,
        scale: &S,
        offset: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_affine_pt_vec_assign(self, dst, scale, offset, scratch)
    }
}

impl<BE: Backend + CKKSMulSubImpl<Rg>, Rg: Ring> CKKSMulSubOps<BE, Rg> for Module<BE> {
    fn ckks_mul_sub_ct_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_mul_sub_ct_tmp_bytes(self, res, a, b, tsk)
    }

    fn ckks_mul_sub_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_sub_pt_vec_tmp_bytes(self, res, a, b)
    }

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_sub_pt_const_tmp_bytes(self, res, a, b)
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
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetTensorKey<BE, Rg>,
    {
        BE::ckks_mul_sub_ct_into(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_sub_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_sub_pt_vec_into(self, dst, a, pt, scratch)
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
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_sub_pt_const_into(self, dst, a, pt, pt_coeff, scratch)
    }
}

impl<BE: Backend + CKKSDotProductImpl<Rg>, Rg: Ring> CKKSDotProductOps<BE, Rg> for Module<BE> {
    fn ckks_dot_product_ct_tmp_bytes<R, A, B, T>(&self, n: usize, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_dot_product_ct_tmp_bytes(self, n, res, a, b, tsk)
    }

    fn ckks_dot_product_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_dot_product_pt_vec_tmp_bytes(self, res, a, b)
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_dot_product_pt_const_tmp_bytes(self, res, a, b)
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
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetTensorKey<BE, Rg>,
    {
        BE::ckks_dot_product_ct(self, dst, a, b, tsk, scratch)
    }

    fn ckks_dot_product_prepared_tmp_bytes<R, A, B, T>(&self, n: usize, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_dot_product_prepared_tmp_bytes(self, n, res, a, b, tsk)
    }

    fn ckks_dot_product_prepared<Dst, H>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>,
    {
        BE::ckks_dot_product_prepared(self, dst, a, b, tsk, scratch)
    }

    fn ckks_dot_product_prepared_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec_tmp_bytes(self, res, a, b)
    }

    fn ckks_dot_product_prepared_pt_vec<Dst>(
        &self,
        dst: &mut Dst,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec(self, dst, a, b, galois, scratch)
    }

    fn ckks_dot_product_prepared_pt_vec_add_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec_add_tmp_bytes(self, res, a, b)
    }

    fn ckks_dot_product_prepared_pt_vec_add<R>(
        &self,
        res: &mut R,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec_add(self, res, a, b, galois, scratch)
    }

    fn ckks_dot_product_prepared_pt_vec_dft_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec_dft_tmp_bytes(self, res, a, b)
    }

    fn ckks_dot_product_prepared_pt_vec_dft<R>(
        &self,
        acc: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: CKKSCtBounds,
    {
        BE::ckks_dot_product_prepared_pt_vec_dft(self, acc, res, a, b, galois, scratch)
    }

    fn ckks_dot_product_prepared_pt_vec_dft_add_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec_dft_add_tmp_bytes(self, res, a, b)
    }

    fn ckks_dot_product_prepared_pt_vec_dft_add<R>(
        &self,
        acc: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        res: &R,
        a: &[&CKKSLeftOperandPrepared<BE, Rg>],
        b: &[&CKKSRightOperandPrepared<BE, Rg>],
        galois: &[i64],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: CKKSCtBounds,
    {
        BE::ckks_dot_product_prepared_pt_vec_dft_add(self, acc, res, a, b, galois, scratch)
    }

    fn ckks_dot_product_prepared_pt_vec_from_dft_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: GLWEInfos,
        P: GLWEInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec_from_dft_tmp_bytes(self, res, a, b)
    }

    fn ckks_dot_product_prepared_pt_vec_from_dft<Dst>(
        &self,
        res: &mut Dst,
        acc: &mut VecZnxDftBackendMut<'_, BE, Rg>,
        a: &CKKSLeftOperandPrepared<BE, Rg>,
        b: &CKKSRightOperandPrepared<BE, Rg>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_dot_product_prepared_pt_vec_from_dft(self, res, acc, a, b, scratch)
    }

    fn ckks_dot_product_pt_vec<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, Rg>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord, Rg>],
        b: &[&E],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord, Rg>: GLWEToBackendMut<BE, Rg>,
        CKKSCiphertext<D, BE::ZnxWord, Rg>: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        E: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_dot_product_pt_vec(self, dst, a, b, scratch)
    }

    fn ckks_dot_product_pt_const<Dst: Data, D: Data, E>(
        &self,
        dst: &mut CKKSCiphertext<Dst, BE::ZnxWord, Rg>,
        a: &[&CKKSCiphertext<D, BE::ZnxWord, Rg>],
        b: &[&E],
        pt_coeffs: &[usize],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst, BE::ZnxWord, Rg>: GLWEToBackendMut<BE, Rg>,
        CKKSCiphertext<D, BE::ZnxWord, Rg>: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        E: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_dot_product_pt_const(self, dst, a, b, pt_coeffs, scratch)
    }
}
