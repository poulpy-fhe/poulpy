use crate::layouts::IntPolyInfos;
use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{
        GGSWRotate, GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWEMulXpMinusOne, GLWENegate, GLWENormalize, GLWEPacking,
        GLWERotate, GLWEShift, GLWESub, GLWETensoring, GLWETrace, GLWEZero,
    },
    layouts::{
        GGLWEInfos, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef, GetAutomorphismKey, GetTensorKey,
    },
    oep::{
        GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl, GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl,
        GLWENormalizeImpl, GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, GLWETraceImpl,
        GLWEZeroImpl,
    },
    operations::{
        GGSWRotateReference, GLWEAddReference, GLWECopyReference, GLWEMulConstReference, GLWEMulPlainReference,
        GLWEMulXpMinusOneReference, GLWENegateReference, GLWENormalizeReference, GLWERotateReference, GLWEShiftReference,
        GLWESubReference, GLWETensoringReference, GLWEZeroReference,
    },
    reference::{glwe_packing::GLWEPackingReference, glwe_trace::GLWETraceReference},
};

macro_rules! impl_operations_delegate {
    ($trait:ty, $impl_trait:path, $default:path, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + $impl_trait,
            Module<BE>: $default,
        {
            $($body)+
        }
    };
}

impl_operations_delegate!(
    GLWEAdd<BE>,
    GLWEAddImpl,
    GLWEAddReference<BE>,
    fn glwe_add_into<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        BE::glwe_add_into(self, res, a, b)
    },
    fn glwe_add_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_add_assign(self, res, a)
    }
);

impl_operations_delegate!(
    GLWENegate<BE>,
    GLWENegateImpl,
    GLWENegateReference<BE>,
    fn glwe_negate<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_negate(self, res, a)
    },
    fn glwe_negate_assign<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_negate_assign(self, res)
    }
);

impl_operations_delegate!(
    GLWESub<BE>,
    GLWESubImpl,
    GLWESubReference<BE>,
    fn glwe_sub<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        BE::glwe_sub(self, res, a, b)
    },
    fn glwe_sub_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_sub_assign(self, res, a)
    },
    fn glwe_sub_negate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_sub_negate_assign(self, res, a)
    }
);

impl_operations_delegate!(
    GLWEZero<BE>,
    GLWEZeroImpl,
    GLWEZeroReference<BE>,
    fn glwe_zero<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_zero(self, res)
    }
);

impl_operations_delegate!(
    GLWECopy<BE>,
    GLWECopyImpl,
    GLWECopyReference<BE>,
    fn glwe_copy_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(&self, res: &R, a: &A) -> usize {
        BE::glwe_copy_tmp_bytes(self, res, a)
    },
    fn glwe_copy<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_copy(self, res, a, scratch)
    }
);

impl_operations_delegate!(
    GLWEMulConst<BE>,
    GLWEMulConstImpl,
    GLWEMulConstReference<BE>,
    fn glwe_mul_const_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_mul_const_tmp_bytes(self, res, a, b)
    },
    fn glwe_mul_const<R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_mul_const(self, cnv_offset, res, a, b, b_coeff, scratch)
    },
    fn glwe_mul_const_assign<R, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_mul_const_assign(self, cnv_offset, res, b, b_coeff, scratch)
    }
);

impl_operations_delegate!(
    GLWEMulPlain<BE>,
    GLWEMulPlainImpl,
    GLWEMulPlainReference<BE>,
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_mul_plain_tmp_bytes(self, res, a, b)
    },
    fn glwe_mul_plain<R, A, B>(&self, cnv_offset: usize, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos,
    {
        BE::glwe_mul_plain(self, cnv_offset, res, a, b, scratch)
    },
    fn glwe_mul_plain_assign<R, A>(&self, cnv_offset: usize, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos,
    {
        BE::glwe_mul_plain_assign(self, cnv_offset, res, a, scratch)
    }
);

impl_operations_delegate!(
    GLWETensoring<BE>,
    GLWETensoringImpl,
    GLWETensoringReference<BE>,
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_tensor_apply_tmp_bytes(self, res, a, b)
    },
    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_tensor_square_apply_tmp_bytes(self, res, a)
    },
    fn glwe_tensor_apply<R, A, B>(&self, cnv_offset: usize, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_tensor_apply(self, cnv_offset, res, a, b, scratch)
    },
    fn glwe_tensor_square_apply<R, A>(&self, cnv_offset: usize, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_tensor_square_apply(self, cnv_offset, res, a, scratch)
    },
    fn glwe_tensor_relinearize<R, A, H>(&self, res: &mut R, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetTensorKey<BE>,
    {
        BE::glwe_tensor_relinearize(self, res, a, tsk, scratch)
    },
    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        BE::glwe_tensor_relinearize_tmp_bytes(self, res, a, tsk)
    }
);

impl_operations_delegate!(
    GLWERotate<BE>,
    GLWERotateImpl,
    GLWERotateReference<BE>,
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        BE::glwe_rotate_tmp_bytes(self)
    },
    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_rotate(self, k, res, a)
    },
    fn glwe_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_rotate_assign(self, k, res, scratch);
    }
);

impl_operations_delegate!(
    GGSWRotate<BE>,
    GGSWRotateImpl,
    GGSWRotateReference<BE>,
    fn ggsw_rotate_tmp_bytes(&self) -> usize {
        BE::ggsw_rotate_tmp_bytes(self)
    },
    fn ggsw_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
    {
        BE::ggsw_rotate(self, k, res, a)
    },
    fn ggsw_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
    {
        BE::ggsw_rotate_assign(self, k, res, scratch)
    }
);

impl_operations_delegate!(
    GLWEMulXpMinusOne<BE>,
    GLWEMulXpMinusOneImpl,
    GLWEMulXpMinusOneReference<BE>,
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_mul_xp_minus_one(self, k, res, a)
    },
    fn glwe_mul_xp_minus_one_assign<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_mul_xp_minus_one_assign(self, k, res, scratch)
    }
);

impl_operations_delegate!(
    GLWEShift<BE>,
    GLWEShiftImpl,
    GLWEShiftReference<BE>,
    fn glwe_shift_tmp_bytes(&self, res_size: usize) -> usize {
        BE::glwe_shift_tmp_bytes(self, res_size)
    },
    fn glwe_rsh<R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_rsh(self, k, res, scratch)
    },
    fn glwe_lsh_assign<R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_lsh_assign(self, res, k, scratch)
    },
    fn glwe_lsh<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_lsh(self, res, a, k, scratch)
    },
    fn glwe_lsh_add<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_lsh_add(self, res, a, k, scratch)
    },
    fn glwe_lsh_sub<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_lsh_sub(self, res, a, k, scratch)
    }
);

impl_operations_delegate!(
    GLWENormalize<BE>,
    GLWENormalizeImpl,
    GLWENormalizeReference<BE>,
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_normalize_tmp_bytes(self)
    },
    fn glwe_normalize<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_normalize(self, res, a, scratch)
    },
    fn glwe_normalize_assign<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_normalize_assign(self, res, scratch)
    }
);

impl_operations_delegate!(
    GLWETrace<BE>,
    GLWETraceImpl,
    GLWETraceReference<BE>,
    fn glwe_trace_galois_elements(&self) -> Vec<i64> {
        BE::glwe_trace_galois_elements(self)
    },
    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_trace_tmp_bytes(self, res_infos, a_infos, key_infos)
    },
    fn glwe_trace<R, A, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::glwe_trace(self, res, skip, a, keys, scratch)
    },
    fn glwe_trace_assign<R, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::glwe_trace_assign(self, res, skip, keys, scratch)
    }
);

impl_operations_delegate!(
    GLWEPacking<BE>,
    GLWEPackImpl,
    GLWEPackingReference<BE>,
    fn glwe_pack_galois_elements(&self) -> Vec<i64> {
        BE::glwe_pack_galois_elements(self)
    },
    fn glwe_pack_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_pack_tmp_bytes(self, res, key)
    },
    fn glwe_pack<R, A, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::glwe_pack(self, res, a, log_gap_out, keys, scratch)
    }
);
