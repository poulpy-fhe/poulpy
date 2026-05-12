use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::{
        GGSWRotate, GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWEMulXpMinusOne, GLWENegate, GLWENormalize, GLWEPacking,
        GLWERotate, GLWEShift, GLWESub, GLWETensoring, GLWETrace, GLWEZero,
    },
    default::{glwe_packing::GLWEPackingDefault, glwe_trace::GLWETraceDefault},
    layouts::{
        GGLWEInfos, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEAutomorphismKeyHelper,
        GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
    oep::{
        GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl, GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl,
        GLWENormalizeImpl, GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, GLWETraceImpl,
        GLWEZeroImpl,
    },
    operations::{
        GGSWRotateDefault, GLWEAddDefault, GLWECopyDefault, GLWEMulConstDefault, GLWEMulPlainDefault, GLWEMulXpMinusOneDefault,
        GLWENegateDefault, GLWENormalizeDefault, GLWERotateDefault, GLWEShiftDefault, GLWESubDefault, GLWETensoringDefault,
        GLWEZeroDefault,
    },
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
    GLWEAddImpl<BE>,
    GLWEAddDefault<BE>,
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
    GLWENegateImpl<BE>,
    GLWENegateDefault<BE>,
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
    GLWESubImpl<BE>,
    GLWESubDefault<BE>,
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
    GLWEZeroImpl<BE>,
    GLWEZeroDefault<BE>,
    fn glwe_zero<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_zero(self, res)
    }
);

impl_operations_delegate!(
    GLWECopy<BE>,
    GLWECopyImpl<BE>,
    GLWECopyDefault<BE>,
    fn glwe_copy<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_copy(self, res, a)
    }
);

impl_operations_delegate!(
    GLWEMulConst<BE>,
    GLWEMulConstImpl<BE>,
    GLWEMulConstDefault<BE>,
    fn glwe_mul_const_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_mul_const_tmp_bytes(self, res, a, b)
    },
    fn glwe_mul_const<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_mul_const(self, cnv_offset, res, a, b, b_coeff, scratch)
    },
    fn glwe_mul_const_assign<'s, R, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_mul_const_assign(self, cnv_offset, res, b, b_coeff, scratch)
    }
);

impl_operations_delegate!(
    GLWEMulPlain<BE>,
    GLWEMulPlainImpl<BE>,
    GLWEMulPlainDefault<BE>,
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_mul_plain_tmp_bytes(self, res, a, b)
    },
    fn glwe_mul_plain<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_mul_plain(self, cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    },
    fn glwe_mul_plain_assign<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_effective_k: usize,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_mul_plain_assign(self, cnv_offset, res, res_effective_k, a, a_effective_k, scratch)
    }
);

impl_operations_delegate!(
    GLWETensoring<BE>,
    GLWETensoringImpl<BE>,
    GLWETensoringDefault<BE>,
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
    fn glwe_tensor_apply<'s, R, A, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_tensor_apply(self, cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    },
    fn glwe_tensor_square_apply<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_tensor_square_apply(self, cnv_offset, res, a, a_effective_k, scratch)
    },
    fn glwe_tensor_relinearize<'s, R, A, T>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::glwe_tensor_relinearize(self, res, a, tsk, tsk_size, scratch)
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
    GLWERotateImpl<BE>,
    GLWERotateDefault<BE>,
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
    fn glwe_rotate_assign<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_rotate_assign(self, k, res, scratch);
    }
);

impl_operations_delegate!(
    GGSWRotate<BE>,
    GGSWRotateImpl<BE>,
    GGSWRotateDefault<BE>,
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
    fn ggsw_rotate_assign<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE> + poulpy_hal::api::ScratchAvailable,
    {
        BE::ggsw_rotate_assign(self, k, res, scratch)
    }
);

impl_operations_delegate!(
    GLWEMulXpMinusOne<BE>,
    GLWEMulXpMinusOneImpl<BE>,
    GLWEMulXpMinusOneDefault<BE>,
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        BE::glwe_mul_xp_minus_one(self, k, res, a)
    },
    fn glwe_mul_xp_minus_one_assign<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::glwe_mul_xp_minus_one_assign(self, k, res, scratch)
    }
);

impl_operations_delegate!(
    GLWEShift<BE>,
    GLWEShiftImpl<BE>,
    GLWEShiftDefault<BE>,
    fn glwe_shift_tmp_bytes(&self) -> usize {
        BE::glwe_shift_tmp_bytes(self)
    },
    fn glwe_rsh<'s, R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_rsh(self, k, res, scratch)
    },
    fn glwe_lsh_assign<'s, R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh_assign(self, res, k, scratch)
    },
    fn glwe_lsh<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh(self, res, a, k, scratch)
    },
    fn glwe_lsh_add<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh_add(self, res, a, k, scratch)
    },
    fn glwe_lsh_sub<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_lsh_sub(self, res, a, k, scratch)
    }
);

impl_operations_delegate!(
    GLWENormalize<BE>,
    GLWENormalizeImpl<BE>,
    GLWENormalizeDefault<BE>,
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_normalize_tmp_bytes(self)
    },
    fn glwe_normalize<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_normalize(self, res, a, scratch)
    },
    fn glwe_normalize_assign<'s, R>(&self, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::glwe_normalize_assign(self, res, scratch)
    }
);

impl_operations_delegate!(
    GLWETrace<BE>,
    GLWETraceImpl<BE>,
    GLWETraceDefault<BE>,
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
    fn glwe_trace<'s, R, A, K, H>(
        &self,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::glwe_trace(self, res, skip, a, keys, key_size, scratch)
    },
    fn glwe_trace_assign<'s, R, K, H>(
        &self,
        res: &mut R,
        skip: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::glwe_trace_assign(self, res, skip, keys, key_size, scratch)
    }
);

impl_operations_delegate!(
    GLWEPacking<BE>,
    GLWEPackImpl<BE>,
    GLWEPackingDefault<BE>,
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
    fn glwe_pack<'s, R, A, K, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        BE::glwe_pack(self, res, a, log_gap_out, keys, key_size, scratch)
    }
);
