use std::collections::HashMap;

use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEAutomorphismKeyHelper,
        GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};

pub trait GLWETrace<BE: Backend> {
    fn glwe_trace_galois_elements(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

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
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;

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
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

pub trait GLWEPacking<BE: Backend> {
    fn glwe_pack_galois_elements(&self) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

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
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's;
}

pub trait GLWEMulConst<BE: Backend> {
    fn glwe_mul_const_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

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
        B: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_mul_const_assign<'s, R, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;
}

pub trait GLWEMulPlain<BE: Backend> {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
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
        B: GLWEToBackendRef<BE> + GLWEInfos;

    #[allow(clippy::too_many_arguments)]
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
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

pub trait GLWETensoring<BE: Backend> {
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
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
        B: GLWEToBackendRef<BE> + GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply<'s, R, A>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

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
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;
}

pub trait GLWEAdd<BE: Backend> {
    fn glwe_add_into<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>;

    fn glwe_add_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

pub trait GLWENegate<BE: Backend> {
    fn glwe_negate<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_negate_assign<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>;
}

pub trait GLWESub<BE: Backend> {
    fn glwe_sub<R, A, B>(&self, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>;

    fn glwe_sub_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_sub_negate_assign<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

pub trait GLWEZero<BE: Backend> {
    fn glwe_zero<R>(&self, res: &mut R)
    where
        R: GLWEToBackendMut<BE>;
}

pub trait GLWERotate<BE: Backend> {
    fn glwe_rotate_tmp_bytes(&self) -> usize;

    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_rotate_assign<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

pub trait GGSWRotate<BE: Backend> {
    fn ggsw_rotate_tmp_bytes(&self) -> usize;

    fn ggsw_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos;

    fn ggsw_rotate_assign<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchAvailable;
}

pub trait GLWEMulXpMinusOne<BE: Backend> {
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_mul_xp_minus_one_assign<'s, R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

pub trait GLWECopy<BE: Backend> {
    fn glwe_copy<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

pub trait GLWEShift<BE: Backend> {
    fn glwe_shift_tmp_bytes(&self) -> usize;

    fn glwe_rsh<'s, R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_assign<'s, R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_add<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_lsh_sub<'s, R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

pub trait GLWENormalize<BE: Backend> {
    fn glwe_normalize_tmp_bytes(&self) -> usize;

    fn glwe_normalize<'s, R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;

    fn glwe_normalize_assign<'s, R>(&self, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}
