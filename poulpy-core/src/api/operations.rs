use crate::layouts::IntPolyInfos;
use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEAutomorphismKeyMap, GLWEInfos,
    GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};

pub trait GLWETrace<BE: Backend> {
    fn glwe_trace_galois_elements(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyMap<K, BE>;

    fn glwe_trace_assign<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyMap<K, BE>;
}

pub trait GLWEPacking<BE: Backend> {
    fn glwe_pack_galois_elements(&self) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack<R, A, K, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyMap<K, BE>;
}

pub trait GLWEMulConst<BE: Backend> {
    fn glwe_mul_const_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

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
        B: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_mul_const_assign<R, B>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Multiplication of a GLWE ciphertext by a **plaintext** operand.
///
/// The plain operand — `b` in [`Self::glwe_mul_plain`], `a` in
/// [`Self::glwe_mul_plain_assign`] — is an **integer polynomial**, not a Torus
/// element: LSB-anchored, every encoded limb carries data. The convolution
/// therefore consumes it at its declared
/// [`encoded_k()`](crate::layouts::IntPolyInfos::encoded_k) — the operand is
/// bounded by [`crate::layouts::IntPolyInfos`], so a type that cannot state
/// its encoded width cannot be passed here. Its `k` labels claimed precision
/// for budget arithmetic only, and `max_k()` is the allocation, never consumed
/// by compute. The ciphertext operand, a Torus element, is processed at its
/// effective `k`.
pub trait GLWEMulPlain<BE: Backend> {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<R, A, B>(&self, cnv_offset: usize, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_assign<R, A>(&self, cnv_offset: usize, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + IntPolyInfos + GLWEInfos;
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

    fn glwe_tensor_apply<R, A, B>(&self, cnv_offset: usize, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_tensor_square_apply<R, A>(&self, cnv_offset: usize, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_tensor_relinearize<R, A, T>(&self, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

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

    fn glwe_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

pub trait GGSWRotate<BE: Backend> {
    fn ggsw_rotate_tmp_bytes(&self) -> usize;

    fn ggsw_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos;

    fn ggsw_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}

pub trait GLWEMulXpMinusOne<BE: Backend> {
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_mul_xp_minus_one_assign<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
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

    fn glwe_rsh<R>(&self, k: usize, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>;

    fn glwe_lsh_assign<R>(&self, res: &mut R, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>;

    fn glwe_lsh<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_lsh_add<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_lsh_sub<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

pub trait GLWENormalize<BE: Backend> {
    fn glwe_normalize_tmp_bytes(&self) -> usize;

    fn glwe_normalize<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_normalize_assign<R>(&self, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>;
}
