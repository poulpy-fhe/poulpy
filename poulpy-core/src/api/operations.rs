use crate::layouts::IntPolyInfos;
use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, CnvPVecRToBackendRef, ScratchArena};

use crate::layouts::{
    GGLWEActiveUse, GGLWEInfos, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef,
    GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedBound, GLWETensorKeyPreparedToBackendRef},
};

pub trait GLWETrace<BE: Backend> {
    fn glwe_trace_galois_elements(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, L, H>(&self, res_infos: &R, a_infos: &A, keys: &H) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>;

    fn glwe_trace<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;

    fn glwe_trace_assign<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;
}

pub trait GLWEPacking<BE: Backend> {
    fn glwe_pack_galois_elements(&self) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, L, H>(&self, res: &R, keys: &H) -> usize
    where
        R: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>;

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
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;
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

/// One [`GLWETensoring::glwe_tensor_apply_relinearize`] in a batch.
///
/// Instantiated with shared references for the `*_tmp_bytes` query and with a
/// mutable destination for execution.
pub struct TensorApplyRelinearizeItem<R, I, A, B> {
    pub cnv_offset: usize,
    pub res: R,
    pub tensor_infos: I,
    pub a: A,
    pub b: B,
}

/// One [`GLWETensoring::glwe_tensor_square_apply_relinearize`] in a batch.
pub struct TensorSquareApplyRelinearizeItem<R, I, A> {
    pub cnv_offset: usize,
    pub res: R,
    pub tensor_infos: I,
    pub a: A,
}

/// One [`GLWETensoring::glwe_tensor_square_relinearize_assign`] in a batch.
pub struct TensorSquareRelinearizeAssignItem<R, I> {
    pub cnv_offset: usize,
    pub res: R,
    pub tensor_infos: I,
}

/// One [`GLWETensoring::glwe_tensor_apply_prepared_right_relinearize_assign`] in a batch.
pub struct TensorPreparedRightRelinearizeAssignItem<R, I, BP> {
    pub cnv_offset: usize,
    pub res: R,
    pub tensor_infos: I,
    pub prepared_right: BP,
    pub prepared_right_size: usize,
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

    /// [`Self::glwe_tensor_apply`] against a caller-prepared right operand:
    /// only `a` is prepared, `b_prep` is reused as-is. `b_size` is the limb
    /// count of the operand `b_prep` was prepared from.
    ///
    /// Scratch: no more than
    /// [`glwe_tensor_apply_tmp_bytes`](Self::glwe_tensor_apply_tmp_bytes) for
    /// the equivalent unprepared layouts, since the right operand is not
    /// prepared here. An override must respect that bound.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_prepared_right<R, A, BP>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b_prep: &BP,
        b_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        BP: CnvPVecRToBackendRef<BE>;

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

    /// Fused [`Self::glwe_tensor_apply`] + [`Self::glwe_tensor_relinearize`].
    ///
    /// `tensor_infos` describes the intermediate the composition materializes;
    /// it cannot be inferred from `res`, which may be narrower. Scratch:
    /// `glwe_tensor_bytes_of_from_infos(tensor_infos) + max(apply, relinearize)`,
    /// i.e. the bound already returned by `ckks_mul_tmp_bytes`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_relinearize<R, I, A, B, T>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        tensor_infos: &I,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        I: GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Fused [`Self::glwe_tensor_square_apply`] + [`Self::glwe_tensor_relinearize`],
    /// with `res` as the implicit source operand. `res` must stay readable
    /// until the tensor product has consumed it.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_relinearize_assign<R, I, T>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        tensor_infos: &I,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        I: GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Fused [`Self::glwe_tensor_apply_prepared_right`] +
    /// [`Self::glwe_tensor_relinearize`], with `res` as the implicit left
    /// operand. `res` must stay readable until the tensor product has consumed it.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_prepared_right_relinearize_assign<R, I, BP, T>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        tensor_infos: &I,
        b_prep: &BP,
        b_size: usize,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        I: GLWEInfos,
        BP: CnvPVecRToBackendRef<BE>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Fused [`Self::glwe_tensor_square_apply`] + [`Self::glwe_tensor_relinearize`],
    /// out of place: `res = a * a`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply_relinearize<R, I, A, T>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        tensor_infos: &I,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        I: GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Scratch bytes for [`Self::glwe_tensor_apply_relinearize_batch`].
    ///
    /// Core tensor batches contain active, positive-precision lanes only;
    /// callers handle [`GGLWEUse::Empty`](crate::layouts::GGLWEUse::Empty)
    /// separately.
    /// The sequential default reuses one arena, so it returns the largest
    /// single-item bound. An override may return more.
    /// `uses[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    fn glwe_tensor_apply_relinearize_batch_tmp_bytes<R, I, A, B>(
        &self,
        items: &[TensorApplyRelinearizeItem<&R, &I, &A, &B>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        R: GLWEInfos,
        I: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    /// Independent [`Self::glwe_tensor_apply_relinearize`] calls, in item order.
    ///
    /// Every item keeps its own `cnv_offset`, tensor layout and operand widths.
    /// `bounds[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    /// Destinations must be mutually non-aliasing and must not alias another
    /// item's readable operand; read-only operands may repeat. An empty slice is
    /// a no-op; a positive-precision singleton is exactly the scalar
    /// computation.
    fn glwe_tensor_apply_relinearize_batch<R, I, A, B>(
        &self,
        items: &mut [TensorApplyRelinearizeItem<&mut R, &I, &A, &B>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        I: GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;

    /// Scratch bytes for [`Self::glwe_tensor_square_apply_relinearize_batch`].
    /// `uses[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    fn glwe_tensor_square_apply_relinearize_batch_tmp_bytes<R, I, A>(
        &self,
        items: &[TensorSquareApplyRelinearizeItem<&R, &I, &A>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        R: GLWEInfos,
        I: GLWEInfos,
        A: GLWEInfos;

    /// Independent [`Self::glwe_tensor_square_apply_relinearize`] calls, in item order.
    /// `bounds[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    fn glwe_tensor_square_apply_relinearize_batch<R, I, A>(
        &self,
        items: &mut [TensorSquareApplyRelinearizeItem<&mut R, &I, &A>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        I: GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    /// Scratch bytes for [`Self::glwe_tensor_square_relinearize_assign_batch`].
    /// `uses[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    fn glwe_tensor_square_relinearize_assign_batch_tmp_bytes<R, I>(
        &self,
        items: &[TensorSquareRelinearizeAssignItem<&R, &I>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        R: GLWEInfos,
        I: GLWEInfos;

    /// Independent [`Self::glwe_tensor_square_relinearize_assign`] calls, in item order.
    /// `bounds[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    fn glwe_tensor_square_relinearize_assign_batch<R, I>(
        &self,
        items: &mut [TensorSquareRelinearizeAssignItem<&mut R, &I>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        I: GLWEInfos;

    /// Scratch bytes for [`Self::glwe_tensor_apply_prepared_right_relinearize_assign_batch`].
    /// `uses[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    fn glwe_tensor_apply_prepared_right_relinearize_assign_batch_tmp_bytes<R, I, BP>(
        &self,
        items: &[TensorPreparedRightRelinearizeAssignItem<&R, &I, &BP>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        R: GLWEInfos,
        I: GLWEInfos;

    /// Independent [`Self::glwe_tensor_apply_prepared_right_relinearize_assign`]
    /// calls, in item order. Prepared operands may repeat across items.
    /// `bounds[i]` is authoritative for `items[i]`; fusion or splitting must
    /// preserve that alignment.
    fn glwe_tensor_apply_prepared_right_relinearize_assign_batch<R, I, BP>(
        &self,
        items: &mut [TensorPreparedRightRelinearizeAssignItem<&mut R, &I, &BP>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        I: GLWEInfos,
        BP: CnvPVecRToBackendRef<BE>;
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
