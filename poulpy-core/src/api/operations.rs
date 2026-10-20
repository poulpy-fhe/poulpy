use crate::layouts::IntPolyInfos;
use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef, GetAutomorphismKey, GetTensorKey,
};

/// Normalized trace onto the subring selected by `skip`.
///
/// Each stage divides by two with the shift operation's rounding, then adds
/// the corresponding Galois conjugate. `skip` must not exceed `log2(N)`.
/// An empty trace (`skip == log2(N)`) copies the input using [`GLWECopy`]
/// semantics; its assign form preserves the destination. Neither needs keys.
/// Invalid skips panic before mutation.
pub trait GLWETrace<BE: Backend> {
    fn glwe_trace_galois_elements(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch required by the selected in-place trace implementation.
    fn glwe_trace_assign_tmp_bytes<A, K>(&self, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<R, A, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;

    fn glwe_trace_assign<R, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;
}

/// Packs the selected coefficients through a merge tree and normalized trace.
///
/// The input map must be nonempty, `log_gap_out <= log2(N)`, and every index
/// must be below `N` and divisible by `2^log_gap_out`. All inputs must have the
/// same degree, radix, precision and rank. Invalid maps, mixed input layouts or
/// gaps panic before changing inputs or destination. Valid calls consume and
/// mutate the input ciphertexts; their final values are not preserved.
pub trait GLWEPacking<BE: Backend> {
    fn glwe_pack_galois_elements(&self) -> Vec<i64>;

    /// Scratch for packing inputs of layout `a` into a destination of layout `res`.
    /// The derived implementation sizes the merge tree at `a` and the final
    /// trace from `a` to `res`, including the trace's intermediate storage.
    fn glwe_pack_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

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
        H: GetAutomorphismKey<BE>;
}

/// Multiplication of a GLWE ciphertext by one coefficient of a plaintext.
///
/// The ciphertext operand is read canonical at its `k` (see the normalized
/// form on [`crate::layouts::GLWE`]): the product reads every bit of its live
/// limbs, so an operand whose canonical flag is clear is normalized first.
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
/// by compute. The ciphertext operand, a Torus element, is consumed at the
/// `k` it declares, canonical at it (see the normalized form on
/// [`crate::layouts::GLWE`]): the convolution reads every bit of its live
/// limbs, so an operand whose canonical flag is clear is normalized first.
pub trait GLWEMulPlain<BE: Backend> {
    /// The right operand may have a degree that is a power-of-two divisor of the
    /// module's, not below the backend floor, which stands for its ring
    /// embedding; the budget is an upper bound for it.
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

/// Tensor products of GLWE ciphertexts.
///
/// Both ciphertext operands are read canonical at their `k` (see the
/// normalized form on [`crate::layouts::GLWE`]): the convolution reads every
/// bit of their live limbs, so an operand whose canonical flag is clear is
/// normalized first.
pub trait GLWETensoring<BE: Backend> {
    /// Scratch required for a tensor product with a reusable prepared right operand.
    /// `a_size` and `b_size` are the operands' effective limb counts.
    fn glwe_tensor_apply_prepared_right_tmp_bytes<R, A>(&self, res: &R, a: &A, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    /// Tensor product with a right operand prepared by [`crate::glwe_prepare_right`].
    /// Preserves both inputs and returns a canonical tensor at `res.k()`.
    /// `b_size` is the effective limb count used to prepare `b`.
    fn glwe_tensor_apply_prepared_right<R, A, BP>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &BP,
        b_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        BP: CnvPVecRToBackendRef<BE>;

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

    fn glwe_tensor_relinearize<R, A, H>(&self, res: &mut R, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetTensorKey<BE>;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    /// Scratch bytes of [`Self::glwe_mul_relinearize`], [`Self::glwe_square_relinearize`],
    /// [`Self::glwe_mul_prepared_relinearize`] and their `_assign` forms, for a result as wide as `res`, operands of
    /// `a_size` and `b_size` limbs and a product of width `tensor_k`, the larger of the two operand widths.
    fn glwe_mul_relinearize_tmp_bytes<R, B>(
        &self,
        res: &R,
        a_size: usize,
        b_size: usize,
        tensor_k: crate::layouts::TorusPrecision,
        tsk: &B,
    ) -> usize
    where
        R: GLWEInfos,
        B: GGLWEInfos;

    /// `res = relinearize(a * b)` in one pass, for rank 1, one `base2k` across `res`,
    /// the operands and the tensor key, and `cnv_offset >= base2k`.
    ///
    /// The result is written at width `res_k`: the caller sets the width of `res` afterwards.
    ///
    /// The columns of the product that are not key-switched stay in the transform domain until the
    /// single final normalization, where [`Self::glwe_tensor_apply`] followed by
    /// [`Self::glwe_tensor_relinearize`] transforms and normalizes them first.
    /// The two paths agree up to rounding, not bit for bit.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_relinearize<R, A, B, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetTensorKey<BE>;

    /// [`Self::glwe_mul_relinearize`] with `res`, read at its current width, as the left operand.
    fn glwe_mul_relinearize_assign<R, A, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetTensorKey<BE>;

    /// [`Self::glwe_mul_relinearize`] of `a` by itself.
    fn glwe_square_relinearize<R, A, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetTensorKey<BE>;

    /// [`Self::glwe_square_relinearize`] of `res`, read at its current width.
    fn glwe_square_relinearize_assign<R, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetTensorKey<BE>;

    /// [`Self::glwe_mul_relinearize`] with a right operand prepared by [`crate::glwe_prepare_right`]:
    /// `b_size` limbs of a ciphertext of width `b_k`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_relinearize<R, A, BP, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        b: &BP,
        b_size: usize,
        b_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        BP: poulpy_hal::layouts::CnvPVecRToBackendRef<BE>,
        H: GetTensorKey<BE>;

    /// [`Self::glwe_mul_prepared_relinearize`] with `res`, read at its current width, as the left operand.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_relinearize_assign<R, AP, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &AP,
        a_size: usize,
        a_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        AP: poulpy_hal::layouts::CnvPVecRToBackendRef<BE>,
        H: GetTensorKey<BE>;
}

/// Sums of products of rank-1 GLWE ciphertexts, relinearized once. The operands are read canonical at their
/// `k`, as for [`GLWETensoring`]: one whose canonical flag is clear is normalized into scratch first.
pub trait GLWEMulRelinearizeSum<BE: Backend> {
    /// Scratch bytes of [`Self::glwe_mul_relinearize_sum`], whatever the number of pairs.
    fn glwe_mul_relinearize_sum_tmp_bytes<R, B>(
        &self,
        res: &R,
        a_size: usize,
        b_size: usize,
        tensor_k: crate::layouts::TorusPrecision,
        tsk: &B,
    ) -> usize
    where
        R: GLWEInfos,
        B: GGLWEInfos;

    /// `res = relinearize(Σ_i a[i] * b[i])`, in one pass, under the conditions of
    /// [`GLWETensoring::glwe_mul_relinearize`]: rank 1, one `base2k` across `res`, the operands and the tensor key,
    /// and `cnv_offset >= base2k`. Every pair shares the first one's layouts; the tensor key is looked up at the
    /// larger of the two operand widths.
    ///
    /// The result is written at width `res_k`, canonical: the caller sets the width of `res` afterwards.
    ///
    /// Each pair's three product columns are added to the first pair's in the transform domain. Only the
    /// key-switched column of the sum is brought back to digits; the other two stay in the transform domain until
    /// the single final normalization. One pair is [`GLWETensoring::glwe_mul_relinearize`], bit for bit.
    /// Relinearizing a sum of [`GLWETensoring::glwe_tensor_apply`] products rounds every term instead: the two agree
    /// up to rounding, not bit for bit.
    ///
    /// Every pair's products accumulate in the transform domain before one inverse transform: pick `base2k` for
    /// `a.len()` times the products of one pair (`Module::max_base2k`), up to `2 * min(a_size, b_size)` per
    /// output limb in the cross column `a0 * b1 + a1 * b0`.
    ///
    /// Panics, before writing `res`, if `a` is empty or `b` is not as long, if a pair does not share the first one's
    /// layouts, if `res` or an operand is not of rank 1 or of `res`'s `base2k`, if `cnv_offset < base2k`, if `tsk`
    /// has no key at the tensor width or one of another `base2k`, or if `scratch` holds less than the call needs.
    #[expect(
        clippy::too_many_arguments,
        reason = "the two operand slices of a sum beside the product's operands"
    )]
    fn glwe_mul_relinearize_sum<R, A, B, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &[&A],
        b: &[&B],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetTensorKey<BE>;
}

/// Products of GLWE ciphertexts over operands prepared once by [`crate::glwe_prepare_left`] and
/// [`crate::glwe_prepare_right`]: the call transforms no operand.
///
/// An operand prepared from a ciphertext of width `k`, under either [`PrepareHint`](poulpy_hal::layouts::PrepareHint),
/// holds it as the unprepared products read it: canonical at `k` (normalized first if its canonical flag is clear),
/// each of its `rank + 1` columns on `ceil(k / base2k)` limbs. Operands prepared so give the unprepared products of
/// the ciphertexts they were prepared from, bit for bit.
pub trait GLWEPreparedTensoring<BE: Backend> {
    /// Scratch bytes of [`Self::glwe_mul_prepared_relinearize_sum`], whatever the number of pairs. One pair needs no
    /// more than [`GLWETensoring::glwe_mul_relinearize_tmp_bytes`].
    fn glwe_mul_prepared_relinearize_sum_tmp_bytes<R, B>(
        &self,
        res: &R,
        a_size: usize,
        b_size: usize,
        tensor_k: crate::layouts::TorusPrecision,
        tsk: &B,
    ) -> usize
    where
        R: GLWEInfos,
        B: GGLWEInfos;

    /// [`GLWEMulRelinearizeSum::glwe_mul_relinearize_sum`] over prepared pairs `(a[i], b[i])`: `a_size` and
    /// `b_size` limbs of rank-1 ciphertexts of widths `a_k` and `b_k`, of the degree and `base2k` of `res`, with
    /// `a_size = ceil(a_k / base2k)` and `b_size = ceil(b_k / base2k)`.
    ///
    /// The pairs feed the product columns where they are, under the conditions of the unprepared sum, whose result
    /// they give.
    ///
    /// Panics, before writing `res`, if `a` is empty or `b` is not as long, if a given size is not the limb count of
    /// its width, if a prepared operand is not of the degree of `res`, of two columns and of its given size, or under
    /// the conditions of [`GLWEMulRelinearizeSum::glwe_mul_relinearize_sum`].
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_relinearize_sum<R, AP, BP, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &[&AP],
        b: &[&BP],
        a_size: usize,
        a_k: crate::layouts::TorusPrecision,
        b_size: usize,
        b_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>,
        H: GetTensorKey<BE>;

    /// Scratch bytes of [`Self::glwe_tensor_apply_prepared`] for the same sizes and radix.
    fn glwe_tensor_apply_prepared_tmp_bytes<R>(
        &self,
        res: &R,
        a_size: usize,
        b_size: usize,
        base2k: crate::layouts::Base2K,
    ) -> usize
    where
        R: GLWEInfos;

    /// [`GLWETensoring::glwe_tensor_apply`] of a prepared left operand of `a_size` limbs and a prepared right operand
    /// of `b_size` limbs, at radix `base2k`, each with one column per column of a ciphertext of the rank of `res`:
    /// the canonical tensor at `res.k()` that the unprepared product gives.
    ///
    /// Panics, before writing `res`, if an operand is not of the degree of `res`, of `rank + 1` columns and of its
    /// given size, or if `scratch` holds less than the call needs.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_prepared<R, AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &AP,
        a_size: usize,
        b: &BP,
        b_size: usize,
        base2k: crate::layouts::Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        AP: CnvPVecLToBackendRef<BE>,
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

    /// Replaces `res` with `a - res`. For a rank-zero `a`, the old mask is negated.
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

/// Multiplies stored polynomial limbs by `X^k - 1` in the negacyclic ring.
///
/// This is raw limb arithmetic: it preserves destination layout metadata and performs
/// neither radix conversion nor normalization, even when source and destination
/// radices differ. Their degrees and ranks must match. Component noise metadata is cleared.
pub trait GLWEMulXpMinusOne<BE: Backend> {
    /// Scratch for the in-place variant on a destination of `res_size` limbs.
    fn glwe_mul_xp_minus_one_assign_tmp_bytes(&self, res_size: usize) -> usize;

    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_mul_xp_minus_one_assign<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

/// Copies a canonical GLWE's torus value into the destination layout.
///
/// The destination keeps its layout metadata and copies the source's component
/// noise metadata. Narrowing rounds to its logical `k`,
/// including partial limbs; differing radices are converted. With equal radices
/// and no precision loss, this is a limb copy with zero-extension. A rank-zero
/// source may be copied into a higher-rank destination, whose mask is zeroed.
/// Equal-layout copies also preserve unnormalized digits verbatim; normalizing
/// such an intermediate remains the caller's responsibility.
/// Exact copies preserve noise metadata; narrowing the torus precision clears it.
pub trait GLWECopy<BE: Backend> {
    /// Scratch for the given source and destination layouts (zero for the
    /// equal-radix, non-narrowing path).
    fn glwe_copy_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(&self, res: &R, a: &A) -> usize;

    fn glwe_copy<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

pub trait GLWEShift<BE: Backend> {
    /// Scratch bytes for the GLWE shift family, on a destination of
    /// `res_size` limbs.
    fn glwe_shift_tmp_bytes(&self, res_size: usize) -> usize;

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
