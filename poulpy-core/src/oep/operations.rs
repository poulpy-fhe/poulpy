use crate::layouts::IntPolyInfos;
use std::{collections::HashMap, ops::Range};

use poulpy_hal::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLToBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef,
    CnvPVecRToBackendRef, Module, ScratchArena, Standard, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut,
};

use crate::layouts::{
    Degree, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
    GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey,
};

/// Backend-provided GLWE constant-multiplication operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEMulConstImpl: Backend {
    /// Scratch bound of [`Self::glwe_mul_const`] and [`Self::glwe_mul_const_assign`]; may report less than the
    /// reference, never less than the body takes.
    fn glwe_mul_const_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    /// Reproduces
    /// [`glwe_mul_const_reference`](crate::reference::operations::GLWEMulConstReference::glwe_mul_const_reference)
    /// bit for bit, within [`Self::glwe_mul_const_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_mul_const<R, A, B>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        B: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_mul_const_assign_reference`](crate::reference::operations::GLWEMulConstReference::glwe_mul_const_assign_reference)
    /// bit for bit, within [`Self::glwe_mul_const_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_mul_const_assign<R, B>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        B: GLWEToBackendRef<Self> + GLWEInfos;

    /// Scratch bytes of [`Self::glwe_const_combination`] for an accumulator of `acc_size` limbs,
    /// terms of at most `a_size` limbs and constants of at most `b_size` limbs.
    fn glwe_const_combination_tmp_bytes(module: &Module<Self>, acc_size: usize, a_size: usize, b_size: usize) -> usize;

    /// A combination of vectors by constants, normalized once:
    ///
    /// `res[col] = normalize(sum_t terms[t].a[col] * (terms[t].b[b_col][b_coeff] >> terms[t].shift))`
    /// for every column of `res`,
    ///
    /// each constant with a non-zero `shift < base2k` being shifted right by that many bits over
    /// one more limb than `b` (`vec_znx_rsh`), each product being `cnv_by_const_apply` at its
    /// `cnv_offset`, the sum being accumulated over `acc_size` limbs and normalized to `res_k`
    /// bits in base `2^base2k`, which is also the base of the terms. `terms` is not empty.
    ///
    /// An internal hook of the CKKS baby-step combination, which records the metadata of `res`.
    /// The reference, [`glwe_const_combination_reference`](crate::reference::operations::glwe_const_combination_reference),
    /// shifts each constant, accumulates with `cnv_by_const_apply` and `cnv_by_const_apply_add`,
    /// then calls `vec_znx_big_normalize`, column by column. Overrides must reproduce it bit for
    /// bit within [`Self::glwe_const_combination_tmp_bytes`].
    fn glwe_const_combination(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        base2k: usize,
        res_k: usize,
        acc_size: usize,
        terms: &[ConstTerm<'_, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    );
}

/// One term of [`GLWEMulConstImpl::glwe_const_combination`]: every column of `a` times
/// coefficient `b_coeff` of column `b_col` of `b`, shifted right by `shift` bits, at
/// convolution offset `cnv_offset`.
pub struct ConstTerm<'a, BE: Backend + 'a> {
    pub a: VecZnxBackendRef<'a, BE>,
    pub b: VecZnxBackendRef<'a, BE>,
    pub b_col: usize,
    pub b_coeff: usize,
    pub shift: usize,
    pub cnv_offset: usize,
}

/// Backend-provided GLWE-by-plaintext multiplication operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEMulPlainImpl: Backend {
    /// Scratch bound of [`Self::glwe_mul_plain`] and [`Self::glwe_mul_plain_assign`]; may report less than the
    /// reference, never less than the body takes.
    fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    /// Reproduces
    /// [`glwe_mul_plain_reference`](crate::reference::operations::GLWEMulPlainReference::glwe_mul_plain_reference)
    /// bit for bit, within [`Self::glwe_mul_plain_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<R, A, B>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        B: GLWEToBackendRef<Self> + IntPolyInfos + GLWEInfos;

    /// Reproduces
    /// [`glwe_mul_plain_assign_reference`](crate::reference::operations::GLWEMulPlainReference::glwe_mul_plain_assign_reference)
    /// bit for bit, within [`Self::glwe_mul_plain_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_mul_plain_assign<R, A>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + IntPolyInfos + GLWEInfos;
}

/// Backend-provided sums of products of prepared GLWE ciphertexts by prepared plaintexts read through Galois elements.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEMulPreparedPlainSumImpl: Backend {
    /// Scratch bound of [`Self::glwe_mul_prepared_plain_sum`], whatever the number of terms; may report less than the
    /// reference, never less than the body takes.
    fn glwe_mul_prepared_plain_sum_tmp_bytes<R>(module: &Module<Self>, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos;

    /// Reproduces
    /// [`glwe_mul_prepared_plain_sum_reference`](crate::reference::operations::GLWEMulPreparedPlainSumReference::glwe_mul_prepared_plain_sum_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_prepared_plain_sum_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_plain_sum<R, AP, BP>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: crate::layouts::Base2K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        AP: CnvPVecLToBackendRef<Self>,
        BP: CnvPVecRToBackendRef<Self>;

    /// Scratch bound of [`Self::glwe_mul_prepared_plain_sum_dft`], whatever the number of terms; may report less than
    /// the reference, never less than the body takes.
    fn glwe_mul_prepared_plain_sum_dft_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize, b_size: usize) -> usize;

    /// Reproduces
    /// [`glwe_mul_prepared_plain_sum_dft_reference`](crate::reference::operations::GLWEMulPreparedPlainSumReference::glwe_mul_prepared_plain_sum_dft_reference)
    /// bit for bit through `idft` under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_prepared_plain_sum_dft_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_plain_sum_dft<AP, BP>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: crate::layouts::Base2K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        AP: CnvPVecLToBackendRef<Self>,
        BP: CnvPVecRToBackendRef<Self>;

    /// Scratch bound of [`Self::glwe_mul_prepared_plain_sum_dft_add`], whatever the number of terms; may report less
    /// than the reference, never less than the body takes.
    fn glwe_mul_prepared_plain_sum_dft_add_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    /// Reproduces
    /// [`glwe_mul_prepared_plain_sum_dft_add_reference`](crate::reference::operations::GLWEMulPreparedPlainSumReference::glwe_mul_prepared_plain_sum_dft_add_reference)
    /// bit for bit through `idft` under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_prepared_plain_sum_dft_add_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_plain_sum_dft_add<AP, BP>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: crate::layouts::Base2K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        AP: CnvPVecLToBackendRef<Self>,
        BP: CnvPVecRToBackendRef<Self>;
}

/// Backend-provided GLWE tensoring and relinearization operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
/// The square hooks ([`Self::glwe_square_prepare`], [`Self::glwe_square_columns`]) are bit for bit
/// taken together: an override of `glwe_square_prepare` that leaves `right` unwritten also
/// overrides `glwe_square_columns`.
pub unsafe trait GLWETensoringImpl: Backend {
    /// Scratch bound of [`Self::glwe_tensor_apply_prepared_right`]; may report less than the reference, never less
    /// than the body takes.
    fn glwe_tensor_apply_prepared_right_tmp_bytes<R, A>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    /// Reproduces
    /// [`glwe_tensor_apply_prepared_right_reference`](crate::reference::operations::GLWETensoringReference::glwe_tensor_apply_prepared_right_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_tensor_apply_prepared_right_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_tensor_apply_prepared_right<R, A, BP>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &BP,
        b_size: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        BP: CnvPVecRToBackendRef<Self>;

    /// Scratch bound of [`Self::glwe_tensor_apply`]; may report less than the reference, never less than the body
    /// takes.
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    /// Scratch bound of [`Self::glwe_tensor_square_apply`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<Self>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    /// Reproduces
    /// [`glwe_tensor_apply_reference`](crate::reference::operations::GLWETensoringReference::glwe_tensor_apply_reference)
    /// bit for bit, within [`Self::glwe_tensor_apply_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_tensor_apply<R, A, B>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        B: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_tensor_square_apply_reference`](crate::reference::operations::GLWETensoringReference::glwe_tensor_square_apply_reference)
    /// bit for bit, within [`Self::glwe_tensor_square_apply_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_tensor_square_apply<R, A>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_tensor_relinearize_reference`](crate::reference::operations::GLWETensoringReference::glwe_tensor_relinearize_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_tensor_relinearize_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_tensor_relinearize<R, A, H>(module: &Module<Self>, res: &mut R, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Scratch bound of [`Self::glwe_tensor_relinearize`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    /// Scratch bound of [`Self::glwe_mul_relinearize`], [`Self::glwe_mul_relinearize_assign`],
    /// [`Self::glwe_square_relinearize`], [`Self::glwe_square_relinearize_assign`],
    /// [`Self::glwe_mul_prepared_relinearize`] and [`Self::glwe_mul_prepared_relinearize_assign`]; may report less
    /// than the reference, never less than the body takes.
    fn glwe_mul_relinearize_tmp_bytes<R, B>(
        module: &Module<Self>,
        res: &R,
        a_size: usize,
        b_size: usize,
        tensor_k: crate::layouts::TorusPrecision,
        tsk: &B,
    ) -> usize
    where
        R: GLWEInfos,
        B: GGLWEInfos;

    /// Reproduces
    /// [`glwe_mul_relinearize_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_relinearize_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_relinearize_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_relinearize<R, A, B, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        B: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`glwe_mul_relinearize_assign_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_relinearize_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_relinearize_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_mul_relinearize_assign<R, A, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`glwe_square_relinearize_reference`](crate::reference::operations::GLWETensoringReference::glwe_square_relinearize_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_relinearize_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_square_relinearize<R, A, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`glwe_square_relinearize_assign_reference`](crate::reference::operations::GLWETensoringReference::glwe_square_relinearize_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_relinearize_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_square_relinearize_assign<R, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`glwe_mul_prepared_relinearize_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_prepared_relinearize_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_relinearize_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_relinearize<R, A, BP, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &A,
        b: &BP,
        b_size: usize,
        b_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        BP: CnvPVecRToBackendRef<Self>,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`glwe_mul_prepared_relinearize_assign_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_prepared_relinearize_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_relinearize_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_relinearize_assign<R, AP, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &AP,
        a_size: usize,
        a_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        AP: CnvPVecRToBackendRef<Self>,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`glwe_mul_columns_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_columns_reference)
    /// bit for bit; free in kernels.
    ///
    /// The caller hands it the prepared rank-one operands, the two column buffers and the scratch left once those are
    /// allocated, within [`Self::glwe_mul_relinearize_tmp_bytes`]; a kernel needing more than the reference
    /// convolutions adds it there. It records no metadata.
    fn glwe_mul_columns(
        module: &Module<Self>,
        cnv_offset_hi: usize,
        diag: &mut VecZnxDftBackendMut<'_, Self>,
        cross: &mut VecZnxDftBackendMut<'_, Self>,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// Scratch bound of [`Self::glwe_mul_relinearize_sum`], whatever the number of pairs; may report less than the
    /// reference, never less than the body takes.
    fn glwe_mul_relinearize_sum_tmp_bytes<R, B>(
        module: &Module<Self>,
        res: &R,
        a_size: usize,
        b_size: usize,
        tensor_k: crate::layouts::TorusPrecision,
        tsk: &B,
    ) -> usize
    where
        R: GLWEInfos,
        B: GGLWEInfos;

    /// Reproduces
    /// [`glwe_mul_relinearize_sum_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_relinearize_sum_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_relinearize_sum_tmp_bytes`]; free in kernels and scratch use.
    #[expect(
        clippy::too_many_arguments,
        reason = "the two operand slices of a sum beside the product's operands"
    )]
    fn glwe_mul_relinearize_sum<R, A, B, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: &[&A],
        b: &[&B],
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        B: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Scratch bound of [`Self::glwe_mul_prepared_relinearize_sum`], whatever the number of pairs; one pair needs no
    /// more than [`Self::glwe_mul_relinearize_tmp_bytes`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_mul_prepared_relinearize_sum_tmp_bytes<R, B>(
        module: &Module<Self>,
        res: &R,
        a_size: usize,
        b_size: usize,
        tensor_k: crate::layouts::TorusPrecision,
        tsk: &B,
    ) -> usize
    where
        R: GLWEInfos,
        B: GGLWEInfos;

    /// Reproduces
    /// [`glwe_mul_prepared_relinearize_sum_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_prepared_relinearize_sum_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_mul_prepared_relinearize_sum_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_relinearize_sum<'a, R, AP, BP, H>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_k: crate::layouts::TorusPrecision,
        a: impl IntoIterator<Item = &'a AP, IntoIter: ExactSizeIterator + Clone>,
        a_size: usize,
        a_k: crate::layouts::TorusPrecision,
        b: impl IntoIterator<Item = &'a BP, IntoIter: ExactSizeIterator + Clone>,
        b_size: usize,
        b_k: crate::layouts::TorusPrecision,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        AP: CnvPVecLToBackendRef<Self> + 'a,
        BP: CnvPVecRToBackendRef<Self> + 'a,
        H: GetTensorKey<Self>;

    /// Scratch bound of [`Self::glwe_tensor_apply_prepared_sum_dft`] for the tensor of the layout `res`; may report less
    /// than the reference, never less than the body takes.
    fn glwe_tensor_apply_prepared_sum_dft_tmp_bytes<R>(module: &Module<Self>, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos;

    /// Internal hook of the linear transformations of ciphertext diagonals, their baby-step kernel as
    /// [`LinearTransformationImpl::glwe_accumulate_giant_steps`](crate::oep::LinearTransformationImpl::glwe_accumulate_giant_steps)
    /// is that of plaintext diagonals: reproduces
    /// [`glwe_tensor_apply_prepared_sum_dft_reference`](crate::reference::operations::GLWETensoringReference::glwe_tensor_apply_prepared_sum_dft_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_tensor_apply_prepared_sum_dft_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_prepared_sum_dft<'a, AP>(
        module: &Module<Self>,
        cnv_offset_hi: usize,
        diag: &mut VecZnxDftBackendMut<'_, Self>,
        cross: &mut VecZnxDftBackendMut<'_, Self>,
        pairs: impl IntoIterator<Item = (&'a AP, crate::layouts::DiagonalOperand<'a, Self>)>,
        b_size: usize,
        b_k: crate::layouts::TorusPrecision,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: 'a,
        AP: CnvPVecLToBackendRef<Self> + 'a;

    /// Scratch bound of [`Self::glwe_tensor_apply_prepared`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_tensor_apply_prepared_tmp_bytes<R>(
        module: &Module<Self>,
        res: &R,
        a_size: usize,
        b_size: usize,
        base2k: crate::layouts::Base2K,
    ) -> usize
    where
        R: GLWEInfos;

    /// Reproduces
    /// [`glwe_tensor_apply_prepared_reference`](crate::reference::operations::GLWETensoringReference::glwe_tensor_apply_prepared_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_tensor_apply_prepared_tmp_bytes`]; free in kernels and scratch use.
    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_prepared<R, AP, BP>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &AP,
        a_size: usize,
        b: &BP,
        b_size: usize,
        base2k: crate::layouts::Base2K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        AP: CnvPVecLToBackendRef<Self>,
        BP: CnvPVecRToBackendRef<Self>;

    /// Prepares the operand of a one-pass square for [`Self::glwe_square_columns`].
    ///
    /// The reference, [`GLWETensoringReference::glwe_square_prepare_reference`](crate::reference::operations::GLWETensoringReference::glwe_square_prepare_reference),
    /// prepares both forms with `cnv_prepare_self`. A backend whose [`Self::glwe_square_columns`]
    /// reads the left form alone may leave `right` unwritten, and only that one: the default
    /// `glwe_square_columns` reads both forms.
    /// Scratch is `cnv_prepare_self_tmp_bytes`, which the one-pass relinearization queries reserve.
    fn glwe_square_prepare(
        module: &Module<Self>,
        left: &mut CnvPVecLBackendMut<'_, Self>,
        right: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// [`Self::glwe_mul_columns`] of a square, on the two forms [`Self::glwe_square_prepare`]
    /// prepared from one operand. The default is [`Self::glwe_mul_columns`].
    /// Overrides must reproduce it bit for bit within the scratch of [`Self::glwe_mul_columns`].
    fn glwe_square_columns(
        module: &Module<Self>,
        cnv_offset_hi: usize,
        diag: &mut VecZnxDftBackendMut<'_, Self>,
        cross: &mut VecZnxDftBackendMut<'_, Self>,
        left: &CnvPVecLBackendRef<'_, Self>,
        right: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        Self::glwe_mul_columns(module, cnv_offset_hi, diag, cross, left, right, scratch)
    }
}

/// Backend-provided GLWE addition operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEAddImpl: Backend {
    /// Reproduces
    /// [`glwe_add_into_reference`](crate::reference::operations::GLWEAddReference::glwe_add_into_reference) bit for
    /// bit; free in kernels.
    fn glwe_add_into<R, A, B>(module: &Module<Self>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
        B: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_add_assign_reference`](crate::reference::operations::GLWEAddReference::glwe_add_assign_reference) bit
    /// for bit; free in kernels.
    fn glwe_add_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;
}

/// Backend-provided GLWE negation operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWENegateImpl: Backend {
    /// Reproduces [`glwe_negate_reference`](crate::reference::operations::GLWENegateReference::glwe_negate_reference)
    /// bit for bit; free in kernels.
    fn glwe_negate<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_negate_assign_reference`](crate::reference::operations::GLWENegateReference::glwe_negate_assign_reference)
    /// bit for bit; free in kernels.
    fn glwe_negate_assign<R>(module: &Module<Self>, res: &mut R)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE subtraction operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWESubImpl: GLWEAddImpl + GLWENegateImpl {
    /// Reproduces [`glwe_sub_reference`](crate::reference::operations::GLWESubReference::glwe_sub_reference) bit for
    /// bit; free in kernels.
    fn glwe_sub<R, A, B>(module: &Module<Self>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
        B: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_sub_assign_reference`](crate::reference::operations::GLWESubReference::glwe_sub_assign_reference) bit
    /// for bit; free in kernels.
    fn glwe_sub_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    /// Default body: `glwe_sub_negate_assign_derived`,
    /// [`GLWENegateImpl::glwe_negate_assign`](crate::oep::GLWENegateImpl::glwe_negate_assign) of `res`, then
    /// [`GLWEAddImpl::glwe_add_assign`](crate::oep::GLWEAddImpl::glwe_add_assign) of `a`.
    fn glwe_sub_negate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
    {
        crate::oep::derived::operations::glwe_sub_negate_assign_derived::<Self, _, _>(module, res, a)
    }
}

/// Backend-provided GLWE zeroing operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEZeroImpl: Backend {
    /// Reproduces [`glwe_zero_reference`](crate::reference::operations::GLWEZeroReference::glwe_zero_reference) bit
    /// for bit; free in kernels.
    fn glwe_zero<R>(module: &Module<Self>, res: &mut R)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE copy operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWECopyImpl: Backend {
    /// Scratch bound of [`Self::glwe_copy`]; may report less than the reference, never less than the body takes.
    fn glwe_copy_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<Self>, res: &R, a: &A) -> usize;

    /// Reproduces [`glwe_copy_reference`](crate::reference::operations::GLWECopyReference::glwe_copy_reference) bit
    /// for bit, within [`Self::glwe_copy_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_copy<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;
}

/// Backend-provided GGLWE copy operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGLWECopyImpl: Backend {
    /// Scratch bound of [`Self::gglwe_copy`]; may report less than the reference, never less than the body takes.
    fn gglwe_copy_tmp_bytes<R: GGLWEInfos, A: GGLWEInfos>(module: &Module<Self>, res: &R, a: &A) -> usize;

    /// Reproduces [`gglwe_copy_reference`](crate::reference::operations::GGLWECopyReference::gglwe_copy_reference)
    /// bit for bit, within [`Self::gglwe_copy_tmp_bytes`]; free in kernels and scratch use.
    fn gglwe_copy<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos;
}

/// Backend-provided GGSW copy operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGSWCopyImpl: Backend {
    /// Scratch bound of [`Self::ggsw_copy`]; may report less than the reference, never less than the body takes.
    fn ggsw_copy_tmp_bytes<R: GGSWInfos, A: GGSWInfos>(module: &Module<Self>, res: &R, a: &A) -> usize;

    /// Reproduces [`ggsw_copy_reference`](crate::reference::operations::GGSWCopyReference::ggsw_copy_reference) bit
    /// for bit, within [`Self::ggsw_copy_tmp_bytes`]; free in kernels and scratch use.
    fn ggsw_copy<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWInfos;
}

/// Backend-provided GLWE rotation operations, on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWERotateImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::glwe_rotate_assign`]; may report less than the reference, never less than the body
    /// takes.
    fn glwe_rotate_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces [`glwe_rotate_reference`](crate::reference::operations::GLWERotateReference::glwe_rotate_reference)
    /// bit for bit; free in kernels.
    fn glwe_rotate<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_rotate_assign_reference`](crate::reference::operations::GLWERotateReference::glwe_rotate_assign_reference)
    /// bit for bit, within [`Self::glwe_rotate_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_rotate_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GGSW rotation operations, on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGSWRotateImpl: GLWERotateImpl {
    /// Default body: `ggsw_rotate_tmp_bytes_derived`,
    /// [`GLWERotateImpl::glwe_rotate_tmp_bytes`](crate::oep::GLWERotateImpl::glwe_rotate_tmp_bytes).
    fn ggsw_rotate_tmp_bytes(module: &Module<Self>) -> usize {
        crate::oep::derived::operations::ggsw_rotate_tmp_bytes_derived::<Self>(module)
    }

    /// Default body: `ggsw_rotate_derived`, [`GLWERotateImpl::glwe_rotate`](crate::oep::GLWERotateImpl::glwe_rotate)
    /// on each entry.
    fn ggsw_rotate<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWAtViewRef<Self> + GGSWInfos,
    {
        crate::oep::derived::operations::ggsw_rotate_derived::<Self, _, _>(module, k, res, a)
    }

    /// Default body: `ggsw_rotate_assign_derived`,
    /// [`GLWERotateImpl::glwe_rotate_assign`](crate::oep::GLWERotateImpl::glwe_rotate_assign) on each entry of `res`,
    /// within [`Self::ggsw_rotate_tmp_bytes`].
    fn ggsw_rotate_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
    {
        crate::oep::derived::operations::ggsw_rotate_assign_derived::<Self, _>(module, k, res, scratch)
    }
}

/// Backend-provided multiplication by `X^p - 1` operations, on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEMulXpMinusOneImpl: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::glwe_mul_xp_minus_one_assign`]; may report less than the reference, never less than
    /// the body takes.
    fn glwe_mul_xp_minus_one_assign_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    /// Reproduces
    /// [`glwe_mul_xp_minus_one_reference`](crate::reference::operations::GLWEMulXpMinusOneReference::glwe_mul_xp_minus_one_reference)
    /// bit for bit; free in kernels.
    fn glwe_mul_xp_minus_one<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_mul_xp_minus_one_assign_reference`](crate::reference::operations::GLWEMulXpMinusOneReference::glwe_mul_xp_minus_one_assign_reference)
    /// bit for bit, within [`Self::glwe_mul_xp_minus_one_assign_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_mul_xp_minus_one_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE shift operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEShiftImpl: Backend {
    /// Scratch bound of [`Self::glwe_rsh`], [`Self::glwe_lsh_assign`], [`Self::glwe_lsh`], [`Self::glwe_lsh_add`] and
    /// [`Self::glwe_lsh_sub`]; may report less than the reference, never less than the body takes.
    fn glwe_shift_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    /// Reproduces [`glwe_rsh_reference`](crate::reference::operations::GLWEShiftReference::glwe_rsh_reference) bit
    /// for bit, within [`Self::glwe_shift_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_rsh<R>(module: &Module<Self>, k: usize, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;

    /// Reproduces
    /// [`glwe_lsh_assign_reference`](crate::reference::operations::GLWEShiftReference::glwe_lsh_assign_reference) bit
    /// for bit, within [`Self::glwe_shift_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_lsh_assign<R>(module: &Module<Self>, res: &mut R, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;

    /// Reproduces [`glwe_lsh_reference`](crate::reference::operations::GLWEShiftReference::glwe_lsh_reference) bit
    /// for bit, within [`Self::glwe_shift_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_lsh<R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_lsh_add_reference`](crate::reference::operations::GLWEShiftReference::glwe_lsh_add_reference) bit for
    /// bit, within [`Self::glwe_shift_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_lsh_add<R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_lsh_sub_reference`](crate::reference::operations::GLWEShiftReference::glwe_lsh_sub_reference) bit for
    /// bit, within [`Self::glwe_shift_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_lsh_sub<R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;
}

/// Backend-provided GLWE normalization operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWENormalizeImpl: Backend {
    /// Scratch bound of [`Self::glwe_normalize`] and [`Self::glwe_normalize_assign`]; may report less than the
    /// reference, never less than the body takes.
    fn glwe_normalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces
    /// [`glwe_normalize_reference`](crate::reference::operations::GLWENormalizeReference::glwe_normalize_reference)
    /// bit for bit, within [`Self::glwe_normalize_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_normalize<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_normalize_assign_reference`](crate::reference::operations::GLWENormalizeReference::glwe_normalize_assign_reference)
    /// bit for bit, within [`Self::glwe_normalize_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_normalize_assign<R>(module: &Module<Self>, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE trace operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWETraceImpl: crate::oep::AutomorphismImpl + GLWEShiftImpl + GLWECopyImpl + GLWENormalizeImpl {
    /// Default body: `glwe_trace_galois_elements_derived`, the Galois elements of the visited levels of a degree-`n`
    /// operand, in visiting order.
    fn glwe_trace_galois_elements(module: &Module<Self>, n: Degree, levels: Range<usize>, reverse: bool) -> Vec<i64> {
        crate::oep::derived::structure::glwe_trace_galois_elements_derived::<Self, _>(module, n, levels, reverse)
    }

    /// Default body: `glwe_trace_tmp_bytes_derived`, a temporary at the key's radix plus the larger of the copies
    /// into and out of it ([`GLWECopyImpl::glwe_copy_tmp_bytes`](crate::oep::GLWECopyImpl::glwe_copy_tmp_bytes)) and
    /// [`Self::glwe_trace_assign_tmp_bytes`] on the temporary.
    fn glwe_trace_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        crate::oep::derived::structure::glwe_trace_tmp_bytes_derived::<Self, _, _, _, _>(module, res_infos, a_infos, key_infos)
    }

    /// Default body: `glwe_trace_assign_tmp_bytes_derived`, the larger of
    /// [`GLWEShiftImpl::glwe_shift_tmp_bytes`](crate::oep::GLWEShiftImpl::glwe_shift_tmp_bytes) and
    /// [`AutomorphismImpl::glwe_automorphism_tmp_bytes`](crate::oep::AutomorphismImpl::glwe_automorphism_tmp_bytes)
    /// and, on a radix mismatch, a converted copy with
    /// [`GLWENormalizeImpl::glwe_normalize_tmp_bytes`](crate::oep::GLWENormalizeImpl::glwe_normalize_tmp_bytes).
    fn glwe_trace_assign_tmp_bytes<A, K>(module: &Module<Self>, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        crate::oep::derived::structure::glwe_trace_assign_tmp_bytes_derived::<Self, _, _, _>(module, a_infos, key_infos)
    }

    /// Default body: `glwe_trace_derived`, [`GLWECopyImpl::glwe_copy`](crate::oep::GLWECopyImpl::glwe_copy) of `a`
    /// into a temporary at the key's radix, [`Self::glwe_trace_assign`] on it, then
    /// [`GLWECopyImpl::glwe_copy`](crate::oep::GLWECopyImpl::glwe_copy) into `res`; an empty window is
    /// [`GLWECopyImpl::glwe_copy`](crate::oep::GLWECopyImpl::glwe_copy) alone, within [`Self::glwe_trace_tmp_bytes`],
    /// under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
    #[expect(
        clippy::too_many_arguments,
        reason = "the trace's operands, levels, direction and normalization, keys and scratch"
    )]
    fn glwe_trace<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        levels: Range<usize>,
        reverse: bool,
        normalize: bool,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::structure::glwe_trace_derived::<Self, _, _, _, _>(
            module, res, levels, reverse, normalize, a, keys, scratch,
        )
    }

    /// Default body: `glwe_trace_assign_derived`, each visited level adds the Galois conjugate of the running sum
    /// through
    /// [`AutomorphismImpl::glwe_automorphism_add_assign`](crate::oep::AutomorphismImpl::glwe_automorphism_add_assign),
    /// after [`GLWEShiftImpl::glwe_rsh`](crate::oep::GLWEShiftImpl::glwe_rsh) by one bit when normalizing; only when
    /// `res`'s radix differs from the keys' does the work run on a copy at the keys' radix, converted by
    /// [`GLWENormalizeImpl::glwe_normalize`](crate::oep::GLWENormalizeImpl::glwe_normalize), within
    /// [`Self::glwe_trace_assign_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn glwe_trace_assign<R, H>(
        module: &Module<Self>,
        res: &mut R,
        levels: Range<usize>,
        reverse: bool,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::structure::glwe_trace_assign_derived::<Self, _, _, _>(
            module, res, levels, reverse, normalize, keys, scratch,
        )
    }
}

/// Backend-provided GLWE packing operations, on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEPackImpl:
    GLWETraceImpl + crate::oep::AutomorphismImpl + GLWERotateImpl + GLWESubImpl + GLWEAddImpl + GLWEShiftImpl + GLWENormalizeImpl
{
    /// Default body: `glwe_pack_galois_elements_derived`,
    /// [`GLWETraceImpl::glwe_trace_galois_elements`](crate::oep::GLWETraceImpl::glwe_trace_galois_elements) over
    /// every level.
    fn glwe_pack_galois_elements(module: &Module<Self>) -> Vec<i64> {
        crate::oep::derived::structure::glwe_pack_galois_elements_derived::<Self, _>(module)
    }

    /// Default body: `glwe_pack_tmp_bytes_derived`, an input-sized temporary plus the larger of
    /// [`GLWERotateImpl::glwe_rotate_tmp_bytes`](crate::oep::GLWERotateImpl::glwe_rotate_tmp_bytes),
    /// [`GLWEShiftImpl::glwe_shift_tmp_bytes`](crate::oep::GLWEShiftImpl::glwe_shift_tmp_bytes),
    /// [`GLWENormalizeImpl::glwe_normalize_tmp_bytes`](crate::oep::GLWENormalizeImpl::glwe_normalize_tmp_bytes) and
    /// [`AutomorphismImpl::glwe_automorphism_tmp_bytes`](crate::oep::AutomorphismImpl::glwe_automorphism_tmp_bytes),
    /// or [`GLWETraceImpl::glwe_trace_tmp_bytes`](crate::oep::GLWETraceImpl::glwe_trace_tmp_bytes), which carries the
    /// trace's radix-conversion term, if larger.
    fn glwe_pack_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        crate::oep::derived::structure::glwe_pack_tmp_bytes_derived::<Self, _, _, _, _>(module, res, a, key)
    }

    /// Default body: `glwe_pack_derived`, a binary merge tree whose steps use
    /// [`GLWERotateImpl::glwe_rotate_assign`](crate::oep::GLWERotateImpl::glwe_rotate_assign),
    /// [`GLWERotateImpl::glwe_rotate`](crate::oep::GLWERotateImpl::glwe_rotate),
    /// [`GLWESubImpl::glwe_sub`](crate::oep::GLWESubImpl::glwe_sub),
    /// [`GLWESubImpl::glwe_sub_assign`](crate::oep::GLWESubImpl::glwe_sub_assign),
    /// [`GLWEAddImpl::glwe_add_assign`](crate::oep::GLWEAddImpl::glwe_add_assign),
    /// [`GLWEShiftImpl::glwe_rsh`](crate::oep::GLWEShiftImpl::glwe_rsh),
    /// [`GLWENormalizeImpl::glwe_normalize_assign`](crate::oep::GLWENormalizeImpl::glwe_normalize_assign),
    /// [`AutomorphismImpl::glwe_automorphism_assign`](crate::oep::AutomorphismImpl::glwe_automorphism_assign),
    /// [`AutomorphismImpl::glwe_automorphism_add_assign`](crate::oep::AutomorphismImpl::glwe_automorphism_add_assign)
    /// and
    /// [`AutomorphismImpl::glwe_automorphism_sub_negate`](crate::oep::AutomorphismImpl::glwe_automorphism_sub_negate),
    /// closed by [`GLWETraceImpl::glwe_trace`](crate::oep::GLWETraceImpl::glwe_trace), within
    /// [`Self::glwe_pack_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared
    /// operands.
    fn glwe_pack<R, A, H>(
        module: &Module<Self>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendMut<Self> + GLWEInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::structure::glwe_pack_derived::<Self, _, _, _, _>(module, res, a, log_gap_out, keys, scratch)
    }
}

/// Selects the reference implementation of `GLWETensoringImpl` for `$be`.
#[macro_export]
macro_rules! impl_glwe_tensoring_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWETensoringImpl for $be {
            fn glwe_tensor_apply_prepared_right_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>, res: &R, a: &A, a_size: usize, b_size: usize,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_right_tmp_bytes_reference(
                    module, res, a, a_size, b_size,
                )
            }

            fn glwe_tensor_apply_prepared_right<R, A, BP>(
                module: &::poulpy_hal::layouts::Module<$be>, cnv_offset: usize, res: &mut R, a: &A, b: &BP, b_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_right_reference(
                    module, cnv_offset, res, a, b, b_size, scratch,
                )
            }

            fn glwe_tensor_apply_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                b: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn glwe_tensor_square_apply_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_square_apply_tmp_bytes_reference(
                    module, res, a,
                )
            }

            fn glwe_tensor_apply<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_reference(
                    module, cnv_offset, res, a, b, scratch,
                )
            }

            fn glwe_tensor_square_apply<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_square_apply_reference(
                    module, cnv_offset, res, a, scratch,
                )
            }

            fn glwe_tensor_relinearize<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_relinearize_reference(
                    module, res, a, tsk, scratch,
                )
            }

            fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                tsk: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_relinearize_tmp_bytes_reference(
                    module, res, a, tsk,
                )
            }

            fn glwe_mul_relinearize_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: $crate::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_relinearize<R, A, B, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: &A,
                b: &B,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_relinearize_assign<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_square_relinearize_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize_assign<R, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_square_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize<R, A, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: &A,
                b: &BP,
                b_size: usize,
                b_k: $crate::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, b_size, b_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_assign<R, AP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: &AP,
                a_size: usize,
                a_k: $crate::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, a_size, a_k, tsk, scratch,
                )
            }

            fn glwe_mul_columns(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset_hi: usize,
                diag: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                cross: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::CnvPVecLBackendRef<'_, $be>,
                b: &::poulpy_hal::layouts::CnvPVecRBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_columns_reference(
                    module, cnv_offset_hi, diag, cross, a, b, scratch,
                )
            }

            fn glwe_square_prepare(
                module: &::poulpy_hal::layouts::Module<$be>,
                left: &mut ::poulpy_hal::layouts::CnvPVecLBackendMut<'_, $be>,
                right: &mut ::poulpy_hal::layouts::CnvPVecRBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_square_prepare_reference(
                    module, left, right, a, scratch,
                )
            }

            fn glwe_mul_relinearize_sum_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: $crate::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            fn glwe_mul_relinearize_sum<R, A, B, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: &[&A],
                b: &[&B],
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                B: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_prepared_relinearize_sum_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: $crate::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                B: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_sum<'a, R, AP, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: impl IntoIterator<Item = &'a AP, IntoIter: ExactSizeIterator + Clone>,
                a_size: usize,
                a_k: $crate::layouts::TorusPrecision,
                b: impl IntoIterator<Item = &'a BP, IntoIter: ExactSizeIterator + Clone>,
                b_size: usize,
                b_k: $crate::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be> + 'a,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be> + 'a,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, a_size, a_k, b, b_size, b_k, tsk, scratch,
                )
            }

            fn glwe_tensor_apply_prepared_sum_dft_tmp_bytes<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_sum_dft_tmp_bytes_reference(
                    module, res, a_size, b_size,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_tensor_apply_prepared_sum_dft<'a, AP>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset_hi: usize,
                diag: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                cross: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                pairs: impl IntoIterator<Item = (&'a AP, $crate::layouts::DiagonalOperand<'a, $be>)>,
                b_size: usize,
                b_k: $crate::layouts::TorusPrecision,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                $be: 'a,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be> + 'a,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_sum_dft_reference(
                    module, cnv_offset_hi, diag, cross, pairs, b_size, b_k, scratch,
                )
            }

            fn glwe_tensor_apply_prepared_tmp_bytes<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                base2k: $crate::layouts::Base2K,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_tmp_bytes_reference(
                    module, res, a_size, b_size, base2k,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_tensor_apply_prepared<R, AP, BP>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &AP,
                a_size: usize,
                b: &BP,
                b_size: usize,
                base2k: $crate::layouts::Base2K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_reference(
                    module, cnv_offset, res, a, a_size, b, b_size, base2k, scratch,
                )
            }
        }
    };
}

/// Selects the core-derived trace defaults.
#[macro_export]
macro_rules! impl_glwe_trace_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWETraceImpl for $be {}
    };
}

/// Selects the core-derived packing defaults.
#[macro_export]
macro_rules! impl_glwe_pack_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEPackImpl for $be {}
    };
}

/// Selects row-wise rotation through the backend's GLWE rotation methods.
#[macro_export]
macro_rules! impl_ggsw_rotate_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGSWRotateImpl for $be {}
    };
}

/// Forwards every elementary arithmetic family except tensoring.
///
/// Use the per-family macros instead when overriding one of these operations.
#[macro_export]
macro_rules! impl_operations_reference_full {
    ($be:ty) => {
        $crate::impl_glwe_mul_const_reference_full!($be);
        $crate::impl_glwe_mul_plain_reference_full!($be);
        $crate::impl_glwe_mul_prepared_plain_sum_reference_full!($be);
        $crate::impl_glwe_add_reference_full!($be);
        $crate::impl_glwe_sub_reference_full!($be);
        $crate::impl_glwe_negate_reference_full!($be);
        $crate::impl_glwe_zero_reference_full!($be);
        $crate::impl_glwe_copy_reference_full!($be);
        $crate::impl_gglwe_copy_reference_full!($be);
        $crate::impl_ggsw_copy_reference_full!($be);
        $crate::impl_glwe_shift_reference_full!($be);
        $crate::impl_glwe_normalize_reference_full!($be);
    };
}
