use crate::layouts::IntPolyInfos;
use std::{collections::HashMap, ops::Range};

use poulpy_hal::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLToBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef,
    CnvPVecRToBackendRef, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut,
};

use crate::layouts::{
    Degree, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
    GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey,
};

/// Backend-provided GLWE constant-multiplication operations.
///
/// # Safety
/// Implementations must respect the provided layout metadata, conversion offset, and scratch-space
/// contracts, and must not read or write outside the specified backend-owned buffers.
pub unsafe trait GLWEMulConstImpl: Backend {
    fn glwe_mul_const_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

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

    /// Scratch bytes of [`Self::glwe_const_combination`] for an accumulator of `acc_size` limbs.
    fn glwe_const_combination_tmp_bytes(module: &Module<Self>, acc_size: usize, a_size: usize, b_size: usize) -> usize
    where
        Module<Self>: poulpy_hal::api::Convolution<Self> + poulpy_hal::api::VecZnxBigNormalizeTmpBytes,
    {
        use poulpy_hal::api::{Convolution, VecZnxBigNormalizeTmpBytes};
        Self::bytes_of_vec_znx_big(module.n(), 1, acc_size)
            + module
                .cnv_by_const_apply_tmp_bytes(0, acc_size, a_size, b_size)
                .max(module.cnv_by_const_apply_add_tmp_bytes(0, acc_size, a_size, b_size))
                .max(module.vec_znx_big_normalize_tmp_bytes())
    }

    /// A combination of vectors by constants, normalized once:
    ///
    /// `res[col] = normalize(sum_t terms[t].a[col] * terms[t].b[b_col][b_coeff])` for every column of `res`,
    ///
    /// each product being `cnv_by_const_apply` at its `cnv_offset`, the sum being accumulated over
    /// `acc_size` limbs and normalized to `res_k` bits in base `2^base2k`, which is also the base of
    /// the terms. `terms` is not empty.
    ///
    /// The default accumulates with `cnv_by_const_apply` and `cnv_by_const_apply_add`, then calls
    /// `vec_znx_big_normalize`. Overrides must reproduce it bit for bit.
    fn glwe_const_combination(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        base2k: usize,
        res_k: usize,
        acc_size: usize,
        terms: &[ConstTerm<'_, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: poulpy_hal::api::Convolution<Self> + poulpy_hal::api::VecZnxBigNormalize<Self>,
    {
        crate::reference::operations::glwe_const_combination_reference::<Self>(
            module, res, base2k, res_k, acc_size, terms, scratch,
        )
    }
}

/// One term of [`GLWEMulConstImpl::glwe_const_combination`]: every column of `a` times
/// coefficient `b_coeff` of column `b_col` of `b`, at convolution offset `cnv_offset`.
pub struct ConstTerm<'a, BE: Backend + 'a> {
    pub a: VecZnxBackendRef<'a, BE>,
    pub b: VecZnxBackendRef<'a, BE>,
    pub b_col: usize,
    pub b_coeff: usize,
    pub cnv_offset: usize,
}

/// Backend-provided GLWE-by-plaintext multiplication operations.
///
/// # Safety
/// Implementations must interpret the plaintext and ciphertext layouts consistently with the
/// backend and preserve all aliasing and buffer-bound invariants.
pub unsafe trait GLWEMulPlainImpl: Backend {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

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
/// Implementations must interpret the prepared operands consistently with the backend's convolution layout and
/// preserve all aliasing and buffer-bound invariants.
pub unsafe trait GLWEMulPreparedPlainSumImpl: Backend {
    fn glwe_mul_prepared_plain_sum_tmp_bytes<R>(module: &Module<Self>, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos;

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
}

/// Backend-provided GLWE tensoring and relinearization operations.
///
/// # Safety
/// Implementations must preserve tensor layout semantics, respect the temporary-size contracts,
/// and only touch backend-owned storage regions that belong to the supplied operands.
pub unsafe trait GLWETensoringImpl: Backend {
    /// Scratch required by the prepared-right tensor implementation.
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

    /// Prepared-right tensor product with the same rounding and mutation contract as the reference.
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

    fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<Self>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

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

    fn glwe_tensor_square_apply<R, A>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_tensor_relinearize<R, A, H>(module: &Module<Self>, res: &mut R, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
        H: GetTensorKey<Self>;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<Self>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    /// Scratch bytes of [`Self::glwe_mul_relinearize`].
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

    /// The one-pass products of [`GLWETensoring`](crate::api::GLWETensoring), one hook per method of the same name.
    /// Overrides must reproduce
    /// [`GLWETensoringReference`](crate::reference::operations::GLWETensoringReference) bit for bit.
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

    /// See [`Self::glwe_mul_relinearize`].
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

    /// See [`Self::glwe_mul_relinearize`].
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

    /// See [`Self::glwe_mul_relinearize`].
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

    /// See [`Self::glwe_mul_relinearize`].
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

    /// See [`Self::glwe_mul_relinearize`].
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

    /// The three product columns of the one-pass products, from prepared rank-1 operands: see
    /// [`GLWETensoringReference::glwe_mul_columns_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_columns_reference).
    ///
    /// It receives the scratch left once the operands and the columns are allocated: a kernel that needs more
    /// than the reference convolutions adds it to [`Self::glwe_mul_relinearize_tmp_bytes`].
    /// Overrides must reproduce the reference bit for bit.
    fn glwe_mul_columns(
        module: &Module<Self>,
        cnv_offset_hi: usize,
        diag: &mut VecZnxDftBackendMut<'_, Self>,
        cross: &mut VecZnxDftBackendMut<'_, Self>,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// Scratch bytes of [`Self::glwe_mul_relinearize_sum`], whatever the number of pairs.
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

    /// The summed one-pass product of [`GLWEMulRelinearizeSum`](crate::api::GLWEMulRelinearizeSum). Overrides must
    /// reproduce
    /// [`GLWETensoringReference::glwe_mul_relinearize_sum_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_relinearize_sum_reference)
    /// bit for bit.
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

    /// Scratch bytes of [`Self::glwe_mul_prepared_relinearize_sum`], whatever the number of pairs.
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

    /// The summed one-pass product over prepared pairs of
    /// [`GLWEPreparedTensoring`](crate::api::GLWEPreparedTensoring). Overrides must reproduce
    /// [`GLWETensoringReference::glwe_mul_prepared_relinearize_sum_reference`](crate::reference::operations::GLWETensoringReference::glwe_mul_prepared_relinearize_sum_reference)
    /// bit for bit, whatever the hint each operand was prepared under, and run one pair within
    /// [`Self::glwe_mul_relinearize_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_relinearize_sum<R, AP, BP, H>(
        module: &Module<Self>,
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
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        AP: CnvPVecLToBackendRef<Self>,
        BP: CnvPVecRToBackendRef<Self>,
        H: GetTensorKey<Self>;

    /// Scratch bytes of [`Self::glwe_tensor_apply_prepared`].
    fn glwe_tensor_apply_prepared_tmp_bytes<R>(
        module: &Module<Self>,
        res: &R,
        a_size: usize,
        b_size: usize,
        base2k: crate::layouts::Base2K,
    ) -> usize
    where
        R: GLWEInfos;

    /// The tensor product of prepared operands of [`GLWEPreparedTensoring`](crate::api::GLWEPreparedTensoring).
    /// Overrides must reproduce
    /// [`GLWETensoringReference::glwe_tensor_apply_prepared_reference`](crate::reference::operations::GLWETensoringReference::glwe_tensor_apply_prepared_reference)
    /// bit for bit, whatever the hint each operand was prepared under.
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
    /// The default prepares both forms with `cnv_prepare_self`. A backend whose
    /// [`Self::glwe_square_columns`] reads the left form alone may leave `right` unwritten.
    fn glwe_square_prepare(
        module: &Module<Self>,
        left: &mut CnvPVecLBackendMut<'_, Self>,
        right: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: poulpy_hal::api::Convolution<Self>,
    {
        use poulpy_hal::api::Convolution;
        module.cnv_prepare_self(left, right, a, scratch)
    }

    /// [`Self::glwe_mul_columns`] of a square, on the two forms [`Self::glwe_square_prepare`]
    /// prepared from one operand. The default is [`Self::glwe_mul_columns`].
    /// Overrides must reproduce it bit for bit.
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
/// Implementations must preserve GLWE layout invariants and respect all backend buffer bounds.
pub unsafe trait GLWEAddImpl: Backend {
    fn glwe_add_into<R, A, B>(module: &Module<Self>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
        B: GLWEToBackendRef<Self>;

    fn glwe_add_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;
}

/// Backend-provided GLWE negation operations.
///
/// # Safety
/// Implementations must preserve GLWE layout invariants and respect all backend buffer bounds.
pub unsafe trait GLWENegateImpl: Backend {
    fn glwe_negate<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    fn glwe_negate_assign<R>(module: &Module<Self>, res: &mut R)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE subtraction operations.
///
/// # Safety
/// Implementations must preserve GLWE layout invariants and respect all backend buffer bounds.
pub unsafe trait GLWESubImpl: GLWEAddImpl + GLWENegateImpl {
    fn glwe_sub<R, A, B>(module: &Module<Self>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
        B: GLWEToBackendRef<Self>;

    fn glwe_sub_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

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
/// Implementations must zero every polynomial column in the GLWE without violating layout or
/// backend buffer invariants.
pub unsafe trait GLWEZeroImpl: Backend {
    fn glwe_zero<R>(module: &Module<Self>, res: &mut R)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE copy operations.
///
/// # Safety
/// Implementations must honor [`crate::GLWECopy`]'s rounding and layout contract,
/// stay within the advertised scratch size, and respect all backend buffer bounds.
pub unsafe trait GLWECopyImpl: Backend {
    fn glwe_copy_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<Self>, res: &R, a: &A) -> usize;

    fn glwe_copy<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;
}

/// Backend-provided GGLWE copy operations.
///
/// # Safety
/// Implementations must honor [`crate::GGLWECopy`]'s rounding and layout contract,
/// stay within the advertised scratch size, and respect all backend buffer bounds.
pub unsafe trait GGLWECopyImpl: Backend {
    fn gglwe_copy_tmp_bytes<R: GGLWEInfos, A: GGLWEInfos>(module: &Module<Self>, res: &R, a: &A) -> usize;

    fn gglwe_copy<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos;
}

/// Backend-provided GGSW copy operations.
///
/// # Safety
/// Implementations must honor [`crate::GGSWCopy`]'s rounding and layout contract,
/// stay within the advertised scratch size, and respect all backend buffer bounds.
pub unsafe trait GGSWCopyImpl: Backend {
    fn ggsw_copy_tmp_bytes<R: GGSWInfos, A: GGSWInfos>(module: &Module<Self>, res: &R, a: &A) -> usize;

    fn ggsw_copy<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWInfos;
}

/// Backend-provided GLWE rotation operations.
///
/// # Safety
/// Implementations must perform rotations according to the polynomial layout without violating
/// scratch-space, aliasing, or buffer-bound guarantees.
pub unsafe trait GLWERotateImpl: Backend {
    fn glwe_rotate_tmp_bytes(module: &Module<Self>) -> usize;

    fn glwe_rotate<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    fn glwe_rotate_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GGSW rotation operations.
///
/// # Safety
/// Implementations must preserve the GGSW structure for the backend and may only use scratch space
/// and in-place mutation in ways compatible with the advertised contracts.
pub unsafe trait GGSWRotateImpl: GLWERotateImpl {
    fn ggsw_rotate_tmp_bytes(module: &Module<Self>) -> usize {
        crate::oep::derived::operations::ggsw_rotate_tmp_bytes_derived::<Self>(module)
    }

    fn ggsw_rotate<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWAtViewRef<Self> + GGSWInfos,
    {
        crate::oep::derived::operations::ggsw_rotate_derived::<Self, _, _>(module, k, res, a)
    }

    fn ggsw_rotate_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
    {
        crate::oep::derived::operations::ggsw_rotate_assign_derived::<Self, _>(module, k, res, scratch)
    }
}

/// Backend-provided multiplication by `X^p - 1` operations.
///
/// # Safety
/// Implementations must apply the requested ring operation without violating the layout or memory
/// invariants of the supplied ciphertext buffers.
pub unsafe trait GLWEMulXpMinusOneImpl: Backend {
    fn glwe_mul_xp_minus_one_assign_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    fn glwe_mul_xp_minus_one<R, A>(module: &Module<Self>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    fn glwe_mul_xp_minus_one_assign<R>(module: &Module<Self>, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE shift operations.
///
/// # Safety
/// Implementations must respect the polynomial/ciphertext layout and scratch requirements, and may
/// not read or write beyond the backend-owned regions described by the inputs.
pub unsafe trait GLWEShiftImpl: Backend {
    fn glwe_shift_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    fn glwe_rsh<R>(module: &Module<Self>, k: usize, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;

    fn glwe_lsh_assign<R>(module: &Module<Self>, res: &mut R, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;

    fn glwe_lsh<R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    fn glwe_lsh_add<R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    fn glwe_lsh_sub<R, A>(module: &Module<Self>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;
}

/// Backend-provided GLWE normalization operations.
///
/// # Safety
/// Implementations must return views that remain valid for the advertised lifetime, preserve
/// normalization semantics, and avoid aliasing or out-of-bounds access across temporary buffers.
pub unsafe trait GLWENormalizeImpl: Backend {
    fn glwe_normalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn glwe_normalize<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>;

    fn glwe_normalize_assign<R>(module: &Module<Self>, res: &mut R, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self>;
}

/// Backend-provided GLWE trace operations.
///
/// # Safety
/// Implementations must apply the requested automorphism sequence faithfully, interpret prepared
/// keys correctly, and keep all accesses within the described ciphertext and scratch regions.
pub unsafe trait GLWETraceImpl: crate::oep::AutomorphismImpl + GLWEShiftImpl + GLWECopyImpl + GLWENormalizeImpl {
    fn glwe_trace_galois_elements(module: &Module<Self>, n: Degree, levels: Range<usize>, reverse: bool) -> Vec<i64> {
        crate::oep::derived::structure::glwe_trace_galois_elements_derived::<Self, _>(module, n, levels, reverse)
    }

    fn glwe_trace_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        crate::oep::derived::structure::glwe_trace_tmp_bytes_derived::<Self, _, _, _, _>(module, res_infos, a_infos, key_infos)
    }

    fn glwe_trace_assign_tmp_bytes<A, K>(module: &Module<Self>, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        crate::oep::derived::structure::glwe_trace_assign_tmp_bytes_derived::<Self, _, _, _>(module, a_infos, key_infos)
    }

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

/// Backend-provided GLWE packing operations.
///
/// # Safety
/// Implementations must maintain ciphertext correctness while combining inputs, and must respect
/// all backend buffer, aliasing, and scratch-space invariants expected by the higher layers.
pub unsafe trait GLWEPackImpl: GLWETraceImpl + GLWERotateImpl + GLWESubImpl + GLWEAddImpl {
    fn glwe_pack_galois_elements(module: &Module<Self>) -> Vec<i64> {
        crate::oep::derived::structure::glwe_pack_galois_elements_derived::<Self, _>(module)
    }

    fn glwe_pack_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        crate::oep::derived::structure::glwe_pack_tmp_bytes_derived::<Self, _, _, _, _>(module, res, a, key)
    }

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

/// Implements the canonical Core tensoring operation for a backend.
///
/// Explicit opt-in lets optimized backends dispatch to specialized kernels.
#[macro_export]
macro_rules! impl_glwe_tensoring_reference {
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
            fn glwe_mul_prepared_relinearize_sum<R, AP, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: $crate::layouts::TorusPrecision,
                a: &[&AP],
                b: &[&BP],
                a_size: usize,
                a_k: $crate::layouts::TorusPrecision,
                b_size: usize,
                b_k: $crate::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, b, a_size, a_k, b_size, b_k, tsk, scratch,
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
macro_rules! impl_glwe_packing_derived_full {
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
