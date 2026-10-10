use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::layouts::{BlasCiphertext, BlasContext, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand};

/// Matrix products of every matrix of the batch, across the blocks of a grid. A product records the larger batch of its
/// operands ([`BlasGridInfos::mat_slots`]); a preparation and the product by public matrices record their input's.
pub trait BlasMatMulOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_matmul`] needs on these layouts.
    fn blas_matmul_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res` receives the product of every matrix of `a` with the matching matrix of `b`.
    ///
    /// Batches of different sizes broadcast: with `m_a` and `m_b` matrices in `a` and `b`, matrix slot `s` of `res` holds
    /// `A_(s mod m_a) B_(s mod m_b)`, so `res` holds the larger batch, at the smaller `log_sparsity` of the two.
    fn blas_matmul<DR, DA, DB, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasCiphertext<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetTensorKey<BE> + GetAutomorphismKey<BE>;

    /// Scratch bytes [`Self::blas_matmul_prepare_left`] needs into `res` from `a` with keys of layout `key`.
    fn blas_matmul_prepare_left_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives sigma of every block of `a`, the left operand of [`Self::blas_matmul_prepared`].
    fn blas_matmul_prepare_left<DR, DA, K>(
        &self,
        res: &mut BlasMatMulLeft<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>;

    /// Scratch bytes [`Self::blas_matmul_prepare_right`] needs into `res` from `b` with keys of layout `key`.
    fn blas_matmul_prepare_right_tmp_bytes<R, B, K>(&self, res: &R, b: &B, key: &K) -> usize
    where
        R: BlasGridInfos,
        B: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives tau of every block of `b`, the right operand of [`Self::blas_matmul_prepared`].
    fn blas_matmul_prepare_right<DR, DB, K>(
        &self,
        res: &mut BlasMatMulRight<DR, BE::ZnxWord, BE::Ring>,
        b: &BlasCiphertext<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>;

    /// Scratch bytes [`Self::blas_matmul_prepared`] needs; `a` and `b` are the grids of the prepared operands
    /// (`left.as_grid()`, `right.as_grid()`), not the original inputs.
    fn blas_matmul_prepared_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res` receives the product of the grids `a` and `b` were prepared from; batches of different sizes broadcast as
    /// in [`Self::blas_matmul`].
    fn blas_matmul_prepared<DR, DA, DB, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasMatMulLeft<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasMatMulRight<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetTensorKey<BE> + GetAutomorphismKey<BE>;

    /// Scratch bytes [`Self::blas_matmul_pt`] needs into `res` from `a` with keys of layout `key`.
    fn blas_matmul_pt_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives the product of every matrix of `a` with the public matrix `b` holds for its matrix slot.
    fn blas_matmul_pt<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasPtOperand<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>;
}
