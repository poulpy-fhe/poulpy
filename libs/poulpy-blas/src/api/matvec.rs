use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_ckks::layouts::CKKSPreparedRight;
use poulpy_core::layouts::GLWEInfos;

use crate::layouts::{
    BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos, BlasDiagonalCiphertextMap, BlasMatVecMaps, BlasVectorInfos,
    BlasVectorMut, BlasVectorRef,
};

/// Products of a vector ([`crate::layouts::BlasVector`]) by a matrix, each named after the matrix's encoding: `_pt` a
/// public operand, `_diagonal` a [`BlasDiagonalCiphertext`].
///
/// In the `# Metadata` sections, `x.k` is the width of the ciphertexts of `x`, for `res` the width they have before the
/// call, `S` the slots of a ciphertext, and `p_W` the scale a public operand was encoded at. For a diagonal ciphertext `m`,
/// `m.k`, `m.log_delta`, `m.log_sparsity` and `m.slots` are those of its diagonals' layout
/// ([`BlasDiagonalCiphertextLayout::ct`](crate::layouts::BlasDiagonalCiphertextLayout)), which appends keep.
pub trait BlasMatVecOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_matvec_pt`] needs into `res` from `a` through `w` with rotation keys of layout
    /// `keys`, resident or streamed by `w`'s form.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matvec_pt`](crate::layouts::validate_blas_matvec_pt) rejects: ciphertexts
    /// of a degree other than the module's.
    fn blas_matvec_pt_tmp_bytes<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos;

    /// `res = W a` for the public matrix `W` of `w` ([`BlasEncodingOps::blas_matvec_operand_encode`]): entry `row` of
    /// `res` is the sum of `W[row][col] a[col]` over the columns, written at every repetition of the operand's output
    /// period, and the other slots of `res` are zero.
    ///
    /// Each output ciphertext is one linear transformation over the `K` input ciphertexts, which share their giant
    /// steps: it costs the giant-step key switches of one of `w`'s maps, whatever `K`, and every output ciphertext
    /// shares one baby-step cache per input ciphertext. `w` is resident ([`BlasMatVecOperandPrepared`]) or streamed
    /// ([`BlasMatVecOperand`], whose plaintext diagonals are prepared at each use, for less memory), with the
    /// same bits. Each giant step adds the products of every term with each of its diagonals in the transform domain
    /// before its rotation, so choose `base2k` for the `K g min(L, P)` limb products it adds in an output limb: `g` the
    /// most diagonals a giant step of `w`'s maps groups, `L` the limbs of `a.k` and `P` those of a diagonal;
    /// [`blas_matvec_pt_max_base2k`](crate::layouts::blas_matvec_pt_max_base2k) returns the largest radix
    /// `Module::max_base2k` allows for that count. Its keys are
    /// [`matvec_pt_galois_elements`](crate::layouts::matvec_pt_galois_elements).
    ///
    /// # Metadata
    ///
    /// Ciphertext `o` of `res` is poulpy-ckks's `ckks_eval_linear_transformation_sum_into` of the `K` terms
    /// `(a_i, w_oi)`, labelled with the period the operand writes:
    ///
    /// ```text
    /// k            = min(res.k, a.k - p_W)
    /// log_delta    = a.log_delta
    /// log_sparsity = log2(S / output_period)
    /// slots        = a.slots.join(Real)
    /// canonical    = true
    /// noise        = None
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_matvec_pt`](crate::layouts::validate_blas_matvec_pt) before any
    ///   write, so on its errors `res` and the operands are untouched: `InvalidOperand` for a vector, shape, layout,
    ///   operand or period it rejects; then, key by key, `MissingAutomorphismKey` for a key `keys` lacks, the baby steps
    ///   at `a.k` and the giant steps at the width of `res`'s result, or `InvalidOperand` for a key of another rank than
    ///   the ciphertexts'.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too:
    ///   `MultiplicationPrecisionUnderflow` when `a.k - a.log_delta` is below `p_W`, and
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below `log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    ///
    /// [`BlasEncodingOps::blas_matvec_operand_encode`]: crate::api::BlasEncodingOps::blas_matvec_operand_encode
    /// [`BlasMatVecOperandPrepared`]: crate::layouts::BlasMatVecOperandPrepared
    /// [`BlasMatVecOperand`]: crate::layouts::BlasMatVecOperand
    fn blas_matvec_pt<R, W, A, K>(&self, res: &mut R, w: &W, a: &A, keys: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matvec_diagonal`] needs into `res` from `a` through a matrix of `m`, with a tensor key of
    /// layout `tsk` and rotation keys of layout `keys`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matvec_diagonal`](crate::layouts::validate_blas_matvec_diagonal) rejects.
    fn blas_matvec_diagonal_tmp_bytes<R, M, A, T, K>(&self, res: &R, m: &M, a: &A, tsk: &T, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res = M a` for the diagonal ciphertext `m` and the whole ciphertext `a`: entry `i` of `res` is the sum, over the
    /// diagonals `t` `m` holds, of `D_t[i] a[i + t]`. It is poulpy-ckks's linear transformation of ciphertext diagonals
    /// (`ckks_eval_linear_transformation_into`): the baby rotations of `a` are made once, each giant step's products
    /// are summed and relinearized once with the tensor key of `keys`, rotated, and summed, with one final normalization. Each giant step
    /// adds the tensor products of its pairs in the transform domain before its relinearization, so choose `base2k` for
    /// the `2 g min(L, P)` limb products it adds in an output limb of the middle term: `g` the most diagonals a giant
    /// step of `m` groups, `L` the limbs of `a.k` and `P` those of a diagonal;
    /// [`blas_matvec_diagonal_max_base2k`](crate::layouts::blas_matvec_diagonal_max_base2k) returns the largest radix `Module::max_base2k`
    /// allows for that count. Its keys are [`matvec_diagonal_galois_elements`](crate::layouts::matvec_diagonal_galois_elements) and the
    /// tensor key. `a` and `res` are vectors of one ciphertext whose every slot is an entry. `m` is resident
    /// ([`BlasDiagonalCiphertextPrepared`](crate::layouts::BlasDiagonalCiphertextPrepared)) or streamed ([`BlasDiagonalCiphertext`], each diagonal prepared at its use,
    /// for less memory and no preparation after appends), with the same bits.
    ///
    /// # Metadata
    ///
    /// The ciphertext of `res` is poulpy-ckks's `ckks_eval_linear_transformation_into` of `m`'s diagonals from `a`, at
    /// poulpy-ckks's ct x ct rule between `a` and a diagonal:
    ///
    /// ```text
    /// log_delta    = min(a.log_delta, m.log_delta)
    /// k            = min(res.k, min(a.k - a.log_delta, m.k - m.log_delta) - max(a.log_delta, m.log_delta) + log_delta)
    /// log_sparsity = min(a.log_sparsity, m.log_sparsity)
    /// slots        = a.slots.join(m.slots)
    /// canonical    = true
    /// noise        = None
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_matvec_diagonal`](crate::layouts::validate_blas_matvec_diagonal) before any write, so on
    ///   its errors `res` and the operands are untouched: `InvalidOperand` for a vector, layout, radix above 62 or matrix
    ///   it rejects; then `MissingAutomorphismKey` for a rotation key `keys` lacks, the baby steps at `a.k` and the
    ///   giant steps at the result's width, `MissingRelinearizationKey` for the tensor key `keys` lacks at
    ///   `max(a.k, m.k)`, or `InvalidOperand` for a key of another rank.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too:
    ///   `MultiplicationPrecisionUnderflow` when the smaller budget of `a` and a diagonal is below the larger scale, and
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below the result's `log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_matvec_diagonal<R, M, A, K>(
        &self,
        res: &mut R,
        m: &M,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasVectorMut<BE>,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_append_row_diagonal`] needs on a matrix of `m` from `v` by a mask of layout `mask`, with
    /// rotation keys of layout `keys`, which it reads only when the append rotates `v`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_append_row_diagonal`](crate::layouts::validate_blas_append_row_diagonal) rejects.
    fn blas_append_row_diagonal_tmp_bytes<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos;

    /// Adds `v` to row `row` of `m`, on the slots `mask` marks: `M[q][q + t] += v[q + t]` for every slot `q = row + s`
    /// with `s` in the mask and every diagonal `t` `m` holds. With `D_t` the diagonals and entry `(b, g)` of the storage
    /// holding `rot(D_(b + g), -g)`, that is `m_(row + g) rot(v, b)` added to entry `(b, g)`: one prepared mask `m_0`,
    /// read through the Galois element of `-(row + g)`, times the baby rotation `rot(v, b)`, by poulpy-ckks's
    /// `ckks_dot_product_prepared_pt_add`. A mask of several slots writes several rows at once. A resident product reads
    /// the diagonals once prepared again
    /// ([`BlasDiagonalCiphertextPreparedFactory::blas_diagonal_ciphertext_prepare`](crate::api::BlasDiagonalCiphertextPreparedFactory::blas_diagonal_ciphertext_prepare)),
    /// a streamed one reads them as they are. Its keys are [`append_row_diagonal_galois_elements`](crate::layouts::append_row_diagonal_galois_elements): the rotations of
    /// `v` by the baby steps `m`'s diagonals read.
    ///
    /// # Metadata
    ///
    /// Every diagonal of `m` holds the frame of the products, `m`'s layout: `v` through the mask's scale, `p_m`. The
    /// append keeps it, and the validator rejects a `v` or `mask` that would change the slot kind or the sparsity:
    ///
    /// ```text
    /// k            = m.k              (= min(m.k, v.k - p_m), checked by poulpy-ckks)
    /// log_delta    = m.log_delta      (= v.log_delta)
    /// log_sparsity = m.log_sparsity   (<= min(v.log_sparsity, log2(v.n / mask.n)))
    /// slots        = m.slots          (= m.slots.join(v.slots).join(mask.slots))
    /// canonical    = true
    /// noise        = None
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_append_row_diagonal`](crate::layouts::validate_blas_append_row_diagonal) before any
    ///   write, so on its errors `m` and the operands are untouched: `InvalidOperand` for a vector, layout, matrix or
    ///   mask it rejects, or an append changing the diagonals' slot kind or sparsity; then `MissingAutomorphismKey` at
    ///   `v.k`, or `InvalidOperand` for a key of another rank.
    /// - poulpy-ckks's errors of the first accumulate come before any write to `m`: `InvalidOperand` for a `v` whose
    ///   product by the mask is not at the diagonals' frame.
    /// - Any other `Err` is a backend fault, and `m` is then unspecified.
    fn blas_append_row_diagonal<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        row: i64,
        v: &V,
        mask: &CKKSPreparedRight<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_append_col_diagonal`] needs, as [`Self::blas_append_row_diagonal_tmp_bytes`].
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_append_col_diagonal`](crate::layouts::validate_blas_append_col_diagonal) rejects.
    fn blas_append_col_diagonal_tmp_bytes<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos;

    /// Adds `v` to column `col` of `m`, on the slots `mask` marks: `M[q - t][q] += v[q - t]` for every slot
    /// `q = col + s` with `s` in the mask and every diagonal `t`. Entry `(b, g)` receives `m_(col - b) rot(v, -g)`: the
    /// mask read through the Galois element of `b - col` times the giant rotation `rot(v, -g)`. Its keys are [`append_col_diagonal_galois_elements`](crate::layouts::append_col_diagonal_galois_elements): the
    /// rotations of `v` by the opposites of `m`'s giant steps. Metadata as [`Self::blas_append_row_diagonal`]; errors too, its
    /// validator [`validate_blas_append_col_diagonal`](crate::layouts::validate_blas_append_col_diagonal).
    fn blas_append_col_diagonal<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        col: i64,
        v: &V,
        mask: &CKKSPreparedRight<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}
