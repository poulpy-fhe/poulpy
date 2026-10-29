use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{BlasMatVecMaps, BlasVectorInfos, BlasVectorMut, BlasVectorRef};

/// Products of a vector ([`crate::layouts::BlasVector`]) by a matrix.
///
/// In the `# Metadata` sections, `x.k` is the width of the ciphertexts of `x`, for `res` the width they have before the
/// call, `S` the slots of a ciphertext, and `p_W` the scale a public operand was encoded at.
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
    /// shares one baby-step cache per input ciphertext. `w` is resident ([`BlasMatVecOperand`]) or streamed
    /// ([`BlasMatVecOperandUnprepared`], whose plaintext diagonals are prepared at each use, for less memory), with the
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
    /// [`BlasMatVecOperand`]: crate::layouts::BlasMatVecOperand
    /// [`BlasMatVecOperandUnprepared`]: crate::layouts::BlasMatVecOperandUnprepared
    fn blas_matvec_pt<R, W, A, K>(&self, res: &mut R, w: &W, a: &A, keys: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;
}
