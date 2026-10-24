use poulpy_ckks::{CKKSResult, api::CKKSEncodingScalar, layouts::CKKSModuleAlloc};
use poulpy_core::layouts::{DiagonalArithmetic, GetDegree};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    BlasContext, BlasContextInfos, BlasContextUnprepared, BlasPtOperand, BlasPtOperandInfos, BlasPtOperandUnprepared, BlasPtSide,
    context, pt_operand,
};

/// Encoding of public operands, and of contexts, into their allocations, at poulpy-ckks's encoding scalar `F`.
///
/// The scalar is a trait parameter, as for poulpy-ckks's encoder: the delegating impl on `Module<BE>` requires
/// [`CKKSEncodingImpl<F>`](poulpy_ckks::oep::CKKSEncodingImpl), and a call that cannot infer `F` from its arguments names
/// it, `BlasEncodingOps::<BE, f64>::blas_context_encode_into(&module, &mut ctx, scratch)`.
pub trait BlasEncodingOps<BE: Backend, F: CKKSEncodingScalar + DiagonalArithmetic> {
    /// Scratch bytes [`Self::blas_pt_operand_encode_into`] needs for an operand of `infos`: the encoding of one diagonal
    /// of a map of period `count d^2`.
    ///
    /// # Panics
    ///
    /// Panics if `count d^2` overflows `usize`.
    fn blas_pt_operand_encode_tmp_bytes<A: BlasPtOperandInfos>(&self, infos: &A) -> usize;

    /// Encodes into `res` the public matrices `entry(m, row, col)`, matrix slot `s` reading matrix `m = s mod count`,
    /// `row < rows d`, `col < cols d`, at the operand's `coeffs_meta`: under [`BlasPtAlgorithm::Diagonal`] block
    /// `(l, j)` as `A -> A B_lj` on the right, `A -> B_lj A` on the left; under [`BlasPtAlgorithm::Jkls`] sigma (tau on
    /// the left), then every block as the map of its results ([`maps::right_jkls`], [`maps::left_jkls`]). Under
    /// [`BlasPtAlgorithm::JklsStored`] and [`BlasPtAlgorithm::JklsGalois`], the same first map, then per block
    /// one-diagonal maps by tau (sigma) of the block rotated by each rotation of its terms ([`maps::slot_product`]), `d`
    /// on the right and `2d - 1` on the left under `JklsStored`, the unrotated one under `JklsGalois`. A left
    /// operand of one public matrix on more than one matrix slot under the diagonal products also holds, per block, the
    /// map of a replicated input ([`maps::left_mul_rep`]). Every diagonal of `res` is overwritten, so encoding again
    /// into the same operand refills it, with no allocation of plaintexts. Operations reject the operand on inputs whose
    /// blocks are not at its radix or are narrower than `coeffs_meta.k`.
    ///
    /// `scratch` must hold [`Self::blas_pt_operand_encode_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// poulpy-ckks's encoding errors, as a value it cannot represent at `coeffs_meta` or a scratch short of the query.
    /// After an error the content of `res` is unspecified: encode it again before use, preparing it before then is
    /// meaningless.
    ///
    /// # Panics
    ///
    /// Panics if `res` was allocated on a module of other slots.
    ///
    /// [`BlasPtAlgorithm::Diagonal`]: crate::layouts::BlasPtAlgorithm::Diagonal
    /// [`BlasPtAlgorithm::Jkls`]: crate::layouts::BlasPtAlgorithm::Jkls
    /// [`BlasPtAlgorithm::JklsStored`]: crate::layouts::BlasPtAlgorithm::JklsStored
    /// [`BlasPtAlgorithm::JklsGalois`]: crate::layouts::BlasPtAlgorithm::JklsGalois
    /// [`maps::slot_product`]: crate::layouts::maps::slot_product
    /// [`maps::right_jkls`]: crate::layouts::maps::right_jkls
    /// [`maps::left_jkls`]: crate::layouts::maps::left_jkls
    /// [`maps::left_mul_rep`]: crate::layouts::maps::left_mul_rep
    fn blas_pt_operand_encode_into<S: BlasPtSide>(
        &self,
        res: &mut BlasPtOperandUnprepared<BE, S>,
        entry: impl Fn(usize, usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>;

    /// Scratch bytes [`Self::blas_context_encode_into`] needs for a context of `infos`: the encoding of one diagonal of a
    /// map of period `d^2`.
    ///
    /// # Panics
    ///
    /// Panics if `d^2` overflows `usize`.
    fn blas_context_encode_tmp_bytes<A: BlasContextInfos>(&self, infos: &A) -> usize;

    /// Encodes every map of `res`'s plan into it at its `coeffs_meta`, each scheduled as the plan schedules it, but tau on
    /// a replicated input ([`BlasContext::tau_rep`]), scheduled at tau's giant step, and psi^k on a replicated input
    /// ([`BlasContext::rotate_rows_rep`]), at no scale. Every diagonal is overwritten, so encoding again refills `res`
    /// with no allocation of plaintexts. Operations reject the context on a module of other slots and on inputs whose
    /// blocks are not at its radix or are narrower than `coeffs_meta.k`.
    ///
    /// `scratch` must hold [`Self::blas_context_encode_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// poulpy-ckks's encoding errors, as a value it cannot represent at `coeffs_meta` or a scratch short of the query.
    /// After an error the content of `res` is unspecified: encode it again before use, preparing it before then is
    /// meaningless.
    ///
    /// # Panics
    ///
    /// Panics if `res` was allocated on a module of another ring.
    fn blas_context_encode_into(&self, res: &mut BlasContextUnprepared<BE>, scratch: &mut ScratchArena<'_, BE>)
    -> CKKSResult<()>;
}

/// Allocation, sizing and preparation of prepared public operands. The allocation is direct on the module; the
/// preparation and its query dispatch through [`BlasEncodingImpl`](crate::oep::BlasEncodingImpl).
pub trait BlasPtOperandPreparedFactory<BE: Backend>: CKKSModuleAlloc<BE> + GetDegree + Sized {
    /// A prepared operand of `infos`, which must be of this module's slots. Under `JklsStored` and `JklsGalois` it holds
    /// the first map and, per term, a right operand of poulpy-ckks's products at the layout of the term's plaintext.
    ///
    /// # Panics
    ///
    /// As [`BlasModuleAlloc::blas_pt_operand_alloc_from_infos`](crate::layouts::BlasModuleAlloc::blas_pt_operand_alloc_from_infos).
    fn blas_pt_operand_prepared_alloc_from_infos<A: BlasPtOperandInfos>(&self, infos: &A) -> BlasPtOperand<BE, A::Side> {
        pt_operand::alloc_prepared(self, &infos.blas_pt_operand_layout())
    }

    /// Scratch bytes [`Self::blas_pt_operand_prepare`] needs for an operand of `infos`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_pt_operand_prepared_alloc_from_infos`].
    fn blas_pt_operand_prepare_tmp_bytes<A: BlasPtOperandInfos>(&self, infos: &A) -> usize;

    /// Prepares every map of `other` into `res`, in place, and under `JklsStored` and `JklsGalois` the one plaintext of
    /// each term's map by `ckks_prepare_right_into`: preparing again after a new encoding refills `res` with no
    /// allocation.
    ///
    /// # Panics
    ///
    /// Panics if `res` and `other` were allocated from different layouts, if `scratch` holds less than
    /// [`Self::blas_pt_operand_prepare_tmp_bytes`], or if a map of `other` has diagonals of more than one scale, width or
    /// slot kind.
    fn blas_pt_operand_prepare<S: BlasPtSide>(
        &self,
        res: &mut BlasPtOperand<BE, S>,
        other: &BlasPtOperandUnprepared<BE, S>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Allocation, sizing and preparation of prepared contexts. The allocation is direct on the module; the preparation and
/// its query dispatch through [`BlasEncodingImpl`](crate::oep::BlasEncodingImpl).
pub trait BlasContextPreparedFactory<BE: Backend>: CKKSModuleAlloc<BE> + GetDegree + Sized {
    /// A prepared context of `infos`, which must be of this module's ring.
    ///
    /// # Panics
    ///
    /// As [`BlasModuleAlloc::blas_context_alloc_from_infos`](crate::layouts::BlasModuleAlloc::blas_context_alloc_from_infos).
    fn blas_context_prepared_alloc_from_infos<A: BlasContextInfos>(&self, infos: &A) -> BlasContext<BE> {
        context::alloc_prepared(self, &infos.blas_context_layout())
    }

    /// Scratch bytes [`Self::blas_context_prepare`] needs for a context of `infos`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_context_prepared_alloc_from_infos`].
    fn blas_context_prepare_tmp_bytes<A: BlasContextInfos>(&self, infos: &A) -> usize;

    /// Prepares every map of `other` into `res`, in place: preparing again after a new encoding refills `res` with no
    /// allocation.
    ///
    /// # Panics
    ///
    /// Panics if `res` and `other` were allocated from different layouts, if `scratch` holds less than
    /// [`Self::blas_context_prepare_tmp_bytes`], or if a map of `other` has diagonals of more than one scale, width or
    /// slot kind.
    fn blas_context_prepare(
        &self,
        res: &mut BlasContext<BE>,
        other: &BlasContextUnprepared<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
