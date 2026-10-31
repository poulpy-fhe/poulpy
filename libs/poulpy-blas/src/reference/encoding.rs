use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSMulOps, CKKSRightOperandPreparedFactory},
    encoding::ckks_linear_transformation_encode_tmp_bytes,
};
use poulpy_core::layouts::DiagonalArithmetic;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    BlasContext, BlasContextInfos, BlasContextPrepared, BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos,
    BlasDiagonalCiphertextPrepared, BlasMatVecOperand, BlasMatVecOperandInfos, BlasMatVecOperandPrepared, BlasPtOperand,
    BlasPtOperandInfos, BlasPtOperandPrepared, BlasPtSide,
    context::{MapId, assert_fits_module, context_map},
    encoding::{encode_map_into, prepare_maps_into, real_map},
    maps, matvec_operand,
    pt_operand::{assert_module_slots, block_map, first_map, masked_rotations, prepared_shapes, public_values, rep_map},
    validate::assert_scratch,
};

/// References of [`crate::oep::BlasEncodingImpl`], which
/// [`impl_blas_encoding_reference_full!`](crate::impl_blas_encoding_reference_full) forwards to: the encodings and
/// preparations of [`crate::api::BlasEncodingOps`], [`crate::api::BlasPtOperandPreparedFactory`],
/// [`crate::api::BlasDiagonalCiphertextPreparedFactory`] and [`crate::api::BlasContextPreparedFactory`]. An encoding builds each map's host diagonals and encodes them by
/// poulpy-ckks's `ckks_linear_transformation_encode_into`, map after map in walking order, and the first error stops it;
/// a preparation prepares each map by poulpy-ckks's `ckks_linear_transformation_rhs_prepare`, and a masked operand's
/// terms by `ckks_right_operand_prepare`, into the allocations of `res`.
pub trait BlasEncodingReference<BE: Backend> {
    /// poulpy-ckks's query for the encoding of one diagonal of a map of period `count d^2`.
    fn blas_pt_operand_encode_tmp_bytes_reference<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasPtOperandInfos;

    /// After asserting its scratch query, every map of `res`, in walking order: the first map under JKLS and the masked paths, the blocks row-major, each
    /// block's terms in order, then the replicated-input maps, every block sharing the first block's schedule.
    fn blas_pt_operand_encode_reference<F, S>(
        &self,
        res: &mut BlasPtOperand<BE, S>,
        entry: impl Fn(usize, usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
        S: BlasPtSide;

    /// poulpy-ckks's query for the encoding of one diagonal of a map of the operand's larger period.
    fn blas_matvec_operand_encode_tmp_bytes_reference<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasMatVecOperandInfos;

    /// After asserting its scratch query, the map of every block, output-major.
    fn blas_matvec_operand_encode_reference<F>(
        &self,
        res: &mut BlasMatVecOperand<BE>,
        entry: impl Fn(usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar;

    /// poulpy-ckks's query for the encoding of one diagonal of a map of period `d^2`.
    fn blas_context_encode_tmp_bytes_reference<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasContextInfos;

    /// After asserting its scratch query, every map of the plan of `res`, in walking order.
    fn blas_context_encode_reference<F>(&self, res: &mut BlasContext<BE>, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar;

    /// The largest of poulpy-ckks's preparation queries over the maps, and over one term's plaintext under `JklsStored`
    /// and `JklsGalois`.
    fn blas_pt_operand_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSMulOps<BE> + CKKSRightOperandPreparedFactory<BE>,
        A: BlasPtOperandInfos;

    /// Every map of `other` prepared into the same position of `res`; under `JklsStored` and `JklsGalois` the first
    /// map, then each term's one plaintext by `ckks_right_operand_prepare`.
    fn blas_pt_operand_prepare_reference<S>(
        &self,
        res: &mut BlasPtOperandPrepared<BE, S>,
        other: &BlasPtOperand<BE, S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE> + CKKSMulOps<BE> + CKKSRightOperandPreparedFactory<BE>,
        S: BlasPtSide;

    /// poulpy-ckks's preparation query of the maps' shared shape.
    fn blas_matvec_operand_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        A: BlasMatVecOperandInfos;

    /// Every map of `other` prepared into the same position of `res`.
    fn blas_matvec_operand_prepare_reference(
        &self,
        res: &mut BlasMatVecOperandPrepared<BE>,
        other: &BlasMatVecOperand<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE>;

    /// poulpy-ckks's preparation query of a diagonal of `infos`.
    fn blas_diagonal_ciphertext_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        A: BlasDiagonalCiphertextInfos;

    /// After asserting its scratch query, every diagonal of `other` prepared into the same position of `res`.
    fn blas_diagonal_ciphertext_prepare_reference(
        &self,
        res: &mut BlasDiagonalCiphertextPrepared<BE>,
        other: &BlasDiagonalCiphertext<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE>;

    /// The largest of poulpy-ckks's preparation queries over the maps.
    fn blas_context_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        A: BlasContextInfos;

    /// Every map of `other` prepared into the same position of `res`.
    fn blas_context_prepare_reference(
        &self,
        res: &mut BlasContextPrepared<BE>,
        other: &BlasContext<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE>;
}

impl<BE: Backend> BlasEncodingReference<BE> for Module<BE> {
    fn blas_pt_operand_encode_tmp_bytes_reference<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasPtOperandInfos,
    {
        let slots = 1usize
            .checked_shl(2 * infos.log_d() as u32)
            .and_then(|d2| d2.checked_mul(infos.count()))
            .expect("count d^2 overflows usize");
        ckks_linear_transformation_encode_tmp_bytes::<BE, Self, F>(self, slots)
    }

    fn blas_pt_operand_encode_reference<F, S>(
        &self,
        res: &mut BlasPtOperand<BE, S>,
        entry: impl Fn(usize, usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
        S: BlasPtSide,
    {
        let layout = res.blas_pt_operand_layout();
        assert_module_slots(self, &layout);
        let bytes = self.blas_pt_operand_encode_tmp_bytes_reference::<F, _>(&layout);
        assert_scratch("blas_pt_operand_encode", bytes, scratch);
        if let (Some(lt), Some(map)) = (&mut res.maps.first, first_map::<F, S>(&layout)) {
            encode_map_into(self, lt, map, scratch)?;
        }
        if layout.plan.pt_algorithm().masked() {
            let rotations = masked_rotations(layout.plan.log_d(), S::LEFT);
            for (t, block) in res.maps.blocks.chunks_mut(res.maps.terms).enumerate() {
                let values = public_values(&layout, &entry, t);
                // `JklsGalois` stores one map per block, the first rotation, 0.
                for (lt, &rotation) in block.iter_mut().zip(&rotations) {
                    let map = maps::slot_product(layout.slots, &values, rotation);
                    let strategy = layout.plan.strategy(&map, false);
                    encode_map_into(self, lt, real_map(map, strategy, layout.coeffs_meta), scratch)?;
                }
            }
        } else {
            let mut strategy = None;
            for (t, lt) in res.maps.blocks.iter_mut().enumerate() {
                let map = block_map(&layout, &entry, t, strategy);
                strategy = Some(map.strategy);
                encode_map_into(self, lt, map, scratch)?;
            }
        }
        if let Some(blocks_rep) = &mut res.maps.blocks_rep {
            let strategy = layout.plan.left_mul_strategy(layout.slots);
            for (t, lt) in blocks_rep.iter_mut().enumerate() {
                encode_map_into(self, lt, rep_map(&layout, &entry, t, strategy), scratch)?;
            }
        }
        Ok(())
    }

    fn blas_matvec_operand_encode_tmp_bytes_reference<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasMatVecOperandInfos,
    {
        let layout = infos.blas_matvec_operand_layout();
        matvec_operand::assert_fits(&layout);
        ckks_linear_transformation_encode_tmp_bytes::<BE, Self, F>(self, layout.map_period())
    }

    fn blas_matvec_operand_encode_reference<F>(
        &self,
        res: &mut BlasMatVecOperand<BE>,
        entry: impl Fn(usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
    {
        let layout = res.blas_matvec_operand_layout();
        matvec_operand::assert_module_slots(self, &layout);
        let bytes = self.blas_matvec_operand_encode_tmp_bytes_reference::<F, _>(&layout);
        assert_scratch("blas_matvec_operand_encode", bytes, scratch);
        let inputs = layout.input.count();
        for (t, lt) in res.maps.iter_mut().enumerate() {
            encode_map_into(
                self,
                lt,
                matvec_operand::block_map(&layout, &entry, t / inputs, t % inputs),
                scratch,
            )?;
        }
        Ok(())
    }

    fn blas_context_encode_tmp_bytes_reference<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasContextInfos,
    {
        let slots = 1usize.checked_shl(2 * infos.log_d() as u32).expect("d^2 overflows usize");
        ckks_linear_transformation_encode_tmp_bytes::<BE, Self, F>(self, slots)
    }

    fn blas_context_encode_reference<F>(&self, res: &mut BlasContext<BE>, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
    {
        let layout = res.blas_context_layout();
        assert_fits_module(self, &layout);
        assert_scratch(
            "blas_context_encode",
            self.blas_context_encode_tmp_bytes_reference::<F, _>(&layout),
            scratch,
        );
        for (id, lt) in MapId::all(layout.plan).zip(res.maps.iter_mut()) {
            encode_map_into(self, lt, context_map::<F>(&layout, id), scratch)?;
        }
        Ok(())
    }

    fn blas_pt_operand_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSMulOps<BE> + CKKSRightOperandPreparedFactory<BE>,
        A: BlasPtOperandInfos,
    {
        let layout = infos.blas_pt_operand_layout();
        let (shapes, terms) = prepared_shapes(self, &layout);
        let terms = terms
            .first()
            .map(|term| self.ckks_right_operand_prepare_tmp_bytes(&term.plaintext::<BE, Self>(self, layout.base2k)));
        shapes
            .iter()
            .map(|map| map.prepare_tmp_bytes::<BE, Self>(self, layout.base2k))
            .chain(terms)
            .max()
            .unwrap_or(0)
    }

    fn blas_pt_operand_prepare_reference<S>(
        &self,
        res: &mut BlasPtOperandPrepared<BE, S>,
        other: &BlasPtOperand<BE, S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE> + CKKSMulOps<BE> + CKKSRightOperandPreparedFactory<BE>,
        S: BlasPtSide,
    {
        const OP: &str = "blas_pt_operand_prepare";
        let layout = res.blas_pt_operand_layout();
        assert!(
            layout == other.blas_pt_operand_layout(),
            "the prepared operand was allocated for another layout"
        );
        if !layout.plan.pt_algorithm().masked() {
            prepare_maps_into(self, OP, res.maps.iter_mut(), other.maps.iter(), 0, scratch);
            return;
        }
        let terms = other.maps.blocks.iter().map(|lt| {
            lt.first_diagonal_plaintext()
                .expect("every term of a masked operand holds one diagonal")
        });
        let bytes = terms
            .clone()
            .map(|pt| self.ckks_right_operand_prepare_tmp_bytes(pt))
            .max()
            .unwrap_or(0);
        prepare_maps_into(self, OP, res.maps.first.iter_mut(), other.maps.first.iter(), bytes, scratch);
        let prepared: Result<()> = res
            .plaintexts
            .iter_mut()
            .zip(terms)
            .try_for_each(|(dst, pt)| self.ckks_right_operand_prepare(dst, pt, scratch));
        prepared.expect("every term's plaintext has the layout its prepared operand was allocated for");
    }

    fn blas_matvec_operand_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        A: BlasMatVecOperandInfos,
    {
        let layout = infos.blas_matvec_operand_layout();
        matvec_operand::assert_module_slots(self, &layout);
        matvec_operand::map_layout(&layout).prepare_tmp_bytes::<BE, Self>(self, layout.base2k)
    }

    fn blas_matvec_operand_prepare_reference(
        &self,
        res: &mut BlasMatVecOperandPrepared<BE>,
        other: &BlasMatVecOperand<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE>,
    {
        assert!(
            res.blas_matvec_operand_layout() == other.blas_matvec_operand_layout(),
            "the prepared operand was allocated for another layout"
        );
        prepare_maps_into(
            self,
            "blas_matvec_operand_prepare",
            res.maps.iter_mut(),
            other.maps.iter(),
            0,
            scratch,
        );
    }

    fn blas_diagonal_ciphertext_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        A: BlasDiagonalCiphertextInfos,
    {
        self.ckks_linear_transformation_rhs_prepare_tmp_bytes(&infos.blas_diagonal_ciphertext_layout().ct)
    }

    fn blas_diagonal_ciphertext_prepare_reference(
        &self,
        res: &mut BlasDiagonalCiphertextPrepared<BE>,
        other: &BlasDiagonalCiphertext<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE>,
    {
        let op = "blas_diagonal_ciphertext_prepare";
        let layout = other.blas_diagonal_ciphertext_layout();
        assert!(
            res.blas_diagonal_ciphertext_layout() == layout,
            "the prepared diagonal ciphertext was allocated for another layout"
        );
        assert_scratch(
            op,
            self.blas_diagonal_ciphertext_prepare_tmp_bytes_reference(&layout),
            scratch,
        );
        if let Err(err) = self.ckks_linear_transformation_rhs_prepare(&mut res.diagonals, &other.diagonals, scratch) {
            panic!("{op}: {err}");
        }
    }

    fn blas_context_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        A: BlasContextInfos,
    {
        let layout = infos.blas_context_layout();
        assert_fits_module(self, &layout);
        MapId::all(layout.plan)
            .map(|id| {
                context_map::<f64>(&layout, id)
                    .layout()
                    .prepare_tmp_bytes::<BE, Self>(self, layout.base2k)
            })
            .max()
            .unwrap_or(0)
    }

    fn blas_context_prepare_reference(
        &self,
        res: &mut BlasContextPrepared<BE>,
        other: &BlasContext<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: CKKSLinearTransformationOps<BE>,
    {
        assert!(
            res.blas_context_layout() == other.blas_context_layout(),
            "the prepared context was allocated for another layout"
        );
        prepare_maps_into(
            self,
            "blas_context_prepare",
            res.maps.iter_mut(),
            other.maps.iter(),
            0,
            scratch,
        );
    }
}
