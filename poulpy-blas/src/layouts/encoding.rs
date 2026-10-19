//! The shape of one encoded map, from which public operands and contexts allocate and prepare their maps, and the
//! encoding of a map into its allocation.

use poulpy_ckks::{
    CKKSLayout, CKKSResult, CoeffsMeta, SlotsKind,
    api::{
        CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, Diagonal, GiantStep,
        LinearTransformation, LinearTransformationLayout, LinearTransformationPlan, LinearTransformationPrepared, LtDiagonalMeta,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
};
use poulpy_core::layouts::{
    Base2K, DiagonalArithmetic, Diagonals, GLWELayout, GetDegree, LWEInfos, LinearTransformationStrategy, Rank, TorusPrecision,
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena},
};

use super::validate::assert_scratch;
use super::{
    BlasContextInfos, BlasContextUnprepared, BlasPtOperandInfos, BlasPtOperandUnprepared, BlasPtSide, context, pt_operand,
};

/// A map encoded on a module's plaintexts.
pub(crate) type Encoded<BE> = LinearTransformation<CKKSPlaintextOwned<BE>>;

/// One real map, the schedule it is evaluated by and the metadata its diagonals are encoded at.
pub(crate) struct Scheduled<F> {
    map: ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
    meta: CoeffsMeta,
}

/// `diagonals` with an empty imaginary part, its slots marked real, scheduled by `strategy` and encoded at `meta`.
pub(crate) fn real_map<F>(diagonals: Diagonals<F>, strategy: LinearTransformationStrategy, mut meta: CoeffsMeta) -> Scheduled<F> {
    let im = Diagonals::new(diagonals.slots());
    meta.meta.slots = SlotsKind::Real;
    Scheduled {
        map: ComplexDiagonals::new(diagonals, im),
        strategy,
        meta,
    }
}

impl<F> Scheduled<F> {
    /// The map's shape, which its values never change.
    pub(crate) fn layout(&self) -> MapLayout {
        MapLayout {
            lt: LinearTransformationLayout {
                indexes: self.map.indexes(),
                slots: self.map.slots(),
                strategy: self.strategy,
            },
            period: self.map.period(),
            meta: self.meta,
        }
    }
}

/// One map's shape: its schedule, the slot count its diagonals repeat over, which sets their compact degree, and the
/// metadata they are encoded at.
#[derive(Clone)]
pub(crate) struct MapLayout {
    lt: LinearTransformationLayout,
    period: usize,
    pub(crate) meta: CoeffsMeta,
}

impl MapLayout {
    /// The layout of every diagonal on `module`: compact at the map's period.
    fn plaintext<BE, M>(&self, module: &M, base2k: Base2K) -> CKKSLayout
    where
        BE: Backend,
        M: CKKSModuleAlloc<BE> + GetDegree,
    {
        CKKSLayout {
            glwe_layout: GLWELayout {
                n: module.ckks_compact_degree(self.period),
                base2k,
                k: self.meta.k,
                rank: Rank(0),
            },
            meta: self.meta.meta,
        }
    }

    /// The schedule a prepared map holds: the layout's, with only the baby steps its diagonals use.
    pub(crate) fn prepared_index(&self) -> LinearTransformationPlan {
        skeleton(&self.lt.index(), ()).index()
    }

    /// The map's zero allocation on `module`, at radix `base2k`.
    pub(crate) fn alloc<BE, M>(&self, module: &M, base2k: Base2K) -> Encoded<BE>
    where
        BE: Backend,
        M: CKKSModuleAlloc<BE> + GetDegree,
    {
        module.ckks_linear_transformation_alloc_from_infos(&self.lt, &self.plaintext::<BE, M>(module, base2k))
    }

    /// The prepared map's allocation on `module`, as poulpy-ckks prepares an encoding of this shape into.
    pub(crate) fn alloc_prepared<BE, M>(&self, module: &M, base2k: Base2K) -> LinearTransformationPrepared<BE>
    where
        BE: Backend,
        M: CKKSModuleAlloc<BE> + CnvPVecAlloc<BE> + GetDegree,
    {
        LinearTransformationPrepared::<BE>::alloc_prepared_from_index(
            module,
            &self.prepared_index(),
            &self.plaintext::<BE, M>(module, base2k),
        )
    }

    /// Scratch of poulpy-ckks's preparation of the map.
    pub(crate) fn prepare_tmp_bytes<BE, M>(&self, module: &M, base2k: Base2K) -> usize
    where
        BE: Backend,
        M: CKKSModuleAlloc<BE> + CKKSLinearTransformationOps<BE> + GetDegree,
    {
        module.ckks_prepare_linear_transformation_rhs_tmp_bytes(&self.plaintext::<BE, M>(module, base2k))
    }
}

/// The transform of schedule `plan` with `diagonal` at every diagonal.
pub(crate) fn skeleton<P: Copy>(plan: &LinearTransformationPlan, diagonal: P) -> LinearTransformation<P> {
    LinearTransformation {
        baby_steps: plan.baby_steps.clone(),
        giant_steps: plan
            .giant_steps
            .iter()
            .zip(&plan.index)
            .map(|(&rot, babies)| GiantStep {
                rot,
                diagonals: babies
                    .iter()
                    .map(|&baby| Diagonal {
                        baby,
                        plaintext: diagonal,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Encodes `map` into `lt`, allocated from its layout: every diagonal at the map's metadata, in the schedule's order,
/// straight from the pre-rotated slots poulpy-ckks's schedule builder hands over, so only the plaintexts are written.
///
/// # Panics
///
/// Panics if the schedule of `lt` is not the map's.
pub(crate) fn encode_map_into<BE, M, F>(
    module: &M,
    lt: &mut Encoded<BE>,
    map: Scheduled<F>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    let Scheduled { map, strategy, meta } = map;
    let plan = LinearTransformationLayout {
        indexes: map.indexes(),
        slots: map.slots(),
        strategy,
    }
    .index();
    let same_schedule = lt.baby_steps == plan.baby_steps
        && lt.giant_steps.len() == plan.giant_steps.len()
        && lt
            .giant_steps
            .iter()
            .zip(plan.giant_steps.iter().zip(&plan.index))
            .all(|(gs, (&rot, babies))| gs.rot == rot && gs.diagonals.iter().map(|d| d.baby).eq(babies.iter().copied()));
    assert!(same_schedule, "the map's schedule is not its allocation's");
    let mut targets = lt.giant_steps.iter_mut().flat_map(|gs| &mut gs.diagonals);
    let mut result = Ok(());
    // The builder visits the diagonals in the schedule's order, the one asserted above.
    map.build_transform(strategy, |re, im| {
        if let (Ok(()), Some(dst)) = (&result, targets.next()) {
            result = encode_diagonal(module, &mut dst.plaintext, meta, re, im, scratch);
        }
    });
    result
}

/// Encodes `re + i im` into `pt` at `meta`.
fn encode_diagonal<BE, M, F>(
    module: &M,
    pt: &mut CKKSPlaintextOwned<BE>,
    meta: CoeffsMeta,
    re: &[F],
    im: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: CKKSEncodingOps<BE, F>,
    F: CKKSEncodingScalar,
{
    pt.set_meta_checked(meta.meta)?;
    module.ckks_encode_reim_into(pt, re, im, scratch)?;
    Ok(())
}

/// Prepares every map of `maps` into the allocation `res` yields at the same position, once `scratch` holds the largest
/// preparation's bytes; `op` names the caller in the scratch assertion.
pub(crate) fn prepare_maps_into<'a, BE, M>(
    module: &M,
    op: &'static str,
    res: impl Iterator<Item = &'a mut LinearTransformationPrepared<BE>>,
    maps: impl Iterator<Item = &'a Encoded<BE>> + Clone,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + 'a,
    M: CKKSLinearTransformationOps<BE>,
{
    let bytes = maps
        .clone()
        .filter_map(LinearTransformation::first_diagonal_plaintext)
        .map(|pt| module.ckks_prepare_linear_transformation_rhs_tmp_bytes(pt))
        .max()
        .unwrap_or(0);
    assert_scratch(op, bytes, scratch);
    for (dst, src) in res.zip(maps) {
        module.ckks_prepare_linear_transformation_rhs(dst, src, scratch);
    }
}

/// Encoding of public operands, and of contexts, into their allocations.
pub trait BlasEncodingOps<BE: Backend>: CKKSModuleAlloc<BE> + GetDegree + Sized {
    /// Scratch bytes [`Self::blas_pt_operand_encode_into`] needs for an operand of `infos`: the encoding of one diagonal
    /// of `count d^2` slots.
    fn blas_pt_operand_encode_tmp_bytes<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasPtOperandInfos,
    {
        let slots = 1usize
            .checked_shl(2 * infos.log_d() as u32)
            .and_then(|d2| d2.checked_mul(infos.count()))
            .expect("count d^2 overflows usize");
        <Self as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(self, slots)
    }

    /// Encodes into `res` the public matrices `entry(m, row, col)`, matrix slot `s` reading matrix `m = s mod count`,
    /// `row < rows d`, `col < cols d`, at the operand's `coeffs_meta`: under [`BlasPtAlgorithm::Diagonal`] block
    /// `(l, j)` as `A -> A B_lj` on the right, `A -> B_lj A` on the left; under [`BlasPtAlgorithm::Jkls`] sigma (tau on
    /// the left), then every block as the map of its results ([`maps::right_jkls`], [`maps::left_jkls`]). A left
    /// operand of one public matrix on more than one matrix slot under the diagonal products also holds, per block, the
    /// map of a replicated input ([`maps::left_mul_rep`]). Every diagonal of `res` is overwritten, so encoding again
    /// into the same operand refills it, with no allocation of plaintexts. Operations reject the operand on inputs whose
    /// blocks are not at its radix, and size their scratch assuming `coeffs_meta.k` is not wider than the ciphertexts.
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
    /// [`maps::right_jkls`]: crate::layouts::maps::right_jkls
    /// [`maps::left_jkls`]: crate::layouts::maps::left_jkls
    /// [`maps::left_mul_rep`]: crate::layouts::maps::left_mul_rep
    fn blas_pt_operand_encode_into<F, S>(
        &self,
        res: &mut BlasPtOperandUnprepared<BE, S>,
        entry: impl Fn(usize, usize, usize) -> F,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
        S: BlasPtSide,
    {
        pt_operand::encode_into(self, res, entry, scratch)
    }
    /// Scratch bytes [`Self::blas_context_encode_into`] needs for a context of `infos`: the encoding of one diagonal of
    /// `d^2` slots.
    fn blas_context_encode_tmp_bytes<F, A>(&self, infos: &A) -> usize
    where
        Self: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
        A: BlasContextInfos,
    {
        let slots = 1usize.checked_shl(2 * infos.log_d() as u32).expect("d^2 overflows usize");
        <Self as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(self, slots)
    }

    /// Encodes every map of `res`'s plan into it at its `coeffs_meta`, each scheduled as the plan schedules it, but tau on
    /// a replicated input ([`BlasContext::tau_rep`](crate::layouts::BlasContext::tau_rep)), scheduled at tau's giant
    /// step, and psi^k on a replicated input
    /// ([`BlasContext::rotate_rows_rep`](crate::layouts::BlasContext::rotate_rows_rep)), at no scale. Every diagonal is
    /// overwritten, so encoding again refills `res` with no allocation of plaintexts. Operations reject the context on a
    /// module of other slots and on inputs whose blocks are not at its radix, and size their scratch assuming
    /// `coeffs_meta.k` is not wider than the ciphertexts.
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
    fn blas_context_encode_into<F>(
        &self,
        res: &mut BlasContextUnprepared<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        Self: CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
    {
        context::encode_into(self, res, scratch)
    }
}

impl<BE: Backend> BlasEncodingOps<BE> for Module<BE> where Module<BE>: CKKSModuleAlloc<BE> {}

/// Width poulpy-ckks gives `lt` evaluated from a ciphertext of width `src_k` into one of width `dst_k`, on an input it
/// evaluates: the diagonals' scale comes off the source, capped at the destination.
pub(crate) fn lt_output_k<P: LtDiagonalMeta>(
    lt: &LinearTransformation<P>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
) -> TorusPrecision {
    let scale = lt.first_diagonal_plaintext().map_or(0, LtDiagonalMeta::lt_log_scale);
    src_k.as_usize().saturating_sub(scale).min(dst_k.as_usize()).into()
}

/// `block` after `lt` evaluated into a ciphertext of width `dst_k`: at the width [`lt_output_k`] gives, its slots joined with
/// the map's. The preparations, the staged blocks of the products and the output-layout queries all read a map's effect
/// on a layout here.
pub(crate) fn mapped_block_into<P: LtDiagonalMeta>(
    mut block: CKKSLayout,
    lt: &LinearTransformation<P>,
    dst_k: TorusPrecision,
) -> CKKSLayout {
    block.glwe_layout.k = lt_output_k(lt, block.k(), dst_k);
    if let Some(diagonal) = lt.first_diagonal_plaintext() {
        block.meta.slots = block.meta.slots.join(diagonal.lt_slots());
    }
    block
}

/// `block` after `lt`: at the width `lt` leaves of it, its slots joined with the map's.
pub(crate) fn mapped_block<P: LtDiagonalMeta>(block: CKKSLayout, lt: &LinearTransformation<P>) -> CKKSLayout {
    let k = block.k();
    mapped_block_into(block, lt, k)
}
