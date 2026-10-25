//! The shape of one encoded map, from which public operands and contexts allocate and prepare their maps, and the
//! encoding of a map into its allocation.

use poulpy_ckks::{
    CKKSLayout, CKKSResult, CoeffsMeta, SlotsKind,
    api::{
        CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSModuleInfos, Diagonal, GiantStep,
        LinearTransformation, LinearTransformationLayout, LinearTransformationPlan, LinearTransformationPrepared, LtDiagonalMeta,
        ckks_linear_transformation_output_layout,
    },
    encoding::ckks_encode_linear_transformation_into,
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
};
use poulpy_core::layouts::{
    Base2K, DiagonalArithmetic, Diagonals, GLWELayout, GetDegree, LWEInfos, LinearTransformationStrategy, Rank, TorusPrecision,
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, ScratchArena},
};

use super::validate::assert_scratch;

/// A map encoded on a module's plaintexts.
pub(crate) type Encoded<BE> = LinearTransformation<CKKSPlaintextOwned<BE>>;

/// One real map, the schedule it is evaluated by and the metadata its diagonals are encoded at.
pub(crate) struct Scheduled<F> {
    map: ComplexDiagonals<F>,
    pub(crate) strategy: LinearTransformationStrategy,
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

/// One map's shape: its schedule, the slot count its diagonals repeat over, which sets their degree, and the
/// metadata they are encoded at.
#[derive(Clone)]
pub(crate) struct MapLayout {
    lt: LinearTransformationLayout,
    period: usize,
    pub(crate) meta: CoeffsMeta,
}

impl MapLayout {
    /// The layout of every diagonal on `module`: at the degree of the map's period.
    pub(crate) fn plaintext<BE, M>(&self, module: &M, base2k: Base2K) -> CKKSLayout
    where
        BE: Backend,
        M: CKKSModuleAlloc<BE> + GetDegree,
    {
        CKKSLayout {
            glwe_layout: GLWELayout {
                n: (self.period * module.ring_degree().as_usize() / module.ckks_max_slots())
                    .max(BE::MIN_DEGREE)
                    .into(),
                base2k,
                k: self.meta.k,
                rank: Rank(0),
            },
            meta: self.meta.meta,
        }
    }

    /// The sparsity the map's period proves on the module's degree: `log2` of its slots over its period.
    pub(crate) fn log_sparsity(&self) -> usize {
        (self.lt.slots / self.period).ilog2() as usize
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

/// What the planning queries read of a map's diagonals: their width, scale, slot kind and the sparsity their period
/// proves on the module's degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiagonalShape {
    pub(crate) k: TorusPrecision,
    pub(crate) log_scale: usize,
    pub(crate) slots: SlotsKind,
    pub(crate) log_sparsity: usize,
}

impl LtDiagonalMeta for DiagonalShape {
    fn lt_log_scale(&self) -> usize {
        self.log_scale
    }

    fn lt_slots(&self) -> SlotsKind {
        self.slots
    }

    fn lt_log_sparsity(&self, _log_n: usize) -> usize {
        self.log_sparsity
    }
}

/// `map` as the planning queries read it: the schedule its prepared form holds, every diagonal of its width, scale and
/// slot kind.
pub(crate) fn shape(map: &MapLayout) -> LinearTransformation<DiagonalShape> {
    let slots = if map.meta.meta.slots.is_real() {
        SlotsKind::Real
    } else {
        SlotsKind::Complex
    };
    let diagonal = DiagonalShape {
        k: map.meta.k,
        log_scale: map.meta.log_delta(),
        slots,
        log_sparsity: map.log_sparsity(),
    };
    skeleton(&map.prepared_index(), diagonal)
}

/// Encodes `map` into `lt`, allocated from its layout.
pub(crate) fn encode_map_into<BE, M, F>(
    module: &M,
    lt: &mut Encoded<BE>,
    map: Scheduled<F>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: CKKSModuleInfos + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    ckks_encode_linear_transformation_into(module, lt, map.meta.meta, &map.map, map.strategy, scratch)
}

/// Prepares every map of `maps` into the allocation `res` yields at the same position, once `scratch` holds the largest
/// preparation's bytes; `op` names the caller in the scratch assertion and prefixes the panic carrying poulpy-ckks's
/// refusal of a map, one whose diagonals do not share one scale, width and slot kind.
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
        if let Err(err) = module.ckks_prepare_linear_transformation_rhs(dst, src, scratch) {
            panic!("{op}: {err}");
        }
    }
}

/// `block` after `lt` evaluated into a ciphertext of width `dst_k`, at `block`'s label: a map never repeats with a longer
/// period than the blocks it reads, so blas keeps their label, where poulpy-ckks reads one off the diagonals' degree,
/// which a backend's minimum degree may raise.
pub(crate) fn mapped_block_into<P: LtDiagonalMeta>(
    block: CKKSLayout,
    lt: &LinearTransformation<P>,
    dst_k: TorusPrecision,
) -> CKKSLayout {
    let mut mapped = ckks_linear_transformation_output_layout(lt, block, dst_k);
    mapped.meta.log_sparsity = block.meta.log_sparsity;
    mapped
}

/// [`mapped_block_into`] at `block`'s width.
pub(crate) fn mapped_block<P: LtDiagonalMeta>(block: CKKSLayout, lt: &LinearTransformation<P>) -> CKKSLayout {
    mapped_block_into(block, lt, block.k())
}
