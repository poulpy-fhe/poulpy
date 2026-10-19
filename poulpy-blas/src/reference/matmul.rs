use std::cell::OnceCell;

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSResult as Result, GLWEPlaintextInfos, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSCopyOps, CKKSDotProductOps, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSMulOps, LinearTransformation,
        LinearTransformationBabySteps, LinearTransformationPrepared, LtDiagonalMeta, ckks_linear_transformation_output_k,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedLeft, CKKSPreparedRight},
};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, TorusPrecision,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, MaxBase2k, Module, PrepareHint, ScratchArena},
};

use super::{BlasKeyUses, MapDiagonal, context_shapes, lt_tmp_bytes, operand_shapes, pt_label};
use crate::{
    layouts::{
        BlasContext, BlasContextInfos, BlasGrid, BlasGridInfos, BlasGridMut, BlasLayout, BlasMatMulOperand,
        BlasMatMulOperandInfos, BlasMatMulOperandLayout, BlasPtOperand, BlasPtOperandInfos, BlasPtSide, Left, Right,
        ciphertext::{layout_of, replicated_at},
        encoding::{mapped_block, mapped_block_into},
        grid_batch,
        matmul_operand::{
            left_operand_layout, left_term_layout, product_terms, right_maps, right_operand_layout, right_term_layout,
        },
        pt_operand::OperandMaps,
        replicated,
        validate::{
            MAX_BASE2K, assert_scratch, ensure_rotation, ensure_tensor_key, left_takes_replicated, left_uses, matmul_tensor,
            matmul_uses, prepared_tensor, prepared_uses, pt_left_uses, pt_right_uses, right_uses, term_uses,
            validate_blas_matmul, validate_blas_matmul_prepare_left, validate_blas_matmul_prepare_right,
            validate_blas_matmul_prepared, validate_blas_matmul_pt_left, validate_blas_matmul_pt_right,
        },
    },
    oep::{BlasJklsImpl, BlasMatIntraImpl, derived::KeyLayouts},
};

/// The largest radix in `1..=62` at which the model of `BE` holds the `products(r)` limb products that a lazy sum adds
/// in one output limb of degree-`n` polynomials at radix `r`, `Some(0)` if none does, `None` without a model.
fn max_radix<BE: MaxBase2k>(n: usize, failure_bits: usize, products: impl Fn(usize) -> usize) -> Option<usize> {
    for r in (1..=MAX_BASE2K).rev() {
        // A zero width adds no product, which the model rejects: count one.
        if Module::<BE>::max_base2k(n, products(r).max(1), failure_bits, false)? >= r {
            return Some(r);
        }
    }
    Some(0)
}

/// The largest `base2k`, at most 62, that [`Module::max_base2k`] allows for every lazy sum of a
/// [`crate::api::BlasMatMulOps::blas_matmul`] of `a` by `b` under `ctx`, `Some(0)` if none does, and `None` for a
/// backend without a model. The JKLS kernel adds the products of its `K d` pairs of terms in the transform domain,
/// `K = a.cols()` and `d` the block dimension: up to `2 L` limb products per pair in an output limb, `L` the limbs of
/// the narrower terms, `min(w_a, w_b)` over `base2k` rounded up, for the widths of
/// [`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout) and
/// [`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout). It counts the one-pass circuit,
/// rank 1 with one `base2k` across the blocks and the tensor key; the two-pass circuit adds at most `2 L` limb products
/// at once in the transform domain. Each map that makes the terms, sigma or tau then the shifts, adds for one term the
/// count [`blas_matmul_pt_right`](crate::api::BlasMatMulOps::blas_matmul_pt_right) documents, read at the block's width
/// and, for a shift, at the width its first map leaves. The largest count binds. Widths are in bits, so the limbs are
/// counted at each candidate radix, whatever the radix of `a`, `b` and `ctx`. For a prepared product, pass the source
/// grids, not a cold operand's term layout. `failure_bits` and the model are those of [`Module::max_base2k`], whose
/// estimate is a Gaussian approximation, not a guarantee. `ctx` is a context or its layout: the query reads the shape
/// of its maps. Name the backend: `blas_matmul_max_base2k::<BE>(..)`.
///
/// # Panics
///
/// As [`Module::max_base2k`]: unless the blocks' degree is a power of two at least the backend's minimum degree and
/// `failure_bits` is positive.
pub fn blas_matmul_max_base2k<BE: MaxBase2k>(
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
    failure_bits: usize,
) -> Option<usize> {
    let maps = context_shapes(ctx);
    let (left, right) = (left_term_layout(a, &maps).block, right_term_layout(b, &maps).block);
    let (width, pairs, shifts) = (
        left.k().min(right.k()).as_usize(),
        a.cols() << a.log_d(),
        1..1usize << maps.log_d(),
    );
    let (first, shift) = right_maps(&maps, replicated_at(maps.slots(), b));
    let terms: Vec<_> = term_maps(
        maps.sigma(),
        shifts.clone().map(|k| maps.rotate_cols(k)),
        a.block_layout().k(),
    )
    .into_iter()
    .chain(term_maps(first, shifts.map(shift), b.block_layout().k()))
    .collect();
    max_radix::<BE>(a.block_layout().n().as_usize(), failure_bits, |r| {
        terms.iter().fold(2 * width.div_ceil(r) * pairs, |count, &(shape, src)| {
            count.max(giant_products(shape, 1, src, r))
        })
    })
}

/// The shape and the input width, in bits, of the maps that make a side's terms: `first` from a block of `k` bits, then
/// the `shifts` from the width `first` leaves.
fn term_maps<'c, P: MapDiagonal + 'c>(
    first: &'c LinearTransformation<P>,
    shifts: impl IntoIterator<Item = &'c LinearTransformation<P>>,
    k: TorusPrecision,
) -> [((usize, usize), usize); 2] {
    [
        (map_shape([first]), k.as_usize()),
        (map_shape(shifts), ckks_linear_transformation_output_k(first, k, k).as_usize()),
    ]
}

/// The most diagonals that a giant step of `maps` groups, and the widest diagonal, in bits.
fn map_shape<'m, P: MapDiagonal + 'm>(maps: impl IntoIterator<Item = &'m LinearTransformation<P>>) -> (usize, usize) {
    maps.into_iter().fold((0, 0), |(diagonals, width), lt| {
        let grouped = lt.giant_steps.iter().map(|gs| gs.diagonals.len()).max().unwrap_or(0);
        let k = lt.first_diagonal_plaintext().map_or(0, |d| d.width().as_usize());
        (diagonals.max(grouped), width.max(k))
    })
}

/// The limb products that one giant step of maps of `shape` adds in an output limb at radix `r`, over `terms` summed
/// terms read at `src` bits: every diagonal of the step, for each term, convolves the `src` bits of the input with its
/// own, and a limb of the result sums at most `min(L, P)` products of limbs.
fn giant_products((diagonals, width): (usize, usize), terms: usize, src: usize, r: usize) -> usize {
    terms * diagonals * src.div_ceil(r).min(width.div_ceil(r))
}

/// [`blas_matmul_pt_right_max_base2k`] and [`blas_matmul_pt_left_max_base2k`] over `inner` terms, the block maps being
/// `b`'s replicated-input ones when `rep`.
fn pt_max_base2k<BE: MaxBase2k, P: MapDiagonal>(
    a: &impl BlasGridInfos,
    b: &OperandMaps<LinearTransformation<P>>,
    inner: usize,
    rep: bool,
    failure_bits: usize,
) -> Option<usize> {
    let (block, maps) = (
        a.block_layout(),
        map_shape((0..b.rows()).flat_map(|l| (0..b.cols()).map(move |j| b.at_for(l, j, rep)))),
    );
    let a_k = block.k();
    let (maps_k, first) = match b.first() {
        None => (a_k, None),
        Some(first) => (ckks_linear_transformation_output_k(first, a_k, a_k), Some(map_shape([first]))),
    };
    max_radix::<BE>(block.n().as_usize(), failure_bits, |r| {
        let blocks = giant_products(maps, inner, maps_k.as_usize(), r);
        first.map_or(blocks, |first| blocks.max(giant_products(first, 1, a_k.as_usize(), r)))
    })
}

/// The largest `base2k`, at most 62, that [`Module::max_base2k`] allows for every lazy sum of a
/// [`crate::api::BlasMatMulOps::blas_matmul_pt_right`] of `a` by `b`, `Some(0)` if none does, and `None` for a backend
/// without a model: the largest radix at which the model holds the count that method documents, with `L` and `P` counted
/// at each candidate radix, whatever the radix of `a` and `b`. `failure_bits` and the model are those of
/// [`Module::max_base2k`]. `b` is a public operand or its layout: the query reads the shape of its maps. Name the
/// backend, as [`blas_matmul_max_base2k`] does.
///
/// # Panics
///
/// As [`blas_matmul_max_base2k`].
pub fn blas_matmul_pt_right_max_base2k<BE: MaxBase2k>(
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Right>,
    failure_bits: usize,
) -> Option<usize> {
    pt_max_base2k::<BE, _>(a, &operand_shapes(b), a.cols(), false, failure_bits)
}

/// [`blas_matmul_pt_right_max_base2k`] for [`crate::api::BlasMatMulOps::blas_matmul_pt_left`]: the terms of a block of
/// `res` number `K = a.rows()`, and the first map under JKLS is tau. The block maps counted are those the product takes
/// for `a`: [`BlasPtOperand::block_replicated`]'s when `a` is replicated ([`replicated`]) and `b` holds them.
///
/// # Panics
///
/// As [`blas_matmul_max_base2k`].
pub fn blas_matmul_pt_left_max_base2k<BE: MaxBase2k>(
    b: &impl BlasPtOperandInfos<Side = Left>,
    a: &impl BlasGridInfos,
    failure_bits: usize,
) -> Option<usize> {
    let maps = operand_shapes(b);
    pt_max_base2k::<BE, _>(a, &maps, a.rows(), left_takes_replicated(&maps, a), failure_bits)
}

/// Fails as [`ensure_rotation`] unless `atk` holds every key of [`term_uses`].
fn ensure_term_keys<'c, BE: Backend + 'c>(
    op: &'static str,
    module: &(impl CKKSModuleInfos + ?Sized),
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    term_uses(a, first, shift, &mut |u| ensure_rotation(op, module, atk, u))
}

/// The metadata of the product of two term blocks written into blocks of width `res_k`, poulpy-ckks's ct x ct rule: the
/// smaller scale and label, a width of `min(res_k, min(wa - da, wb - db) - max(da, db) + min(da, db))` for term widths
/// `wa`, `wb` and scales `da`, `db`, each difference saturating at `0` where the product returns its budget error, and
/// the slots of both joined with the maps' `Real`.
fn product_block(left: CKKSLayout, right: CKKSLayout, res_k: TorusPrecision) -> CKKSLayout {
    let (da, db) = (left.log_delta(), right.log_delta());
    let budget = (left.k().as_usize().saturating_sub(da))
        .min(right.k().as_usize().saturating_sub(db))
        .saturating_sub(da.max(db));
    let mut block = left;
    block.glwe_layout.k = res_k.as_usize().min(da.min(db) + budget).into();
    block.meta.slots = left.meta.slots.join(right.meta.slots).join(SlotsKind::Real);
    block.meta.scale = Scale::Log(da.min(db));
    block.meta.log_sparsity = left.log_sparsity().min(right.log_sparsity());
    block
}

/// The layout [`crate::api::BlasMatMulOps::blas_matmul`] stamps on `res` for factors `a` and `b` under `ctx`: the
/// terms' product ([`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout),
/// [`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout)) in blocks of `res`'s width,
/// `a.rows x b.cols` blocks. The record is that of the validator, the larger of the factors' batches. `ctx` is a
/// context or its layout: the query reads the shape of its maps.
pub fn blas_matmul_output_layout(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    let maps = context_shapes(ctx);
    let (left, right) = (left_term_layout(a, &maps).block, right_term_layout(b, &maps).block);
    BlasLayout::new(
        product_block(left, right, res.block_layout().k()),
        a.log_d(),
        a.rows(),
        b.cols(),
    )
}

/// [`blas_matmul_output_layout`] for [`crate::api::BlasMatMulOps::blas_matmul_prepared`], with each side's terms as the
/// product reads them: a cold operand's own, a hot one's under `ctx`. `ctx` is a context or its layout: the query
/// reads the shape of its maps.
pub fn blas_matmul_prepared_output_layout(
    res: &impl BlasGridInfos,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    let maps = OnceCell::new();
    let shapes = || maps.get_or_init(|| context_shapes(ctx));
    let (left, right) = (
        product_terms(a, || left_term_layout(a, shapes())),
        product_terms(b, || right_term_layout(b, shapes())),
    );
    BlasLayout::new(
        product_block(left, right, res.block_layout().k()),
        a.log_d(),
        a.rows(),
        b.cols(),
    )
}

/// The block a product of `a` by the public matrices of `b` leaves in blocks of `res`'s width: `first`'s effect, if
/// any, then that of a block map `map`, at the product's label, [`pt_label`].
fn pt_block<P: LtDiagonalMeta, S: BlasPtSide>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = S>,
    first: Option<&LinearTransformation<P>>,
    map: &LinearTransformation<P>,
) -> CKKSLayout {
    let block = a.block_layout();
    let staged = first.map_or(block, |first| mapped_block_into(block, first, block.k()));
    let mut block = mapped_block_into(staged, map, res.block_layout().k());
    block.meta.log_sparsity = pt_label(b.slots(), a, b.count());
    block
}

/// The layout [`crate::api::BlasMatMulOps::blas_matmul_pt_right`] stamps on `res` for `a` and `b`: `a`'s block through
/// `b`'s first map, if any, and a block map, into `res`'s width, at the label that method documents, `a.rows x b.cols`
/// blocks. The record is `a`'s. `b` is a public operand or its layout: the query reads the shape of its maps.
pub fn blas_matmul_pt_right_output_layout(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Right>,
) -> BlasLayout {
    let maps = operand_shapes(b);
    BlasLayout::new(
        pt_block(res, a, b, maps.first(), maps.at(0, 0)),
        a.log_d(),
        a.rows(),
        b.cols(),
    )
}

/// As [`blas_matmul_pt_right_output_layout`], for [`crate::api::BlasMatMulOps::blas_matmul_pt_left`], `b.rows x a.cols`
/// blocks, with the block maps of the replicated-input choice.
pub fn blas_matmul_pt_left_output_layout(
    res: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Left>,
    a: &impl BlasGridInfos,
) -> BlasLayout {
    let maps = operand_shapes(b);
    let rep = left_takes_replicated(&maps, a);
    BlasLayout::new(
        pt_block(res, a, b, maps.first(), maps.at_for(0, 0, rep)),
        a.log_d(),
        b.rows(),
        a.cols(),
    )
}

/// The keys [`crate::api::BlasMatMulOps::blas_matmul`] looks up for factors `a` and `b` under `ctx`: the terms'
/// rotation keys and the tensor key. The result grid takes no part in them. See [`BlasKeyUses`]. `ctx` is a context
/// or its layout: the query reads the shape of its maps.
pub fn blas_matmul_key_uses<BE: Backend>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, Some(matmul_tensor(a, b, &maps).0), |f| matmul_uses(a, b, &maps, f))
}

/// The keys [`crate::api::BlasMatMulOps::blas_matmul_prepare_left`] looks up for `a` under `ctx`, under either hint.
/// See [`BlasKeyUses`]. `ctx` is a context or its layout: the query reads the shape of its maps.
pub fn blas_matmul_prepare_left_key_uses<BE: Backend>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| left_uses(a, &maps, f))
}

/// [`blas_matmul_prepare_left_key_uses`] for [`crate::api::BlasMatMulOps::blas_matmul_prepare_right`].
pub fn blas_matmul_prepare_right_key_uses<BE: Backend>(
    module: &Module<BE>,
    b: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| right_uses(b, &maps, f))
}

/// The keys [`crate::api::BlasMatMulOps::blas_matmul_prepared`] looks up for the operands `a` and `b` under `ctx`: the
/// terms' rotation keys of each hot side, and the tensor key. See [`BlasKeyUses`]. `ctx` is a context or its
/// layout: the query reads the shape of its maps.
pub fn blas_matmul_prepared_key_uses<BE: Backend>(
    module: &Module<BE>,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, Some(prepared_tensor(a, b, &maps).0), |f| {
        prepared_uses(a, b, &maps, f)
    })
}

/// The keys [`crate::api::BlasMatMulOps::blas_matmul_pt_right`] looks up for `a` into `res`, with the replicated-input
/// choice of `b`'s maps. See [`BlasKeyUses`]. `b` is a public operand or its layout: the query reads the shape of
/// its maps.
pub fn blas_matmul_pt_right_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Right>,
) -> BlasKeyUses {
    let maps = operand_shapes(b);
    BlasKeyUses::collect(module, None, |f| pt_right_uses(res, a, &maps, f))
}

/// The keys [`crate::api::BlasMatMulOps::blas_matmul_pt_left`] looks up for `a` into `res`, with the replicated-input
/// choice of `pt_left`: `b`'s replicated-input maps when `a` is replicated. See [`BlasKeyUses`]. `b` is a public
/// operand or its layout: the query reads the shape of its maps.
pub fn blas_matmul_pt_left_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Left>,
    a: &impl BlasGridInfos,
) -> BlasKeyUses {
    let maps = operand_shapes(b);
    BlasKeyUses::collect(module, None, |f| pt_left_uses(res, &maps, a, f))
}

/// Reference compositions of [`crate::api::BlasMatMulOps`] and the JKLS kernel. Each operation runs its validator, the
/// matching `validate_*` of [`crate::layouts`], before any write, so its errors leave `res`, its batch record and the
/// operands untouched, then asserts its scratch and returns poulpy-ckks's budget errors before any write to `res`; on
/// `Ok`, a product stamps `res` with the batch the validator returns, and a preparation records its source's batch on
/// its operand. The JKLS kernel runs on inputs the prepared product's validator accepted and checks nothing but its
/// scratch. A hot preparation copies its input; a cold one makes its terms through the map hook of [`BlasMatIntraImpl`],
/// block row after block row, each converted once by poulpy-ckks's preparations. The prepared product, and
/// `blas_matmul` on the borrowed grids, make a hot side's terms the same way under their context, a block row of `a` or
/// a block column of `b` at a time, holding, of two hot sides, the smaller one's terms and streaming the other's, and
/// read the terms one output block at a time through the JKLS kernel of [`BlasJklsImpl`]. The products by public
/// matrices evaluate one poulpy-ckks summed linear transformation per output block, under JKLS of the results of the
/// operand's first map, made through the map hook a block row (column) of `a` at a time. They run block after block on
/// one scratch, and the first error stops them.
pub trait BlasMatMulReference<BE: Backend> {
    /// Under `OneShot`, the copy of a block of `a` into a block of its layout. Under `Reuse`, the largest of the map
    /// hook's queries through sigma, from a block row of `a` into blocks of its layout at the width sigma leaves, and
    /// through every shift, from those into blocks of the terms' layout, of the copy of one into a block of the terms'
    /// layout, and of the conversion of a term.
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Fills `res` from `a` under `res`'s hint, with no allocation of `res`'s storage. Under `OneShot`, `ckks_copy` of
    /// every block of `a` into the block of its layout `res` holds, its live limbs, metadata, canonical flag and noise
    /// tag. Under `Reuse`, the terms of every block of `a`, row-major, block row after block row: one hook call
    /// evaluates `S = sigma(a_il)` for every block of the row into a heap block of `a`'s layout at the width sigma
    /// leaves; one hook call evaluates `phi^k(S)` for every `S` and every `k` in `1..d` into heap blocks of the terms'
    /// layout, one baby-step cache per `S`; then `ckks_copy` narrows each `S` into a block of the terms' layout, its
    /// term `0`, exactly: it charges the shift to the budget. Each term is converted once by `ckks_prepare_left_into`
    /// into its place in `res`, and has the layout
    /// [`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout) gives. `res` then reports the
    /// infos [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) gives for `a` under
    /// its hint and `a`'s batch, [`grid_batch`].
    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// As [`Self::blas_matmul_prepare_left_tmp_bytes_reference`], with the right maps
    /// [`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout) reads.
    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGridInfos,
        K: GGLWEInfos;

    /// As [`Self::blas_matmul_prepare_left_reference`], with the right maps
    /// [`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout) reads, converted by
    /// `ckks_prepare_right_into`. For a replicated `b`, whose shifts spend no scale, the copy keeps `S`'s width.
    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// The largest of the JKLS kernel's query for a block of `res` and the `K d` pairs of terms of the operands `a` and
    /// `b`, or of the layouts [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) and
    /// [`blas_matmul_right_operand_layout`](crate::layouts::blas_matmul_right_operand_layout) give, a hot side's terms at
    /// the layout [`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout)
    /// ([`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout)) gives under `ctx`, and for each
    /// hot side, the query of the terms of one of its units, `K` blocks of its source: a block row of `a`, a block column
    /// of `b`.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// For every block `(i, j)` of `res`, one JKLS kernel, [`BlasJklsImpl::blas_jkls_block_impl`], over the `K d` pairs
    /// `(a_il[k], b_lj[k])`, `l`-major: ascending `l`, then ascending `k` within each `l`. A cold side brings its terms.
    /// A hot side's are made under `ctx` as the cold preparation of its source makes them, but converted under
    /// `OneShot`, one unit at a time: a block row of `a` or a block column of `b`, `K` blocks, through two hook calls,
    /// sigma (tau) over the unit, then every shift of every result. Each unit is made once and every term converted
    /// once:
    /// - one hot side streams: each of its units serves its row (column) of `res`, then is dropped, the other side's
    ///   terms read in place;
    /// - two hot sides: one is held and the other streams. For `res` of `m x n` blocks and terms of `La` and `Lb` limbs,
    ///   `a` is held when `(m - 1) La <= (n - 1) Lb`, the fewer limbs for the held terms plus one unit of the other
    ///   side; `b` otherwise, after `a`'s first unit.
    ///
    /// So beside `res` and the operands the heap holds the held terms, if any, in the backend's transform-domain form,
    /// and one unit with its sigma (tau) results; the blocks of `res` are written row by row, or column by column when
    /// `b` streams. poulpy-ckks's budget errors come before any write to `res`: the left terms', then the right terms',
    /// then the first block's ct x ct rule.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared_reference<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        b: &BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync;

    /// `res = a b`: [`Self::blas_matmul_prepared_reference`] of two hot operands of `a` and `b`, bit for bit, reading
    /// the blocks of `a` and `b` in place, without the copies of the hot preparations. It asserts the reference's
    /// composition of the `blas_matmul` query, the largest of both reference hot preparations' queries and the
    /// reference prepared product's on the layouts they give, over the keys it looks up: the provided query under
    /// [`crate::impl_blas_matmul_reference!`], so that query stays exact.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_reference<R, A, B, T, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync;

    /// poulpy-ckks's `ckks_dot_product_ct` query for `terms` pairs, which covers `ckks_dot_product_prepared` on pairs
    /// prepared from terms of these layouts.
    fn blas_jkls_block_tmp_bytes_reference<R, L, Rh, T>(&self, res: &R, lhs: &L, rhs: &Rh, terms: usize, tsk: &T) -> usize
    where
        Self: CKKSDotProductOps<BE>,
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_dot_product_ct_tmp_bytes(terms, res, lhs, rhs, tsk)
    }

    /// The JKLS kernel: `res = sum_t lhs[t] rhs[t]` by poulpy-ckks's `ckks_dot_product_prepared`, which is
    /// `ckks_dot_product_ct` of the terms the pairs were prepared from, bit for bit, with no operand transformed. On
    /// pairs prepared from terms that share one layout per side, it relinearizes once, with the tensor key at
    /// `max(lhs.k, rhs.k)`. On rank 1, with one `base2k` across `res`, the terms and the key, and an offset of at least
    /// `base2k` in poulpy-ckks's ct x ct rule, that is poulpy-core's `glwe_mul_prepared_relinearize_sum`: the pairs'
    /// products are added in the transform domain, in slice order, before `res` is stamped. Otherwise it rounds each
    /// `glwe_tensor_apply_prepared` at that width, sums them exactly, normalizes the sum once, right before the
    /// relinearization reads it, and stamps `res` before relinearizing.
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds less than the query reports for the tensor key it looks up.
    fn blas_jkls_block_reference<R, T>(
        &self,
        res: &mut R,
        lhs: &[&CKKSPreparedLeft<BE>],
        rhs: &[&CKKSPreparedRight<BE>],
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSDotProductOps<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GetTensorKey<BE> + Sync,
    {
        if let (Some(&l), Some(&r)) = (lhs.first(), rhs.first()) {
            let op = "blas_jkls_block";
            let used = KeyLayouts::new(tsk);
            let _ = ensure_tensor_key(op, &used, l.k().max(r.k()), l.rank());
            let bytes = used.max_bytes(|tsk| self.blas_jkls_block_tmp_bytes_reference(&*res, l, r, lhs.len(), tsk));
            assert_scratch(op, bytes, scratch);
        }
        self.ckks_dot_product_prepared(res, lhs, rhs, tsk, scratch)
    }

    /// The baby steps of one block and one summed linear transformation from blocks of `a`'s layout into a block of
    /// `res`'s; under JKLS, from blocks of the layout `b`'s first map leaves, and the map hook's query of that map from
    /// a block row of `a` into such blocks.
    fn blas_matmul_pt_right_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// For every block row `i` of `res`, in order: under JKLS, one map-hook call first evaluates `b`'s first map,
    /// sigma, of every block `a_il` into a heap block `s_il` of the layout it leaves, labelled as `a_il`; then one
    /// baby-step cache per block `a_il` (`s_il` under JKLS), `l < K`, covering the baby steps of every block map of
    /// `b`, then every block `res_ij` is poulpy-ckks's summed linear transformation of the `K` terms `(a_il, b_lj)`
    /// (`(s_il, b_lj)`), in ascending `l`, which shares the giant steps across them, stamped with the label
    /// `min(a.log_sparsity, log2(S / (count d^2)))`, for `S` the slots of a block and `count` that of `b`.
    fn blas_matmul_pt_right_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// As [`Self::blas_matmul_pt_right_tmp_bytes_reference`], under JKLS from a block column of `a`.
    fn blas_matmul_pt_left_tmp_bytes_reference<R, A, K>(&self, res: &R, b: &BlasPtOperand<BE, Left>, a: &A, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res = b a`. For every block column `j` of `res`, in order: under JKLS, one map-hook call first evaluates `b`'s
    /// first map, tau, of every block `a_lj` into a heap block `s_lj` of the layout it leaves, labelled as `a_lj`; then
    /// one baby-step cache per block `a_lj` (`s_lj` under JKLS), `l < K`, covering the baby steps of every block map of
    /// `b`, then every block `res_ij` is poulpy-ckks's summed linear transformation of the `K` terms `(a_lj, b_il)`
    /// (`(s_lj, b_il)`), in ascending `l`, which shares the giant steps across them, stamped with the label of
    /// [`Self::blas_matmul_pt_right_reference`].
    fn blas_matmul_pt_left_reference<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}

impl<BE: Backend> BlasMatMulReference<BE> for Module<BE> {
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk, hint)
    }

    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_matmul_prepare_left(self, &*res, a, ctx, atk)?;
        let convert = |res: &mut CKKSPreparedLeft<BE>, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| {
            self.ckks_prepare_left_into(res, term, s)
        };
        let (first, shift) = (ctx.sigma(), |k| ctx.rotate_cols(k));
        let layout = left_operand_layout(a, &ctx.maps, res.hint());
        prepare(
            self,
            "blas_matmul_prepare_left",
            res,
            layout,
            a,
            first,
            shift,
            atk,
            convert,
            scratch,
        )
    }

    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        let (first, shift) = right_maps(&ctx.maps, replicated::<BE::Ring>(b));
        prepare_tmp_bytes(self, b, first, shift, atk, hint)
    }

    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_matmul_prepare_right(self, &*res, b, ctx, atk)?;
        let convert = |res: &mut CKKSPreparedRight<BE>, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| {
            self.ckks_prepare_right_into(res, term, s)
        };
        let (first, shift) = right_maps(&ctx.maps, replicated::<BE::Ring>(b));
        let layout = right_operand_layout(b, &ctx.maps, res.hint());
        prepare(
            self,
            "blas_matmul_prepare_right",
            res,
            layout,
            b,
            first,
            shift,
            atk,
            convert,
            scratch,
        )
    }

    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        kernel_tmp_bytes(self, res, a, b, ctx, tsk).max(hot_tmp_bytes(self, a, b, ctx, atk))
    }

    fn blas_matmul_prepared_reference<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        b: &BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_prepared";
        let batch = validate_blas_matmul_prepared(self, &*res, a, b, ctx, tsk, atk)?;
        assert_scratch(op, product_scratch(self, op, &*res, a, b, ctx, tsk, atk), scratch);
        let b_rep = replicated::<BE::Ring>(b);
        product(
            self,
            res,
            a.side(),
            b.side(),
            a.cols(),
            a.log_d(),
            ctx,
            b_rep,
            tsk,
            atk,
            scratch,
        )?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_reference<R, A, B, T, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul";
        let batch = validate_blas_matmul(self, &*res, a, b, ctx, tsk, atk)?;
        let hot = PrepareHint::OneShot;
        let (left, right) = (
            left_operand_layout(a, &ctx.maps, hot),
            right_operand_layout(b, &ctx.maps, hot),
        );
        let bytes = product_scratch(self, op, &*res, &left, &right, ctx, tsk, atk)
            .max(copy_tmp_bytes(self, a))
            .max(copy_tmp_bytes(self, b));
        assert_scratch(op, bytes, scratch);
        let (a_side, b_side) = (Side::Hot(a.blocks()), Side::Hot(b.blocks()));
        let b_rep = replicated::<BE::Ring>(b);
        product(self, res, a_side, b_side, a.cols(), a.log_d(), ctx, b_rep, tsk, atk, scratch)?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_pt_right_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        pt_tmp_bytes(self, res, a, b.first(), a.cols(), atk)
    }

    fn blas_matmul_pt_right_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_pt_right";
        let batch = validate_blas_matmul_pt_right(self, &*res, a, b, atk)?;
        let (inner, n) = (a.cols(), b.cols());
        let maps: Vec<_> = (0..inner).flat_map(|l| (0..n).map(move |j| b.maps.at(l, j))).collect();
        let used = KeyLayouts::new(atk);
        let _ = pt_right_uses(&*res, a, &b.maps, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_matmul_pt_right_tmp_bytes_reference(&*res, a, b, key));
        assert_scratch(op, bytes, scratch);
        let (blocks, label) = (a.blocks(), pt_label(b.slots(), a, b.count()));
        let rows: Vec<Vec<_>> = res.blocks_mut().chunks_mut(n).map(|row| row.iter_mut().collect()).collect();
        let map = |_: usize, j: usize, l: usize| b.maps.at(l, j);
        match b.first() {
            None => {
                let mut caches = baby_caches(self, &maps, &blocks[0], inner);
                let source = |i: usize, l: usize| &blocks[i * inner + l];
                shared_giants(self, &mut caches, rows, source, map, label, atk, scratch)?;
            }
            Some(first) => {
                let mut staged = alloc_blocks(self, &mapped_block(layout_of(&blocks[0]), first), inner);
                let mut caches = baby_caches(self, &maps, &staged[0], inner);
                for (i, row) in rows.into_iter().enumerate() {
                    let unit: Vec<_> = blocks[i * inner..(i + 1) * inner].iter().collect();
                    BE::blas_map_blocks_impl(self, &mut staged, &unit, &[first], atk, scratch)?;
                    shared_giants(self, &mut caches, vec![row], |_, l| &staged[l], map, label, atk, scratch)?;
                }
            }
        }
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_pt_left_tmp_bytes_reference<R, A, K>(&self, res: &R, b: &BlasPtOperand<BE, Left>, a: &A, atk: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        pt_tmp_bytes(self, res, a, b.first(), a.rows(), atk)
    }

    fn blas_matmul_pt_left_reference<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE> + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_pt_left";
        let batch = validate_blas_matmul_pt_left(self, &*res, b, a, atk)?;
        let (inner, n) = (a.rows(), a.cols());
        let rep = left_takes_replicated(&b.maps, a);
        let maps: Vec<_> = (0..b.rows())
            .flat_map(|i| (0..inner).map(move |l| b.maps.at_for(i, l, rep)))
            .collect();
        let used = KeyLayouts::new(atk);
        let _ = pt_left_uses(&*res, &b.maps, a, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_matmul_pt_left_tmp_bytes_reference(&*res, b, a, key));
        assert_scratch(op, bytes, scratch);
        let mut columns: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
        for (t, block) in res.blocks_mut().iter_mut().enumerate() {
            columns[t % n].push(block);
        }
        let (blocks, label) = (a.blocks(), pt_label(b.slots(), a, b.count()));
        let map = |_: usize, i: usize, l: usize| b.maps.at_for(i, l, rep);
        match b.first() {
            None => {
                let mut caches = baby_caches(self, &maps, &blocks[0], inner);
                let source = |j: usize, l: usize| &blocks[l * n + j];
                shared_giants(self, &mut caches, columns, source, map, label, atk, scratch)?;
            }
            Some(first) => {
                let mut staged = alloc_blocks(self, &mapped_block(layout_of(&blocks[0]), first), inner);
                let mut caches = baby_caches(self, &maps, &staged[0], inner);
                for (j, column) in columns.into_iter().enumerate() {
                    let unit: Vec<_> = (0..inner).map(|l| &blocks[l * n + j]).collect();
                    BE::blas_map_blocks_impl(self, &mut staged, &unit, &[first], atk, scratch)?;
                    shared_giants(self, &mut caches, vec![column], |_, l| &staged[l], map, label, atk, scratch)?;
                }
            }
        }
        res.set_mat_slots(batch);
        Ok(())
    }
}

impl<BE: Backend, P> BlasMatMulOperand<BE, P> {
    /// The side the operand is in a product: its source blocks when hot, its terms when cold.
    fn side(&self) -> Side<'_, CKKSCiphertextOwned<BE>, P> {
        match self.layout.hint {
            PrepareHint::OneShot => Side::Hot(&self.blocks),
            PrepareHint::Reuse => Side::Cold(&self.terms),
        }
    }
}

/// One side of a product: a hot side's source blocks, row-major, whose terms the product makes under its context, or a
/// cold side's terms, block `(i, j)`'s term `k` at `(i * cols + j) * d + k`.
enum Side<'a, S, P> {
    Hot(&'a [S]),
    Cold(&'a [P]),
}

/// The prepared product's query on the layouts of the keys it looks up in `tsk` and `atk`, for `res` and the operands,
/// or their layouts, `a` and `b` under `ctx`: the tensor key's for the kernel, a hot side's rotation keys for its terms.
/// The keys passed `op`'s validator, so the look-ups only record their layouts; a call that looks up none asks for `0`.
#[allow(clippy::too_many_arguments)]
fn product_scratch<BE, T, K>(
    module: &Module<BE>,
    op: &'static str,
    res: &impl BlasGridInfos,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    tsk: &T,
    atk: &K,
) -> usize
where
    BE: BlasJklsImpl + BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE>,
    T: GetTensorKey<BE>,
    K: GetAutomorphismKey<BE>,
{
    let (tsks, atks) = (KeyLayouts::new(tsk), KeyLayouts::new(atk));
    let (k, rank) = prepared_tensor(a, b, &ctx.maps);
    let _ = ensure_tensor_key(op, &tsks, k, rank);
    let _ = prepared_uses(a, b, &ctx.maps, &mut |u| ensure_rotation(op, module, &atks, u));
    tsks.max_bytes(|tsk| kernel_tmp_bytes(module, res, a, b, ctx, tsk))
        .max(atks.max_bytes(|atk| hot_tmp_bytes(module, a, b, ctx, atk)))
}

/// `res = a b` under `ctx`, `res` of `m x n` blocks, for the left side `a` of `m x K` blocks and the right side `b` of
/// `K x n`, `K = inner`, as [`BlasMatMulReference::blas_matmul_prepared_reference`] states: one JKLS kernel per block
/// `(i, j)` of `res` over the `K d` pairs `(a_il[k], b_lj[k])`, `l`-major, a hot side's terms made one unit at a time,
/// a block row of `a` or a block column of `b`, by [`unit_terms`] under `OneShot`, a hot `b`'s through [`right_maps`]
/// of `b_rep`, whether `b` is replicated. Each unit is made once, and `a`'s first before any of `b`'s, so the budget
/// errors come in the order the contract fixes, before any write to `res`: of two hot sides, the held side's units come
/// first, after `a`'s first when `b` is held.
#[allow(clippy::too_many_arguments)]
fn product<BE, R, SA, SB, T, K>(
    module: &Module<BE>,
    res: &mut R,
    a: Side<'_, SA, CKKSPreparedLeft<BE>>,
    b: Side<'_, SB, CKKSPreparedRight<BE>>,
    inner: usize,
    log_d: usize,
    ctx: &BlasContext<BE>,
    b_rep: bool,
    tsk: &T,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasJklsImpl + BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE>,
    R: BlasGridMut<BE>,
    SA: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    SB: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    T: GetTensorKey<BE> + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (m, n, d, hot) = (res.rows(), res.cols(), 1usize << log_d, PrepareHint::OneShot);
    let (b_first, b_shift) = right_maps(&ctx.maps, b_rep);
    // The terms of block row `i` of a hot `a`, and of block column `j` of a hot `b`, `l`-major.
    let a_row = |blocks: &[SA], i: usize, scratch: &mut ScratchArena<'_, BE>| {
        let sources: Vec<_> = blocks[i * inner..(i + 1) * inner].iter().collect();
        let convert = |term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| module.ckks_prepare_left(term, hot, s);
        new_unit_terms(
            module,
            &sources,
            log_d,
            ctx.sigma(),
            |k| ctx.rotate_cols(k),
            atk,
            convert,
            scratch,
        )
    };
    let b_col = |blocks: &[SB], j: usize, scratch: &mut ScratchArena<'_, BE>| {
        let sources: Vec<_> = (0..inner).map(|l| &blocks[l * n + j]).collect();
        let convert = |term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| module.ckks_prepare_right(term, hot, s);
        new_unit_terms(module, &sources, log_d, b_first, b_shift, atk, convert, scratch)
    };
    let mut jkls =
        |i: usize, j: usize, lhs: &[&CKKSPreparedLeft<BE>], rhs: &[&CKKSPreparedRight<BE>], s: &mut ScratchArena<'_, BE>| {
            BE::blas_jkls_block_impl(module, &mut res.blocks_mut()[i * n + j], lhs, rhs, tsk, s)
        };
    match (a, b) {
        (Side::Cold(at), Side::Cold(bt)) => {
            for i in 0..m {
                let lhs = cold_row(at, i, inner, d);
                for j in 0..n {
                    jkls(i, j, &lhs, &cold_col(bt, j, inner, n, d), scratch)?;
                }
            }
        }
        (Side::Hot(ab), Side::Cold(bt)) => {
            for i in 0..m {
                let row = a_row(ab, i, scratch)?;
                let lhs: Vec<_> = row.iter().collect();
                for j in 0..n {
                    jkls(i, j, &lhs, &cold_col(bt, j, inner, n, d), scratch)?;
                }
            }
        }
        (Side::Cold(at), Side::Hot(bb)) => {
            for j in 0..n {
                let col = b_col(bb, j, scratch)?;
                let rhs: Vec<_> = col.iter().collect();
                for i in 0..m {
                    jkls(i, j, &cold_row(at, i, inner, d), &rhs, scratch)?;
                }
            }
        }
        (Side::Hot(ab), Side::Hot(bb)) => {
            // Held terms stay in the transform domain: as coefficients, each streamed unit would transform them again.
            let limbs = |block: CKKSLayout, first, shift| mapped_block(mapped_block(block, first), shift).size();
            let la = limbs(layout_of(&ab[0]), ctx.sigma(), ctx.rotate_cols(1));
            let lb = limbs(layout_of(&bb[0]), b_first, b_shift(1));
            if (m - 1) * la <= (n - 1) * lb {
                let held = (0..m).map(|i| a_row(ab, i, scratch)).collect::<Result<Vec<_>>>()?;
                let lhs: Vec<Vec<_>> = held.iter().map(|row| row.iter().collect()).collect();
                for j in 0..n {
                    let col = b_col(bb, j, scratch)?;
                    let rhs: Vec<_> = col.iter().collect();
                    for (i, lhs) in lhs.iter().enumerate() {
                        jkls(i, j, lhs, &rhs, scratch)?;
                    }
                }
            } else {
                let mut first = Some(a_row(ab, 0, scratch)?);
                let held = (0..n).map(|j| b_col(bb, j, scratch)).collect::<Result<Vec<_>>>()?;
                let rhs: Vec<Vec<_>> = held.iter().map(|col| col.iter().collect()).collect();
                for i in 0..m {
                    let row = match first.take() {
                        Some(row) => row,
                        None => a_row(ab, i, scratch)?,
                    };
                    let lhs: Vec<_> = row.iter().collect();
                    for (j, rhs) in rhs.iter().enumerate() {
                        jkls(i, j, &lhs, rhs, scratch)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// The terms of block row `i` of a cold left side of `inner` block columns and `d` terms per block, `l`-major.
fn cold_row<P>(terms: &[P], i: usize, inner: usize, d: usize) -> Vec<&P> {
    terms[i * inner * d..(i + 1) * inner * d].iter().collect()
}

/// The terms of block column `j` of a cold right side of `inner` block rows, `n` block columns and `d` terms per block,
/// `l`-major.
fn cold_col<P>(terms: &[P], j: usize, inner: usize, n: usize, d: usize) -> Vec<&P> {
    (0..inner)
        .flat_map(|l| &terms[(l * n + j) * d..(l * n + j + 1) * d])
        .collect()
}

/// The JKLS kernel's query for a block of `res` and the `K d` pairs of the terms of `a` and `b` in a product under `ctx`.
fn kernel_tmp_bytes<BE: BlasJklsImpl>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    tsk: &impl GGLWEInfos,
) -> usize {
    let (res, lhs, rhs) = (
        res.block_layout(),
        product_terms(a, || left_term_layout(a, &ctx.maps)),
        product_terms(b, || right_term_layout(b, &ctx.maps)),
    );
    BE::blas_jkls_block_tmp_bytes_impl(module, &res, &lhs, &rhs, a.cols() << a.log_d(), tsk)
}

/// The scratch of the terms of the hot sides of `a` and `b` in a product under `ctx`, a unit at a time, `K` blocks each:
/// a block row of `a`, a block column of `b`; `0` without a hot side.
fn hot_tmp_bytes<BE>(
    module: &Module<BE>,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    atk: &impl GGLWEInfos,
) -> usize
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE>,
{
    let left = match a.hint() {
        PrepareHint::OneShot => unit_tmp_bytes(module, a, ctx.sigma(), |k| ctx.rotate_cols(k), atk, a.cols()),
        PrepareHint::Reuse => 0,
    };
    let right = match b.hint() {
        PrepareHint::OneShot => {
            let (first, shift) = right_maps(&ctx.maps, replicated::<BE::Ring>(b));
            unit_tmp_bytes(module, b, first, shift, atk, b.rows())
        }
        PrepareHint::Reuse => 0,
    };
    left.max(right)
}

/// Scratch of a hot preparation of `a`: the copy of a block of `a` into a block of its layout.
fn copy_tmp_bytes<BE: BlasMatIntraImpl>(module: &Module<BE>, a: &impl BlasGridInfos) -> usize {
    let block = a.block_layout();
    module.ckks_copy_tmp_bytes(&block, &block)
}

/// Scratch of a preparation of `a` under `hint` with `first` (sigma or tau) and `shift(k)` (phi^k or psi^k): the copy
/// of a block of `a` into a block of its layout when hot, the query of one block row's terms when cold.
fn prepare_tmp_bytes<'c, BE, K>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    hint: PrepareHint,
) -> usize
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE>,
    K: GGLWEInfos,
{
    match hint {
        PrepareHint::OneShot => copy_tmp_bytes(module, a),
        PrepareHint::Reuse => unit_tmp_bytes(module, a, first, shift, atk, a.cols()),
    }
}

/// Scratch of the terms of `sources` blocks of `a`'s layout through `first` and `shift(k)`, each converted once: the
/// largest of the map hook's queries through `first`, from those blocks into blocks of the layout it leaves, and
/// through every shift, from those into blocks of the terms' layout, of the copy of one into a block of the terms'
/// layout, and of the conversion of a term.
fn unit_tmp_bytes<'c, BE, K>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    sources: usize,
) -> usize
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE>,
    K: GGLWEInfos,
{
    let block = a.block_layout();
    let s = mapped_block(block, first);
    let term = mapped_block(s, shift(1));
    let shifts: Vec<_> = (1..1usize << a.log_d()).map(shift).collect();
    BE::blas_map_blocks_tmp_bytes_impl(module, &s, &block, sources, &[first], atk)
        .max(BE::blas_map_blocks_tmp_bytes_impl(module, &term, &s, sources, &shifts, atk))
        .max(module.ckks_copy_tmp_bytes(&term, &s))
        .max(module.ckks_prepare_tmp_bytes(&term))
}

/// Fills `res` from `a` under its hint: hot, `ckks_copy` of every block of `a` into `res`'s; cold, the terms of every
/// block, block row after block row ([`unit_terms`]), each converted once by `convert` into its place in `res`. `res`
/// then reports `layout`, the infos of this preparation, and `a`'s batch.
#[allow(clippy::too_many_arguments)]
fn prepare<'c, BE, A, K, P>(
    module: &Module<BE>,
    op: &'static str,
    res: &mut BlasMatMulOperand<BE, P>,
    layout: BlasMatMulOperandLayout,
    a: &A,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    convert: impl Fn(&mut P, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    match res.layout.hint {
        PrepareHint::OneShot => {
            assert_scratch(op, copy_tmp_bytes(module, a), scratch);
            for (block, source) in res.blocks.iter_mut().zip(a.blocks()) {
                module.ckks_copy(block, source, scratch)?;
            }
        }
        PrepareHint::Reuse => {
            let used = KeyLayouts::new(atk);
            let _ = ensure_term_keys(op, module, a, first, &shift, &used);
            let bytes = used.max_bytes(|key| prepare_tmp_bytes(module, a, first, &shift, key, PrepareHint::Reuse));
            assert_scratch(op, bytes, scratch);
            let row_terms = a.cols() << a.log_d();
            for (row, out) in a.blocks().chunks(a.cols()).zip(res.terms.chunks_mut(row_terms)) {
                let sources: Vec<_> = row.iter().collect();
                let emit = |t: usize, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| convert(&mut out[t], term, s);
                unit_terms(module, &sources, a.log_d(), first, &shift, atk, emit, scratch)?;
            }
        }
    }
    res.layout = layout;
    res.mat_slots = grid_batch::<BE::Ring>(a);
    Ok(())
}

/// `count` ciphertexts of layout `layout`.
fn alloc_blocks<BE: BlasMatIntraImpl>(module: &Module<BE>, layout: &CKKSLayout, count: usize) -> Vec<CKKSCiphertextOwned<BE>> {
    (0..count).map(|_| module.ckks_ciphertext_alloc_from_infos(layout)).collect()
}

/// The terms of `sources`, blocks of one layout, through the map hook: `S = first(s)` for every source into a heap block
/// at the width `first` leaves, then `shift(k)` of every `S` for every `k` in `1..d` into heap blocks of the terms' layout,
/// then each `S` narrowed by `ckks_copy` into a block of the terms' layout, its term `0`, and dropped. Each term is passed
/// once to `emit` with its index, term `k` of source `t` at `t * d + k`, then dropped.
#[allow(clippy::too_many_arguments)]
fn unit_terms<'c, BE, S, K>(
    module: &Module<BE>,
    sources: &[&S],
    log_d: usize,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    mut emit: impl FnMut(usize, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl + 'c,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (block, d) = (layout_of(sources[0]), 1usize << log_d);
    let staged = mapped_block(block, first);
    let term = mapped_block(staged, shift(1));
    let mut sigma = alloc_blocks(module, &staged, sources.len());
    BE::blas_map_blocks_impl(module, &mut sigma, sources, &[first], atk, scratch)?;
    let (shifts, results): (Vec<_>, Vec<_>) = ((1..d).map(&shift).collect(), sigma.iter().collect());
    let mut shifted = alloc_blocks(module, &term, sigma.len() * (d - 1));
    BE::blas_map_blocks_impl(module, &mut shifted, &results, &shifts, atk, scratch)?;
    let mut shifted = shifted.into_iter();
    for (t, s) in sigma.into_iter().enumerate() {
        let mut narrowed = module.ckks_ciphertext_alloc_from_infos(&term);
        module.ckks_copy(&mut narrowed, &s, scratch)?;
        drop(s);
        emit(t * d, &narrowed, scratch)?;
        for (k, shifted) in shifted.by_ref().take(d - 1).enumerate() {
            emit(t * d + k + 1, &shifted, scratch)?;
        }
    }
    Ok(())
}

/// [`unit_terms`] of `sources`, each term converted once by `convert` into a new operand, in their order.
#[allow(clippy::too_many_arguments)]
fn new_unit_terms<'c, BE, S, K, P>(
    module: &Module<BE>,
    sources: &[&S],
    log_d: usize,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    atk: &K,
    convert: impl Fn(&CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<P>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<P>>
where
    BE: BlasMatIntraImpl + 'c,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let mut terms = Vec::with_capacity(sources.len() << log_d);
    let emit = |_: usize, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| {
        terms.push(convert(term, s)?);
        Ok(())
    };
    unit_terms(module, sources, log_d, first, shift, atk, emit, scratch)?;
    Ok(terms)
}

/// Scratch of a product by public matrices of blocks of layout `a` into blocks of layout `res`, `sources` blocks of `a`
/// per unit: the baby steps of one block and one summed linear transformation, from the blocks `first` leaves, and
/// under JKLS the map hook's query of `first` over a unit.
fn pt_tmp_bytes<BE: BlasMatIntraImpl>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    first: Option<&LinearTransformationPrepared<BE>>,
    sources: usize,
    atk: &impl GGLWEInfos,
) -> usize
where
    Module<BE>: CKKSLinearTransformationOps<BE>,
{
    let (res, block) = (res.block_layout(), a.block_layout());
    match first {
        None => lt_tmp_bytes(module, &res, &block, atk, true),
        Some(first) => {
            let staged = mapped_block(block, first);
            BE::blas_map_blocks_tmp_bytes_impl(module, &staged, &block, sources, &[first], atk)
                .max(lt_tmp_bytes(module, &res, &staged, atk, true))
        }
    }
}

/// `count` baby-step caches for sources of `src`'s layout, each covering the baby steps of every map of `maps`.
fn baby_caches<BE, M, S>(
    module: &M,
    maps: &[&LinearTransformationPrepared<BE>],
    src: &S,
    count: usize,
) -> Vec<LinearTransformationBabySteps<BE>>
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let rotations: Vec<i64> = maps.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
    (0..count)
        .map(|_| LinearTransformationBabySteps::alloc(module, &rotations, src))
        .collect()
}

/// The products by public matrices, which share their giant steps across their terms, one per cache of `caches`
/// ([`baby_caches`]): group `g` of `groups` holds the blocks `o` of `res` it writes, block `o` receiving poulpy-ckks's
/// summed linear transformation of the terms `(source(g, l), map(g, o, l))` in ascending `l`, then `label`, the
/// product's [`pt_label`]. The caches are filled from a group's sources and serve all its blocks. The groups run in
/// order, refilling the same caches, and the first error stops them.
#[allow(clippy::too_many_arguments)]
fn shared_giants<'s, 'm, BE, M, D, S, K>(
    module: &M,
    caches: &mut [LinearTransformationBabySteps<BE>],
    groups: Vec<Vec<&mut D>>,
    source: impl Fn(usize, usize) -> &'s S,
    map: impl Fn(usize, usize, usize) -> &'m LinearTransformationPrepared<BE>,
    label: usize,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + 's + 'm,
    M: CKKSLinearTransformationOps<BE>,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + 's,
    K: GetAutomorphismKey<BE>,
{
    let inner = caches.len();
    let (mut srcs, mut lts) = (Vec::with_capacity(inner), Vec::with_capacity(inner));
    for (g, blocks) in groups.into_iter().enumerate() {
        for (l, cache) in caches.iter_mut().enumerate() {
            module.ckks_prepare_linear_transformation_baby_steps(cache, source(g, l), atk, scratch)?;
        }
        let babies: Vec<_> = caches.iter().collect();
        srcs.clear();
        srcs.extend((0..inner).map(|l| source(g, l)));
        for (o, dst) in blocks.into_iter().enumerate() {
            lts.clear();
            lts.extend((0..inner).map(|l| map(g, o, l)));
            module.ckks_eval_linear_transformation_sum_into(&mut *dst, &srcs, &babies, &lts, atk, scratch)?;
            dst.set_log_sparsity(label);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use poulpy_ckks::{CoeffsMeta, layouts::CKKSModuleAlloc};
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::layouts::HostBytesBackend;

    use super::*;
    use crate::layouts::{BlasPlan, BlasPtOperandLayout};

    /// One block of `layout` recording `mat_slots` matrices.
    struct Recorded(BlasLayout, usize);

    impl BlasGridInfos for Recorded {
        fn log_d(&self) -> usize {
            self.0.log_d
        }

        fn rows(&self) -> usize {
            self.0.rows
        }

        fn cols(&self) -> usize {
            self.0.cols
        }

        fn block_layout(&self) -> CKKSLayout {
            self.0.block
        }

        fn mat_slots(&self) -> Option<usize> {
            Some(self.1)
        }
    }

    fn block(module: &Module<HostBytesBackend>, k: usize, log_delta: usize) -> CKKSLayout {
        let mut block = layout_of(&module.ckks_ciphertext_alloc(19usize.into(), k.into()));
        block.meta.scale = Scale::Log(log_delta);
        block
    }

    #[test]
    fn pt_output_layouts_stamp_the_label_of_the_public_matrices() {
        // 128 slots: eight matrix slots of `4 x 4` blocks; label 3 proves one matrix, the record holds eight.
        let module = Module::<HostBytesBackend>::new(256);
        let mut block = block(&module, 60, 20);
        block.meta.log_sparsity = 3;
        let a = Recorded(BlasLayout::new(block, 2, 1, 1), 8);
        for (count, label) in [(1, 3), (2, 2), (8, 0)] {
            let meta = CoeffsMeta::from_delta_budget(20, 19);
            let right = BlasPtOperandLayout::<Right>::new(BlasPlan::new(2), 128, 1, 1, count, Base2K(19), meta);
            let left = BlasPtOperandLayout::<Left>::new(BlasPlan::new(2), 128, 1, 1, count, Base2K(19), meta);
            assert_eq!(
                blas_matmul_pt_right_output_layout(&a, &a, &right).block.log_sparsity(),
                label,
                "count={count}"
            );
            assert_eq!(
                blas_matmul_pt_left_output_layout(&a, &left, &a).block.log_sparsity(),
                label,
                "count={count}"
            );
        }
    }

    #[test]
    fn product_block_saturates_an_exhausted_budget() {
        let module = Module::<HostBytesBackend>::new(256);
        let res_k = TorusPrecision(90);
        assert_eq!(
            product_block(block(&module, 90, 30), block(&module, 90, 30), res_k)
                .k()
                .as_usize(),
            60
        );
        assert_eq!(
            product_block(block(&module, 40, 30), block(&module, 90, 30), res_k)
                .k()
                .as_usize(),
            30
        );
    }
}
