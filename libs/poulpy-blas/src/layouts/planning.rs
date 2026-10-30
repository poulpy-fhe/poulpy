//! The planning queries of every operation, from layouts alone: the keys it looks up ([`BlasKeyUses`]), the layout it
//! stamps on `res`, and the largest radix of a product.

use std::cell::OnceCell;

use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSResult as Result, GLWEPlaintextInfos, Scale, SlotsKind,
    api::{CKKSModuleInfos, LinearTransformation, LtDiagonalMeta, PreparedDiagonal, ckks_linear_transformation_output_k},
};
use poulpy_core::layouts::{LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, Data, MaxBase2k, Module};

use super::{
    BlasContextInfos, BlasGridInfos, BlasLayout, BlasMatMulOperandInfos, BlasMatVecOperandInfos, BlasPlan, BlasPtOperandInfos,
    BlasPtSide, BlasVectorInfos, BlasVectorLayout, Left, Right,
    ciphertext::{mat_slots_in, replicated_at},
    context::{ContextMaps, context_shapes, reduction_projection, shift_maps, shift_rotates},
    encoding::{DiagonalShape, copied_block, mapped_block, mapped_block_into, shape},
    matmul_operand::{left_term_layout, product_terms, right_maps, right_term_layout},
    matvec_operand::{map_layout, output_label},
    pt_operand::{OperandMaps, layouts_of},
    validate::{
        MAX_BASE2K, UseSink, context_masks, left_takes_replicated, left_uses, matmul_tensor, matmul_uses, matvec_pt_uses,
        prepared_tensor, prepared_uses, pt_left_uses, pt_right_uses, reduction_uses, right_uses, rotate_uses, shift_uses,
        sum_uses, transpose_uses,
    },
};

/// The keys an operation looks up, by the widths it looks them up at: the Galois elements that rotating by each
/// rotation of its linear transformations, rotations and sums takes, with the width of the input it key-switches, the
/// identity (element 1) which poulpy-ckks applies without a key left out, and the width of the tensor key where it
/// relinearizes. These are the very look-ups its validator makes, in sets: pairs sorted by element then width, with no
/// repeats.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlasKeyUses {
    /// `(Galois element, width)` of every rotation key looked up.
    pub rotations: Vec<(i64, TorusPrecision)>,
    /// The width of the tensor key looked up, if the operation relinearizes.
    pub tensor: Option<TorusPrecision>,
}

impl BlasKeyUses {
    /// Collects the look-ups an enumerator makes, and the tensor key's width.
    fn collect<BE: Backend>(
        module: &Module<BE>,
        tensor: Option<TorusPrecision>,
        uses: impl FnOnce(UseSink<'_>) -> Result<()>,
    ) -> Self {
        let mut rotations = Vec::new();
        let _ = uses(&mut |u| {
            rotations.push((module.ckks_galois_element(u.rotation), u.k));
            Ok(())
        });
        rotations.retain(|&(p, _)| p != 1);
        rotations.sort_unstable_by_key(|&(p, k)| (p, k.as_usize()));
        rotations.dedup();
        Self { rotations, tensor }
    }
}

/// The label of a product of `a` by `count` public matrices, with `slots` slots per block, the operand's:
/// `min(a.log_sparsity, log2(slots / (count d^2)))`, the smaller of `a`'s label and the largest one `count` distinct
/// matrices allow, as poulpy-ckks's plaintext product keeps the smaller of its operands' labels; `0` when `count`
/// matrices do not fit.
pub(crate) fn pt_label(slots: usize, a: &impl BlasGridInfos, count: usize) -> usize {
    let block = a.block_layout();
    mat_slots_in(slots, a.log_d(), 0)
        .checked_div(count)
        .and_then(usize::checked_ilog2)
        .map_or(0, |log| block.log_sparsity().min(log as usize))
}

/// A map's diagonal as the queries read it: its scale and slot kind, and its width.
pub(crate) trait MapDiagonal: LtDiagonalMeta {
    fn width(&self) -> TorusPrecision;
}

impl MapDiagonal for DiagonalShape {
    fn width(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, BE: Backend> MapDiagonal for PreparedDiagonal<D, BE> {
    fn width(&self) -> TorusPrecision {
        self.k()
    }
}

/// The maps of a public operand of `infos` as the queries read them, without encoding any.
///
/// # Panics
///
/// As [`crate::layouts::BlasModuleAlloc::blas_pt_operand_alloc`], for a layout that does not fit.
pub(crate) fn operand_shapes<S: BlasPtSide>(
    infos: &impl BlasPtOperandInfos<Side = S>,
) -> OperandMaps<LinearTransformation<DiagonalShape>> {
    layouts_of(&infos.blas_pt_operand_layout()).map(shape)
}

/// `f` of the masks of a context of `ctx` that a product by public matrices of an operand of `plan` evaluates, on the
/// `left` or the right, as the queries read them ([`context_masks`]): none, `ctx` unread, but under `JklsStored` and
/// `JklsGalois`.
///
/// # Panics
///
/// Under those paths, as [`crate::layouts::BlasModuleAlloc::blas_context_alloc`] for a layout that does not fit, and on
/// the `left` for a context whose plan selects neither, which holds no column halves of psi^k.
fn with_mask_shapes<R>(
    plan: BlasPlan,
    ctx: &impl BlasContextInfos,
    left: bool,
    f: impl FnOnce(&[&LinearTransformation<DiagonalShape>]) -> R,
) -> R {
    let maps = plan.pt_algorithm().masked().then(|| context_shapes(ctx));
    f(&maps.as_ref().map_or_else(Vec::new, |maps| context_masks(plan, maps, left)))
}

/// The layout [`crate::api::BlasMatInterOps::blas_rotate`] and [`crate::api::BlasMatInterOps::blas_sum`] stamp on `res`
/// for `a`: `a`'s block copied into `res`'s width, `a`'s dimensions and record.
pub fn blas_rotate_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos) -> BlasLayout {
    BlasLayout::new(
        copied_block(a.block_layout(), res.block_layout().k()),
        a.log_d(),
        a.rows(),
        a.cols(),
    )
}

/// [`blas_rotate_output_layout`], named for [`crate::api::BlasMatInterOps::blas_sum`].
pub fn blas_sum_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos) -> BlasLayout {
    blas_rotate_output_layout(res, a)
}

/// The keys [`crate::api::BlasMatInterOps::blas_rotate`] by `r` looks up for `a` into `res`; see [`BlasKeyUses`].
pub fn blas_rotate_key_uses<BE: Backend>(
    module: &Module<BE>,
    r: i64,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
) -> BlasKeyUses {
    BlasKeyUses::collect(module, None, |f| rotate_uses(r, res, a, f))
}

/// The keys [`crate::api::BlasMatInterOps::blas_sum`] looks up for `a` into `res`; see [`BlasKeyUses`].
pub fn blas_sum_key_uses<BE: Backend>(module: &Module<BE>, res: &impl BlasGridInfos, a: &impl BlasGridInfos) -> BlasKeyUses {
    BlasKeyUses::collect(module, None, |f| sum_uses::<BE::Ring>(res, a, f))
}

/// The layout [`crate::api::BlasMatIntraOps::blas_transpose`] stamps on `res` for `a`: `a`'s block through the transpose
/// map into `res`'s width, `a.cols x a.rows` blocks. The record is `a`'s. `ctx` is a context or its layout: the
/// query reads the shape of its maps.
///
/// # Panics
///
/// On a `ctx` layout no context is allocated at, which
/// [`validate_blas_transpose`](crate::layouts::validate_blas_transpose) therefore never receives: blocks that do not
/// fit its slots, maps without a torus width, or a radix outside `1..=63`.
pub fn blas_transpose_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    let block = mapped_block_into(a.block_layout(), context_shapes(ctx).transpose(), res.block_layout().k());
    BlasLayout::new(block, a.log_d(), a.cols(), a.rows())
}

/// The layout of a row (`rows`) or column shift by `k`: a copy of `a`'s block for a shift by a multiple of `d` or one
/// that rotates ([`shift_rotates`]), else its block through the shift's maps ([`shift_maps`], of one scale) into
/// `res`'s width.
fn shift_output<P: LtDiagonalMeta>(
    k: usize,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    rows: bool,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    let extent = if rows { a.rows() } else { a.cols() };
    let lts = if shift_rotates(rows, extent, replicated_at(maps.slots(), a)) {
        Vec::new()
    } else {
        shift_maps(maps, k % (1 << maps.log_d()), rows, extent)
    };
    let block = match lts.first() {
        None => copied_block(block, res_k),
        Some(lt) => mapped_block_into(block, lt, res_k),
    };
    BlasLayout::new(block, a.log_d(), a.rows(), a.cols())
}

/// The layout [`crate::api::BlasMatIntraOps::blas_rotate_rows`] by `k` stamps on `res` for `a`; see
/// [`blas_transpose_output_layout`].
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_rotate_rows`](crate::layouts::validate_blas_rotate_rows).
pub fn blas_rotate_rows_output_layout(
    k: usize,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    shift_output(k, res, a, true, &context_shapes(ctx))
}

/// [`blas_rotate_rows_output_layout`] for [`crate::api::BlasMatIntraOps::blas_rotate_cols`].
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_rotate_cols`](crate::layouts::validate_blas_rotate_cols).
pub fn blas_rotate_cols_output_layout(
    k: usize,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    shift_output(k, res, a, false, &context_shapes(ctx))
}

/// The keys [`crate::api::BlasMatIntraOps::blas_transpose`] looks up for `a` into `res`; see [`BlasKeyUses`]. `ctx`
/// is a context or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_transpose`](crate::layouts::validate_blas_transpose).
pub fn blas_transpose_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| transpose_uses(res, a, &maps, f))
}

/// The keys [`crate::api::BlasMatIntraOps::blas_rotate_rows`] by `k` looks up for `a` into `res`; see [`BlasKeyUses`].
/// The shifts of two values of `k` look up different keys: a client rotating by several takes the union. `ctx` is a
/// context or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_rotate_rows`](crate::layouts::validate_blas_rotate_rows).
pub fn blas_rotate_rows_key_uses<BE: Backend>(
    module: &Module<BE>,
    k: usize,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| shift_uses(k, res, a, true, &maps, f))
}

/// [`blas_rotate_rows_key_uses`] for [`crate::api::BlasMatIntraOps::blas_rotate_cols`].
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_rotate_cols`](crate::layouts::validate_blas_rotate_cols).
pub fn blas_rotate_cols_key_uses<BE: Backend>(
    module: &Module<BE>,
    k: usize,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| shift_uses(k, res, a, false, &maps, f))
}

/// The layout of an axis reduction along the rows (`rows`) or the columns: `a`'s block copied into `res`'s width without
/// a projection ([`reduction_projection`]), else through it into `res`'s width.
fn reduction_output<P: LtDiagonalMeta>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    rows: bool,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    let block = match reduction_projection(maps, rows, replicated_at(maps.slots(), a)) {
        None => copied_block(block, res_k),
        Some(projection) => mapped_block_into(block, projection, res_k),
    };
    BlasLayout::new(block, a.log_d(), a.rows(), a.cols())
}

/// The layout [`crate::api::BlasMatIntraOps::blas_sum_rows`] stamps on `res` for `a`. `ctx` is a context or its layout:
/// the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_sum_rows`](crate::layouts::validate_blas_sum_rows).
pub fn blas_sum_rows_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    reduction_output(res, a, true, &context_shapes(ctx))
}

/// [`blas_sum_rows_output_layout`] for [`crate::api::BlasMatIntraOps::blas_sum_cols`].
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_sum_cols`](crate::layouts::validate_blas_sum_cols).
pub fn blas_sum_cols_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    reduction_output(res, a, false, &context_shapes(ctx))
}

/// The keys [`crate::api::BlasMatIntraOps::blas_sum_rows`] looks up for `a` into `res`; see [`BlasKeyUses`]. `ctx` is a
/// context or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_sum_rows`](crate::layouts::validate_blas_sum_rows).
pub fn blas_sum_rows_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| reduction_uses(res, a, true, &maps, f))
}

/// [`blas_sum_rows_key_uses`] for [`crate::api::BlasMatIntraOps::blas_sum_cols`].
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_sum_cols`](crate::layouts::validate_blas_sum_cols).
pub fn blas_sum_cols_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| reduction_uses(res, a, false, &maps, f))
}

/// The largest radix in `1..=62` at which the model of `BE` holds the `products(r)` limb products that a lazy sum adds
/// in one output limb of degree-`n` polynomials at radix `r`, 0 if none does.
fn max_radix<BE: MaxBase2k>(n: usize, failure_bits: usize, products: impl Fn(usize) -> usize) -> usize {
    // A zero width adds no product, which the model rejects: count one.
    (1..=MAX_BASE2K)
        .rev()
        .find(|&r| Module::<BE>::max_base2k(n, products(r).max(1), failure_bits, false) >= r)
        .unwrap_or(0)
}

/// The largest `base2k`, at most 62, that [`Module::max_base2k`] allows for every lazy sum of a
/// [`crate::api::BlasMatMulOps::blas_matmul`] of `a` by `b` under `ctx`, 0 if none does. The JKLS kernel adds the
/// products of its `K d` pairs of terms in the transform domain, `K = a.cols()` and `d` the block dimension: up to
/// `2 L` limb products per pair in an output limb, `L` the limbs of the narrower terms, `min(w_a, w_b)` over `base2k`
/// rounded up, for the widths of [`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout) and
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
/// `failure_bits` is positive; and as [`blas_matmul_output_layout`] on `ctx`.
pub fn blas_matmul_max_base2k<BE: MaxBase2k>(
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
    failure_bits: usize,
) -> usize {
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
/// `b`'s replicated-input ones when `rep`, under the masked paths through the context's `masks`.
fn pt_max_base2k<BE: MaxBase2k, P: MapDiagonal>(
    a: &impl BlasGridInfos,
    b: &OperandMaps<LinearTransformation<P>>,
    masks: &[&LinearTransformation<P>],
    inner: usize,
    rep: bool,
    failure_bits: usize,
) -> usize {
    let (block, maps) = (
        a.block_layout(),
        map_shape((0..b.rows()).flat_map(|l| (0..b.cols()).map(move |j| b.at_for(l, j, rep)))),
    );
    let a_k = block.k();
    let (maps_k, first) = match b.first() {
        None => (a_k, None),
        Some(first) => (ckks_linear_transformation_output_k(first, a_k, a_k), Some(map_shape([first]))),
    };
    let (terms, src_k, masked) = match masks.first() {
        None => (inner, maps_k, None),
        Some(&mask) => (
            inner * (masks.len() + 1),
            ckks_linear_transformation_output_k(mask, maps_k, maps_k),
            Some(map_shape(masks.iter().copied())),
        ),
    };
    max_radix::<BE>(block.n().as_usize(), failure_bits, |r| {
        let blocks = giant_products(maps, terms, src_k.as_usize(), r);
        let blocks = masked.map_or(blocks, |shape| blocks.max(giant_products(shape, 1, maps_k.as_usize(), r)));
        first.map_or(blocks, |first| blocks.max(giant_products(first, 1, a_k.as_usize(), r)))
    })
}

/// The largest `base2k`, at most 62, that [`Module::max_base2k`] allows for every lazy sum of a
/// [`crate::api::BlasMatMulOps::blas_matmul_pt_right`] of `a` by `b`, 0 if none does: the largest radix at which the
/// model holds the count that method documents, with `L` and `P` counted at each candidate radix, whatever the radix of
/// `a` and `b`. `failure_bits` and the model are those of [`Module::max_base2k`]. `b` is a public operand or its
/// layout: the query reads the shape of its maps. Under `JklsStored` and `JklsGalois` it also reads the shape of
/// `ctx`'s masks, which add one diagonal per giant step from the first map's width, and the final step, poulpy-ckks's
/// summed product by prepared plaintexts, adds `K d` products from the masks' width in the transform domain. `ctx` is a
/// context or its layout, read only then. Name the backend, as [`blas_matmul_max_base2k`] does.
///
/// # Panics
///
/// As [`blas_matmul_max_base2k`] on the degree and `failure_bits`, and as [`blas_matmul_pt_right_output_layout`] on `b`
/// and `ctx`.
pub fn blas_matmul_pt_right_max_base2k<BE: MaxBase2k>(
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Right>,
    ctx: &impl BlasContextInfos,
    failure_bits: usize,
) -> usize {
    let maps = operand_shapes(b);
    with_mask_shapes(b.plan(), ctx, false, |masks| {
        pt_max_base2k::<BE, _>(a, &maps, masks, a.cols(), false, failure_bits)
    })
}

/// [`blas_matmul_pt_right_max_base2k`] for [`crate::api::BlasMatMulOps::blas_matmul_pt_left`]: the terms of a block of
/// `res` number `K = a.rows()`, and the first map under JKLS is tau. The block maps counted are those the product takes
/// for `a`: [`BlasPtOperandPrepared::block_replicated`](super::BlasPtOperandPrepared::block_replicated)'s when `a` is replicated
/// ([`replicated`](super::replicated)) and `b` holds them. Under the masked paths the final step adds `K (2d - 1)`
/// products.
///
/// # Panics
///
/// As [`blas_matmul_max_base2k`] on the degree and `failure_bits`, and as [`blas_matmul_pt_left_output_layout`] on `b`
/// and `ctx`.
pub fn blas_matmul_pt_left_max_base2k<BE: MaxBase2k>(
    b: &impl BlasPtOperandInfos<Side = Left>,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
    failure_bits: usize,
) -> usize {
    let maps = operand_shapes(b);
    let rep = left_takes_replicated(&maps, a);
    with_mask_shapes(b.plan(), ctx, true, |masks| {
        pt_max_base2k::<BE, _>(a, &maps, masks, a.rows(), rep, failure_bits)
    })
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
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_matmul`](crate::layouts::validate_blas_matmul).
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
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for
/// [`validate_blas_matmul_prepared`](crate::layouts::validate_blas_matmul_prepared).
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
/// any, then that of a context mask `mask`, if any, then that of a block map `map`, at the product's label, [`pt_label`].
fn pt_block<P: LtDiagonalMeta, S: BlasPtSide>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = S>,
    first: Option<&LinearTransformation<P>>,
    mask: Option<&LinearTransformation<P>>,
    map: &LinearTransformation<P>,
) -> CKKSLayout {
    let block = a.block_layout();
    let staged = first.map_or(block, |first| mapped_block_into(block, first, block.k()));
    let staged = mask.map_or(staged, |mask| mapped_block(staged, mask));
    let mut block = mapped_block_into(staged, map, res.block_layout().k());
    block.meta.log_sparsity = pt_label(b.slots(), a, b.count());
    block
}

/// The layout [`crate::api::BlasMatMulOps::blas_matmul_pt_right`] stamps on `res` for `a` and `b`: `a`'s block through
/// `b`'s first map, if any, and a block map, into `res`'s width, at the label that method documents, `a.rows x b.cols`
/// blocks. The record is `a`'s. `b` is a public operand or its layout: the query reads the shape of its maps. Under
/// `JklsStored` and `JklsGalois` the block also goes through a mask of `ctx`, at its scale. `ctx` is a context or its
/// layout, read only then.
///
/// # Panics
///
/// On a layout of `b` no operand is allocated at, as [`crate::layouts::BlasModuleAlloc::blas_pt_operand_alloc`], a
/// count that does not divide its matrix slots included, and under `JklsStored` and `JklsGalois` on a `ctx` layout no
/// context is allocated at or at another block dimension than the operand's ("a context at another block dimension than
/// the operand's"), which [`validate_blas_matmul_pt_right`](crate::layouts::validate_blas_matmul_pt_right) rejects.
pub fn blas_matmul_pt_right_output_layout(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Right>,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    let maps = operand_shapes(b);
    let block = with_mask_shapes(b.plan(), ctx, false, |masks| {
        pt_block(res, a, b, maps.first(), masks.first().copied(), maps.at(0, 0))
    });
    BlasLayout::new(block, a.log_d(), a.rows(), b.cols())
}

/// As [`blas_matmul_pt_right_output_layout`], for [`crate::api::BlasMatMulOps::blas_matmul_pt_left`], `b.rows x a.cols`
/// blocks, with the block maps of the replicated-input choice.
///
/// # Panics
///
/// As [`blas_matmul_pt_right_output_layout`], for
/// [`validate_blas_matmul_pt_left`](crate::layouts::validate_blas_matmul_pt_left), and under `JklsStored` and
/// `JklsGalois` on a `ctx` without the column halves of psi^k, whose plan selects neither.
pub fn blas_matmul_pt_left_output_layout(
    res: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Left>,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    let maps = operand_shapes(b);
    let rep = left_takes_replicated(&maps, a);
    let block = with_mask_shapes(b.plan(), ctx, true, |masks| {
        pt_block(res, a, b, maps.first(), masks.first().copied(), maps.at_for(0, 0, rep))
    });
    BlasLayout::new(block, a.log_d(), b.rows(), a.cols())
}

/// The keys [`crate::api::BlasMatMulOps::blas_matmul`] looks up for factors `a` and `b` under `ctx`: the terms'
/// rotation keys and the tensor key. The result grid takes no part in them. See [`BlasKeyUses`]. `ctx` is a context
/// or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for [`validate_blas_matmul`](crate::layouts::validate_blas_matmul).
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
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for
/// [`validate_blas_matmul_prepare_left`](crate::layouts::validate_blas_matmul_prepare_left).
pub fn blas_matmul_prepare_left_key_uses<BE: Backend>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| left_uses(a, &maps, f))
}

/// [`blas_matmul_prepare_left_key_uses`] for [`crate::api::BlasMatMulOps::blas_matmul_prepare_right`].
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for
/// [`validate_blas_matmul_prepare_right`](crate::layouts::validate_blas_matmul_prepare_right).
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
///
/// # Panics
///
/// As [`blas_transpose_output_layout`], for
/// [`validate_blas_matmul_prepared`](crate::layouts::validate_blas_matmul_prepared).
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
/// its maps. Under `JklsStored` and `JklsGalois`, the first map's and `ctx`'s masks', `ctx` a context or its layout,
/// read only then.
///
/// # Panics
///
/// As [`blas_matmul_pt_right_output_layout`].
pub fn blas_matmul_pt_right_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Right>,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = operand_shapes(b);
    with_mask_shapes(b.plan(), ctx, false, |masks| {
        BlasKeyUses::collect(module, None, |f| pt_right_uses(res, a, &maps, masks, f))
    })
}

/// The keys [`crate::api::BlasMatMulOps::blas_matmul_pt_left`] looks up for `a` into `res`, with the replicated-input
/// choice of `pt_left`: `b`'s replicated-input maps when `a` is replicated. See [`BlasKeyUses`]. `b` is a public
/// operand or its layout: the query reads the shape of its maps. Under `JklsStored` and `JklsGalois`, the first map's
/// and `ctx`'s masks', `ctx` a context or its layout, read only then.
///
/// # Panics
///
/// As [`blas_matmul_pt_left_output_layout`].
pub fn blas_matmul_pt_left_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    b: &impl BlasPtOperandInfos<Side = Left>,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = operand_shapes(b);
    with_mask_shapes(b.plan(), ctx, true, |masks| {
        BlasKeyUses::collect(module, None, |f| pt_left_uses(res, &maps, masks, a, f))
    })
}

/// The layout [`crate::api::BlasMatVecOps::blas_matvec_pt`] stamps on `res` for `w` and `a`: `a`'s ciphertext through a
/// map of `w` into `res`'s width, labelled with the output period `w` proves, of `w`'s output shape. `w` is a public
/// operand or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// On a layout of `w` no operand is allocated at, as
/// [`crate::layouts::BlasModuleAlloc::blas_matvec_operand_alloc`].
pub fn blas_matvec_pt_output_layout(
    res: &impl BlasVectorInfos,
    w: &impl BlasMatVecOperandInfos,
    a: &impl BlasVectorInfos,
) -> BlasVectorLayout {
    let map = shape(&map_layout(&w.blas_matvec_operand_layout()));
    let mut ct = mapped_block_into(a.ct_layout(), &map, res.ct_layout().k());
    ct.meta.log_sparsity = output_label(w);
    BlasVectorLayout::new(ct, w.output())
}

/// The keys [`crate::api::BlasMatVecOps::blas_matvec_pt`] looks up for `a` into `res` through `w`: every map's baby
/// steps at `a`'s width and giant steps at the width they leave into `res`. See [`BlasKeyUses`]. `w` is a public operand
/// or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_matvec_pt_output_layout`].
pub fn blas_matvec_pt_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasVectorInfos,
    w: &impl BlasMatVecOperandInfos,
    a: &impl BlasVectorInfos,
) -> BlasKeyUses {
    let layout = w.blas_matvec_operand_layout();
    let map = shape(&map_layout(&layout));
    BlasKeyUses::collect(module, None, |f| {
        matvec_pt_uses(res, std::iter::repeat_n(&map, layout.blocks()), a, f)
    })
}

/// The largest `base2k`, at most 62, that [`Module::max_base2k`] allows for every lazy sum of a
/// [`crate::api::BlasMatVecOps::blas_matvec_pt`] of `w` by `a`, 0 if none does: each giant step adds, over the `K`
/// input ciphertexts, `K g min(L, P)` limb products in an output limb, `g` the most diagonals a giant step of `w`'s maps
/// groups, `L` the limbs of `a.k` and `P` those of a diagonal, counted at each candidate radix. `failure_bits` and the
/// model are those of [`Module::max_base2k`]. Name the backend, as [`blas_matmul_max_base2k`] does.
///
/// # Panics
///
/// As [`blas_matmul_max_base2k`] on the degree and `failure_bits`, and as [`blas_matvec_pt_output_layout`] on `w`.
pub fn blas_matvec_pt_max_base2k<BE: MaxBase2k>(
    w: &impl BlasMatVecOperandInfos,
    a: &impl BlasVectorInfos,
    failure_bits: usize,
) -> usize {
    let map = shape(&map_layout(&w.blas_matvec_operand_layout()));
    let (ct, terms) = (a.ct_layout(), w.input().count());
    max_radix::<BE>(ct.n().as_usize(), failure_bits, |r| {
        giant_products(map_shape([&map]), terms, ct.k().as_usize(), r)
    })
}

#[cfg(test)]
mod tests {
    use poulpy_ckks::{CoeffsMeta, layouts::CKKSModuleAlloc};
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::layouts::{CyclotomicOrder, HostBytesBackend};

    use super::*;
    use crate::layouts::{
        BlasPlan, BlasPtAlgorithm, BlasPtOperandLayout, ciphertext::layout_of, context::module_layout, validate::MASKLESS,
    };

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
        let ctx = module_layout(&module, BlasPlan::new(2), Base2K(19), CoeffsMeta::from_delta_budget(20, 19));
        for (count, label) in [(1, 3), (2, 2), (8, 0)] {
            let meta = CoeffsMeta::from_delta_budget(20, 19);
            let right = BlasPtOperandLayout::<Right>::new(BlasPlan::new(2), 128, 1, 1, count, Base2K(19), meta);
            let left = BlasPtOperandLayout::<Left>::new(BlasPlan::new(2), 128, 1, 1, count, Base2K(19), meta);
            assert_eq!(
                blas_matmul_pt_right_output_layout(&a, &a, &right, &ctx).block.log_sparsity(),
                label,
                "count={count}"
            );
            assert_eq!(
                blas_matmul_pt_left_output_layout(&a, &left, &a, &ctx).block.log_sparsity(),
                label,
                "count={count}"
            );
        }
    }

    /// The masked paths stamp three scales, the operand's twice and the context's masks' once, and look up the keys of
    /// the first map and the masks alone: those of their plan's sets.
    #[test]
    fn masked_pt_queries_read_the_context_masks() {
        let module = Module::<HostBytesBackend>::new(256);
        let a = Recorded(BlasLayout::new(block(&module, 90, 20), 2, 1, 1), 8);
        let meta = CoeffsMeta::from_delta_budget(20, 19);
        let order = module.cyclotomic_order();
        for algorithm in [BlasPtAlgorithm::JklsStored, BlasPtAlgorithm::JklsGalois] {
            let plan = BlasPlan::new(2).with_pt_algorithm(algorithm);
            let ctx = module_layout(&module, plan, Base2K(19), CoeffsMeta::from_delta_budget(15, 19));
            let right = BlasPtOperandLayout::<Right>::new(plan, 128, 1, 1, 1, Base2K(19), meta);
            let left = BlasPtOperandLayout::<Left>::new(plan, 128, 1, 1, 1, Base2K(19), meta);
            // 90 bits through sigma (tau), the masks and the plaintexts: 90 - 20 - 15 - 20.
            assert_eq!(
                blas_matmul_pt_right_output_layout(&a, &a, &right, &ctx).block.k().as_usize(),
                35
            );
            assert_eq!(
                blas_matmul_pt_left_output_layout(&a, &left, &a, &ctx).block.k().as_usize(),
                35
            );
            let elements = |uses: &BlasKeyUses| {
                let mut elements: Vec<i64> = uses.rotations.iter().map(|&(p, _)| p).collect();
                elements.dedup();
                elements
            };
            let right_uses = blas_matmul_pt_right_key_uses(&module, &a, &a, &right, &ctx);
            let left_uses = blas_matmul_pt_left_key_uses(&module, &a, &left, &a, &ctx);
            assert_eq!(elements(&right_uses), plan.matmul_pt_right_galois_elements(order));
            assert_eq!(elements(&left_uses), plan.matmul_pt_left_galois_elements(order));
            // The first map's baby steps at 90 and giant steps at 70, the masks' at 90 - 20 - 15.
            for uses in [&right_uses, &left_uses] {
                let mut widths: Vec<usize> = uses.rotations.iter().map(|&(_, k)| k.as_usize()).collect();
                widths.sort_unstable();
                widths.dedup();
                assert_eq!(widths, [55, 70, 90], "{algorithm:?}");
            }
        }
    }

    /// A left product under the masked paths reads the column halves of psi^k, which a context holds only when its plan
    /// selects one of those paths: the queries panic on another, with a static message.
    #[test]
    fn masked_left_queries_panic_without_the_masks() {
        let module = Module::<HostBytesBackend>::new(256);
        let a = Recorded(BlasLayout::new(block(&module, 90, 20), 2, 1, 1), 8);
        let ctx = module_layout(&module, BlasPlan::new(2), Base2K(19), CoeffsMeta::from_delta_budget(15, 19));
        let plan = BlasPlan::new(2).with_pt_algorithm(BlasPtAlgorithm::JklsGalois);
        let meta = CoeffsMeta::from_delta_budget(20, 19);
        let left = BlasPtOperandLayout::<Left>::new(plan, 128, 1, 1, 1, Base2K(19), meta);
        let queries: [(&str, &dyn Fn()); 2] = [
            ("output layout", &|| {
                blas_matmul_pt_left_output_layout(&a, &left, &a, &ctx);
            }),
            ("key uses", &|| {
                blas_matmul_pt_left_key_uses(&module, &a, &left, &a, &ctx);
            }),
        ];
        for (label, query) in queries {
            let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(query)).expect_err(label);
            assert_eq!(
                payload.downcast_ref::<String>().map(String::as_str),
                Some(MASKLESS),
                "{label}"
            );
        }
    }

    /// The masked paths read the masks of a context at the operand's block dimension: a query panics on another, with a
    /// static message.
    #[test]
    fn masked_queries_panic_at_another_dimension() {
        let module = Module::<HostBytesBackend>::new(256);
        let a = Recorded(BlasLayout::new(block(&module, 90, 20), 1, 1, 1), 32);
        let plan = |log_d| BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::JklsGalois);
        let ctx = module_layout(&module, plan(2), Base2K(19), CoeffsMeta::from_delta_budget(15, 19));
        let meta = CoeffsMeta::from_delta_budget(20, 19);
        let right = BlasPtOperandLayout::<Right>::new(plan(1), 128, 1, 1, 1, Base2K(19), meta);
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            blas_matmul_pt_right_output_layout(&a, &a, &right, &ctx);
        }))
        .expect_err("the query read a context at another dimension");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("a context at another block dimension than the operand's")
        );
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
