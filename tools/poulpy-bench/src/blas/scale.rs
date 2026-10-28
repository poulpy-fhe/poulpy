//! `poulpy-blas` at large blocks: blocks of `d = 128` on `n = 2^15` and `2^16`, grids of up to 224 blocks, an inner
//! dimension of up to 112 blocks.
//!
//! - [`max_base2k_table`]: the radix [`blas_matmul_max_base2k`], [`blas_matmul_pt_right_max_base2k`] and
//!   [`blas_matmul_pt_left_max_base2k`] allow for products by public matrices of `32 x 48`, `32 x 32`, `32 x 224` and
//!   `112 x 32` blocks, on both public sides and under every [`BlasPtAlgorithm`], read from layouts without encoding or
//!   encrypting anything.
//! - [`validate`]: one product per shape class on grids of many blocks, the longest inner dimension in full, the
//!   product of two ciphertext blocks through prepared operands, and the reductions over 32 blocks, each checked against
//!   [`error_bound`] and the radix the model allows, returning the checks that failed. It prints the heap each prepared
//!   operand holds, and the most its preparation held at once, when [`crate::heap::Counting`] is the global allocator.
//!
//! Neither runs under Criterion: a product at the longest inner dimension takes minutes, so a bench binary of its own
//! calls them once from its `main`.

use std::{
    collections::{BTreeMap, BTreeSet},
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

use poulpy_blas::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasContextLayout, BlasLayout, BlasMatMulOperandPreparedFactory, BlasPlan, BlasPtAlgorithm, BlasPtOperandLayout, Left,
        Right, blas_matmul_left_operand_layout, blas_matmul_max_base2k, blas_matmul_prepare_left_key_uses,
        blas_matmul_prepare_right_key_uses, blas_matmul_prepared_key_uses, blas_matmul_pt_left_key_uses,
        blas_matmul_pt_left_max_base2k, blas_matmul_pt_right_key_uses, blas_matmul_pt_right_max_base2k,
        blas_matmul_right_operand_layout, blas_sum_cols_key_uses, blas_sum_key_uses, blas_sum_rows_key_uses, replicated,
    },
    oep::BlasImpl,
    test_suite::helpers::{HINTS, MASKED_PT_ALGORITHMS, MIXES, PT_ALGORITHMS},
};
use poulpy_ckks::{CKKSLayout, CoeffsMeta, SlotsKind};
use poulpy_core::layouts::Degree;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ConjugateInvariant, MaxBase2k, Module, Ring, ScratchOwned, Standard},
};

use crate::{
    backend_name,
    blas::{
        intra::elements_of,
        matmul::{
            BlasBenchModule, Client, FAILURE_BITS, HEADROOM, a_entry, atk_layout, base2k, block_layout, entry, error_bound,
            tsk_layout, verify, verify_within,
        },
        params::{BlasBenchParams, default_bench_params_blas},
    },
    heap,
};

/// A product of an encrypted grid by a public matrix, in blocks: `inner` block columns of the encrypted grid, one block
/// high, by `inner x outer` blocks of the public matrix.
#[derive(Clone, Copy)]
struct PtProduct {
    inner: usize,
    outer: usize,
}

/// The products by public matrices the table covers: square, wide and long inner dimensions.
const PT_PRODUCTS: [PtProduct; 4] = [
    PtProduct { inner: 32, outer: 48 },
    PtProduct { inner: 32, outer: 32 },
    PtProduct { inner: 32, outer: 224 },
    PtProduct { inner: 112, outer: 32 },
];

/// Blocks a reduction adds.
const REDUCED_BLOCKS: usize = 32;

/// Every algorithm of the products by public matrices.
fn pt_algorithms() -> impl Iterator<Item = BlasPtAlgorithm> {
    PT_ALGORITHMS.into_iter().chain(MASKED_PT_ALGORITHMS)
}

/// The points of the sweep at the block dimension `d = 128`.
fn scale_params() -> Vec<BlasBenchParams> {
    default_bench_params_blas().into_iter().filter(|bp| bp.log_d == 7).collect()
}

/// Prints, for each point of the BLAS sweep at `d = 128` on `BE`'s ring whose label one of `filter`'s words selects
/// (every one for an empty filter), the radix the model of `BE` allows at 128 failure bits for products by public
/// matrices of `32 x 48`, `32 x 32`, `32 x 224` and `112 x 32` blocks, on the right (`X W`, an encrypted grid of
/// `1 x inner` blocks) and on the left (`W^T X^T`, `inner x 1`), under every [`BlasPtAlgorithm`], and for the ciphertext
/// by ciphertext product of one block. The encrypted grids are replicated, one matrix whose label proves it; where a
/// block holds several matrices, the table repeats for a dense grid of distinct ones.
///
/// Every block of a product's output sums the same `inner` terms through maps of one shape, so the radix of the whole
/// product is that of one block of its output, and the products of one inner dimension share it: the queries run once
/// per inner dimension, on one block column of the public matrix (row on the left).
pub fn max_base2k_table<BE: Backend + MaxBase2k>(filter: &[String]) {
    let real = BE::Ring::CYCLOTOMIC_ORDER_FACTOR == ConjugateInvariant::CYCLOTOMIC_ORDER_FACTOR;
    let backend = backend_name::<BE>();
    for bp in scale_params() {
        if !selected(filter, &format!("{backend}/blas/max_base2k/{bp}")) {
            continue;
        }
        let slots = bp.n * BE::Ring::CYCLOTOMIC_ORDER_FACTOR as usize / 4;
        let log_batch = (slots >> (2 * bp.log_d)).ilog2() as usize;
        let labels = if log_batch > 0 {
            vec![("replicated", log_batch), ("dense", 0)]
        } else {
            vec![("replicated", 0)]
        };
        let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
        let ctx = |plan| BlasContextLayout {
            plan,
            base2k: base2k(&bp),
            coeffs_meta: meta,
            n: Degree(bp.n as u32),
            slots,
        };
        let plan = |algorithm| BlasPlan::new(bp.log_d).with_pt_algorithm(algorithm);
        for (grid, log_sparsity) in labels {
            let mut block: CKKSLayout = block_layout(&bp);
            block.meta.log_sparsity = log_sparsity;
            if real {
                block.meta.slots = SlotsKind::Real;
            }
            let encrypted = |rows, cols| BlasLayout::new(block, bp.log_d, rows, cols);
            let per_algorithm = |radix: &dyn Fn(BlasPlan) -> usize| {
                pt_algorithms()
                    .map(|algorithm| format!("{algorithm:?}={}", radix(plan(algorithm))))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let by_inner: BTreeMap<usize, (String, String)> = PT_PRODUCTS
                .iter()
                .map(|w| w.inner)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|inner| {
                    let right = per_algorithm(&|plan| {
                        let b = BlasPtOperandLayout::<Right>::new(plan, slots, inner, 1, 1, base2k(&bp), meta);
                        blas_matmul_pt_right_max_base2k::<BE>(&encrypted(1, inner), &b, &ctx(plan), FAILURE_BITS)
                    });
                    let left = per_algorithm(&|plan| {
                        let b = BlasPtOperandLayout::<Left>::new(plan, slots, 1, inner, 1, base2k(&bp), meta);
                        blas_matmul_pt_left_max_base2k::<BE>(&b, &encrypted(inner, 1), &ctx(plan), FAILURE_BITS)
                    });
                    (inner, (right, left))
                })
                .collect();
            let line = |product: &str, shape: String, values: &str| {
                println!("MAX_BASE2K backend={backend} params={bp} grid={grid} product={product} shape={shape} {values}")
            };
            for PtProduct { inner, outer } in PT_PRODUCTS {
                let (right, left) = &by_inner[&inner];
                line("pt_right", format!("1x{inner}*{inner}x{outer}"), right);
                line("pt_left", format!("{outer}x{inner}*{inner}x1"), left);
            }
            let radix = blas_matmul_max_base2k::<BE>(
                &encrypted(1, 1),
                &encrypted(1, 1),
                &ctx(BlasPlan::new(bp.log_d)),
                FAILURE_BITS,
            );
            line("ct_x_ct", "1x1*1x1".to_string(), &format!("max_base2k={radix}"));
        }
    }
}

/// Whether the table or check named `label` runs: its label holds no word of `filter` that starts with `!` (after it),
/// and one of the others, or there are none.
fn selected(filter: &[String], label: &str) -> bool {
    let (excluded, included): (Vec<_>, Vec<_>) = filter.iter().partition(|word| word.starts_with('!'));
    !excluded.iter().any(|word| label.contains(&word[1..]))
        && (included.is_empty() || included.iter().any(|word| label.contains(word.as_str())))
}

/// The bytes the allocator holds above `base` and the most it held since, in MiB, or `-` without the counting
/// allocator.
fn heap_since(base: usize) -> String {
    if !heap::counting() {
        return "held=- peak=-".to_string();
    }
    let mib = |bytes: usize| bytes as f64 / (1u64 << 20) as f64;
    format!(
        "held={:.1}MiB peak={:.1}MiB",
        mib(heap::live().saturating_sub(base)),
        mib(heap::peak().saturating_sub(base))
    )
}

/// Runs the checks of every point of the BLAS sweep at `d = 128` whose label one of `filter`'s words selects, all of
/// them for an empty filter, and returns the labels of those that failed: a result beyond [`error_bound`], or a product
/// whose radix the model rejects at `base2k`, which still runs so that its error shows. A bench binary fails on any.
/// The checks:
///
/// - the products by public matrices on either side, one block of output, which is what each block of a wider output
///   computes: at an inner dimension of 112 blocks, which binds the radix, under every [`BlasPtAlgorithm`], and at 32
///   blocks under `Diagonal`;
/// - the product of two ciphertext blocks through operands prepared under every mix of hints;
/// - `blas_sum_rows` of a `32 x 1` grid, `blas_sum_cols` of a `1 x 32` one, and where a block holds several matrices,
///   `blas_sum` of a `1 x 32` grid of distinct ones; the row and column sums against the bound times `sqrt(32)`, since
///   they add the independent noises of 32 blocks.
///
/// The encrypted grids are replicated. Entries are scaled so that every result stays in `[-1, 1)`. It prints a
/// `PRECISION` line per check, a `SCRATCH` line before each operation allocates its scratch, a `TIME` line per
/// operation with the heap the call allocates beyond its scratch, and a `HEAP` line per prepared operand.
pub fn validate<BE>(filter: &[String]) -> Vec<String>
where
    BE: Backend<Ring = Standard> + 'static + BlasImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let inners: BTreeSet<usize> = PT_PRODUCTS.iter().map(|w| w.inner).collect();
    let widest = inners.last().copied();
    let mut failed = Vec::new();
    let backend = backend_name::<BE>();
    let mut run = |op: String, bp: &BlasBenchParams, check: &dyn Fn(&str)| {
        let label = format!("{backend}/blas/{op}/{bp}");
        if selected(filter, &label) && catch_unwind(AssertUnwindSafe(|| check(&op))).is_err() {
            failed.push(label);
        }
    };
    for bp in scale_params() {
        for &inner in &inners {
            let every = Some(inner) == widest;
            for algorithm in pt_algorithms().filter(|&algorithm| every || algorithm == BlasPtAlgorithm::Diagonal) {
                let name = format!("{algorithm:?}").to_lowercase();
                run(format!("pt_right_{name}/1x{inner}*{inner}x1"), &bp, &|op| {
                    pt_right::<BE>(&bp, algorithm, inner, op)
                });
                run(format!("pt_left_{name}/1x{inner}*{inner}x1"), &bp, &|op| {
                    pt_left::<BE>(&bp, algorithm, inner, op)
                });
            }
        }
        run("ct_prepared/1x1*1x1".to_string(), &bp, &|op| ct_product::<BE>(&bp, op));
        run(format!("sum_rows/{REDUCED_BLOCKS}x1"), &bp, &|op| {
            reduction::<BE>(&bp, Reduction::SumRows, REDUCED_BLOCKS, 1, op)
        });
        run(format!("sum_cols/1x{REDUCED_BLOCKS}"), &bp, &|op| {
            reduction::<BE>(&bp, Reduction::SumCols, 1, REDUCED_BLOCKS, op)
        });
        if bp.batch() > 1 {
            run(format!("sum/1x{REDUCED_BLOCKS}"), &bp, &|op| {
                reduction::<BE>(&bp, Reduction::Sum, 1, REDUCED_BLOCKS, op)
            });
        }
    }
    failed
}

/// Fails the check of `op` when the model rejects `base2k` for its lazy sums, after its error was verified.
fn assert_radix(op: &str, bp: &BlasBenchParams, max_base2k: usize) {
    assert!(
        max_base2k >= bp.base2k,
        "{op}: base2k exceeds the radix its lazy sums allow at 128 failure bits"
    );
}

/// The public matrix of a product summing `inner` blocks, scaled so that it stays in `[-1, 1)`.
fn public_entry(inner: usize, d: usize) -> impl Fn(usize, usize, usize) -> f64 {
    move |s, r, c| entry(3, s, r, c) / (inner * d) as f64
}

/// Times one call of `op`, with the heap it held at the end and at most during the call, scratch aside.
fn timed_call<BE>(op: &str, bp: &BlasBenchParams, call: impl FnOnce()) {
    let base = heap::reset_peak();
    let start = Instant::now();
    call();
    println!(
        "TIME backend={} op={op} params={bp} secs={:.2} {}",
        backend_name::<BE>(),
        start.elapsed().as_secs_f64(),
        heap_since(base)
    );
}

/// The scratch of `op`, `bytes` long, allocated after printing its size: a product's scratch can outgrow the host.
fn scratch_for<BE: Backend>(op: &str, bp: &BlasBenchParams, bytes: usize) -> ScratchOwned<BE>
where
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    println!(
        "SCRATCH backend={} op={op} params={bp} bytes={:.1}MiB",
        backend_name::<BE>(),
        bytes as f64 / (1u64 << 20) as f64
    );
    ScratchOwned::alloc(bytes)
}

/// Prints the heap the operand `operand` of `op` holds above `base`, and the most its preparation held.
fn print_heap<BE>(op: &str, operand: &str, bp: &BlasBenchParams, base: usize) {
    println!(
        "HEAP backend={} op={op} operand={operand} params={bp} {}",
        backend_name::<BE>(),
        heap_since(base)
    );
}

/// `blas_matmul_pt_right` of a replicated grid of `1 x inner` blocks by `inner x 1` blocks of a public matrix.
fn pt_right<BE>(bp: &BlasBenchParams, algorithm: BlasPtAlgorithm, inner: usize, op: &str)
where
    BE: Backend<Ring = Standard> + 'static + BlasImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut client = Client::<BE>::new(bp);
    let w = public_entry(inner, bp.d());
    let base = heap::reset_peak();
    let pt = client.public::<Right>(bp, algorithm, inner, 1, &w);
    print_heap::<BE>(op, "public", bp, base);
    let ctx = client.context(bp, BlasPlan::new(bp.log_d).with_pt_algorithm(algorithm));
    let ct_a = client.encrypt(bp, 1, inner, 1, a_entry);
    assert!(replicated::<Standard>(&ct_a), "{op}: the encrypted grid is not replicated");
    let max_base2k = blas_matmul_pt_right_max_base2k::<BE>(&ct_a, &pt, &ctx, FAILURE_BITS);
    let mut res = client.result(bp, 1, 1);
    let uses = blas_matmul_pt_right_key_uses(&client.module, &res, &ct_a, &pt, &ctx);
    let atks = client.rotation_keys(bp, elements_of(&uses));
    let bytes = client
        .module
        .blas_matmul_pt_right_tmp_bytes(&res, &ct_a, &pt, &ctx, &atk_layout(bp));
    let mut scratch = scratch_for::<BE>(op, bp, bytes);
    timed_call::<BE>(op, bp, || {
        client
            .module
            .blas_matmul_pt_right(&mut res, &ct_a, &pt, &ctx, &atks, &mut scratch.borrow())
            .unwrap()
    });
    let d = bp.d();
    let max_err = client.max_error(bp, &res, |_, r, c| {
        (0..inner * d).map(|l| a_entry(0, r, l) * w(0, l, c)).sum()
    });
    verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
    assert_radix(op, bp, max_base2k);
}

/// `blas_matmul_pt_left` of `1 x inner` blocks of a public matrix by a replicated grid of `inner x 1` blocks.
fn pt_left<BE>(bp: &BlasBenchParams, algorithm: BlasPtAlgorithm, inner: usize, op: &str)
where
    BE: Backend<Ring = Standard> + 'static + BlasImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut client = Client::<BE>::new(bp);
    let w = public_entry(inner, bp.d());
    let base = heap::reset_peak();
    let pt = client.public::<Left>(bp, algorithm, 1, inner, &w);
    print_heap::<BE>(op, "public", bp, base);
    let ctx = client.context(bp, BlasPlan::new(bp.log_d).with_pt_algorithm(algorithm));
    let ct_a = client.encrypt(bp, inner, 1, 1, a_entry);
    assert!(replicated::<Standard>(&ct_a), "{op}: the encrypted grid is not replicated");
    let max_base2k = blas_matmul_pt_left_max_base2k::<BE>(&pt, &ct_a, &ctx, FAILURE_BITS);
    let mut res = client.result(bp, 1, 1);
    let uses = blas_matmul_pt_left_key_uses(&client.module, &res, &pt, &ct_a, &ctx);
    let atks = client.rotation_keys(bp, elements_of(&uses));
    let bytes = client
        .module
        .blas_matmul_pt_left_tmp_bytes(&res, &pt, &ct_a, &ctx, &atk_layout(bp));
    let mut scratch = scratch_for::<BE>(op, bp, bytes);
    timed_call::<BE>(op, bp, || {
        client
            .module
            .blas_matmul_pt_left(&mut res, &pt, &ct_a, &ctx, &atks, &mut scratch.borrow())
            .unwrap()
    });
    let d = bp.d();
    let max_err = client.max_error(bp, &res, |_, r, c| {
        (0..inner * d).map(|l| w(0, r, l) * a_entry(0, l, c)).sum()
    });
    verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
    assert_radix(op, bp, max_base2k);
}

/// `blas_matmul_prepared` of two replicated blocks through operands prepared under every mix of hints, among them a
/// right operand under `Reuse`, the one several products share.
fn ct_product<BE>(bp: &BlasBenchParams, op: &str)
where
    BE: Backend<Ring = Standard> + 'static + BlasImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut client = Client::<BE>::new(bp);
    let ctx = client.context(bp, BlasPlan::new(bp.log_d));
    let d = bp.d();
    let b_entry = move |s, r, c| entry(2, s, r, c) / d as f64;
    let (a, b) = (client.encrypt(bp, 1, 1, 1, a_entry), client.encrypt(bp, 1, 1, 1, b_entry));
    let max_base2k = blas_matmul_max_base2k::<BE>(&a, &b, &ctx, FAILURE_BITS);
    let module = &client.module;
    let mut elements = elements_of(&blas_matmul_prepare_left_key_uses(module, &a, &ctx));
    elements.extend(elements_of(&blas_matmul_prepare_right_key_uses(module, &b, &ctx)));
    for (hl, hr) in MIXES {
        let (left, right) = (
            blas_matmul_left_operand_layout(&a, &ctx, hl),
            blas_matmul_right_operand_layout(&b, &ctx, hr),
        );
        elements.extend(elements_of(&blas_matmul_prepared_key_uses(module, &left, &right, &ctx)));
    }
    let atks = client.rotation_keys(bp, elements);
    let tsk = client.tensor_key(bp);
    let mut prepared_left = Vec::new();
    let mut prepared_right = Vec::new();
    for hint in HINTS {
        let module = &client.module;
        let base = heap::reset_peak();
        let left = {
            let mut scratch =
                ScratchOwned::<BE>::alloc(module.blas_matmul_prepare_left_tmp_bytes(&a, &ctx, &atk_layout(bp), hint));
            let mut left = module.blas_matmul_left_alloc_from_infos(&blas_matmul_left_operand_layout(&a, &ctx, hint));
            module
                .blas_matmul_prepare_left(&mut left, &a, &ctx, &atks, &mut scratch.borrow())
                .unwrap();
            left
        };
        print_heap::<BE>(op, &format!("left/{hint:?}"), bp, base);
        prepared_left.push(left);
        let base = heap::reset_peak();
        let right = {
            let mut scratch =
                ScratchOwned::<BE>::alloc(module.blas_matmul_prepare_right_tmp_bytes(&b, &ctx, &atk_layout(bp), hint));
            let mut right = module.blas_matmul_right_alloc_from_infos(&blas_matmul_right_operand_layout(&b, &ctx, hint));
            module
                .blas_matmul_prepare_right(&mut right, &b, &ctx, &atks, &mut scratch.borrow())
                .unwrap();
            right
        };
        print_heap::<BE>(op, &format!("right/{hint:?}"), bp, base);
        prepared_right.push(right);
    }
    for (left, hl) in prepared_left.iter().zip(HINTS) {
        for (right, hr) in prepared_right.iter().zip(HINTS) {
            let mut res = client.result(bp, 1, 1);
            let bytes = client
                .module
                .blas_matmul_prepared_tmp_bytes(&res, left, right, &ctx, &tsk_layout(bp), &atk_layout(bp));
            let mix = format!("{op}/{hl:?}x{hr:?}");
            let mut scratch = scratch_for::<BE>(&mix, bp, bytes);
            timed_call::<BE>(&mix, bp, || {
                client
                    .module
                    .blas_matmul_prepared(&mut res, left, right, &ctx, &tsk, &atks, &mut scratch.borrow())
                    .unwrap()
            });
            let max_err = client.max_error(bp, &res, |_, r, c| (0..d).map(|l| a_entry(0, r, l) * b_entry(0, l, c)).sum());
            verify::<BE>(&mix, bp, max_err, Some(max_base2k), atks.len());
        }
    }
    assert_radix(op, bp, max_base2k);
}

/// The reductions [`validate`] checks.
#[derive(Clone, Copy)]
enum Reduction {
    Sum,
    SumRows,
    SumCols,
}

/// `blas_sum_rows` or `blas_sum_cols` of a replicated grid of `rows x cols` blocks, or `blas_sum` of one of distinct
/// matrices, entries scaled so that the sums stay in `[-1, 1)`, against [`error_bound`], times the square root of the
/// blocks summed for the row and column sums.
fn reduction<BE>(bp: &BlasBenchParams, kind: Reduction, rows: usize, cols: usize, op: &str)
where
    BE: Backend<Ring = Standard> + 'static + BlasImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut client = Client::<BE>::new(bp);
    let ctx = client.context(bp, BlasPlan::new(bp.log_d));
    let d = bp.d();
    let (matrices, terms) = match kind {
        Reduction::Sum => (bp.batch(), bp.batch()),
        Reduction::SumRows => (1, rows * d),
        Reduction::SumCols => (1, cols * d),
    };
    let e = move |s, r, c| entry(4, s, r, c) / terms as f64;
    let ct_a = client.encrypt(bp, rows, cols, matrices, e);
    let mut res = client.result(bp, rows, cols);
    let atk = atk_layout(bp);
    let module = &client.module;
    let (uses, bytes) = match kind {
        Reduction::Sum => (
            blas_sum_key_uses(module, &res, &ct_a),
            module.blas_sum_tmp_bytes(&res, &ct_a, &atk),
        ),
        Reduction::SumRows => (
            blas_sum_rows_key_uses(module, &res, &ct_a, &ctx),
            module.blas_sum_rows_tmp_bytes(&res, &ct_a, &ctx, &atk),
        ),
        Reduction::SumCols => (
            blas_sum_cols_key_uses(module, &res, &ct_a, &ctx),
            module.blas_sum_cols_tmp_bytes(&res, &ct_a, &ctx, &atk),
        ),
    };
    let atks = client.rotation_keys(bp, elements_of(&uses));
    let mut scratch = scratch_for::<BE>(op, bp, bytes);
    timed_call::<BE>(op, bp, || {
        let module = &client.module;
        match kind {
            Reduction::Sum => module.blas_sum(&mut res, &ct_a, &atks, &mut scratch.borrow()),
            Reduction::SumRows => module.blas_sum_rows(&mut res, &ct_a, &ctx, &atks, &mut scratch.borrow()),
            Reduction::SumCols => module.blas_sum_cols(&mut res, &ct_a, &ctx, &atks, &mut scratch.borrow()),
        }
        .unwrap()
    });
    let max_err = client.max_error(bp, &res, |s, r, c| match kind {
        Reduction::Sum => (0..matrices).map(|t| e(t, r, c)).sum(),
        Reduction::SumRows => (0..rows * d).map(|l| e(s % matrices, l, c)).sum(),
        Reduction::SumCols => (0..cols * d).map(|l| e(s % matrices, r, l)).sum(),
    });
    // The row and column sums add the independent noises of every block they sum; `blas_sum` stays within a block.
    let summed = match kind {
        Reduction::Sum => 1,
        Reduction::SumRows | Reduction::SumCols => rows * cols,
    };
    let bound = error_bound(bp) * (summed as f64).sqrt();
    verify_within::<BE>(op, bp, max_err, bound, None, atks.len());
}
