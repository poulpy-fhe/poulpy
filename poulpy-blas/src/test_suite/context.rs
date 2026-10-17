//! The plan's key sets against the encoded maps of a context and of public operands.

use std::collections::{BTreeMap, BTreeSet};

use poulpy_ckks::{
    CKKSModuleInfos, CoeffsMeta,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, LinearTransformation, LinearTransformationPrepared, PreparedDiagonal},
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch},
    },
};
use poulpy_core::layouts::{Base2K, DiagonalArithmetic, GetDegree, LWEInfos};
use poulpy_hal::{
    AlignedBuf,
    api::{CnvPVecAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, DataView, DataViewMut, HostBytesBackend, Module, ScratchArena, Standard},
};

use super::helpers::{
    assert_panics, assert_short_scratch_panics, context_maps, encoded_context, map_galois_elements, pt_operand,
};
use crate::layouts::{
    BlasBytesOf, BlasContext, BlasContextInfos, BlasContextLayout, BlasContextPreparedFactory, BlasContextUnprepared,
    BlasEncodingOps, BlasModuleAlloc, BlasPlan, BlasPtAlgorithm, BlasPtOperand, BlasPtOperandInfos, BlasPtOperandLayout,
    BlasPtOperandPreparedFactory, BlasPtOperandUnprepared, BlasPtSide, Left, Right,
};

/// Every block map of `b`.
fn operand_maps<BE: Backend, S: BlasPtSide>(b: &BlasPtOperand<BE, S>) -> Vec<&LinearTransformationPrepared<BE>> {
    (0..b.rows())
        .flat_map(|l| (0..b.cols()).map(move |j| b.block(l, j)))
        .collect()
}

/// At each `log_d` from 1 to 3 whose blocks fit the `S` slots, `d^2 <= S`, each key set of the plan equals the Galois
/// elements of the maps its operation evaluates, as the context encodes them: the transpose, the row and column shifts
/// by every `k` below `2 d`, the matrix products, and the products by public operands, on either side, of any values,
/// grid shape and count, of the counts that divide the matrix slots. Public operands of either algorithm, side and count
/// store their blocks at one degree, whatever their values, and a JKLS operand's first map and blocks rotate by its
/// side's set. The union of these sets, and the plan's own, equals the context's. A `log_d` whose blocks do not fit is
/// skipped with a message. The context panics at its allocation on matrices wider than the slots, a zero torus
/// width, `coeffs_meta.k = 0`, a radix outside `1..=63` and a prepared allocation or size of the same; a public
/// operand whose count does not divide the matrix slots, or of a zero torus width, panics at its allocation, prepared
/// or not, and so does its prepared size. The prepare scratch query panics on a layout of other slots, and a
/// preparation into an operand of another layout, or one scratch byte short of its query, panics.
pub fn test_blas_context<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
{
    let mut scratch = alloc_scratch(&params, module);
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let order = module.cyclotomic_order();
    type Entry = fn(usize, usize, usize) -> f64;
    // Generic, identity, constant and zero matrices, one per matrix slot; then generic, equal and zero pairs.
    let operands: [(&str, usize, usize, usize, Entry); 7] = [
        ("generic", 3, 2, 1, |m, r, c| ((m + r * 3 + c) % 5) as f64 - 2.0),
        ("identity", 1, 1, 1, |_, r, c| f64::from(r == c)),
        ("constant", 2, 1, 1, |_, _, _| 1.0),
        ("zero", 1, 2, 1, |_, _, _| 0.0),
        ("generic pair", 3, 2, 2, |m, r, c| ((m + r * 3 + c) % 5) as f64 - 2.0),
        ("equal pair", 1, 1, 2, |_, r, c| ((r * 3 + c) % 5) as f64 - 2.0),
        ("zero pair", 2, 2, 2, |_, _, _| 0.0),
    ];
    // The largest `log_d` whose blocks fit the slots, `d^2 <= S`; the sweep skips the others with a message.
    let slots = params.n / 2;
    let fits = slots.ilog2() as usize / 2;
    let (sweep, skipped): (Vec<usize>, Vec<usize>) = (1..=3).partition(|&log_d| log_d <= fits);
    if !skipped.is_empty() {
        eprintln!("test_blas_context skips log_d {skipped:?}: it needs d^2 <= {slots} slots");
    }
    for log_d in sweep {
        let plan = BlasPlan::new(log_d);
        let ctx = encoded_context::<BE, F>(module, plan, base2k, meta, &mut scratch.borrow());
        assert_eq!((ctx.plan(), ctx.log_d()), (plan, log_d));
        let d = 1 << log_d;
        let encoded = |maps: Vec<_>| map_galois_elements(maps, order);
        let mut union = plan.transpose_galois_elements(order);
        assert_eq!(union, encoded(vec![ctx.transpose()]), "transpose log_d={log_d}");

        let matmul = plan.matmul_galois_elements(order);
        let mut maps = vec![ctx.sigma(), ctx.tau(), ctx.tau_rep()];
        (1..d).for_each(|k| maps.extend([ctx.rotate_cols(k), ctx.rotate_rows(k), ctx.rotate_rows_rep(k)]));
        assert_eq!(matmul, encoded(maps), "matmul log_d={log_d}");
        union.extend(matmul);

        for k in 0..2 * d {
            let k1 = k % d;
            let (rows, cols) = if k1 == 0 {
                (vec![], vec![])
            } else {
                (
                    vec![ctx.rotate_rows(k1), ctx.rows_upper(k1), ctx.rows_lower(k1)],
                    vec![ctx.rotate_cols(k1), ctx.cols_left(k1), ctx.cols_right(k1)],
                )
            };
            let (plan_rows, plan_cols) = (
                plan.rotate_rows_galois_elements(order, k),
                plan.rotate_cols_galois_elements(order, k),
            );
            assert_eq!(plan_rows, encoded(rows), "rotate_rows log_d={log_d} k={k}");
            assert_eq!(plan_cols, encoded(cols), "rotate_cols log_d={log_d} k={k}");
            union.extend(plan_rows.into_iter().chain(plan_cols));
        }

        let (matmul_pt_right, matmul_pt_left) = (
            plan.matmul_pt_right_galois_elements(order),
            plan.matmul_pt_left_galois_elements(order),
        );
        // The degrees the blocks of each side's, count's and algorithm's operands store their diagonals at.
        let mut degrees = BTreeMap::<(&str, usize, bool), BTreeSet<usize>>::new();
        for (label, rows, cols, count, entry) in operands {
            if !(slots >> (2 * log_d)).is_multiple_of(count) {
                continue;
            }
            for algorithm in [BlasPtAlgorithm::Diagonal, BlasPtAlgorithm::Jkls] {
                let (plan, jkls) = (plan.with_pt_algorithm(algorithm), algorithm == BlasPtAlgorithm::Jkls);
                let f = |m: usize, r: usize, c: usize| F::from_f64(entry(m, r, c)).unwrap();
                let right = pt_operand::<BE, F, Right>(module, plan, rows, cols, count, f, base2k, meta, &mut scratch.borrow());
                let left = pt_operand::<BE, F, Left>(module, plan, rows, cols, count, f, base2k, meta, &mut scratch.borrow());
                for (side, set, first, blocks) in [
                    ("matmul_pt_right", &matmul_pt_right, right.first(), operand_maps(&right)),
                    ("matmul_pt_left", &matmul_pt_left, left.first(), operand_maps(&left)),
                ] {
                    assert_eq!(first.is_some(), jkls, "{side} {algorithm:?} {label} log_d={log_d}: first map");
                    assert_eq!(
                        *set,
                        map_galois_elements(first.into_iter().chain(blocks.iter().copied()), order),
                        "{side} {algorithm:?} {label} log_d={log_d}"
                    );
                    let diagonals = blocks
                        .iter()
                        .flat_map(|lt| lt.giant_steps.iter().flat_map(|gs| &gs.diagonals));
                    degrees
                        .entry((side, count, jkls))
                        .or_default()
                        .extend(diagonals.map(|diagonal| diagonal.plaintext.n().as_usize()));
                }
            }
        }
        for ((side, count, jkls), degrees) in degrees {
            assert_eq!(
                degrees.len(),
                1,
                "{side} degrees {degrees:?} count={count} jkls={jkls} log_d={log_d}"
            );
        }
        union.extend(matmul_pt_right.into_iter().chain(matmul_pt_left));
        union.sort_unstable();
        union.dedup();
        let every = encoded(context_maps(&ctx));
        assert_eq!(union, every, "union log_d={log_d}");
        assert_eq!(plan.galois_elements(order), every, "galois_elements log_d={log_d}");
    }

    // The slots hold no matrix one dimension past the largest that fits; 2^40 x 2^40 has more entries than a usize
    // counts.
    for log_d in [fits + 1, 40] {
        assert_panics(&format!("log_d={log_d}"), || {
            module.blas_context_alloc(BlasPlan::new(log_d), base2k, meta)
        });
    }

    let ctx_base = module
        .blas_context_alloc(BlasPlan::new(fits.min(2)), base2k, meta)
        .blas_context_layout();
    let ctx_with = |edit: fn(&mut BlasContextLayout)| {
        let mut layout = ctx_base;
        edit(&mut layout);
        layout
    };
    for bad in [
        ctx_with(|l| l.coeffs_meta = CoeffsMeta::from_delta_budget(0, 0)),
        ctx_with(|l| l.base2k = Base2K(0)),
        ctx_with(|l| l.plan = BlasPlan::new(0)),
    ] {
        assert_panics("context prepared alloc", || {
            module.blas_context_prepared_alloc_from_infos(&bad)
        });
        assert_panics("context prepared bytes_of", || {
            module.blas_context_prepared_bytes_of_from_infos(&bad)
        });
    }
    // Twice the slots fit the blocks, so only the slot check rejects them.
    let ctx_other_slots = ctx_with(|l| l.slots *= 2);
    assert_panics("context prepare_tmp_bytes slots", || {
        module.blas_context_prepare_tmp_bytes(&ctx_other_slots)
    });
    let mut ctx = module.blas_context_alloc_from_infos(&ctx_base);
    module.blas_context_encode_into::<F>(&mut ctx, &mut scratch.borrow()).unwrap();
    let mut ctx_other = module.blas_context_prepared_alloc_from_infos(&ctx_with(|l| l.base2k = Base2K(l.base2k.as_u32() + 1)));
    assert_panics("context prepare into another layout", || {
        module.blas_context_prepare(&mut ctx_other, &ctx, &mut scratch.borrow())
    });
    let mut ctx_prepared = module.blas_context_prepared_alloc_from_infos(&ctx_base);
    let ctx_bytes = module.blas_context_prepare_tmp_bytes(&ctx_base);
    if ctx_bytes > 0 {
        assert_short_scratch_panics::<BE, _>("blas_context_prepare", ctx_bytes, |s| {
            module.blas_context_prepare(&mut ctx_prepared, &ctx, s)
        });
    }

    // Matrix slots of 4 x 4 blocks, or of 2 x 2 where those do not fit, are a power of two: 0 and 3 do not divide them.
    let (log_d, generic) = (fits.min(2), operands[0].4);
    for count in [0, 3] {
        assert_panics(&format!("count={count}"), || {
            module.blas_pt_operand_alloc::<Right>(BlasPlan::new(log_d), 3, 2, count, base2k, meta)
        });
    }
    let count = (slots >> (2 * log_d)).min(2);
    let f = |m: usize, r: usize, c: usize| F::from_f64(generic(m, r, c)).unwrap();
    let b = pt_operand::<BE, F, Right>(
        module,
        BlasPlan::new(log_d),
        3,
        2,
        count,
        f,
        base2k,
        meta,
        &mut scratch.borrow(),
    );
    assert_eq!((b.log_d(), b.rows(), b.cols(), b.count()), (log_d, 3, 2, count));

    let base = BlasPtOperandLayout::<Right>::new(BlasPlan::new(log_d), slots, 1, 1, 1, base2k, meta);
    let with = |edit: fn(&mut BlasPtOperandLayout<Right>)| {
        let mut layout = base;
        edit(&mut layout);
        layout
    };
    for bad in [with(|l| l.count = 3), with(|l| l.base2k = Base2K(0))] {
        assert_panics("prepared alloc", || module.blas_pt_operand_prepared_alloc_from_infos(&bad));
        assert_panics("prepared bytes_of", || {
            module.blas_pt_operand_prepared_bytes_of_from_infos(&bad)
        });
    }
    // Twice the slots fit the blocks, so only the slot check rejects them.
    let other_slots = with(|l| l.slots *= 2);
    assert_panics("prepare_tmp_bytes slots", || {
        module.blas_pt_operand_prepare_tmp_bytes(&other_slots)
    });
    let mut op = module.blas_pt_operand_alloc_from_infos(&base);
    module.blas_pt_operand_encode_into(&mut op, f, &mut scratch.borrow()).unwrap();
    let mut other = module.blas_pt_operand_prepared_alloc_from_infos(&with(|l| l.rows = 2));
    assert_panics("prepare into another layout", || {
        module.blas_pt_operand_prepare(&mut other, &op, &mut scratch.borrow())
    });
    let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&base);
    let bytes = module.blas_pt_operand_prepare_tmp_bytes(&base);
    // A backend whose preparation needs no scratch has no short scratch to test.
    if bytes > 0 {
        assert_short_scratch_panics::<BE, _>("blas_pt_operand_prepare", bytes, |s| {
            module.blas_pt_operand_prepare(&mut prepared, &op, s)
        });
    }

    // `coeffs_meta.k = 0` is rejected at allocation: the context and every public operand, either side.
    let zero = CoeffsMeta::from_delta_budget(0, 0);
    assert_panics("context", || module.blas_context_alloc(BlasPlan::new(log_d), base2k, zero));
    for algorithm in [BlasPtAlgorithm::Diagonal, BlasPtAlgorithm::Jkls] {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        assert_panics(&format!("right operand {algorithm:?}"), || {
            module.blas_pt_operand_alloc::<Right>(plan, 1, 1, 1, base2k, zero)
        });
        assert_panics(&format!("left operand {algorithm:?}"), || {
            module.blas_pt_operand_alloc::<Left>(plan, 1, 1, 1, base2k, zero)
        });
    }
}

/// The address and the bytes of every diagonal buffer of a map set.
type Buffers = Vec<(*const u8, Vec<u8>)>;

const POISON: u8 = 0xA5;

fn diagonals<'a, P: 'a>(maps: impl IntoIterator<Item = &'a LinearTransformation<P>>) -> impl Iterator<Item = &'a P> {
    maps.into_iter()
        .flat_map(|lt| lt.giant_steps.iter().flat_map(|gs| gs.diagonals.iter().map(|d| &d.plaintext)))
}

fn diagonals_mut<'a, P: 'a>(maps: impl IntoIterator<Item = &'a mut LinearTransformation<P>>) -> impl Iterator<Item = &'a mut P> {
    maps.into_iter().flat_map(|lt| {
        lt.giant_steps
            .iter_mut()
            .flat_map(|gs| gs.diagonals.iter_mut().map(|d| &mut d.plaintext))
    })
}

fn buffers<'a, P: 'a>(maps: impl IntoIterator<Item = &'a LinearTransformation<P>>, bytes: impl Fn(&P) -> &[u8]) -> Buffers {
    diagonals(maps)
        .map(|p| {
            let b = bytes(p);
            (b.as_ptr(), b.to_vec())
        })
        .collect()
}

fn plaintext<BE: Backend<OwnedBuf = AlignedBuf>>(pt: &CKKSPlaintextOwned<BE>) -> &[u8] {
    pt.data().data()
}

fn prepared_diagonal<BE: Backend<OwnedBuf = AlignedBuf>>(d: &PreparedDiagonal<AlignedBuf, BE>) -> &[u8] {
    d.cnv().data()
}

fn held(buffers: &Buffers) -> usize {
    buffers.iter().map(|(_, b)| b.len()).sum()
}

fn poison_plaintexts<'a, BE: Backend<OwnedBuf = AlignedBuf> + 'a>(
    maps: impl IntoIterator<Item = &'a mut LinearTransformation<CKKSPlaintextOwned<BE>>>,
) {
    diagonals_mut(maps).for_each(|pt| pt.data_mut().data_mut().as_mut().fill(POISON));
}

fn poison_prepared<'a, BE: Backend<OwnedBuf = AlignedBuf> + 'a>(
    maps: impl IntoIterator<Item = &'a mut LinearTransformation<PreparedDiagonal<AlignedBuf, BE>>>,
) {
    diagonals_mut(maps).for_each(|d| d.cnv_mut().data_mut().as_mut().fill(POISON));
}

/// A pair, unprepared and prepared, filled from `first` then refilled from `second` in the same allocations after
/// every byte of both was poisoned, against a pair filled from `second` alone; and its byte queries against its
/// buffers.
fn check_refill<U, P>(
    label: &str,
    alloc: impl Fn() -> (U, P),
    (bytes, prepared_bytes): (usize, usize),
    view: fn(&U) -> Buffers,
    view_prepared: fn(&P) -> Buffers,
    poison: fn(&mut U, &mut P),
    mut fill: impl FnMut(&mut U, &mut P, bool),
) {
    let (mut u, mut p) = alloc();
    assert_eq!(bytes, held(&view(&u)), "{label}: bytes_of");
    assert_eq!(prepared_bytes, held(&view_prepared(&p)), "{label}: prepared_bytes_of");
    fill(&mut u, &mut p, false);
    let (first, first_prepared) = (view(&u), view_prepared(&p));
    poison(&mut u, &mut p);
    fill(&mut u, &mut p, true);
    let (mut fresh, mut fresh_prepared) = alloc();
    fill(&mut fresh, &mut fresh_prepared, true);
    for (name, first, refilled, fresh) in [
        ("unprepared", first, view(&u), view(&fresh)),
        ("prepared", first_prepared, view_prepared(&p), view_prepared(&fresh_prepared)),
    ] {
        assert_eq!(refilled.len(), fresh.len(), "{label} {name}: diagonals");
        assert_eq!(refilled.len(), first.len(), "{label} {name}: diagonals");
        assert!(
            refilled.iter().zip(&first).all(|((a, _), (b, _))| a == b),
            "{label} {name}: a refill reallocated"
        );
        assert!(
            refilled.iter().zip(&fresh).all(|((_, a), (_, b))| a == b),
            "{label} {name}: a refill differs from a fresh build"
        );
    }
}

/// A public operand of `plan` on side `S` through [`check_refill`], its second fill from other matrices than its first.
fn refill_operand<BE, F, S>(
    module: &Module<BE>,
    plan: BlasPlan,
    base2k: Base2K,
    meta: CoeffsMeta,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE>,
    F: TestScalar + DiagonalArithmetic,
    S: BlasPtSide,
{
    let first: fn(usize, usize, usize) -> f64 = |m, r, c| ((m + r * 3 + c) % 5) as f64 - 2.0;
    let second: fn(usize, usize, usize) -> f64 = |m, r, c| ((m * 7 + r + c * 2) % 9) as f64 / 4.0 - 1.0;
    let layout = BlasPtOperandLayout::<S>::new(plan, module.ckks_max_slots(), 2, 2, 1, base2k, meta);
    check_refill(
        &format!("{:?} left={}", plan.pt_algorithm(), S::LEFT),
        || {
            (
                module.blas_pt_operand_alloc_from_infos(&layout),
                module.blas_pt_operand_prepared_alloc_from_infos(&layout),
            )
        },
        (
            module.blas_pt_operand_bytes_of_from_infos(&layout),
            module.blas_pt_operand_prepared_bytes_of_from_infos(&layout),
        ),
        |op: &BlasPtOperandUnprepared<BE, S>| buffers(op.maps(), plaintext::<BE>),
        |p: &BlasPtOperand<BE, S>| buffers(p.maps(), prepared_diagonal::<BE>),
        |op, p| {
            poison_plaintexts::<BE>(op.maps.iter_mut());
            poison_prepared::<BE>(p.maps.iter_mut());
        },
        |op, p, again| {
            let entry = if again { second } else { first };
            module
                .blas_pt_operand_encode_into(op, |m, r, c| F::from_f64(entry(m, r, c)).unwrap(), scratch)
                .unwrap();
            module.blas_pt_operand_prepare(p, op, scratch);
        },
    );
}

/// Encoding and preparing again into the same allocations overwrites every byte and equals a fresh build, for the
/// context and for public operands of either side and algorithm, the operands from other matrices than their first;
/// every byte query equals the buffers allocated from its layout.
pub fn test_blas_refill<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE>,
    F: TestScalar + DiagonalArithmetic,
{
    let mut scratch = alloc_scratch(&params, module);
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let layout = BlasContextLayout {
        plan: BlasPlan::new(log_d),
        base2k,
        coeffs_meta: meta,
        n: module.ring_degree(),
        slots: module.ckks_max_slots(),
    };
    check_refill(
        "context",
        || {
            (
                module.blas_context_alloc_from_infos(&layout),
                module.blas_context_prepared_alloc_from_infos(&layout),
            )
        },
        (
            module.blas_context_bytes_of_from_infos(&layout),
            module.blas_context_prepared_bytes_of_from_infos(&layout),
        ),
        |c: &BlasContextUnprepared<BE>| buffers(c.maps(), plaintext::<BE>),
        |p: &BlasContext<BE>| buffers(p.maps(), prepared_diagonal::<BE>),
        |c, p| {
            poison_plaintexts::<BE>(c.maps.iter_mut());
            poison_prepared::<BE>(p.maps.iter_mut());
        },
        |c, p, _| {
            module.blas_context_encode_into::<F>(c, &mut scratch.borrow()).unwrap();
            module.blas_context_prepare(p, c, &mut scratch.borrow());
        },
    );
    for algorithm in [BlasPtAlgorithm::Diagonal, BlasPtAlgorithm::Jkls] {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        refill_operand::<BE, F, Right>(module, plan, base2k, meta, &mut scratch.borrow());
        refill_operand::<BE, F, Left>(module, plan, base2k, meta, &mut scratch.borrow());
    }
}
