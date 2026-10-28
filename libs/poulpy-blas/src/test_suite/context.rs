//! The plan's key sets against the encoded maps of a context and of public operands.

use std::collections::{BTreeMap, BTreeSet};

use poulpy_ckks::{
    CKKSInfos, CKKSModuleInfos, CoeffsMeta, GLWEPlaintextInfos, Scale, SetCKKSInfos, SetGLWEPlaintextInfos, SlotsKind,
    api::{
        CKKSEncodingOps, CKKSLinearTransformationOps, CKKSMulOps, LinearTransformation, LinearTransformationPrepared,
        PreparedDiagonal,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar},
        parity::helpers::with_scratch,
    },
};
use poulpy_core::layouts::{Base2K, DiagonalArithmetic, GetDegree, LWEInfos};
use poulpy_hal::{
    AlignedBuf,
    api::{CnvPVecAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, DataView, DataViewMut, HostBytesBackend, Module, Standard, galois_elements_from_rotations,
    },
};

use super::helpers::{
    MASKED_PT_ALGORITHMS, PT_ALGORITHMS, assert_panics, assert_short_scratch_panics, encoded_context, map_galois_elements,
    poisoned_scratch, pt_operand,
};
use crate::{
    api::{BlasContextPreparedFactory, BlasEncodingOps, BlasPtOperandPreparedFactory},
    layouts::{
        BlasContext, BlasContextInfos, BlasContextLayout, BlasContextUnprepared, BlasModuleAlloc, BlasPlan, BlasPtAlgorithm,
        BlasPtOperand, BlasPtOperandInfos, BlasPtOperandLayout, BlasPtOperandUnprepared, BlasPtSide, Left, Right,
    },
    oep::BlasEncodingImpl,
};

/// Every block map of `b`; none under `JklsStored` and `JklsGalois`, whose blocks are plaintexts.
fn operand_maps<BE: Backend, S: BlasPtSide>(b: &BlasPtOperand<BE, S>) -> Vec<&LinearTransformationPrepared<BE>> {
    if b.blas_pt_operand_layout().plan.pt_algorithm().masked() {
        return Vec::new();
    }
    (0..b.rows())
        .flat_map(|l| (0..b.cols()).map(move |j| b.block(l, j)))
        .collect()
}

/// On blocks of `2^log_d`, each key set of the plan equals the Galois elements of the maps its operation evaluates, as
/// the context encodes them: the transpose, the row and column shifts by every `k` below `2 d`, the matrix products,
/// and the products by public operands, on either side, of any values, grid shape and count, of the counts that divide
/// the matrix slots. Public operands of `Diagonal` and `Jkls`, side and count store their blocks at one degree,
/// whatever their values, and a JKLS operand's first map and blocks rotate by its side's set; under `JklsStored` and
/// `JklsGalois`, a context of a masked plan holds the masks, and an operand's first map and the context's masks of its
/// side, phi^k on the right and the column halves of psi^k on the left, rotate by that side's set. The union of these
/// sets, and the plan's own, equals the context's, for a plan of each kind. The row and column sums rotate by `2^i d`
/// and `2^i` and their opposites, within the context's set, and their projections take no key. The context panics at
/// its allocation on matrices wider than the slots, a zero torus width, `coeffs_meta.k = 0`, a radix outside `1..=63`
/// and a prepared allocation or size of the same; a public operand whose count does not divide the matrix slots, or of
/// a zero torus width, panics at its allocation, prepared or not, and so does its prepared size. The prepare scratch
/// query panics on a layout of other slots, and a preparation into an operand of another layout, or one scratch byte
/// short of its query, masked or not, panics, as does the preparation of a context map with mixed slot kinds or of an
/// operand map whose diagonals differ in scale, before any map is prepared, and an encoding one scratch byte short of
/// its query, before any write. Each panic carries its own message.
pub fn test_blas_context<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
{
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
    let slots = params.n / 2;
    let d = 1 << log_d;
    let masked_plan = BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::JklsGalois);
    for (plan, algorithms) in [(BlasPlan::new(log_d), PT_ALGORITHMS), (masked_plan, MASKED_PT_ALGORITHMS)] {
        let ctx = encoded_context::<BE, F>(module, plan, base2k, meta);
        assert_eq!((ctx.plan(), ctx.log_d()), (plan, log_d));
        let encoded = |maps: Vec<_>| map_galois_elements(maps, order);
        let mut union = plan.transpose_galois_elements(order);
        assert_eq!(union, encoded(vec![ctx.transpose()]), "transpose {plan:?}");

        let matmul = plan.matmul_galois_elements(order);
        let mut maps = vec![ctx.sigma(), ctx.tau(), ctx.tau_rep()];
        (1..d).for_each(|k| maps.extend([ctx.rotate_cols(k), ctx.rotate_rows(k), ctx.rotate_rows_rep(k)]));
        assert_eq!(matmul, encoded(maps), "matmul {plan:?}");
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
            assert_eq!(plan_rows, encoded(rows), "rotate_rows {plan:?} k={k}");
            assert_eq!(plan_cols, encoded(cols), "rotate_cols {plan:?} k={k}");
            union.extend(plan_rows.into_iter().chain(plan_cols));
        }

        // The context's masks each side's masked products evaluate after the operand's first map.
        let (right_masks, left_masks): (Vec<_>, Vec<_>) = if plan.pt_algorithm().masked() {
            (
                (1..d).map(|k| ctx.rotate_cols(k)).collect(),
                (1..d).flat_map(|k| [ctx.rows_left(k), ctx.rows_right(k)]).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
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
            for algorithm in algorithms {
                let plan = plan.with_pt_algorithm(algorithm);
                let f = |m: usize, r: usize, c: usize| F::from_f64(entry(m, r, c)).unwrap();
                let right = pt_operand::<BE, F, Right>(module, plan, rows, cols, count, f, base2k, meta);
                let left = pt_operand::<BE, F, Left>(module, plan, rows, cols, count, f, base2k, meta);
                for (side, set, first, blocks, masks) in [
                    (
                        "matmul_pt_right",
                        &matmul_pt_right,
                        right.first(),
                        operand_maps(&right),
                        &right_masks,
                    ),
                    (
                        "matmul_pt_left",
                        &matmul_pt_left,
                        left.first(),
                        operand_maps(&left),
                        &left_masks,
                    ),
                ] {
                    assert_eq!(
                        first.is_some(),
                        algorithm != BlasPtAlgorithm::Diagonal,
                        "{side} {algorithm:?} {label} log_d={log_d}: first map"
                    );
                    assert_eq!(
                        *set,
                        map_galois_elements(
                            first.into_iter().chain(blocks.iter().copied()).chain(masks.iter().copied()),
                            order
                        ),
                        "{side} {algorithm:?} {label} log_d={log_d}"
                    );
                    if algorithm.masked() {
                        continue;
                    }
                    let diagonals = blocks
                        .iter()
                        .flat_map(|lt| lt.giant_steps.iter().flat_map(|gs| &gs.diagonals));
                    degrees
                        .entry((side, count, algorithm == BlasPtAlgorithm::Jkls))
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
        let every = encoded(ctx.maps().collect());
        assert_eq!(union, every, "union {plan:?}");
        assert_eq!(plan.galois_elements(order), every, "galois_elements {plan:?}");
        assert!(
            encoded(vec![ctx.first_row(), ctx.first_col()]).is_empty(),
            "projections {plan:?}"
        );
        for (label, set, stride) in [
            ("sum_rows", plan.sum_rows_galois_elements(order), d),
            ("sum_cols", plan.sum_cols_galois_elements(order), 1),
        ] {
            let strides = (0..log_d).flat_map(|i| [(stride << i) as i64, (slots - (stride << i)) as i64]);
            assert_eq!(set, galois_elements_from_rotations(strides, order), "{label} {plan:?}");
            assert!(
                set.iter().all(|p| every.contains(p)),
                "{label} {plan:?}: outside the context's set"
            );
        }
    }

    // The slots hold no matrix one dimension past the largest that fits; 2^40 x 2^40 has more entries than a usize
    // counts.
    let fits = slots.ilog2() as usize / 2;
    for bad in [fits + 1, 40] {
        assert_panics(&format!("log_d={bad}"), "the context's blocks do not fit its slots", || {
            module.blas_context_alloc(BlasPlan::new(bad), base2k, meta)
        });
    }

    let ctx_base = module
        .blas_context_alloc(BlasPlan::new(log_d), base2k, meta)
        .blas_context_layout();
    let ctx_with = |edit: fn(&mut BlasContextLayout)| {
        let mut layout = ctx_base;
        edit(&mut layout);
        layout
    };
    for (bad, expected) in [
        (
            ctx_with(|l| l.coeffs_meta = CoeffsMeta::from_delta_budget(0, 0)),
            "the context's maps have no torus width",
        ),
        (ctx_with(|l| l.base2k = Base2K(0)), "the context's radix is outside 1..=63"),
        (
            ctx_with(|l| l.plan = BlasPlan::new(0)),
            "the context's blocks do not fit its slots",
        ),
    ] {
        assert_panics("context prepared alloc", expected, || {
            module.blas_context_prepared_alloc_from_infos(&bad)
        });
    }
    // Twice the slots fit the blocks, so only the slot check rejects them.
    let ctx_other_slots = ctx_with(|l| l.slots *= 2);
    assert_panics(
        "context prepare_tmp_bytes slots",
        "the context's ring is not the module's",
        || module.blas_context_prepare_tmp_bytes(&ctx_other_slots),
    );
    let ctx_encode_bytes = module.blas_context_encode_tmp_bytes(&ctx_base);
    let mut ctx = module.blas_context_alloc_from_infos(&ctx_base);
    let before = buffers(ctx.maps(), plaintext::<BE>);
    assert_short_scratch_panics::<BE, _>("blas_context_encode", ctx_encode_bytes, |s| {
        module.blas_context_encode(&mut ctx, s)
    });
    assert!(
        buffers(ctx.maps(), plaintext::<BE>) == before,
        "an encoding one byte short changed its context"
    );
    with_scratch::<BE, _>(ctx_encode_bytes, |s| module.blas_context_encode(&mut ctx, s)).unwrap();
    let mut ctx_other = module.blas_context_prepared_alloc_from_infos(&ctx_with(|l| l.base2k = Base2K(l.base2k.as_u32() + 1)));
    assert_panics(
        "context prepare into another layout",
        "the prepared context was allocated for another layout",
        || with_scratch::<BE, _>(0, |s| module.blas_context_prepare(&mut ctx_other, &ctx, s)),
    );
    let mut ctx_prepared = module.blas_context_prepared_alloc_from_infos(&ctx_base);
    let ctx_bytes = module.blas_context_prepare_tmp_bytes(&ctx_base);
    assert_short_scratch_panics::<BE, _>("blas_context_prepare", ctx_bytes, |s| {
        module.blas_context_prepare(&mut ctx_prepared, &ctx, s)
    });
    // A later map with mixed diagonals: the refusal comes before the first map is prepared, on an empty arena and on
    // the query, every prepared map keeping its bytes.
    with_scratch::<BE, _>(ctx_bytes, |s| module.blas_context_prepare(&mut ctx_prepared, &ctx, s));
    let before = buffers(ctx_prepared.maps(), prepared_diagonal::<BE>);
    let mut ctx_mixed = module.blas_context_alloc_from_infos(&ctx_base);
    with_scratch::<BE, _>(ctx_encode_bytes, |s| module.blas_context_encode(&mut ctx_mixed, s)).unwrap();
    let map = ctx_mixed
        .maps
        .iter_mut()
        .skip(1)
        .find(|lt| lt.giant_steps.iter().map(|gs| gs.diagonals.len()).sum::<usize>() > 1)
        .expect("a later context map of two diagonals");
    let last = diagonals_mut([map]).last().unwrap();
    last.set_slots(if last.slots() == SlotsKind::Complex {
        SlotsKind::Real
    } else {
        SlotsKind::Complex
    });
    for arena in [0, ctx_bytes] {
        assert_panics(
            &format!("context prepare of a later map with mixed diagonals, arena of {arena} bytes"),
            "blas_context_prepare: a map's diagonals do not share one scale, width and slot kind",
            || with_scratch::<BE, _>(arena, |s| module.blas_context_prepare(&mut ctx_prepared, &ctx_mixed, s)),
        );
        assert!(
            buffers(ctx_prepared.maps(), prepared_diagonal::<BE>) == before,
            "a refused context preparation changed a prepared map"
        );
    }

    // 0 and 3 do not divide the matrix slots, a power of two.
    let generic = operands[0].4;
    for count in [0, 3] {
        assert_panics(
            &format!("count={count}"),
            "the operand's count does not divide its matrix slots",
            || module.blas_pt_operand_alloc::<Right>(BlasPlan::new(log_d), 3, 2, count, base2k, meta),
        );
    }
    let count = (slots >> (2 * log_d)).min(2);
    let f = |m: usize, r: usize, c: usize| F::from_f64(generic(m, r, c)).unwrap();
    let b = pt_operand::<BE, F, Right>(module, BlasPlan::new(log_d), 3, 2, count, f, base2k, meta);
    assert_eq!((b.log_d(), b.rows(), b.cols(), b.count()), (log_d, 3, 2, count));

    let base = BlasPtOperandLayout::<Right>::new(BlasPlan::new(log_d), slots, 1, 1, 1, base2k, meta);
    let with = |edit: fn(&mut BlasPtOperandLayout<Right>)| {
        let mut layout = base;
        edit(&mut layout);
        layout
    };
    for (bad, expected) in [
        (with(|l| l.count = 3), "the operand's count does not divide its matrix slots"),
        (with(|l| l.base2k = Base2K(0)), "the operand's radix is outside 1..=63"),
    ] {
        assert_panics("prepared alloc", expected, || {
            module.blas_pt_operand_prepared_alloc_from_infos(&bad)
        });
    }
    // Twice the slots fit the blocks, so only the slot check rejects them.
    let other_slots = with(|l| l.slots *= 2);
    assert_panics("prepare_tmp_bytes slots", "the operand's slots are not the module's", || {
        module.blas_pt_operand_prepare_tmp_bytes(&other_slots)
    });
    let encode_bytes = module.blas_pt_operand_encode_tmp_bytes(&base);
    let mut op = module.blas_pt_operand_alloc_from_infos(&base);
    let before = buffers(op.maps(), plaintext::<BE>);
    assert_short_scratch_panics::<BE, _>("blas_pt_operand_encode", encode_bytes, |s| {
        module.blas_pt_operand_encode(&mut op, f, s)
    });
    assert!(
        buffers(op.maps(), plaintext::<BE>) == before,
        "an encoding one byte short changed its operand"
    );
    with_scratch::<BE, _>(encode_bytes, |s| module.blas_pt_operand_encode(&mut op, f, s)).unwrap();
    let mut other = module.blas_pt_operand_prepared_alloc_from_infos(&with(|l| l.rows = 2));
    assert_panics(
        "prepare into another layout",
        "the prepared operand was allocated for another layout",
        || with_scratch::<BE, _>(0, |s| module.blas_pt_operand_prepare(&mut other, &op, s)),
    );
    let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&base);
    let bytes = module.blas_pt_operand_prepare_tmp_bytes(&base);
    assert_short_scratch_panics::<BE, _>("blas_pt_operand_prepare", bytes, |s| {
        module.blas_pt_operand_prepare(&mut prepared, &op, s)
    });
    // One diagonal at another scale makes its map non-uniform.
    let map = op
        .maps
        .iter_mut()
        .find(|lt| lt.giant_steps.iter().map(|gs| gs.diagonals.len()).sum::<usize>() > 1)
        .expect("an operand map of two diagonals");
    let last = &mut map.giant_steps.last_mut().unwrap().diagonals.last_mut().unwrap().plaintext;
    let mut last_meta = last.meta();
    last_meta.scale = Scale::Log(last.log_delta() + 1);
    last.set_meta(last_meta);
    assert_panics(
        "a non-uniform map",
        "blas_pt_operand_prepare: a map's diagonals do not share one scale, width and slot kind",
        || with_scratch::<BE, _>(bytes, |s| module.blas_pt_operand_prepare(&mut prepared, &op, s)),
    );
    // A later map non-uniform, the operand encoded again from other matrices: the refusal comes before the first map
    // is prepared, on an empty arena and on the query, every prepared map keeping its bytes.
    let grid = with(|l| {
        l.rows = 2;
        l.cols = 2;
    });
    let mut op = module.blas_pt_operand_alloc_from_infos(&grid);
    with_scratch::<BE, _>(module.blas_pt_operand_encode_tmp_bytes(&grid), |s| {
        module.blas_pt_operand_encode(&mut op, f, s)
    })
    .unwrap();
    let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&grid);
    let bytes = module.blas_pt_operand_prepare_tmp_bytes(&grid);
    with_scratch::<BE, _>(bytes, |s| module.blas_pt_operand_prepare(&mut prepared, &op, s));
    let before = buffers(prepared.maps(), prepared_diagonal::<BE>);
    let other = |m: usize, r: usize, c: usize| F::from_f64(((m * 7 + r + c * 2) % 9) as f64 / 4.0 - 1.0).unwrap();
    with_scratch::<BE, _>(module.blas_pt_operand_encode_tmp_bytes(&grid), |s| {
        module.blas_pt_operand_encode(&mut op, other, s)
    })
    .unwrap();
    let map = op
        .maps
        .iter_mut()
        .skip(1)
        .find(|lt| lt.giant_steps.iter().map(|gs| gs.diagonals.len()).sum::<usize>() > 1)
        .expect("a later operand map of two diagonals");
    let last = &mut map.giant_steps.last_mut().unwrap().diagonals.last_mut().unwrap().plaintext;
    let mut last_meta = last.meta();
    last_meta.scale = Scale::Log(last.log_delta() + 1);
    last.set_meta(last_meta);
    for arena in [0, bytes] {
        assert_panics(
            &format!("a later non-uniform map, arena of {arena} bytes"),
            "blas_pt_operand_prepare: a map's diagonals do not share one scale, width and slot kind",
            || with_scratch::<BE, _>(arena, |s| module.blas_pt_operand_prepare(&mut prepared, &op, s)),
        );
        assert!(
            buffers(prepared.maps(), prepared_diagonal::<BE>) == before,
            "a refused preparation changed a prepared map"
        );
    }
    // A masked operand one byte short of its query, encoded again from other matrices: the refusal comes before its
    // first map is prepared.
    let masked = with(|l| l.plan = l.plan.with_pt_algorithm(BlasPtAlgorithm::JklsStored));
    let (masked_encode, masked_bytes) = (
        module.blas_pt_operand_encode_tmp_bytes(&masked),
        module.blas_pt_operand_prepare_tmp_bytes(&masked),
    );
    let mut op = module.blas_pt_operand_alloc_from_infos(&masked);
    with_scratch::<BE, _>(masked_encode, |s| module.blas_pt_operand_encode(&mut op, f, s)).unwrap();
    let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&masked);
    with_scratch::<BE, _>(masked_bytes, |s| module.blas_pt_operand_prepare(&mut prepared, &op, s));
    let before = buffers(prepared.maps(), prepared_diagonal::<BE>);
    with_scratch::<BE, _>(masked_encode, |s| module.blas_pt_operand_encode(&mut op, other, s)).unwrap();
    assert_short_scratch_panics::<BE, _>("blas_pt_operand_prepare masked", masked_bytes, |s| {
        module.blas_pt_operand_prepare(&mut prepared, &op, s)
    });
    assert!(
        buffers(prepared.maps(), prepared_diagonal::<BE>) == before,
        "a masked preparation one byte short changed its first map"
    );

    // `coeffs_meta.k = 0` is rejected at allocation: the context and every public operand, either side.
    let zero = CoeffsMeta::from_delta_budget(0, 0);
    assert_panics("context", "the context's maps have no torus width", || {
        module.blas_context_alloc(BlasPlan::new(log_d), base2k, zero)
    });
    for algorithm in PT_ALGORITHMS.into_iter().chain(MASKED_PT_ALGORITHMS) {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        let expected = "the operand's maps have no torus width";
        assert_panics(&format!("right operand {algorithm:?}"), expected, || {
            module.blas_pt_operand_alloc::<Right>(plan, 1, 1, 1, base2k, zero)
        });
        assert_panics(&format!("left operand {algorithm:?}"), expected, || {
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

/// The prepared maps of `p`, then its terms' prepared plaintexts under `JklsStored` and `JklsGalois`.
fn prepared_operand<BE: Backend<OwnedBuf = AlignedBuf>, S: BlasPtSide>(p: &BlasPtOperand<BE, S>) -> Buffers {
    let mut maps = buffers(p.maps(), prepared_diagonal::<BE>);
    maps.extend(p.plaintexts.iter().map(|pt| {
        let b: &[u8] = pt.cnv().data();
        (b.as_ptr(), b.to_vec())
    }));
    maps
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
/// every byte of both was poisoned, against a pair filled from `second` alone.
fn check_refill<U, P>(
    label: &str,
    alloc: impl Fn() -> (U, P),
    view: fn(&U) -> Buffers,
    view_prepared: fn(&P) -> Buffers,
    poison: fn(&mut U, &mut P),
    mut fill: impl FnMut(&mut U, &mut P, bool),
) {
    let (mut u, mut p) = alloc();
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

/// A public operand of `plan` on side `S` through [`check_refill`], its second fill from other matrices than its first;
/// the terms' prepared plaintexts of a masked operand, which offer no mutable view, are compared but not poisoned.
fn refill_operand<BE, F, S>(module: &Module<BE>, plan: BlasPlan, base2k: Base2K, meta: CoeffsMeta)
where
    BE: TestContextBackend + BlasEncodingImpl,
    Module<BE>:
        CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CKKSMulOps<BE>,
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
        |op: &BlasPtOperandUnprepared<BE, S>| buffers(op.maps(), plaintext::<BE>),
        prepared_operand::<BE, S>,
        |op, p| {
            poison_plaintexts::<BE>(op.maps.iter_mut());
            poison_prepared::<BE>(p.maps.iter_mut());
        },
        |op, p, again| {
            let entry = if again { second } else { first };
            let mut encode_scratch = poisoned_scratch::<BE>(module.blas_pt_operand_encode_tmp_bytes(&layout));
            module
                .blas_pt_operand_encode(
                    op,
                    |m, r, c| F::from_f64(entry(m, r, c)).unwrap(),
                    &mut encode_scratch.borrow(),
                )
                .unwrap();
            let mut prepare_scratch = poisoned_scratch::<BE>(module.blas_pt_operand_prepare_tmp_bytes(&layout));
            module.blas_pt_operand_prepare(p, op, &mut prepare_scratch.borrow());
        },
    );
}

/// Encoding and preparing again into the same allocations overwrites every byte and equals a fresh build, for the
/// context of a plan with and without masks and for public operands of either side and algorithm, the operands from
/// other matrices than their first; every byte query equals the buffers allocated from its layout.
pub fn test_blas_refill<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CKKSLinearTransformationOps<BE>,
    F: TestScalar + DiagonalArithmetic,
{
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let masked = BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::JklsGalois);
    for plan in [BlasPlan::new(log_d), masked] {
        let layout = BlasContextLayout {
            plan,
            base2k,
            coeffs_meta: meta,
            n: module.ring_degree(),
            slots: module.ckks_max_slots(),
        };
        check_refill(
            &format!("context {:?}", plan.pt_algorithm()),
            || {
                (
                    module.blas_context_alloc_from_infos(&layout),
                    module.blas_context_prepared_alloc_from_infos(&layout),
                )
            },
            |c: &BlasContextUnprepared<BE>| buffers(c.maps(), plaintext::<BE>),
            |p: &BlasContext<BE>| buffers(p.maps(), prepared_diagonal::<BE>),
            |c, p| {
                poison_plaintexts::<BE>(c.maps.iter_mut());
                poison_prepared::<BE>(p.maps.iter_mut());
            },
            |c, p, _| {
                let mut encode_scratch = poisoned_scratch::<BE>(module.blas_context_encode_tmp_bytes(&layout));
                module.blas_context_encode(c, &mut encode_scratch.borrow()).unwrap();
                let mut prepare_scratch = poisoned_scratch::<BE>(module.blas_context_prepare_tmp_bytes(&layout));
                module.blas_context_prepare(p, c, &mut prepare_scratch.borrow());
            },
        );
    }
    for algorithm in PT_ALGORITHMS.into_iter().chain(MASKED_PT_ALGORITHMS) {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        refill_operand::<BE, F, Right>(module, plan, base2k, meta);
        refill_operand::<BE, F, Left>(module, plan, base2k, meta);
    }
}
