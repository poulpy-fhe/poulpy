//! The plan's key sets against the encoded maps of a context and of public operands.

use std::collections::{BTreeMap, BTreeSet};

use poulpy_ckks::{
    CoeffsMeta,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, LinearTransformationPrepared},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch},
    },
};
use poulpy_core::layouts::{DiagonalArithmetic, LWEInfos};
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, Module, ScratchArena, Standard},
};

use super::helpers::{Rejection, assert_rejected, context_maps, map_galois_elements};
use crate::layouts::{BlasContextUnprepared, BlasPlan, BlasPtOperand, BlasPtOperandUnprepared};

/// Every block map of `b`.
fn operand_maps<BE: Backend, S>(b: &BlasPtOperand<BE, S>) -> Vec<&LinearTransformationPrepared<BE>> {
    (0..b.rows())
        .flat_map(|l| (0..b.cols()).map(move |j| b.block(l, j)))
        .collect()
}

/// At each `log_d` from 1 to 3 whose blocks fit the `S` slots, `d^2 <= S`, each key set of the plan equals the Galois
/// elements of the maps its operation evaluates, as the context encodes them: the transpose, the row and column shifts
/// by every `k` below `2 d`, the matrix products, and the products by public operands, on either side, of any values,
/// grid shape and count, of the counts that divide the matrix slots. Public operands of one side and count store their
/// diagonals at one degree, whatever their values. The union of these sets, and the plan's own, equals the context's.
/// A `log_d` whose blocks do not fit is skipped with a message. The context rejects matrices wider than the slots; a
/// public operand's count divides the matrix slots. Both reject a zero torus width, `coeffs_meta.k = 0`, with
/// poulpy-ckks's `InvalidOperand`.
pub fn test_blas_context<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
{
    let mut scratch = alloc_scratch(&params, module);
    let order = module.cyclotomic_order();
    type Entry = fn(usize, usize, usize) -> f64;
    let encode = |log_d: usize, rows: usize, cols: usize, count: usize, entry: Entry, scratch: &mut ScratchArena<'_, BE>| {
        BlasPtOperandUnprepared::encode(
            module,
            log_d,
            rows,
            cols,
            count,
            move |m, r, c| F::from_f64(entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
            scratch,
        )
    };
    let encode_left = |log_d: usize, rows: usize, cols: usize, count: usize, entry: Entry, scratch: &mut ScratchArena<'_, BE>| {
        BlasPtOperandUnprepared::encode_left(
            module,
            log_d,
            rows,
            cols,
            count,
            move |m, r, c| F::from_f64(entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
            scratch,
        )
    };
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
        let ctx = BlasContextUnprepared::encode::<F>(
            module,
            plan,
            params.base2k.into(),
            params.prec().into(),
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(module, &mut scratch.borrow());
        assert_eq!((*ctx.plan(), ctx.log_d()), (plan, log_d));
        let d = 1 << log_d;
        let encoded = |maps: Vec<_>| map_galois_elements(maps, order);
        let mut union = plan.transpose_galois_elements(order);
        assert_eq!(union, encoded(vec![ctx.transpose()]), "transpose log_d={log_d}");

        let matmul = plan.matmul_galois_elements(order);
        let mut maps = vec![ctx.sigma(), ctx.tau()];
        (1..d).for_each(|k| maps.extend([ctx.rotate_cols(k), ctx.rotate_rows(k)]));
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

        let (matmul_pt, matmul_pt_left) = (
            plan.matmul_pt_galois_elements(order),
            plan.matmul_pt_left_galois_elements(order),
        );
        // The degrees the diagonals of each side's and count's operands are stored at.
        let mut degrees = BTreeMap::<(&str, usize), BTreeSet<usize>>::new();
        for (label, rows, cols, count, entry) in operands {
            if !(slots >> (2 * log_d)).is_multiple_of(count) {
                continue;
            }
            let right = encode(log_d, rows, cols, count, entry, &mut scratch.borrow())
                .unwrap()
                .prepare(module, &mut scratch.borrow());
            let left = encode_left(log_d, rows, cols, count, entry, &mut scratch.borrow())
                .unwrap()
                .prepare(module, &mut scratch.borrow());
            for (side, set, blocks) in [
                ("matmul_pt", &matmul_pt, operand_maps(&right)),
                ("matmul_pt_left", &matmul_pt_left, operand_maps(&left)),
            ] {
                assert_eq!(
                    *set,
                    map_galois_elements(blocks.iter().copied(), order),
                    "{side} {label} log_d={log_d}"
                );
                let diagonals = blocks
                    .iter()
                    .flat_map(|lt| lt.giant_steps.iter().flat_map(|gs| &gs.diagonals));
                degrees
                    .entry((side, count))
                    .or_default()
                    .extend(diagonals.map(|diagonal| diagonal.plaintext.n().as_usize()));
            }
        }
        for ((side, count), degrees) in degrees {
            assert_eq!(degrees.len(), 1, "{side} degrees {degrees:?} count={count} log_d={log_d}");
        }
        union.extend(matmul_pt.into_iter().chain(matmul_pt_left));
        union.sort_unstable();
        union.dedup();
        let every = encoded(context_maps(&ctx));
        assert_eq!(union, every, "union log_d={log_d}");
        assert_eq!(plan.galois_elements(order), every, "galois_elements log_d={log_d}");
    }

    // The slots hold no matrix one dimension past the largest that fits; 2^40 x 2^40 has more entries than a usize
    // counts.
    for log_d in [fits + 1, 40] {
        let res = BlasContextUnprepared::encode::<F>(
            module,
            BlasPlan::new(log_d),
            params.base2k.into(),
            params.prec().into(),
            &mut scratch.borrow(),
        );
        assert!(res.is_err());
    }

    // Matrix slots of 4 x 4 blocks, or of 2 x 2 blocks where those do not fit, are a power of two: 3 and 0 public
    // matrices do not divide them; 2 does, or 1 on one matrix slot.
    let (log_d, generic) = (fits.min(2), operands[0].4);
    for count in [0, 3] {
        assert!(encode(log_d, 3, 2, count, generic, &mut scratch.borrow()).is_err());
    }
    let count = (slots >> (2 * log_d)).min(2);
    let b = encode(log_d, 3, 2, count, generic, &mut scratch.borrow())
        .unwrap()
        .prepare(module, &mut scratch.borrow());
    assert_eq!((b.log_d(), b.rows(), b.cols(), b.count()), (log_d, 3, 2, count));

    // poulpy-ckks's encoder rejects a zero torus width with `InvalidOperand`, for the context and for a public operand
    // on either side.
    let zero = CoeffsMeta::from_delta_budget(0, 0);
    let one = |_: usize, _: usize, _: usize| F::from_f64(1.0).unwrap();
    let rejected = [
        (
            "context",
            BlasContextUnprepared::encode::<F>(
                module,
                BlasPlan::new(log_d),
                params.base2k.into(),
                zero,
                &mut scratch.borrow(),
            )
            .map(drop),
        ),
        (
            "right operand",
            BlasPtOperandUnprepared::encode(module, log_d, 1, 1, 1, one, params.base2k.into(), zero, &mut scratch.borrow())
                .map(drop),
        ),
        (
            "left operand",
            BlasPtOperandUnprepared::encode_left(module, log_d, 1, 1, 1, one, params.base2k.into(), zero, &mut scratch.borrow())
                .map(drop),
        ),
    ];
    for (label, res) in rejected {
        assert_rejected(
            label,
            "ckks_encode_linear_transformation_from_diagonals",
            Rejection::InvalidOperand,
            res.map_err(Into::into),
        );
    }
}
