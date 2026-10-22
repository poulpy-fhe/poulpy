//! The widths every operation looks its keys up at.

use std::collections::{BTreeMap, HashSet};

use poulpy_ckks::{
    CKKSInfos, CKKSResult, CoeffsMeta, SlotsKind,
    api::{
        CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSMulOps, LinearTransformation,
        LinearTransformationPrepared,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_atk, gen_sk_with_raw, gen_tsk},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    layouts::{
        Base2K, DiagonalArithmetic, GGLWELayout, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory,
        GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos,
        ModuleCoreAlloc, TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedBackendRef,
            GLWETensorKeyPreparedToBackendRef,
        },
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, Data, HostBytesBackend, HostDataMut, HostDataRef, Module,
        PrepareHint::{OneShot, Reuse},
        Ring, ScratchArena, Standard, ZnxWord,
    },
    source::Source,
};

use super::helpers::{
    Grid, HINTS, MASKED_PT_ALGORITHMS, MIXES, PT_ALGORITHMS,
    Rejection::{MissingKey, MissingTensorKey},
    StrictKeys, TestKeys, alloc_grid, assert_grid_decrypt, assert_guard_on, at_scale, blas_context, context_and_keys,
    encoded_context, encrypt_grid, grid_snapshot, lt_key_uses, mapped_layout, matmul_left_operand, matmul_right_operand,
    other_scales, poisoned_scratch, product_layout, pt_operand,
};
use super::matmul::public_entry;
use crate::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContextInfos, BlasEncodingOps, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight,
        BlasModuleAlloc, BlasPlan, BlasPtAlgorithm, BlasPtOperandInfos, BlasPtOperandPreparedFactory, BlasPtSide, Left, Right,
        context::context_shapes, replicated, sum_galois_elements,
    },
    oep::BlasImpl,
    reference::{
        BlasKeyUses, MapDiagonal, blas_matmul_key_uses, blas_matmul_prepare_left_key_uses, blas_matmul_prepare_right_key_uses,
        blas_matmul_prepared_key_uses, blas_matmul_pt_left_key_uses, blas_matmul_pt_right_key_uses, blas_rotate_cols_key_uses,
        blas_rotate_key_uses, blas_rotate_rows_key_uses, blas_sum_key_uses, blas_transpose_key_uses, operand_shapes,
    },
};

fn width<DG: Data, W: ZnxWord, R: Ring>(g: &BlasCiphertext<DG, W, R>) -> usize {
    g.block_layout().k().as_usize()
}

/// Asserts `call` rejects `keys` answering only the rotation keys `atks`, as `(Galois element, width)`, and the tensor
/// keys at the widths `tsks`, short of any one of these keys, before the unwritten `res` changes; then runs `call` on
/// `res` under all of them and asserts it writes `full` bit for bit. The identity takes no key. `query` is the operation's
/// key-use query, which must list exactly these keys.
fn assert_key_widths<BE, X>(
    label: &str,
    op: &str,
    keys: &TestKeys<BE>,
    query: BlasKeyUses,
    atks: impl IntoIterator<Item = (i64, usize)>,
    tsks: impl IntoIterator<Item = usize>,
    full: &BlasCiphertextOwned<BE>,
    res: &mut X,
    grid: impl Fn(&X) -> &BlasCiphertextOwned<BE>,
    mut call: impl FnMut(&mut X, &StrictKeys<'_, BE>) -> CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    let mut atks: Vec<(i64, usize)> = atks.into_iter().filter(|&(p, _)| p != 1).collect();
    atks.sort_unstable();
    atks.dedup();
    let mut tsks: Vec<usize> = tsks.into_iter().collect();
    tsks.sort_unstable();
    tsks.dedup();
    let want = BlasKeyUses {
        rotations: atks.iter().map(|&(p, k)| (p, k.into())).collect(),
        tensor: tsks.first().map(|&k| k.into()),
    };
    assert_eq!(
        query, want,
        "{label}: the key-use query differs from the widths read off the results"
    );
    for (i, &(p, k)) in atks.iter().enumerate() {
        let others = atks.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &key)| key);
        let short = keys.at_widths(others, tsks.iter().copied());
        assert_guard_on::<BE, _, _>(
            &format!("{label} without p={p} at k={k}"),
            op,
            MissingKey,
            res,
            &grid,
            |res| call(res, &short),
        );
    }
    for (i, &k) in tsks.iter().enumerate() {
        let others = tsks.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &k)| k);
        let short = keys.at_widths(atks.iter().copied(), others);
        assert_guard_on::<BE, _, _>(
            &format!("{label} without the tensor key at k={k}"),
            op,
            MissingTensorKey,
            res,
            &grid,
            |res| call(res, &short),
        );
    }
    call(res, &keys.at_widths(atks.iter().copied(), tsks.iter().copied())).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert!(
        grid_snapshot::<BE, _>(grid(res)) == grid_snapshot::<BE, _>(full),
        "{label}: the result under the width-restricted keys differs"
    );
}

/// With a key store answering only at the widths poulpy-ckks uses each key at, read off real results (a rotation at its
/// input's width, a sum's rotations at its result's, baby steps at the source's width, giant steps at the result's, the
/// tensor key at the wider of the products' term widths), every operation runs and matches its result under the full
/// store bit for bit, and rejects the store short of any one of these keys before it writes. Rotations, sums,
/// transposes, row and column shifts and products by public matrices, on either side and under `Diagonal` and `Jkls`,
/// write results half as wide as their inputs, narrower than any map's output, then one limb wider than their inputs; a
/// JKLS product by public matrices looks up its first map's keys from its input's width and its blocks' from the width
/// that map leaves. The matrix products run on factors of one width, then on a narrower left or right factor; a right
/// factor holding one matrix takes [`crate::layouts::BlasContext::tau_rep`] and
/// [`crate::layouts::BlasContext::rotate_rows_rep`]. The context's maps and the public matrices sit a quarter of the
/// inputs' scale below it ([`other_scales`]), where a width taken from the inputs' scale in place of the maps' `p` or
/// the public matrices' `p_B`, or capped by it, finds no key. The preparations, under either hint, look up every
/// rotation key of their terms, each checked through the prepared product of their operand beside a cold one; the
/// prepared product looks up those of each hot side's terms and the tensor key, and of two cold operands the tensor key
/// alone. Every operation runs on exactly the scratch its own query reports.
pub fn test_blas_key_widths<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let (d, lanes) = (1 << log_d, slots >> (2 * log_d));
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    // The maps and the public matrices below the inputs' scale, where even a width capped by it differs from theirs.
    let below = at_scale(&params, other_scales(&params)[0]);
    let (ctx, mut atks) = context_and_keys::<BE, F>(&below, module, log_d, &sk_raw, &mut scratch.borrow());
    let entry = |m: usize, r: usize, c: usize| F::from_f64(((m + r * 4 + c) * 3 % 5) as f64 / 4.0 - 0.5).unwrap();
    let operands = PT_ALGORITHMS.map(|algorithm| {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        let right = pt_operand::<BE, F, Right>(
            module,
            plan,
            2,
            1,
            1,
            entry,
            params.base2k.into(),
            below.prec().into(),
            &mut scratch.borrow(),
        );
        // `1 x 2` blocks on the left, whose products take a grid of two block rows.
        let left = pt_operand::<BE, F, Left>(
            module,
            plan,
            1,
            2,
            1,
            entry,
            params.base2k.into(),
            below.prec().into(),
            &mut scratch.borrow(),
        );
        (algorithm, right, left)
    });
    let p_b = below.prec().log_delta();
    let encrypt = |rows: usize, cols: usize, k: usize, seed: u64, scratch: &mut ScratchArena<'_, BE>| {
        let grid = Grid::model(log_d, rows, cols, lanes, seed);
        encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, 0, &grid, scratch)
    };
    let a_k = params.k;
    let row = encrypt(1, 2, a_k, 11, &mut scratch.borrow());
    let column = encrypt(2, 1, a_k, 12, &mut scratch.borrow());
    let one = Grid::model(log_d, 2, 1, 1, 15);
    let spread = Grid::from_fn(log_d, 2, 1, lanes, |_, r, c| one.at(0, r, c));
    let column_one = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        a_k,
        lanes.ilog2() as usize,
        &spread,
        &mut scratch.borrow(),
    );
    assert!(replicated::<Standard>(&column_one), "the fixture holds one matrix");
    let order = module.cyclotomic_order();
    for p in ctx
        .plan()
        .matmul_pt_right_galois_elements(order)
        .into_iter()
        .chain(ctx.plan().matmul_pt_left_galois_elements(order))
        .chain(sum_galois_elements(module.cyclotomic_order(), &row))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let keys = TestKeys::<BE> {
        atks,
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let alloc = |rows: usize, cols: usize, k: usize| alloc_grid(&params, module, log_d, rows, cols, k);
    let uses = |lt: &LinearTransformationPrepared<BE>, src_k, res_k| lt_key_uses(module, lt, src_k, res_k);
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    // Every operation runs on exactly the scratch its own query reports.
    let own = |bytes: usize| poisoned_scratch::<BE>(bytes);
    // Half as wide as the inputs, so that the result's width caps the maps' giant steps, then one limb wider.
    let (narrow, wide) = (a_k / 2, a_k + params.base2k);

    for res_k in [narrow, wide] {
        let mut full = alloc(1, 2, res_k);
        let bytes = module.blas_rotate_tmp_bytes(&full, &row, &atk_layout);
        module
            .blas_rotate(&mut full, &row, 1, &keys, &mut own(bytes).borrow())
            .unwrap();
        let rotation = [(module.ckks_galois_element((d * d) as i64), a_k)];
        assert_key_widths(
            &format!("rotate res_k={res_k}"),
            "blas_rotate",
            &keys,
            blas_rotate_key_uses(module, &row, 1),
            rotation,
            [],
            &full,
            &mut alloc(1, 2, res_k),
            |g| g,
            |res, keys| module.blas_rotate(res, &row, 1, keys, &mut own(bytes).borrow()),
        );

        let mut full = alloc(1, 2, res_k);
        let bytes = module.blas_sum_tmp_bytes(&full, &row, &atk_layout);
        module.blas_sum(&mut full, &row, &keys, &mut own(bytes).borrow()).unwrap();
        let sum = sum_galois_elements(module.cyclotomic_order(), &row)
            .into_iter()
            .map(|p| (p, width(&full)));
        assert_key_widths(
            &format!("sum res_k={res_k}"),
            "blas_sum",
            &keys,
            blas_sum_key_uses(module, &full, &row),
            sum,
            [],
            &full,
            &mut alloc(1, 2, res_k),
            |g| g,
            |res, keys| module.blas_sum(res, &row, keys, &mut own(bytes).borrow()),
        );

        let mut full = alloc(2, 1, res_k);
        let bytes = module.blas_transpose_tmp_bytes(&full, &row, &ctx, &atk_layout);
        module
            .blas_transpose(&mut full, &row, &ctx, &keys, &mut own(bytes).borrow())
            .unwrap();
        let transpose = uses(ctx.transpose(), a_k, width(&full));
        assert_key_widths(
            &format!("transpose res_k={res_k}"),
            "blas_transpose",
            &keys,
            blas_transpose_key_uses(module, &full, &row, &ctx),
            transpose,
            [],
            &full,
            &mut alloc(2, 1, res_k),
            |g| g,
            |res, keys| module.blas_transpose(res, &row, &ctx, keys, &mut own(bytes).borrow()),
        );

        // A shift by 1 takes both half maps along two blocks, the whole map along one.
        for a in [&row, &column] {
            let (rows, cols) = (a.rows(), a.cols());
            let case = format!("{rows}x{cols} res_k={res_k}");
            let mut full = alloc(rows, cols, res_k);
            let bytes = module.blas_rotate_rows_tmp_bytes(&full, a, &ctx, &atk_layout);
            module
                .blas_rotate_rows(&mut full, a, 1, &ctx, &keys, &mut own(bytes).borrow())
                .unwrap();
            let maps = if rows == 1 {
                vec![ctx.rotate_rows(1)]
            } else {
                vec![ctx.rows_upper(1), ctx.rows_lower(1)]
            };
            let used: Vec<(i64, usize)> = maps.into_iter().flat_map(|lt| uses(lt, a_k, width(&full))).collect();
            assert_key_widths(
                &format!("rotate_rows {case}"),
                "blas_rotate_rows",
                &keys,
                blas_rotate_rows_key_uses(module, &full, a, 1, &ctx),
                used,
                [],
                &full,
                &mut alloc(rows, cols, res_k),
                |g| g,
                |res, keys| module.blas_rotate_rows(res, a, 1, &ctx, keys, &mut own(bytes).borrow()),
            );
            let mut full = alloc(rows, cols, res_k);
            let bytes = module.blas_rotate_cols_tmp_bytes(&full, a, &ctx, &atk_layout);
            module
                .blas_rotate_cols(&mut full, a, 1, &ctx, &keys, &mut own(bytes).borrow())
                .unwrap();
            let maps = if cols == 1 {
                vec![ctx.rotate_cols(1)]
            } else {
                vec![ctx.cols_left(1), ctx.cols_right(1)]
            };
            let used: Vec<(i64, usize)> = maps.into_iter().flat_map(|lt| uses(lt, a_k, width(&full))).collect();
            assert_key_widths(
                &format!("rotate_cols {case}"),
                "blas_rotate_cols",
                &keys,
                blas_rotate_cols_key_uses(module, &full, a, 1, &ctx),
                used,
                [],
                &full,
                &mut alloc(rows, cols, res_k),
                |g| g,
                |res, keys| module.blas_rotate_cols(res, a, 1, &ctx, keys, &mut own(bytes).borrow()),
            );
        }

        for (algorithm, pt, pt_left) in &operands {
            let staged = |first: Option<&LinearTransformationPrepared<BE>>| match first {
                None => (Vec::new(), a_k),
                Some(first) => (uses(first, a_k, a_k - p_b), a_k - p_b),
            };
            let mut full = alloc(1, 1, res_k);
            let bytes = module.blas_matmul_pt_right_tmp_bytes(&full, &row, pt, &ctx, &atk_layout);
            module
                .blas_matmul_pt_right(&mut full, &row, pt, &ctx, &keys, &mut own(bytes).borrow())
                .unwrap();
            let (first, src_k) = staged(pt.first());
            let used: Vec<(i64, usize)> = first
                .into_iter()
                .chain((0..2).flat_map(|l| uses(pt.block(l, 0), src_k, width(&full))))
                .collect();
            assert_key_widths(
                &format!("matmul_pt_right {algorithm:?} res_k={res_k}"),
                "blas_matmul_pt_right",
                &keys,
                blas_matmul_pt_right_key_uses(module, &full, &row, pt, &ctx),
                used,
                [],
                &full,
                &mut alloc(1, 1, res_k),
                |g| g,
                |res, keys| module.blas_matmul_pt_right(res, &row, pt, &ctx, keys, &mut own(bytes).borrow()),
            );

            let mut full = alloc(1, 1, res_k);
            let bytes = module.blas_matmul_pt_left_tmp_bytes(&full, pt_left, &column, &ctx, &atk_layout);
            module
                .blas_matmul_pt_left(&mut full, pt_left, &column, &ctx, &keys, &mut own(bytes).borrow())
                .unwrap();
            let (first, src_k) = staged(pt_left.first());
            let used: Vec<(i64, usize)> = first
                .into_iter()
                .chain((0..2).flat_map(|l| uses(pt_left.block(0, l), src_k, width(&full))))
                .collect();
            assert_key_widths(
                &format!("matmul_pt_left {algorithm:?} res_k={res_k}"),
                "blas_matmul_pt_left",
                &keys,
                blas_matmul_pt_left_key_uses(module, &full, pt_left, &column, &ctx),
                used,
                [],
                &full,
                &mut alloc(1, 1, res_k),
                |g| g,
                |res, keys| module.blas_matmul_pt_left(res, pt_left, &column, &ctx, keys, &mut own(bytes).borrow()),
            );

            // A replicated input, one matrix over the matrix slots: a `Diagonal` operand of one matrix takes its
            // replicated-input maps, so keys at their widths alone suffice and each one is needed.
            let mut full = alloc(1, 1, res_k);
            let bytes = module.blas_matmul_pt_left_tmp_bytes(&full, pt_left, &column_one, &ctx, &atk_layout);
            module
                .blas_matmul_pt_left(&mut full, pt_left, &column_one, &ctx, &keys, &mut own(bytes).borrow())
                .unwrap();
            let (first, src_k) = staged(pt_left.first());
            let block = |l: usize| pt_left.block_replicated(0, l).unwrap_or_else(|| pt_left.block(0, l));
            assert_eq!(
                pt_left.block_replicated(0, 0).is_some(),
                *algorithm == BlasPtAlgorithm::Diagonal && lanes > 1,
                "replicated maps of {algorithm:?}"
            );
            let used: Vec<(i64, usize)> = first
                .into_iter()
                .chain((0..2).flat_map(|l| uses(block(l), src_k, width(&full))))
                .collect();
            assert_key_widths(
                &format!("matmul_pt_left of a replicated input {algorithm:?} res_k={res_k}"),
                "blas_matmul_pt_left",
                &keys,
                blas_matmul_pt_left_key_uses(module, &full, pt_left, &column_one, &ctx),
                used,
                [],
                &full,
                &mut alloc(1, 1, res_k),
                |g| g,
                |res, keys| module.blas_matmul_pt_left(res, pt_left, &column_one, &ctx, keys, &mut own(bytes).borrow()),
            );
        }
    }

    // Factors of one width, then a left or a right factor one limb narrower.
    let short_k = a_k - params.base2k;
    let short_row = encrypt(1, 2, short_k, 13, &mut scratch.borrow());
    let short_column = encrypt(2, 1, short_k, 14, &mut scratch.borrow());
    for (a, b) in [(&row, &column), (&short_row, &column), (&row, &short_column)] {
        let factors = format!("a_k={} b_k={}", width(a), width(b));
        // The widths of the terms, read off real results: sigma (tau) evaluated into a block of its input's width,
        // then phi^1 (psi^1) of that into a block of its width, which gives the width of every term.
        let term_widths = |src: &BlasCiphertextOwned<BE>, first, shift| {
            let bytes = module.blas_matmul_prepare_left_tmp_bytes(src, &ctx, &atk_layout, Reuse);
            let mut s = alloc(1, 1, width(src));
            module
                .ckks_eval_linear_transformation_self_into(
                    s.block_mut(0, 0),
                    src.block(0, 0),
                    first,
                    &keys,
                    &mut own(bytes).borrow(),
                )
                .unwrap();
            let mut p = alloc(1, 1, width(&s));
            module
                .ckks_eval_linear_transformation_self_into(
                    p.block_mut(0, 0),
                    s.block(0, 0),
                    shift,
                    &keys,
                    &mut own(bytes).borrow(),
                )
                .unwrap();
            (width(&s), width(&p))
        };
        let (s_k, p_k) = term_widths(a, ctx.sigma(), ctx.rotate_cols(1));
        let rep = replicated::<Standard>(b);
        let (tau, psi) = (if rep { ctx.tau_rep() } else { ctx.tau() }, |k| {
            if rep { ctx.rotate_rows_rep(k) } else { ctx.rotate_rows(k) }
        });
        let (t_k, q_k) = term_widths(b, tau, psi(1));
        // The keys of each preparation: its first map from its input's width, every shift from the first map's result.
        let left_uses: Vec<(i64, usize)> = uses(ctx.sigma(), width(a), s_k)
            .into_iter()
            .chain((1..d).flat_map(|k| uses(ctx.rotate_cols(k), s_k, p_k)))
            .collect();
        let right_uses: Vec<(i64, usize)> = uses(tau, width(b), t_k)
            .into_iter()
            .chain((1..d).flat_map(|k| uses(psi(k), t_k, q_k)))
            .collect();
        // Every term sits at the shifts' width, the one width the product looks the tensor key up at.
        let products = [p_k.max(q_k)];

        let mut full = alloc(1, 1, a_k);
        let bytes = module.blas_matmul_tmp_bytes(&full, a, b, &ctx, &tsk_layout, &atk_layout);
        module
            .blas_matmul(&mut full, a, b, &ctx, &keys, &keys, &mut own(bytes).borrow())
            .unwrap();
        assert_key_widths(
            &format!("matmul {factors}"),
            "blas_matmul",
            &keys,
            blas_matmul_key_uses(module, a, b, &ctx),
            left_uses.iter().chain(&right_uses).copied(),
            products,
            &full,
            &mut alloc(1, 1, a_k),
            |g| g,
            |res, keys| module.blas_matmul(res, a, b, &ctx, keys, keys, &mut own(bytes).borrow()),
        );

        // Each preparation, under either hint, under the restricted rotation keys alone, its product with the other
        // operand cold under the full store: a missing key stops the preparation before `res` changes, whatever the
        // hint, and the operand's every term reaches `res`.
        let prepared_bytes = |left: &BlasMatMulLeft<BE>, right: &BlasMatMulRight<BE>| {
            module.blas_matmul_prepared_tmp_bytes(&alloc(1, 1, a_k), left, right, &ctx, &tsk_layout, &atk_layout)
        };
        let prepare_left = |hint, keys: &StrictKeys<'_, BE>| {
            let bytes = module.blas_matmul_prepare_left_tmp_bytes(a, &ctx, &atk_layout, hint);
            matmul_left_operand(module, a, &ctx, keys, hint, &mut own(bytes).borrow())
        };
        let prepare_right = |hint, keys: &StrictKeys<'_, BE>| {
            let bytes = module.blas_matmul_prepare_right_tmp_bytes(b, &ctx, &atk_layout, hint);
            matmul_right_operand(module, b, &ctx, keys, hint, &mut own(bytes).borrow())
        };
        let all = keys.at_widths(left_uses.iter().chain(&right_uses).copied(), products);
        let lefts = HINTS.map(|hint| prepare_left(hint, &all).unwrap());
        let rights = HINTS.map(|hint| prepare_right(hint, &all).unwrap());
        // Hot operands at `side(OneShot)`, cold ones at `side(Reuse)`.
        let side = |hint| usize::from(hint == Reuse);
        let (cold_left, cold_right) = (&lefts[side(Reuse)], &rights[side(Reuse)]);
        for hint in HINTS {
            assert_key_widths(
                &format!("prepare_left {hint:?} {factors}"),
                "blas_matmul_prepare_left",
                &keys,
                blas_matmul_prepare_left_key_uses(module, a, &ctx),
                left_uses.iter().copied(),
                [],
                &full,
                &mut alloc(1, 1, a_k),
                |g| g,
                |res, strict| {
                    let left = prepare_left(hint, strict)?;
                    let bytes = prepared_bytes(&left, cold_right);
                    module.blas_matmul_prepared(res, &left, cold_right, &ctx, &keys, &keys, &mut own(bytes).borrow())
                },
            );
            assert_key_widths(
                &format!("prepare_right {hint:?} {factors}"),
                "blas_matmul_prepare_right",
                &keys,
                blas_matmul_prepare_right_key_uses(module, b, &ctx),
                right_uses.iter().copied(),
                [],
                &full,
                &mut alloc(1, 1, a_k),
                |g| g,
                |res, strict| {
                    let right = prepare_right(hint, strict)?;
                    let bytes = prepared_bytes(cold_left, &right);
                    module.blas_matmul_prepared(res, cold_left, &right, &ctx, &keys, &keys, &mut own(bytes).borrow())
                },
            );
        }
        // The prepared product of every mix: the keys of each hot side's terms, as its preparation looks them up, and
        // the tensor key; of two cold operands, the tensor key alone.
        for (hl, hr) in MIXES {
            let (left, right) = (&lefts[side(hl)], &rights[side(hr)]);
            let hot = |hint, uses: &[(i64, usize)]| if hint == OneShot { uses.to_vec() } else { Vec::new() };
            let bytes = prepared_bytes(left, right);
            assert_key_widths(
                &format!("matmul_prepared {hl:?} x {hr:?} {factors}"),
                "blas_matmul_prepared",
                &keys,
                blas_matmul_prepared_key_uses(module, left, right, &ctx),
                hot(hl, &left_uses).into_iter().chain(hot(hr, &right_uses)),
                products,
                &full,
                &mut alloc(1, 1, a_k),
                |g| g,
                |res, keys| module.blas_matmul_prepared(res, left, right, &ctx, keys, keys, &mut own(bytes).borrow()),
            );
        }
    }
}

/// Rotation keys by `(element, width)` and tensor keys by width, each generated for its width: a look-up at `k` answers
/// with the narrowest key at least `k` wide.
struct WidthKeys<A, T> {
    atks: BTreeMap<(i64, u32), A>,
    tsks: BTreeMap<u32, T>,
}

fn no_key_this_wide(op: &'static str) -> poulpy_core::CoreError {
    poulpy_core::CoreError::GGLWEKeyUse {
        op,
        detail: "no key is wide enough".into(),
    }
}

impl<BE: Backend, A: GLWEAutomorphismKeyPreparedToBackendRef<BE>, T> GetAutomorphismKey<BE> for WidthKeys<A, T> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let (_, key) = self
            .atks
            .range((p, k.as_u32())..=(p, u32::MAX))
            .next()
            .ok_or_else(|| no_key_this_wide("get_automorphism_key"))?;
        Ok(GLWEAutomorphismKeyPreparedToBackendRef::to_backend_ref(key))
    }
}

impl<BE: Backend, A, T: GLWETensorKeyPreparedToBackendRef<BE>> GetTensorKey<BE> for WidthKeys<A, T> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        let (_, key) = self
            .tsks
            .range(k.as_u32()..)
            .next()
            .ok_or_else(|| no_key_this_wide("get_tensor_key"))?;
        Ok(GLWETensorKeyPreparedToBackendRef::to_backend_ref(key))
    }
}

/// With a store holding one key per `(element, width)` the key-use queries list, generated for that width and no other,
/// a product of two encrypted grids and a product by public matrices run and decrypt as under the full keys.
pub fn test_blas_key_stores_from_queries<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let ctx = blas_context::<BE, F>(&params, module, log_d, &mut scratch.borrow());
    let encrypt = |rows: usize, cols: usize, seed: u64, scratch: &mut ScratchArena<'_, BE>| {
        let grid = Grid::model(log_d, rows, cols, lanes, seed);
        let ct = encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, params.k, 0, &grid, scratch);
        (grid, ct)
    };
    let ((grid_a, a), (grid_b, b)) = (
        encrypt(1, 2, 21, &mut scratch.borrow()),
        encrypt(2, 1, 22, &mut scratch.borrow()),
    );
    let public = pt_operand::<BE, F, Right>(
        module,
        BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::Diagonal),
        2,
        1,
        1,
        |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    );

    let alloc = |k: usize| alloc_grid(&params, module, log_d, 1, 1, k);
    let uses = [
        blas_matmul_key_uses(module, &a, &b, &ctx),
        blas_matmul_pt_right_key_uses(module, &alloc(params.k), &a, &public, &ctx),
    ];

    // One key per `(element, width)` the queries list, laid out for that width; then one key per element, at the widest
    // width listed, which serves the narrower look-ups of its element.
    for widest_only in [false, true] {
        let mode = if widest_only {
            "one key per element"
        } else {
            "one key per pair"
        };
        let (mut product, mut by_public) = (alloc(params.k), alloc(params.k));
        let mut keys = WidthKeys {
            atks: BTreeMap::new(),
            tsks: BTreeMap::new(),
        };
        let (mut xa, mut xe) = (Source::new([3u8; 32]), Source::new([4u8; 32]));
        let mut pairs: Vec<(i64, TorusPrecision)> = uses.iter().flat_map(|u| u.rotations.iter().copied()).collect();
        if widest_only {
            let mut widest: BTreeMap<i64, TorusPrecision> = BTreeMap::new();
            for &(p, width) in &pairs {
                let w = widest.entry(p).or_insert(width);
                if width.as_usize() > w.as_usize() {
                    *w = width;
                }
            }
            pairs = widest.into_iter().collect();
        }
        let mut seen = HashSet::new();
        let dnum_at = |width: TorusPrecision| GGLWELayout::dnum_for_input(width, params.atk_layout().k_digit);
        for (p, width) in pairs {
            if !seen.insert((p, width.as_usize())) {
                continue;
            }
            let infos = GLWEAutomorphismKeyLayout {
                dnum: dnum_at(width),
                ..params.atk_layout()
            };
            let mut raw = module.glwe_automorphism_key_alloc_from_infos(&infos);
            module.glwe_automorphism_key_encrypt_sk(&mut raw, p, &sk_raw, &mut xe, &mut xa, &mut scratch.borrow());
            let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&infos);
            module.glwe_automorphism_key_prepare(&mut prepared, &raw, &mut scratch.borrow());
            keys.atks.insert((p, width.as_u32()), prepared);
        }
        for width in uses.iter().filter_map(|u| u.tensor).take(1) {
            let infos = GLWETensorKeyLayout {
                dnum: dnum_at(width),
                ..params.tsk_layout()
            };
            let mut raw = module.glwe_tensor_key_alloc_from_infos(&infos);
            module.glwe_tensor_key_encrypt_sk(&mut raw, &sk_raw, &mut xe, &mut xa, &mut scratch.borrow());
            let mut prepared = module.alloc_tensor_key_prepared_from_infos(&infos);
            module.prepare_tensor_key(&mut prepared, &raw, &mut scratch.borrow());
            keys.tsks.insert(width.as_u32(), prepared);
        }

        let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
        let bytes = module.blas_matmul_tmp_bytes(&product, &a, &b, &ctx, &tsk_layout, &atk_layout);
        module
            .blas_matmul(
                &mut product,
                &a,
                &b,
                &ctx,
                &keys,
                &keys,
                &mut poisoned_scratch::<BE>(bytes).borrow(),
            )
            .unwrap();
        assert_grid_decrypt::<BE, F, E>(
            &format!("matmul under the queries' keys, {mode}"),
            &params,
            module,
            &encoder,
            &product,
            &sk,
            &grid_a.matmul(&grid_b),
            &product_layout::<BE::Ring>(&a, &b, 1, 1, params.k, params.prec().log_delta()),
            &mut scratch.borrow(),
        );
        let bytes = module.blas_matmul_pt_right_tmp_bytes(&by_public, &a, &public, &ctx, &atk_layout);
        module
            .blas_matmul_pt_right(
                &mut by_public,
                &a,
                &public,
                &ctx,
                &keys,
                &mut poisoned_scratch::<BE>(bytes).borrow(),
            )
            .unwrap();
        assert_grid_decrypt::<BE, F, E>(
            &format!("matmul_pt_right under the queries' keys, {mode}"),
            &params,
            module,
            &encoder,
            &by_public,
            &sk,
            &grid_a.matmul_public(1, 1, public_entry),
            &mapped_layout(&a, 1, 1, params.k, params.prec().log_delta()),
            &mut scratch.borrow(),
        );
    }
}

type Shape = (Vec<i64>, Vec<(i64, Vec<i64>)>, Option<(TorusPrecision, usize, SlotsKind)>);

impl<D: Data, W: ZnxWord, R: Ring> MapDiagonal for CKKSPlaintext<D, W, R> {
    fn width(&self) -> TorusPrecision {
        self.k()
    }
}

/// A map's schedule and its first diagonal's width, scale and slot kind: what the queries read of it.
fn shape_of<P: MapDiagonal>(lt: &LinearTransformation<P>) -> Shape {
    (
        lt.baby_steps.clone(),
        lt.giant_steps
            .iter()
            .map(|gs| (gs.rot, gs.diagonals.iter().map(|d| d.baby).collect()))
            .collect(),
        lt.first_diagonal_plaintext()
            .map(|d| (d.width(), d.lt_log_scale(), d.lt_slots())),
    )
}

/// The shapes the queries read from the layout of a public operand of `plan` on side `S` are those of its encoded maps,
/// and their first ones those of its prepared maps: under `JklsStored` and `JklsGalois` the first map alone, its terms
/// being plaintexts.
fn assert_operand_shapes<BE, F, S>(
    module: &Module<BE>,
    plan: BlasPlan,
    base2k: Base2K,
    meta: CoeffsMeta,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CKKSMulOps<BE>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
    S: BlasPtSide,
{
    let label = format!("left={} {:?} log_d={}", S::LEFT, plan.pt_algorithm(), plan.log_d());
    let entry = |m: usize, r: usize, c: usize| F::from_f64(((m + r * 3 + c) % 5) as f64 - 2.0).unwrap();
    let mut op = module.blas_pt_operand_alloc::<S>(plan, 2, 2, 1, base2k, meta);
    module.blas_pt_operand_encode_into(&mut op, entry, scratch).unwrap();
    let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&op);
    module.blas_pt_operand_prepare(&mut prepared, &op, scratch);
    let shapes: Vec<Shape> = operand_shapes(&op.blas_pt_operand_layout()).iter().map(shape_of).collect();
    assert!(op.maps().map(shape_of).eq(shapes.iter().cloned()), "{label}: encoded");
    let prepared: Vec<Shape> = prepared.maps().map(shape_of).collect();
    assert!(shapes.starts_with(&prepared) && !prepared.is_empty(), "{label}: prepared");
}

/// The shapes the queries read from a layout are those of the maps encoded and prepared from it, for the context of a
/// plan of each kind and for public operands of either side and algorithm: the same schedules, and diagonals of the same
/// width, scale and slot kind. So every query gives on a layout what it gives on the encoded maps.
pub fn test_blas_queries_from_layouts<BE, F, E>(
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
    let masked = BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::JklsGalois);
    for plan in [BlasPlan::new(log_d), masked] {
        let ctx = encoded_context::<BE, F>(module, plan, base2k, meta, &mut scratch.borrow());
        let shapes = context_shapes(&ctx.blas_context_layout());
        assert!(
            shapes.iter().map(shape_of).eq(ctx.maps.iter().map(shape_of)),
            "context {plan:?}"
        );
    }
    for algorithm in PT_ALGORITHMS.into_iter().chain(MASKED_PT_ALGORITHMS) {
        let plan = BlasPlan::new(log_d).with_pt_algorithm(algorithm);
        assert_operand_shapes::<BE, F, Right>(module, plan, base2k, meta, &mut scratch.borrow());
        assert_operand_shapes::<BE, F, Left>(module, plan, base2k, meta, &mut scratch.borrow());
    }
}
