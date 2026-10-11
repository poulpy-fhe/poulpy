//! The widths every operation looks its keys up at.

use poulpy_ckks::{
    CKKSResult,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, CKKSModuleInfos},
    layouts::CKKSCiphertext,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_atk, gen_sk_with_raw, gen_tsk},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{DiagonalArithmetic, GLWEToBackendRef, LWEInfos};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, Data, HostBytesBackend, HostDataMut, HostDataRef, Module, Ring, ScratchArena, ScratchOwned,
        Standard, ZnxWord,
    },
};

use super::helpers::{
    Grid,
    Rejection::{MissingKey, MissingTensorKey},
    StrictKeys, TestKeys, alloc_grid, assert_guard_on, context_and_keys, encrypt_grid, grid_snapshot, lt_key_uses,
};
use crate::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{BlasCiphertext, BlasCiphertextOwned, BlasGridInfos, BlasPtOperandUnprepared, sum_galois_elements},
    oep::BlasImpl,
};

fn width<DG: Data, W: ZnxWord, R: Ring>(g: &BlasCiphertext<DG, W, R>) -> usize {
    g.block_layout().k().as_usize()
}

/// Asserts `call` rejects `keys` answering only the rotation keys `atks`, as `(Galois element, width)`, and the tensor
/// keys at the widths `tsks`, short of any one of these keys, before the unwritten `res` changes; then runs `call` on
/// `res` under all of them and asserts it writes `full` bit for bit. The identity takes no key.
#[allow(clippy::too_many_arguments)]
fn assert_key_widths<BE, X>(
    label: &str,
    op: &str,
    keys: &TestKeys<BE>,
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
/// tensor key at the wider factor of each product), every operation runs and matches its result under the full store
/// bit for bit, and rejects the store short of any one of these keys before it writes. Rotations, sums, transposes, row
/// and column shifts and products by public matrices write results half as wide as their inputs, narrower than any
/// map's output, then one limb wider than their inputs; the matrix products run on factors of one width, then on a
/// narrower left or right factor. The preparations look up every rotation key of the product, each checked through
/// the prepared product of its operand under the full store, and the prepared product the tensor key alone. Every call
/// runs on exactly the scratch its own query reports.
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
    let (ctx, mut atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let pt = BlasPtOperandUnprepared::encode::<F>(
        module,
        log_d,
        2,
        1,
        1,
        |m, r, c| F::from_f64(((m + r * 4 + c) * 3 % 5) as f64 / 4.0 - 0.5).unwrap(),
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    )
    .unwrap()
    .prepare(module, &mut scratch.borrow());
    let encrypt = |rows: usize, cols: usize, k: usize, seed: u64, scratch: &mut ScratchArena<'_, BE>| {
        let grid = Grid::model(log_d, rows, cols, lanes, seed);
        encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, 0, &grid, scratch)
    };
    let a_k = params.k;
    let row = encrypt(1, 2, a_k, 11, &mut scratch.borrow());
    let column = encrypt(2, 1, a_k, 12, &mut scratch.borrow());
    for p in ctx
        .plan()
        .matmul_pt_galois_elements(module.cyclotomic_order())
        .into_iter()
        .chain(sum_galois_elements(module, &row))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let keys = TestKeys::<BE> {
        atks,
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let alloc = |rows: usize, cols: usize, k: usize| alloc_grid(&params, module, log_d, rows, cols, k);
    let uses = |lt, src_k, res_k| lt_key_uses(module, lt, src_k, res_k);
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    // Every call runs on exactly the scratch its own query reports.
    let own = |bytes: usize| ScratchOwned::<BE>::alloc(bytes);
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
        let sum = sum_galois_elements(module, &row).into_iter().map(|p| (p, width(&full)));
        assert_key_widths(
            &format!("sum res_k={res_k}"),
            "blas_sum",
            &keys,
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
                used,
                [],
                &full,
                &mut alloc(rows, cols, res_k),
                |g| g,
                |res, keys| module.blas_rotate_cols(res, a, 1, &ctx, keys, &mut own(bytes).borrow()),
            );
        }

        let mut full = alloc(1, 1, res_k);
        let bytes = module.blas_matmul_pt_tmp_bytes(&full, &row, &pt, &atk_layout);
        module
            .blas_matmul_pt(&mut full, &row, &pt, &keys, &mut own(bytes).borrow())
            .unwrap();
        let used: Vec<(i64, usize)> = (0..2).flat_map(|l| uses(pt.block(l, 0), a_k, width(&full))).collect();
        assert_key_widths(
            &format!("matmul_pt res_k={res_k}"),
            "blas_matmul_pt",
            &keys,
            used,
            [],
            &full,
            &mut alloc(1, 1, res_k),
            |g| g,
            |res, keys| module.blas_matmul_pt(res, &row, &pt, keys, &mut own(bytes).borrow()),
        );
    }

    // Factors of one width, then a left or a right factor one limb narrower.
    let short_k = a_k - params.base2k;
    let short_row = encrypt(1, 2, short_k, 13, &mut scratch.borrow());
    let short_column = encrypt(2, 1, short_k, 14, &mut scratch.borrow());
    for (a, b) in [(&row, &column), (&short_row, &column), (&row, &short_column)] {
        let factors = format!("a_k={} b_k={}", width(a), width(b));
        // The widths of the terms, read off real results: sigma (tau) evaluated into a block of its input's width,
        // then phi^1 (psi^1) of that into a block of its width, which gives the width of every shift.
        let term_widths = |src: &BlasCiphertextOwned<BE>, first, shift| {
            let bytes = module.blas_matmul_prepare_left_tmp_bytes(src, &ctx, &atk_layout);
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
        let (t_k, q_k) = term_widths(b, ctx.tau(), ctx.rotate_rows(1));
        // The keys of each preparation: its first map from its input's width, every shift from the first map's result.
        let left_uses: Vec<(i64, usize)> = uses(ctx.sigma(), width(a), s_k)
            .into_iter()
            .chain((1..d).flat_map(|k| uses(ctx.rotate_cols(k), s_k, p_k)))
            .collect();
        let right_uses: Vec<(i64, usize)> = uses(ctx.tau(), width(b), t_k)
            .into_iter()
            .chain((1..d).flat_map(|k| uses(ctx.rotate_rows(k), t_k, q_k)))
            .collect();
        let products = [s_k.max(t_k), p_k.max(q_k)];

        let mut full = alloc(1, 1, a_k);
        let bytes = module.blas_matmul_tmp_bytes(&full, a, b, &ctx, &tsk_layout, &atk_layout);
        module
            .blas_matmul(&mut full, a, b, &ctx, &keys, &keys, &mut own(bytes).borrow())
            .unwrap();
        assert_key_widths(
            &format!("matmul {factors}"),
            "blas_matmul",
            &keys,
            left_uses.iter().chain(&right_uses).copied(),
            products,
            &full,
            &mut alloc(1, 1, a_k),
            |g| g,
            |res, keys| module.blas_matmul(res, a, b, &ctx, keys, keys, &mut own(bytes).borrow()),
        );

        // Each preparation under the restricted rotation keys alone, its product with the other operand under the full
        // store: a missing key stops the preparation before `res` changes, and the operand's every term reaches `res`.
        let (left_bytes, right_bytes) = (
            module.blas_matmul_prepare_left_tmp_bytes(a, &ctx, &atk_layout),
            module.blas_matmul_prepare_right_tmp_bytes(b, &ctx, &atk_layout),
        );
        let left = module
            .blas_matmul_prepare_left(a, &ctx, &keys, &mut own(left_bytes).borrow())
            .unwrap();
        let right = module
            .blas_matmul_prepare_right(b, &ctx, &keys, &mut own(right_bytes).borrow())
            .unwrap();
        let prepared_bytes = module.blas_matmul_prepared_tmp_bytes(&alloc(1, 1, a_k), &left, &right, &tsk_layout);
        assert_key_widths(
            &format!("prepare_left {factors}"),
            "blas_matmul_prepare_left",
            &keys,
            left_uses.iter().copied(),
            [],
            &full,
            &mut alloc(1, 1, a_k),
            |g| g,
            |res, strict| {
                let left = module.blas_matmul_prepare_left(a, &ctx, strict, &mut own(left_bytes).borrow())?;
                module.blas_matmul_prepared(res, &left, &right, &keys, &mut own(prepared_bytes).borrow())
            },
        );
        assert_key_widths(
            &format!("prepare_right {factors}"),
            "blas_matmul_prepare_right",
            &keys,
            right_uses.iter().copied(),
            [],
            &full,
            &mut alloc(1, 1, a_k),
            |g| g,
            |res, strict| {
                let right = module.blas_matmul_prepare_right(b, &ctx, strict, &mut own(right_bytes).borrow())?;
                module.blas_matmul_prepared(res, &left, &right, &keys, &mut own(prepared_bytes).borrow())
            },
        );
        assert_key_widths(
            &format!("matmul_prepared {factors}"),
            "blas_matmul_prepared",
            &keys,
            [],
            products,
            &full,
            &mut alloc(1, 1, a_k),
            |g| g,
            |res, keys| module.blas_matmul_prepared(res, &left, &right, keys, &mut own(prepared_bytes).borrow()),
        );
    }
}
