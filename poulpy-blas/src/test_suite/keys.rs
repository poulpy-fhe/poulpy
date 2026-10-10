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
    D, Grid, LOG_D,
    Rejection::{MissingKey, MissingTensorKey},
    StrictKeys, TestKeys, assert_guard_on, context_and_keys, encrypt_grid, grid_snapshot, lt_key_uses,
};
use crate::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight, BlasModuleAlloc,
        BlasPtOperandUnprepared, sum_galois_elements,
    },
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
/// and column shifts, products by public matrices and the preparations write results half as wide as their inputs,
/// narrower than any map's output, then one limb wider than their inputs; the matrix products run on factors of one
/// width, then on a narrower left or right factor.
pub fn test_blas_key_widths<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots / (D * D);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, mut atks) = context_and_keys::<BE, F>(&params, module, &sk_raw, &mut scratch.borrow());
    let pt = BlasPtOperandUnprepared::encode::<F>(
        module,
        LOG_D,
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
        let grid = Grid::model(rows, cols, lanes, seed);
        encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, 0, &grid, scratch)
    };
    let a_k = params.k;
    let row = encrypt(1, 2, a_k, 11, &mut scratch.borrow());
    let column = encrypt(2, 1, a_k, 12, &mut scratch.borrow());
    for p in pt
        .galois_elements(module.cyclotomic_order())
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
    let alloc =
        |rows: usize, cols: usize, k: usize| module.blas_ciphertext_alloc(LOG_D, rows, cols, params.base2k.into(), k.into());
    let uses = |lt, src_k, res_k| lt_key_uses(module, lt, src_k, res_k);
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    // Half as wide as the inputs, so that the result's width caps the maps' giant steps, then one limb wider.
    let (narrow, wide) = (a_k / 2, a_k + params.base2k);
    let bytes = module
        .blas_matmul_tmp_bytes(&alloc(1, 1, a_k), &row, &column, &tsk_layout, &atk_layout)
        .max(module.blas_matmul_pt_tmp_bytes(&alloc(1, 1, wide), &row, &atk_layout))
        .max(module.blas_rotate_rows_tmp_bytes(&alloc(2, 1, wide), &column, &atk_layout))
        .max(module.blas_rotate_tmp_bytes(&alloc(1, 2, wide), &row, &atk_layout))
        .max(module.blas_sum_tmp_bytes(&alloc(1, 2, wide), &row, &atk_layout));
    let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);

    for res_k in [narrow, wide] {
        let mut full = alloc(1, 2, res_k);
        module
            .blas_rotate(&mut full, &row, 1, &keys, &mut op_scratch.borrow())
            .unwrap();
        let rotation = [(module.ckks_galois_element((D * D) as i64), a_k)];
        assert_key_widths(
            &format!("rotate res_k={res_k}"),
            "blas_rotate",
            &keys,
            rotation,
            [],
            &full,
            &mut alloc(1, 2, res_k),
            |g| g,
            |res, keys| module.blas_rotate(res, &row, 1, keys, &mut op_scratch.borrow()),
        );

        let mut full = alloc(1, 2, res_k);
        module.blas_sum(&mut full, &row, &keys, &mut op_scratch.borrow()).unwrap();
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
            |res, keys| module.blas_sum(res, &row, keys, &mut op_scratch.borrow()),
        );

        let mut full = alloc(2, 1, res_k);
        module
            .blas_transpose(&mut full, &row, &ctx, &keys, &mut op_scratch.borrow())
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
            |res, keys| module.blas_transpose(res, &row, &ctx, keys, &mut op_scratch.borrow()),
        );

        // A shift by 1 takes both half maps along two blocks, the whole map along one.
        for a in [&row, &column] {
            let (rows, cols) = (a.rows(), a.cols());
            let case = format!("{rows}x{cols} res_k={res_k}");
            let mut full = alloc(rows, cols, res_k);
            module
                .blas_rotate_rows(&mut full, a, 1, &ctx, &keys, &mut op_scratch.borrow())
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
                |res, keys| module.blas_rotate_rows(res, a, 1, &ctx, keys, &mut op_scratch.borrow()),
            );
            let mut full = alloc(rows, cols, res_k);
            module
                .blas_rotate_cols(&mut full, a, 1, &ctx, &keys, &mut op_scratch.borrow())
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
                |res, keys| module.blas_rotate_cols(res, a, 1, &ctx, keys, &mut op_scratch.borrow()),
            );
        }

        let mut full = alloc(1, 1, res_k);
        module
            .blas_matmul_pt(&mut full, &row, &pt, &keys, &mut op_scratch.borrow())
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
            |res, keys| module.blas_matmul_pt(res, &row, &pt, keys, &mut op_scratch.borrow()),
        );

        let mut full = BlasMatMulLeft::from_grid(alloc(1, 2, res_k));
        module
            .blas_matmul_prepare_left(&mut full, &row, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_key_widths(
            &format!("prepare_left res_k={res_k}"),
            "blas_matmul_prepare_left",
            &keys,
            uses(ctx.sigma(), a_k, width(full.as_grid())),
            [],
            full.as_grid(),
            &mut BlasMatMulLeft::from_grid(alloc(1, 2, res_k)),
            |left| left.as_grid(),
            |left, keys| module.blas_matmul_prepare_left(left, &row, &ctx, keys, &mut op_scratch.borrow()),
        );
        let mut full = BlasMatMulRight::from_grid(alloc(2, 1, res_k));
        module
            .blas_matmul_prepare_right(&mut full, &column, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_key_widths(
            &format!("prepare_right res_k={res_k}"),
            "blas_matmul_prepare_right",
            &keys,
            uses(ctx.tau(), a_k, width(full.as_grid())),
            [],
            full.as_grid(),
            &mut BlasMatMulRight::from_grid(alloc(2, 1, res_k)),
            |right| right.as_grid(),
            |right, keys| module.blas_matmul_prepare_right(right, &column, &ctx, keys, &mut op_scratch.borrow()),
        );
    }

    // Factors of one width, then a left or a right factor one limb narrower.
    let short_k = a_k - params.base2k;
    let short_row = encrypt(1, 2, short_k, 13, &mut scratch.borrow());
    let short_column = encrypt(2, 1, short_k, 14, &mut scratch.borrow());
    for (a, b) in [(&row, &column), (&short_row, &column), (&row, &short_column)] {
        let factors = format!("a_k={} b_k={}", width(a), width(b));
        // The transformed operands are allocated as wide as the inputs, so every step runs at the widths `blas_matmul`
        // uses.
        let mut left = BlasMatMulLeft::from_grid(alloc(1, 2, width(a)));
        module
            .blas_matmul_prepare_left(&mut left, a, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let mut right = BlasMatMulRight::from_grid(alloc(2, 1, width(b)));
        module
            .blas_matmul_prepare_right(&mut right, b, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let (s_k, t_k) = (width(left.as_grid()), width(right.as_grid()));

        // The shifts are carved at the widths of sigma's and tau's results; phi^1 and psi^1 give the width of every
        // shift.
        let mut p = alloc(1, 1, s_k);
        module
            .ckks_eval_linear_transformation_self_into(
                p.block_mut(0, 0),
                left.as_grid().block(0, 0),
                ctx.rotate_cols(1),
                &keys,
                &mut op_scratch.borrow(),
            )
            .unwrap();
        let mut q = alloc(1, 1, t_k);
        module
            .ckks_eval_linear_transformation_self_into(
                q.block_mut(0, 0),
                right.as_grid().block(0, 0),
                ctx.rotate_rows(1),
                &keys,
                &mut op_scratch.borrow(),
            )
            .unwrap();
        let (p_k, q_k) = (width(&p), width(&q));
        let shifts: Vec<(i64, usize)> = (1..D)
            .flat_map(|k| {
                uses(ctx.rotate_cols(k), s_k, p_k)
                    .into_iter()
                    .chain(uses(ctx.rotate_rows(k), t_k, q_k))
            })
            .collect();
        let transforms: Vec<(i64, usize)> = uses(ctx.sigma(), width(a), s_k)
            .into_iter()
            .chain(uses(ctx.tau(), width(b), t_k))
            .collect();
        let products = [s_k.max(t_k), p_k.max(q_k)];

        let mut full = alloc(1, 1, a_k);
        module
            .blas_matmul(&mut full, a, b, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_key_widths(
            &format!("matmul {factors}"),
            "blas_matmul",
            &keys,
            transforms.iter().chain(&shifts).copied(),
            products,
            &full,
            &mut alloc(1, 1, a_k),
            |g| g,
            |res, keys| module.blas_matmul(res, a, b, &ctx, keys, &mut op_scratch.borrow()),
        );
        assert_key_widths(
            &format!("matmul_prepared {factors}"),
            "blas_matmul_prepared",
            &keys,
            shifts.iter().copied(),
            products,
            &full,
            &mut alloc(1, 1, a_k),
            |g| g,
            |res, keys| module.blas_matmul_prepared(res, &left, &right, &ctx, keys, &mut op_scratch.borrow()),
        );
    }
}
