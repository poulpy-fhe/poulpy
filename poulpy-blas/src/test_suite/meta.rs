//! The metadata operations leave on their results: the batch's sparsity and record, the noise tag and the slot kind.

use poulpy_ckks::{
    GLWEPlaintextInfos, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSEncodingOps, CKKSLinearTransformationOps},
    layouts::CKKSModuleAlloc,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_scratch, ckks_encrypt_pt, gen_atk, gen_sk_with_raw, gen_tsk,
            upload_pt,
        },
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{DiagonalArithmetic, LWEInfos};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        CyclotomicOrder, Data, HostBytesBackend, HostDataMut, HostDataRef, Module, Ring, ScratchArena, ScratchOwned, Standard,
        ZnxWord,
    },
};

use super::helpers::{
    D, Grid, LOG_D, Rejection::InvalidPlan, TestKeys, assert_grid_decrypt, assert_guard, context_and_keys, encrypt_grid, relabel,
    to_f,
};
use crate::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight, BlasModuleAlloc, BlasPtOperandUnprepared,
        sum_galois_elements,
    },
    oep::BlasImpl,
};

/// Asserts `g` records `mat_slots` matrices and every block has sparsity `log_sparsity` and no noise tag.
fn assert_meta<DG: Data, W: ZnxWord, R: Ring>(label: &str, g: &BlasCiphertext<DG, W, R>, log_sparsity: usize, mat_slots: usize) {
    assert_eq!(g.mat_slots(), Some(mat_slots), "{label}: mat_slots");
    for block in g.blocks() {
        assert_eq!(block.log_sparsity(), log_sparsity, "{label}: log_sparsity");
        assert!(block.noise().is_none(), "{label}: kept a noise tag");
    }
}

/// Every per-matrix map and the matrix-slot rotation give their result the sparsity and the batch record of their
/// input, whatever `res` held: transpose, row and column rotations across blocks, within one block and by whole
/// blocks, the operand preparations, the product by public matrices and the rotation, from a half-full batch into a
/// fresh grid and from a full batch into a grid labelled and recorded half full. Summing a transposed half-full batch
/// adds each of its matrices once and records them. Copies, the shifts by whole blocks and the sum of one matrix leave
/// no noise tag, and transpose keeps real slots real.
pub fn test_blas_result_meta<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    // Grids encrypted from `grid` and labelled `log_sparsity`: `grid` holds `lanes >> log_sparsity` matrices, each
    // repeated over the slots, and the grid records them.
    let encrypt = |grid: &Grid, log_sparsity: usize, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            log_sparsity,
            grid,
            scratch,
        )
    };
    let half = Grid::model(1, 2, lanes / 2, 1);
    let (half_row, half_column) = (
        encrypt(&half, 1, &mut scratch.borrow()),
        encrypt(&Grid::model(2, 1, lanes / 2, 2), 1, &mut scratch.borrow()),
    );
    let (full_row, full_column) = (
        encrypt(&Grid::model(1, 2, lanes, 3), 0, &mut scratch.borrow()),
        encrypt(&Grid::model(2, 1, lanes, 4), 0, &mut scratch.borrow()),
    );
    let one_log_sparsity = lanes.ilog2() as usize;
    let one = encrypt(&Grid::model(1, 2, 1, 5), one_log_sparsity, &mut scratch.borrow());
    let mut real = Grid::model(1, 2, lanes, 6);
    real.im.fill(0.0);
    let real = relabel(&encrypt(&real, 0, &mut scratch.borrow()), |meta| meta.slots = SlotsKind::Real);
    let order = module.cyclotomic_order();
    for p in ctx
        .plan()
        .matmul_pt_galois_elements(order)
        .into_iter()
        .chain(sum_galois_elements(module, &full_row))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let keys = TestKeys::<BE> { atks, tsk: None };
    let atk = params.atk_layout();
    let bytes = module
        .blas_rotate_rows_tmp_bytes(&full_row, &full_row, &atk)
        .max(module.blas_sum_tmp_bytes(&full_row, &full_row, &atk))
        .max(module.blas_rotate_tmp_bytes(&full_row, &full_row, &atk));
    let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
    // A fresh grid labelled `log_sparsity`, recording the batch that label proves.
    let alloc = |rows: usize, cols: usize, log_sparsity: usize| {
        let grid = module.blas_ciphertext_alloc(LOG_D, rows, cols, params.base2k.into(), params.k.into());
        BlasCiphertext::from_blocks(
            relabel(&grid, |meta| meta.log_sparsity = log_sparsity).into_blocks(),
            LOG_D,
            rows,
            cols,
        )
        .unwrap()
    };

    // A shift by 1 crosses blocks along the axis two blocks long and stays in the block along the other; a shift by `D`
    // copies whole blocks.
    for (row, column, want, stale) in [(&half_row, &half_column, 1, 0), (&full_row, &full_column, 0, 1)] {
        let label = |op: &str| format!("{op} from log_sparsity {want} into {stale}");
        let batch = lanes >> want;
        let mut res = alloc(2, 1, stale);
        module
            .blas_transpose(&mut res, row, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("transpose"), &res, want, batch);
        for k in [1, D] {
            for a in [row, column] {
                let shape = format!("{}x{} k={k}", a.rows(), a.cols());
                let mut res = alloc(a.rows(), a.cols(), stale);
                module
                    .blas_rotate_rows(&mut res, a, k, &ctx, &keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_meta(&label(&format!("rotate_rows {shape}")), &res, want, batch);
                let mut res = alloc(a.rows(), a.cols(), stale);
                module
                    .blas_rotate_cols(&mut res, a, k, &ctx, &keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_meta(&label(&format!("rotate_cols {shape}")), &res, want, batch);
            }
        }
        let mut left = BlasMatMulLeft::from_grid(alloc(1, 2, stale));
        module
            .blas_matmul_prepare_left(&mut left, row, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("prepare_left"), left.as_grid(), want, batch);
        let mut right = BlasMatMulRight::from_grid(alloc(2, 1, stale));
        module
            .blas_matmul_prepare_right(&mut right, column, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("prepare_right"), right.as_grid(), want, batch);
        let mut res = alloc(1, 1, stale);
        module
            .blas_matmul_pt(&mut res, row, &pt, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("matmul_pt"), &res, want, batch);
        let mut res = alloc(1, 2, stale);
        module.blas_rotate(&mut res, row, 1, &keys, &mut op_scratch.borrow()).unwrap();
        assert_meta(&label("rotate"), &res, want, batch);
    }

    let mut transposed = alloc(2, 1, 0);
    module
        .blas_transpose(&mut transposed, &half_row, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    let mut res = alloc(2, 1, 0);
    module
        .blas_sum(&mut res, &transposed, &keys, &mut op_scratch.borrow())
        .unwrap();
    assert_eq!(res.mat_slots(), Some(lanes / 2), "sum of a half-full transpose: mat_slots");
    assert_grid_decrypt::<BE, F, E>(
        "sum of a half-full transpose",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &half.transpose().sum_lanes(),
        &mut scratch.borrow(),
    );

    let mut res = alloc(1, 2, 0);
    module.blas_sum(&mut res, &one, &keys, &mut op_scratch.borrow()).unwrap();
    assert_meta("sum of one matrix", &res, one_log_sparsity, 1);

    let mut res = alloc(2, 1, 0);
    module
        .blas_transpose(&mut res, &real, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    assert!(
        res.blocks().iter().all(|block| block.slots() == SlotsKind::Real),
        "the transpose of a real batch has complex slots"
    );
}

/// The batch record. Blocks encrypted from compact plaintexts of `count d^2` slots record `count` matrices, and the sum
/// of a compact pair adds each matrix once. A dense plaintext added to its blocks drops their label below that record,
/// so the next operation rejects the grid before `res` changes. A product, and its prepared form, records the larger
/// batch of its factors, with either factor the sparse one, and broadcasts the smaller.
pub fn test_blas_batch_record<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let prec = params.prec();
    // Every block of `grid` encrypted from a compact plaintext of `lanes d^2` slots, at the degree the backend allocates
    // it.
    let encrypt_compact = |grid: &Grid, scratch: &mut ScratchArena<'_, BE>| {
        let count = grid.lanes();
        let encoder = ReferenceEncoder::<E>::new(count * D * D).unwrap();
        let degree = (2 * count * D * D).max(BE::MIN_DEGREE).min(params.n);
        let blocks = grid
            .blocks(count * D * D)
            .iter()
            .map(|(re, im)| {
                let mut pt = host_module.ckks_pt_coeffs_alloc(degree, params.base2k.into(), prec.k());
                pt.set_meta(prec.meta);
                encoder.encode_reim(&mut pt, &to_f::<F>(re), &to_f::<F>(im)).unwrap();
                ckks_encrypt_pt(&params, module, &sk, params.k, &pt, scratch)
            })
            .collect();
        BlasCiphertext::from_blocks(blocks, LOG_D, grid.rows, grid.cols).unwrap()
    };
    for count in [1, 2, lanes / 2] {
        let ct = encrypt_compact(&Grid::model(1, 1, count, count as u64), &mut scratch.borrow());
        assert_eq!(ct.mat_slots(), Some(count), "compact count={count}");
    }

    let pair = Grid::model(1, 2, 2, 1);
    let ct_pair = encrypt_compact(&pair, &mut scratch.borrow());
    for p in sum_galois_elements(module, &ct_pair) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let keys = TestKeys::<BE> {
        atks,
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let bytes = module.blas_sum_tmp_bytes(&ct_pair, &ct_pair, &atk_layout);
    let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
    let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 2, params.base2k.into(), params.k.into());
    module.blas_sum(&mut res, &ct_pair, &keys, &mut op_scratch.borrow()).unwrap();
    assert_eq!(res.mat_slots(), Some(2), "sum of a compact pair: mat_slots");
    assert_grid_decrypt::<BE, F, E>(
        "sum of a compact pair",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &pair.sum_lanes(),
        &mut scratch.borrow(),
    );

    let (re, im) = &Grid::model(1, 1, lanes, 2).blocks(slots)[0];
    let mut dense = host_module.ckks_pt_vec_alloc(params.base2k.into(), prec.k());
    dense.set_meta(prec.meta);
    encoder.encode_reim(&mut dense, &to_f::<F>(re), &to_f::<F>(im)).unwrap();
    let dense = upload_pt(module, &dense);
    let mut added = ct_pair.clone();
    for block in added.blocks_mut() {
        module.ckks_add_pt_vec_assign(block, &dense, &mut scratch.borrow()).unwrap();
    }
    assert_guard::<BE, _>("dense plaintext added", "blas_sum", InvalidPlan, &mut res, |res| {
        module.blas_sum(res, &added, &keys, &mut op_scratch.borrow())
    });

    let full = Grid::model(1, 1, lanes, 3);
    let ct_full = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &full,
        &mut scratch.borrow(),
    );
    let half = Grid::model(1, 1, 2, 4);
    let ct_half = encrypt_compact(&half, &mut scratch.borrow());
    let spread = Grid::from_fn(1, 1, lanes, |s, r, c| half.at(s % 2, r, c));
    for (label, (a, ct_a), (b, ct_b)) in [
        ("pair times full", (&spread, &ct_half), (&full, &ct_full)),
        ("full times pair", (&full, &ct_full), (&spread, &ct_half)),
    ] {
        // `res` starts recording the pair's batch, so that only the stamp makes it the larger one.
        let mut res = module.blas_ciphertext_alloc_from_infos(&ct_half);
        let bytes = module.blas_matmul_tmp_bytes(&res, ct_a, ct_b, &tsk_layout, &atk_layout);
        module
            .blas_matmul(
                &mut res,
                ct_a,
                ct_b,
                &ctx,
                &keys,
                &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
            )
            .unwrap();
        assert_eq!(res.mat_slots(), Some(lanes), "{label}: mat_slots");
        assert_grid_decrypt::<BE, F, E>(
            label,
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &a.matmul(b),
            &mut scratch.borrow(),
        );

        let mut left = BlasMatMulLeft::from_grid(module.blas_ciphertext_alloc_from_infos(ct_a));
        let mut right = BlasMatMulRight::from_grid(module.blas_ciphertext_alloc_from_infos(ct_b));
        let bytes = module
            .blas_matmul_prepare_left_tmp_bytes(left.as_grid(), ct_a, &atk_layout)
            .max(module.blas_matmul_prepare_right_tmp_bytes(right.as_grid(), ct_b, &atk_layout));
        let mut prepare_scratch = ScratchOwned::<BE>::alloc(bytes);
        module
            .blas_matmul_prepare_left(&mut left, ct_a, &ctx, &keys, &mut prepare_scratch.borrow())
            .unwrap();
        module
            .blas_matmul_prepare_right(&mut right, ct_b, &ctx, &keys, &mut prepare_scratch.borrow())
            .unwrap();
        let mut prepared = module.blas_ciphertext_alloc_from_infos(&ct_half);
        let bytes = module.blas_matmul_prepared_tmp_bytes(&prepared, left.as_grid(), right.as_grid(), &tsk_layout, &atk_layout);
        module
            .blas_matmul_prepared(
                &mut prepared,
                &left,
                &right,
                &ctx,
                &keys,
                &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
            )
            .unwrap();
        assert_eq!(prepared.mat_slots(), Some(lanes), "{label}: prepared mat_slots");
    }
}
