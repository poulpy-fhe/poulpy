//! The metadata operations leave on their results: the batch's sparsity and record, the noise tag and the slot kind.

use poulpy_ckks::{
    CKKSInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, SetCKKSInfos, SetGLWEPlaintextInfos, SlotsKind,
    api::{CKKSAddOps, CKKSEncodingOps, CKKSLinearTransformationOps, CKKSModuleInfos, LtDiagonalMeta},
    layouts::CKKSPlaintextAlloc,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_scratch, ckks_encrypt_pt, gen_atk, gen_sk_with_raw, gen_tsk,
            upload_pt,
        },
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{DiagonalArithmetic, GLWE, LWEInfos};
use poulpy_hal::{
    api::{CnvPVecAlloc, ReimNew, ScratchOwnedBorrow},
    layouts::{
        CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module,
        PrepareHint::{OneShot, Reuse},
        ScratchArena, Standard,
    },
};

use super::helpers::{
    Grid, HINTS, MIXES, Rejection::InvalidOperand, TestKeys, alloc_grid, assert_grid_decrypt, assert_guard, context_and_keys,
    copied_layout, encrypt_grid, mapped_layout, matmul_left_operand, matmul_right_operand, operand_infos, poisoned_scratch,
    product_layout, pt_operand, relabel, rotation_keys, to_f, want_operand_infos,
};
use super::keys::slot_mask;
use super::matvec::{
    append, append_tmp_bytes, assert_vector_decrypt, diagonal_ciphertext_elements, diagonal_ciphertext_layout, encrypt_vector,
    matvec_diagonal_layout, matvec_layouts, matvec_model, matvec_operands, matvec_operands_of, prepared_of, vector_entry,
};
use super::repack::{
    GRID, grid_model, grid_to_vector_cases, line_values, matrix_times, placed, repack_fits, vector_to_grid_cases,
};
use crate::{
    api::{BlasEncodingOps, BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps, BlasMatVecOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContextInfos, BlasGridInfos, BlasGridMut, BlasLayout, BlasMatVecOperandLayout,
        BlasModuleAlloc, BlasPlan, BlasVectorInfos, BlasVectorShape, Left, Right, blas_grid_to_vector_output_layout,
        blas_matmul_left_operand_layout, blas_matmul_left_term_layout, blas_matmul_output_layout,
        blas_matmul_prepared_output_layout, blas_matmul_pt_left_output_layout, blas_matmul_pt_right_output_layout,
        blas_matmul_right_operand_layout, blas_matmul_right_term_layout, blas_matvec_diagonal_output_layout,
        blas_matvec_pt_output_layout, blas_rotate_cols_output_layout, blas_rotate_output_layout, blas_rotate_rows_output_layout,
        blas_sum_cols_output_layout, blas_sum_output_layout, blas_sum_rows_output_layout, blas_transpose_output_layout,
        blas_vector_to_grid_output_layout, diagonal_ciphertext::AppendAxis, matvec_pt_galois_elements, replicated,
        sum_galois_elements,
    },
    oep::BlasImpl,
};

/// Every per-matrix map, the matrix-slot rotation and the sum give their result the metadata [`copied_layout`] or
/// [`mapped_layout`] derives from their input, whatever `res` held, and the values of their cleartext model: transpose,
/// row and column rotations across blocks, within one block and by whole blocks, row rotations of a grid one block high
/// holding one matrix, which copy its layout, row and column sums, canonical, the products by public matrices on either side and
/// the rotation, from a half-full batch into a fresh grid and from a full batch into a grid labelled and recorded half
/// full. The operand preparations report their hint, their input's dimensions, batch and layout as
/// their source, and as their layout their input's when hot and that of their terms when cold, sigma (tau) then a
/// shift of its blocks, at no scale on a right input holding one matrix, which [`blas_matmul_left_term_layout`]
/// ([`blas_matmul_right_term_layout`]) gives, as [`blas_matmul_left_operand_layout`]
/// ([`blas_matmul_right_operand_layout`]) gives them all before preparing. Summing a transposed half-full batch adds
/// each of its matrices once, canonical, and records them. The sum of one matrix copies it, the transpose of a real batch keeps its
/// slots real, and the terms of an integer batch are real, while its hot operand reports its own slots. On one matrix
/// slot no batch is half full: only the full one runs. A product of a vector into a `res` that held another
/// encryption, tagged and flag-clear, stamps the metadata of its output-layout query, canonical and untagged, and so
/// does a repacking each way, its label proving the operand's output period and a grid result recording `output_period
/// / d^2`, where its cases fit the slots. The appends of a diagonal ciphertext keep its layout, and its product is real by a real vector, complex by a complex
/// one, at the layout its doc gives, from 16 slots.
pub fn test_blas_result_meta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let (d, lanes) = (1 << log_d, slots >> (2 * log_d));
    let (k, p) = (params.k, params.prec().log_delta());
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, mut atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let entry = |m: usize, r: usize, c: usize| ((m + r * 4 + c) * 3 % 5) as f64 / 4.0 - 0.5;
    let pt = pt_operand::<BE, F, Right>(
        module,
        BlasPlan::new(log_d),
        2,
        1,
        1,
        |m, r, c| F::from_f64(entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
    );
    // `1 x 2` blocks on the left, whose product takes the two block rows of a column.
    let pt_left = pt_operand::<BE, F, Left>(
        module,
        BlasPlan::new(log_d),
        1,
        2,
        1,
        |m, r, c| F::from_f64(entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
    );
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
    let full = (Grid::model(log_d, 1, 2, lanes, 3), Grid::model(log_d, 2, 1, lanes, 4));
    let (full_row, full_column) = (
        encrypt(&full.0, 0, &mut scratch.borrow()),
        encrypt(&full.1, 0, &mut scratch.borrow()),
    );
    let one_log_sparsity = lanes.ilog2() as usize;
    let one_model = Grid::model(log_d, 1, 2, 1, 5);
    let one = encrypt(&one_model, one_log_sparsity, &mut scratch.borrow());
    let mut real_model = Grid::model(log_d, 1, 2, lanes, 6);
    real_model.im.fill(0.0);
    let real = relabel(&encrypt(&real_model, 0, &mut scratch.borrow()), |meta| {
        meta.slots = SlotsKind::Real
    });
    let order = module.cyclotomic_order();
    for p in ctx
        .plan()
        .matmul_pt_right_galois_elements(order)
        .into_iter()
        .chain(ctx.plan().matmul_pt_left_galois_elements(order))
        .chain(sum_galois_elements(module.cyclotomic_order(), &full_row))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let keys = TestKeys::<BE> { atks, tsk: None };
    let atk = params.atk_layout();
    let bytes = module
        .blas_transpose_tmp_bytes(&full_column, &full_row, &ctx, &atk)
        .max(module.blas_transpose_tmp_bytes(&full_row, &full_column, &ctx, &atk))
        .max(module.blas_rotate_rows_tmp_bytes(&full_row, &full_row, &ctx, &atk))
        .max(module.blas_rotate_cols_tmp_bytes(&full_row, &full_row, &ctx, &atk))
        .max(module.blas_rotate_rows_tmp_bytes(&full_column, &full_column, &ctx, &atk))
        .max(module.blas_rotate_cols_tmp_bytes(&full_column, &full_column, &ctx, &atk))
        .max(module.blas_sum_rows_tmp_bytes(&full_row, &full_row, &ctx, &atk))
        .max(module.blas_sum_cols_tmp_bytes(&full_row, &full_row, &ctx, &atk))
        .max(module.blas_sum_rows_tmp_bytes(&full_column, &full_column, &ctx, &atk))
        .max(module.blas_sum_cols_tmp_bytes(&full_column, &full_column, &ctx, &atk))
        .max(module.blas_sum_tmp_bytes(&full_row, &full_row, &atk))
        .max(module.blas_rotate_tmp_bytes(&full_row, &full_row, &atk))
        .max(module.blas_matmul_prepare_left_tmp_bytes(&full_row, &ctx, &atk, Reuse))
        .max(module.blas_matmul_prepare_right_tmp_bytes(&full_column, &ctx, &atk, Reuse))
        .max(module.blas_matmul_prepare_left_tmp_bytes(&full_row, &ctx, &atk, OneShot))
        .max(module.blas_matmul_prepare_right_tmp_bytes(&full_column, &ctx, &atk, OneShot))
        .max(module.blas_matmul_pt_right_tmp_bytes(&full_row, &full_row, &pt, &ctx, &atk))
        .max(module.blas_matmul_pt_left_tmp_bytes(&full_row, &pt_left, &full_column, &ctx, &atk));
    let mut op_scratch = poisoned_scratch::<BE>(bytes);
    // A fresh grid labelled `log_sparsity`, recording the batch that label proves.
    let alloc = |rows: usize, cols: usize, log_sparsity: usize| {
        let grid = alloc_grid(&params, module, log_d, rows, cols, params.k);
        BlasCiphertext::from_blocks(
            relabel(&grid, |meta| meta.log_sparsity = log_sparsity).into_blocks(),
            log_d,
            rows,
            cols,
        )
        .unwrap()
    };
    let check =
        |label: &str, res: &BlasCiphertextOwned<BE>, want: &Grid, layout: &BlasLayout, scratch: &mut ScratchArena<'_, BE>| {
            assert_grid_decrypt::<BE, F, E>(label, &params, module, &encoder, res, &sk, want, layout, scratch)
        };

    // A half-full batch into a fresh grid, then a full batch into a grid labelled and recorded half full; on one matrix
    // slot, the full batch into a fresh grid alone.
    let half = (Grid::model(log_d, 1, 2, lanes / 2, 1), Grid::model(log_d, 2, 1, lanes / 2, 2));
    let half_grids = (lanes > 1).then(|| {
        (
            encrypt(&half.0, 1, &mut scratch.borrow()),
            encrypt(&half.1, 1, &mut scratch.borrow()),
        )
    });
    let mut cases: Vec<_> = half_grids.iter().map(|(row, column)| (row, column, &half, 1, 0)).collect();
    cases.push((&full_row, &full_column, &full, 0, usize::from(lanes > 1)));
    // A shift by 1 crosses blocks along the axis two blocks long and stays in the block along the other; a shift by `d`
    // copies whole blocks, and a row shift of a grid one block high holding one matrix rotates them.
    for (row, column, (row_model, column_model), want, stale) in cases {
        let label = |op: &str| format!("{op} from log_sparsity {want} into {stale}");
        let mut res = alloc(2, 1, stale);
        module
            .blas_transpose(&mut res, row, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let layout = mapped_layout(row, 2, 1, k, p);
        assert_eq!(
            blas_transpose_output_layout(&res, row, &ctx),
            layout,
            "{}",
            label("transpose query")
        );
        check(
            &label("transpose"),
            &res,
            &row_model.transpose(),
            &layout,
            &mut scratch.borrow(),
        );
        for shift in [1, d] {
            for (a, model) in [(row, row_model), (column, column_model)] {
                let (rows, cols) = (a.rows(), a.cols());
                let shape = format!("{rows}x{cols} k={shift}");
                let layout = if shift == d {
                    copied_layout(a, rows, cols, k)
                } else {
                    mapped_layout(a, rows, cols, k, p)
                };
                let rotates = rows == 1 && replicated::<BE::Ring>(a);
                let rows_layout = if rotates { copied_layout(a, rows, cols, k) } else { layout };
                let mut res = alloc(rows, cols, stale);
                module
                    .blas_rotate_rows(shift, &mut res, a, &ctx, &keys, &mut op_scratch.borrow())
                    .unwrap();
                let op = label(&format!("rotate_rows {shape}"));
                assert_eq!(
                    blas_rotate_rows_output_layout(shift, &res, a, &ctx),
                    rows_layout,
                    "{op}: query"
                );
                check(&op, &res, &model.rotate_rows(shift), &rows_layout, &mut scratch.borrow());
                let mut res = alloc(rows, cols, stale);
                module
                    .blas_rotate_cols(shift, &mut res, a, &ctx, &keys, &mut op_scratch.borrow())
                    .unwrap();
                let op = label(&format!("rotate_cols {shape}"));
                assert_eq!(blas_rotate_cols_output_layout(shift, &res, a, &ctx), layout, "{op}: query");
                check(&op, &res, &model.rotate_cols(shift), &layout, &mut scratch.borrow());
            }
        }
        // A replicated grid sums its rows without the projection.
        for (a, model) in [(row, row_model), (column, column_model)] {
            let (rows, cols) = (a.rows(), a.cols());
            for along_rows in [true, false] {
                let layout = if along_rows && replicated::<BE::Ring>(a) {
                    copied_layout(a, rows, cols, k)
                } else {
                    mapped_layout(a, rows, cols, k, p)
                };
                let mut res = alloc(rows, cols, stale);
                let (op, query, want) = if along_rows {
                    module
                        .blas_sum_rows(&mut res, a, &ctx, &keys, &mut op_scratch.borrow())
                        .unwrap();
                    ("sum_rows", blas_sum_rows_output_layout(&res, a, &ctx), model.sum_rows())
                } else {
                    module
                        .blas_sum_cols(&mut res, a, &ctx, &keys, &mut op_scratch.borrow())
                        .unwrap();
                    ("sum_cols", blas_sum_cols_output_layout(&res, a, &ctx), model.sum_cols())
                };
                let op = label(&format!("{op} {rows}x{cols}"));
                assert_eq!(query, layout, "{op}: query");
                // The last inner sum or replicate leaves every block canonical.
                assert!(res.blocks().iter().all(|block| block.is_canonical()), "{op}: canonical");
                check(&op, &res, &want, &layout, &mut scratch.borrow());
            }
        }
        for hint in HINTS {
            let left = matmul_left_operand(module, row, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap();
            let right = matmul_right_operand(module, column, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap();
            for (op, operand, a, is_right) in [
                ("prepare_left", operand_infos(&left), row, false),
                ("prepare_right", operand_infos(&right), column, true),
            ] {
                let op = &label(&format!("{op} {hint:?}"));
                let scale = if is_right && replicated::<BE::Ring>(a) { p } else { 2 * p };
                let terms = mapped_layout(a, a.rows(), a.cols(), k, scale);
                let term_layout = if is_right {
                    blas_matmul_right_term_layout(a, &ctx)
                } else {
                    blas_matmul_left_term_layout(a, &ctx)
                };
                assert_eq!(term_layout, terms, "{op}: term layout");
                let want = want_operand_infos(a, hint, terms);
                assert_eq!(operand, want, "{op}: operand infos");
                let before = if is_right {
                    blas_matmul_right_operand_layout(a, &ctx, hint)
                } else {
                    blas_matmul_left_operand_layout(a, &ctx, hint)
                };
                assert_eq!(
                    (before.grid, before.mat_slots(), before.hint, before.source),
                    (want.0, None, want.2, want.3),
                    "{op}: layout before preparing"
                );
            }
        }
        let mut res = alloc(1, 1, stale);
        module
            .blas_matmul_pt_right(&mut res, row, &pt, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let (want_pt, layout) = (row_model.matmul_public(1, 1, entry), mapped_layout(row, 1, 1, k, p));
        assert_eq!(
            blas_matmul_pt_right_output_layout(&res, row, &pt, &ctx),
            layout,
            "{}",
            label("pt_right query")
        );
        check(&label("matmul_pt_right"), &res, &want_pt, &layout, &mut scratch.borrow());
        let mut res = alloc(1, 1, stale);
        module
            .blas_matmul_pt_left(&mut res, &pt_left, column, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let (want_left, layout) = (column_model.public_matmul(1, 1, entry), mapped_layout(column, 1, 1, k, p));
        assert_eq!(
            blas_matmul_pt_left_output_layout(&res, &pt_left, column, &ctx),
            layout,
            "{}",
            label("pt_left query")
        );
        check(&label("matmul_pt_left"), &res, &want_left, &layout, &mut scratch.borrow());
        let mut res = alloc(1, 2, stale);
        module.blas_rotate(1, &mut res, row, &keys, &mut op_scratch.borrow()).unwrap();
        let layout = copied_layout(row, 1, 2, k);
        assert_eq!(blas_rotate_output_layout(&res, row), layout, "{}", label("rotate query"));
        check(
            &label("rotate"),
            &res,
            &row_model.rotate_lanes(1),
            &layout,
            &mut scratch.borrow(),
        );
    }

    if let Some((half_row, _)) = &half_grids {
        let mut transposed = alloc(2, 1, 0);
        module
            .blas_transpose(&mut transposed, half_row, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let mut res = alloc(2, 1, 0);
        module
            .blas_sum(&mut res, &transposed, &keys, &mut op_scratch.borrow())
            .unwrap();
        let want = half.0.transpose().sum_lanes();
        let layout = copied_layout(&transposed, 2, 1, k);
        assert_eq!(blas_sum_output_layout(&res, &transposed), layout, "sum query");
        assert!(res.blocks().iter().all(|block| block.is_canonical()), "sum: canonical");
        check("sum of a half-full transpose", &res, &want, &layout, &mut scratch.borrow());
    }

    let mut res = alloc(1, 2, 0);
    module.blas_sum(&mut res, &one, &keys, &mut op_scratch.borrow()).unwrap();
    let layout = copied_layout(&one, 1, 2, k);
    assert_eq!(blas_sum_output_layout(&res, &one), layout, "sum of one query");
    check("sum of one matrix", &res, &one_model, &layout, &mut scratch.borrow());

    let mut res = alloc(2, 1, 0);
    module
        .blas_transpose(&mut res, &real, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    let layout = mapped_layout(&real, 2, 1, k, p);
    assert_eq!(
        blas_transpose_output_layout(&res, &real, &ctx),
        layout,
        "real transpose query"
    );
    assert_eq!(layout.block.slots(), SlotsKind::Real);
    check(
        "transpose of a real batch",
        &res,
        &real_model.transpose(),
        &layout,
        &mut scratch.borrow(),
    );

    // Labelled integer, the batch's terms take the maps' `Real` slots; its hot operand reports its own.
    let integer = relabel(&real, |meta| meta.slots = SlotsKind::Integer);
    let layout = mapped_layout(&integer, 1, 2, k, 2 * p);
    assert_eq!(layout.block.slots(), SlotsKind::Real);
    assert_eq!(
        blas_matmul_left_term_layout(&integer, &ctx),
        layout,
        "integer batch: term layout"
    );
    for hint in HINTS {
        let left = matmul_left_operand(module, &integer, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap();
        assert_eq!(
            operand_infos(&left),
            want_operand_infos(&integer, hint, layout),
            "integer batch: {hint:?} operand infos"
        );
    }

    // A product of a vector into a `res` that held another encryption, tagged with its fresh noise and one ciphertext
    // flag-clear: the metadata its output-layout query gives, canonical and untagged, and the values of its model.
    if slots >= 16 {
        let layout = matvec_layouts(&params, slots).pop().unwrap();
        let (_, w) = matvec_operands::<BE, F>(module, &layout);
        let elements = matvec_pt_galois_elements(module.cyclotomic_order(), &layout);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
            tsk: None,
        };
        let vector = |shape: BlasVectorShape, period: usize, scratch: &mut ScratchArena<'_, BE>| {
            let v: Vec<f64> = (0..shape.len).map(vector_entry).collect();
            let ct = encrypt_vector::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, shape, period, &v, scratch);
            (v, ct)
        };
        let (v, x) = vector(layout.input, layout.input_period, &mut scratch.borrow());
        let (_, mut res) = vector(layout.output, layout.output_period, &mut scratch.borrow());
        GLWE::set_canonical(res.ct_mut(0), false);
        let want = blas_matvec_pt_output_layout(&res, &layout, &x);
        let bytes = module.blas_matvec_pt_tmp_bytes(&res, &w, &x, &params.atk_layout());
        module
            .blas_matvec_pt(&mut res, &w, &x, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            .unwrap();
        assert!(
            res.cts().iter().all(|ct| GLWE::is_canonical(ct)),
            "matvec_pt over a used res: canonical"
        );
        assert_vector_decrypt::<BE, F, E>(
            "matvec_pt over a used res",
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &matvec_model(layout.output.len, &v),
            layout.output_period,
            &want,
            &mut scratch.borrow(),
        );
    }

    // A repacking each way into a `res` that held another encryption, tagged with its fresh noise and one ciphertext
    // flag-clear: the metadata its output-layout query gives, canonical and untagged, its label proving the operand's
    // output period, a grid recording `output_period / d^2` over the dense batch it held, and the values of its model.
    if repack_fits(slots, log_d) {
        let proves = |layout: &BlasMatVecOperandLayout| (slots / layout.output_period).ilog2() as usize;
        let keys_of = |layout: &BlasMatVecOperandLayout, scratch: &mut ScratchArena<'_, BE>| TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, matvec_pt_galois_elements(order, layout), scratch),
            tsk: None,
        };

        let case = grid_to_vector_cases(&params, slots, log_d).swap_remove(0);
        let (line, layout) = (case.line, case.layout);
        let (_, w) = matvec_operands_of::<BE, F>(module, &layout, case.entry);
        let keys = keys_of(&layout, &mut scratch.borrow());
        let model = grid_model(log_d, case.proven);
        let mut a = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            k,
            case.label(slots),
            &model,
            &mut scratch.borrow(),
        );
        BlasGridMut::<BE>::set_mat_slots(&mut a, GRID.2);
        let used: Vec<f64> = (0..layout.output.len).map(vector_entry).collect();
        let mut res = encrypt_vector::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            k,
            layout.output,
            slots,
            &used,
            &mut scratch.borrow(),
        );
        GLWE::set_canonical(res.ct_mut(0), false);
        let want = blas_grid_to_vector_output_layout(&res, &layout, &a);
        assert_eq!(
            want.ct.log_sparsity(),
            proves(&layout),
            "grid_to_vector over a used res: label"
        );
        let bytes = module.blas_grid_to_vector_tmp_bytes(&res, &w, &a, &params.atk_layout());
        module
            .blas_grid_to_vector(&mut res, &w, &a, line, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            .unwrap();
        assert!(
            res.cts().iter().all(|ct| GLWE::is_canonical(ct)),
            "grid_to_vector over a used res: canonical"
        );
        assert_vector_decrypt::<BE, F, E>(
            "grid_to_vector over a used res",
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &matrix_times(case.entry, layout.output.len, &line_values(&model, line, case.proven)),
            layout.output_period,
            &want,
            &mut scratch.borrow(),
        );

        let case = vector_to_grid_cases(&params, slots, log_d).swap_remove(0);
        let (line, layout) = (case.line, case.layout);
        let (_, w) = matvec_operands_of::<BE, F>(module, &layout, case.entry);
        let keys = keys_of(&layout, &mut scratch.borrow());
        let v: Vec<f64> = (0..layout.input.len).map(vector_entry).collect();
        let a = encrypt_vector::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            k,
            layout.input,
            layout.input_period,
            &v,
            &mut scratch.borrow(),
        );
        let used = Grid::model(log_d, case.rows, case.cols, lanes, 41);
        let mut res = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            k,
            0,
            &used,
            &mut scratch.borrow(),
        );
        GLWE::set_canonical(&mut res.blocks_mut()[0], false);
        let want = blas_vector_to_grid_output_layout(&res, &layout, &a);
        assert_eq!(
            want.block.log_sparsity(),
            proves(&layout),
            "vector_to_grid over a used res: label"
        );
        let bytes = module.blas_vector_to_grid_tmp_bytes(&res, &w, &a, &params.atk_layout());
        module
            .blas_vector_to_grid(&mut res, line, &w, &a, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            .unwrap();
        assert!(
            res.blocks().iter().all(|block| block.is_canonical()),
            "vector_to_grid over a used res: canonical"
        );
        assert_eq!(
            res.mat_slots(),
            Some(layout.output_period >> (2 * log_d)),
            "vector_to_grid over a used res: batch record"
        );
        assert_grid_decrypt::<BE, F, E>(
            "vector_to_grid over a used res",
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &placed(&case, &matrix_times(case.entry, layout.output.len, &v)),
            &want,
            &mut scratch.borrow(),
        );
    }

    // A diagonal ciphertext of real diagonals keeps its layout through appends of a real vector by a real mask, its
    // diagonals canonical and untagged; its product into a `res` of other metadata is real by a real vector and complex
    // by a complex one, canonical and untagged, at the layout of its doc and of its output query.
    if slots < 16 {
        eprintln!("result_meta: the diagonal ciphertext skipped, its diagonals span 16 slots");
        return;
    }
    let real_params = CKKSTestParams {
        prec_meta: GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            ..params.prec_meta
        },
        ..params
    };
    let whole = BlasVectorShape::new(slots, slots, 0, 1);
    let layout = diagonal_ciphertext_layout(&real_params, slots, k - p);
    let keys = TestKeys::<BE> {
        atks: rotation_keys(
            &params,
            module,
            &sk_raw,
            diagonal_ciphertext_elements(order, &layout),
            &mut scratch.borrow(),
        ),
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let values: Vec<f64> = (0..slots).map(vector_entry).collect();
    let complex = encrypt_vector::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        k,
        whole,
        slots,
        &values,
        &mut scratch.borrow(),
    );
    let mut real = complex.clone();
    real.cts_mut()[0].set_slots(SlotsKind::Real);
    let mask = slot_mask::<BE, F, E>(&real_params, module, host_module, &encoder, &[0]);
    let mut m = module.blas_diagonal_ciphertext_alloc_from_infos(&layout);
    for (axis, at) in [(AppendAxis::Row, 0), (AppendAxis::Col, 2)] {
        let bytes = append_tmp_bytes(module, axis, &m, &real, &mask, &params.atk_layout());
        append(
            module,
            axis,
            &mut m,
            at,
            &real,
            &mask,
            &keys,
            &mut poisoned_scratch::<BE>(bytes).borrow(),
        )
        .unwrap();
        for d in m.diagonals().giant_steps.iter().flat_map(|gs| &gs.diagonals) {
            let diagonal = &d.plaintext;
            assert_eq!(
                (diagonal.k(), diagonal.meta()),
                (layout.ct.k(), layout.ct.meta),
                "{}: the diagonals' layout",
                axis.op()
            );
            assert!(
                diagonal.is_canonical() && diagonal.noise().is_none(),
                "{}: canonical, untagged",
                axis.op()
            );
        }
    }
    let prepared = prepared_of(module, &m);
    assert!(
        prepared.diagonals().first_diagonal_plaintext().map(|d| d.lt_slots()) == Some(layout.ct.slots()),
        "prepared: the diagonals' slots"
    );
    for (a, slots_kind) in [(&real, SlotsKind::Real), (&complex, SlotsKind::Complex)] {
        let want = matvec_diagonal_layout(a.ct_layout(), layout.ct, k);
        assert_eq!(want.slots(), slots_kind);
        for form in ["resident", "streamed"] {
            let mut res = complex.clone();
            res.cts_mut()[0].set_log_delta(p / 2);
            assert_eq!(
                blas_matvec_diagonal_output_layout(&res, &m, a).ct,
                want,
                "matvec {slots_kind:?}: output query"
            );
            let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
            if form == "resident" {
                let bytes = module.blas_matvec_diagonal_tmp_bytes(&res, &prepared, a, &tsk_layout, &atk_layout);
                module.blas_matvec_diagonal(&mut res, &prepared, a, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            } else {
                let bytes = module.blas_matvec_diagonal_tmp_bytes(&res, &m, a, &tsk_layout, &atk_layout);
                module.blas_matvec_diagonal(&mut res, &m, a, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            }
            .unwrap();
            let ct = &res.cts()[0];
            assert_eq!(res.ct_layout(), want, "matvec {slots_kind:?} {form}: layout");
            assert!(
                ct.is_canonical() && ct.noise().is_none(),
                "matvec {slots_kind:?} {form}: canonical, untagged"
            );
        }
    }
}

/// The batch record. Blocks encrypted from compact plaintexts of `count d^2` slots record `count` matrices, and the sum
/// of a compact pair adds each matrix once. A dense plaintext added to its blocks drops their label below that record,
/// so the next operation rejects the grid before `res` changes. A product, and its prepared form of every mix of hot
/// and cold operands, records the larger batch of its factors, with either factor the sparse one or a replicated one,
/// holding one matrix, and broadcasts the smaller; a prepared operand records its input's, under either hint. A vector
/// written as a grid's line records `output_period / d^2` over the dense batch `res` held, for every vector-to-grid
/// case. The pair must leave matrix slots free, so below four of them the test is skipped with a message.
pub fn test_blas_batch_record<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let (d, lanes) = (1 << log_d, slots >> (2 * log_d));
    if lanes < 4 {
        eprintln!("test_blas_batch_record skips log_d {log_d}: it needs 4 d^2 <= {slots} slots");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, mut atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let prec = params.prec();
    // Every block of `grid` encrypted from a compact plaintext of `lanes d^2` slots, at the degree the backend allocates
    // it.
    let encrypt_compact = |grid: &Grid, scratch: &mut ScratchArena<'_, BE>| {
        let count = grid.lanes();
        let encoder = ReferenceEncoder::<E>::new(count * d * d).unwrap();
        let degree = (2 * count * d * d).max(BE::MIN_DEGREE).min(params.n);
        let blocks = grid
            .blocks(count * d * d)
            .iter()
            .map(|(re, im)| {
                let mut pt = host_module.ckks_plaintext_alloc(degree.into(), params.base2k.into(), prec.k());
                pt.set_meta(prec.meta);
                encoder.encode_reim(&mut pt, &to_f::<F>(re), &to_f::<F>(im)).unwrap();
                ckks_encrypt_pt(&params, module, &sk, params.k, &pt, scratch)
            })
            .collect();
        BlasCiphertext::from_blocks(blocks, log_d, grid.rows, grid.cols).unwrap()
    };
    for count in [1, 2, lanes / 2] {
        let ct = encrypt_compact(&Grid::model(log_d, 1, 1, count, count as u64), &mut scratch.borrow());
        assert_eq!(ct.mat_slots(), Some(count), "compact count={count}");
    }

    let pair = Grid::model(log_d, 1, 2, 2, 1);
    let ct_pair = encrypt_compact(&pair, &mut scratch.borrow());
    for p in sum_galois_elements(module.cyclotomic_order(), &ct_pair) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let keys = TestKeys::<BE> {
        atks,
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let bytes = module.blas_sum_tmp_bytes(&ct_pair, &ct_pair, &atk_layout);
    let mut op_scratch = poisoned_scratch::<BE>(bytes);
    let mut res = alloc_grid(&params, module, log_d, 1, 2, params.k);
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
        &copied_layout(&ct_pair, 1, 2, params.k),
        &mut scratch.borrow(),
    );

    let (re, im) = &Grid::model(log_d, 1, 1, lanes, 2).blocks(slots)[0];
    let mut dense = host_module.ckks_plaintext_alloc_from_slots(host_module.ckks_max_slots(), params.base2k.into(), prec.k());
    dense.set_meta(prec.meta);
    encoder.encode_reim(&mut dense, &to_f::<F>(re), &to_f::<F>(im)).unwrap();
    let dense = upload_pt(module, &dense);
    let mut added = ct_pair.clone();
    for block in added.blocks_mut() {
        module.ckks_add_pt_vec_assign(block, &dense, &mut scratch.borrow()).unwrap();
    }
    assert_guard::<BE, _>("dense plaintext added", "blas_sum", InvalidOperand, &mut res, |res| {
        module.blas_sum(res, &added, &keys, &mut op_scratch.borrow())
    });

    let full = Grid::model(log_d, 1, 1, lanes, 3);
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
    let half = Grid::model(log_d, 1, 1, 2, 4);
    let ct_half = encrypt_compact(&half, &mut scratch.borrow());
    let spread = Grid::from_fn(log_d, 1, 1, lanes, |s, r, c| half.at(s % 2, r, c));
    // One matrix repeated over the slots: on the right its psi^k are rotations at no scale.
    let one = Grid::model(log_d, 1, 1, 1, 5);
    let ct_one = encrypt_compact(&one, &mut scratch.borrow());
    assert!(replicated::<BE::Ring>(&ct_one), "the fixture holds one matrix");
    let spread_one = Grid::from_fn(log_d, 1, 1, lanes, |_, r, c| one.at(0, r, c));
    for (label, (a, ct_a), (b, ct_b)) in [
        ("pair times full", (&spread, &ct_half), (&full, &ct_full)),
        ("full times pair", (&full, &ct_full), (&spread, &ct_half)),
        ("one times full", (&spread_one, &ct_one), (&full, &ct_full)),
        ("full times one", (&full, &ct_full), (&spread_one, &ct_one)),
    ] {
        // `res` starts recording the pair's batch, so that only the stamp makes it the larger one.
        let mut res = module.blas_ciphertext_alloc_from_infos(&ct_half);
        let bytes = module.blas_matmul_tmp_bytes(&res, ct_a, ct_b, &ctx, &tsk_layout, &atk_layout);
        module
            .blas_matmul(
                &mut res,
                ct_a,
                ct_b,
                &ctx,
                &keys,
                &keys,
                &mut poisoned_scratch::<BE>(bytes).borrow(),
            )
            .unwrap();
        assert_eq!(res.mat_slots(), Some(lanes), "{label}: mat_slots");
        let product = product_layout::<BE::Ring>(ct_a, ct_b, 1, 1, params.k, params.prec().log_delta());
        assert_eq!(blas_matmul_output_layout(&res, ct_a, ct_b, &ctx), product, "{label}: query");
        assert_grid_decrypt::<BE, F, E>(
            label,
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &a.matmul(b),
            &product,
            &mut scratch.borrow(),
        );

        let prepare = |hint| {
            let bytes = module
                .blas_matmul_prepare_left_tmp_bytes(ct_a, &ctx, &atk_layout, hint)
                .max(module.blas_matmul_prepare_right_tmp_bytes(ct_b, &ctx, &atk_layout, hint));
            let mut prepare_scratch = poisoned_scratch::<BE>(bytes);
            let left = matmul_left_operand(module, ct_a, &ctx, &keys, hint, &mut prepare_scratch.borrow()).unwrap();
            let right = matmul_right_operand(module, ct_b, &ctx, &keys, hint, &mut prepare_scratch.borrow()).unwrap();
            assert_eq!(
                (left.mat_slots(), right.mat_slots()),
                (ct_a.mat_slots(), ct_b.mat_slots()),
                "{label}: {hint:?} operand records"
            );
            (left, right)
        };
        let [hot, cold] = HINTS.map(prepare);
        for (hl, hr) in MIXES {
            let left = if hl == OneShot { &hot.0 } else { &cold.0 };
            let right = if hr == OneShot { &hot.1 } else { &cold.1 };
            let mut prepared = module.blas_ciphertext_alloc_from_infos(&ct_half);
            let bytes = module.blas_matmul_prepared_tmp_bytes(&prepared, left, right, &ctx, &tsk_layout, &atk_layout);
            module
                .blas_matmul_prepared(
                    &mut prepared,
                    left,
                    right,
                    &ctx,
                    &keys,
                    &keys,
                    &mut poisoned_scratch::<BE>(bytes).borrow(),
                )
                .unwrap();
            let product = product_layout::<BE::Ring>(ct_a, ct_b, 1, 1, params.k, params.prec().log_delta());
            assert_eq!(
                blas_matmul_prepared_output_layout(&prepared, left, right, &ctx),
                product,
                "{label}: prepared {hl:?} x {hr:?} query"
            );
            assert_eq!(
                prepared.mat_slots(),
                Some(lanes),
                "{label}: prepared {hl:?} x {hr:?} mat_slots"
            );
        }
    }

    // A vector written as a grid's line records the batch of the operand's output period over the dense batch `res`
    // held.
    if !repack_fits(slots, log_d) {
        eprintln!("test_blas_batch_record: the repacking skipped, its cases span 4 d^2 and 16 slots");
        return;
    }
    for case in vector_to_grid_cases(&params, slots, log_d) {
        let (line, layout) = (case.line, case.layout);
        let (_, w) = matvec_operands_of::<BE, F>(module, &layout, case.entry);
        let elements = matvec_pt_galois_elements(module.cyclotomic_order(), &layout);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
            tsk: None,
        };
        let v: Vec<f64> = (0..layout.input.len).map(vector_entry).collect();
        let a = encrypt_vector::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            layout.input,
            layout.input_period,
            &v,
            &mut scratch.borrow(),
        );
        let mut res = alloc_grid(&params, module, log_d, case.rows, case.cols, params.k);
        assert_eq!(
            res.mat_slots(),
            Some(lanes),
            "vector_to_grid {line:?}: the dense batch before"
        );
        let bytes = module.blas_vector_to_grid_tmp_bytes(&res, &w, &a, &atk_layout);
        module
            .blas_vector_to_grid(&mut res, line, &w, &a, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            .unwrap();
        assert_eq!(
            res.mat_slots(),
            Some(layout.output_period >> (2 * log_d)),
            "vector_to_grid {line:?}: mat_slots"
        );
        assert_grid_decrypt::<BE, F, E>(
            &format!("vector_to_grid {line:?}"),
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &placed(&case, &matrix_times(case.entry, layout.output.len, &v)),
            &blas_vector_to_grid_output_layout(&res, &layout, &a),
            &mut scratch.borrow(),
        );
    }
}
