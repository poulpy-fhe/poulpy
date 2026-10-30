//! Every operation on the conjugate-invariant ring, whose slots are real.

use std::collections::BTreeSet;

use poulpy_ckks::{
    CKKSInfos, CKKSModuleInfos, GLWEPlaintextInfos, SetCKKSInfos, SetGLWEPlaintextInfos, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSLinearTransformationOps, CKKSMulOps},
    layouts::CKKSModuleAlloc,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextSharedModule, alloc_scratch, assert_precision, ckks_decrypt_with_prec,
            ckks_encrypt_pt, gen_sk_with_raw, gen_tsk, precision_at,
        },
        parity::helpers::with_scratch,
    },
};
use poulpy_core::layouts::{LWEInfos, prepared::GLWESecretPrepared};
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedBorrow},
    layouts::{
        ConjugateInvariant, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, PrepareHint, ScratchArena,
    },
};

use super::{
    helpers::{
        Grid, HINTS, MASKED_PT_ALGORITHMS, MIXES, PT_ALGORITHMS, TestKeys, alloc_grid, copied_layout, encoded_context,
        mapped_layout, matmul_left_operand, matmul_right_operand, product_layout, pt_operand, rotation_keys,
    },
    matmul::public_entry,
    matvec::{
        DiagonalCiphertextModel, alloc_vector, append, append_tmp_bytes, diagonal_ciphertext_elements,
        diagonal_ciphertext_layout, matvec_layouts, matvec_model, matvec_operands, prepared_of, vector_entry, vector_slots,
    },
};
use crate::{
    api::{BlasEncodingOps, BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps, BlasMatVecOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContextInfos, BlasGridInfos, BlasLayout, BlasModuleAlloc, BlasPlan,
        BlasPtAlgorithm, BlasPtOperandPrepared, BlasVector, BlasVectorInfos, BlasVectorLayout, BlasVectorOwned, BlasVectorShape,
        Left, Right, blas_matvec_diagonal_output_layout, blas_matvec_pt_output_layout, ciphertext::layout_of,
        diagonal_ciphertext::AppendAxis, matvec_pt_galois_elements, replicated, sum_galois_elements,
    },
    oep::BlasImpl,
};

/// Every operation once on the conjugate-invariant ring, on real matrices of `4 x 4` blocks encoded with the module's
/// own encoder (the standard decoding helper reads complex slots): `blas_rotate`, `blas_sum`, `blas_transpose`,
/// `blas_rotate_rows` and `blas_rotate_cols` across blocks, `blas_sum_rows` and `blas_sum_cols`, of distinct matrices
/// and of one matrix one block high, which also shifts its rows by rotations, both preparations under either hint then
/// `blas_matmul_prepared` of every mix and `blas_matmul`, by a right factor of distinct matrices and by one whose
/// compact encryption proves it holds one matrix, `blas_matmul_pt_right` and `blas_matmul_pt_left` under every
/// algorithm, the masked ones under the context's masks and on the right of one public matrix, each on exactly the
/// scratch its query reports, `blas_matvec_pt` of a real vector, resident and streamed, and the row and column appends of
/// a diagonal ciphertext by real masks then its product by a real vector. Every output decrypts to its
/// cleartext model at the metadata [`copied_layout`],
/// [`mapped_layout`] or [`product_layout`] gives, and its slots, like the prepared operands', are `Real`. On fewer than
/// 16 slots, which hold no `4 x 4` block, it is skipped with a message.
pub fn test_blas_conjugate_invariant<BE>(
    mut params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<ConjugateInvariant>>,
) where
    BE: TestContextBackend<Ring = ConjugateInvariant> + BlasImpl,
    Module<BE>: TestContextSharedModule<BE>
        + CKKSEncodingOps<BE, f64>
        + BlasEncodingOps<BE, f64>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    params.prec_meta.slots = SlotsKind::Real;
    let (log_d, slots) = (2, module.ckks_max_slots());
    if 2 * log_d > slots.ilog2() as usize {
        eprintln!("test_blas_conjugate_invariant skips log_d {log_d}: it needs d^2 <= {slots} slots");
        return;
    }
    let (k, p) = (params.k, params.prec().log_delta());
    let lanes = slots >> (2 * log_d);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let masked = BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::JklsGalois);
    let ctx = encoded_context::<BE, f64>(module, masked, params.base2k.into(), params.prec().into());
    // `A` of 2 x 3 blocks and `B` of 3 x 2.
    let real = |rows: usize, cols: usize, seed: u64| {
        let mut grid = Grid::model(log_d, rows, cols, lanes, seed);
        grid.im.fill(0.0);
        grid
    };
    let (a, b) = (real(2, 3, 11), real(3, 2, 12));
    let ct_a = encrypt_real(&params, module, &sk, &a, &mut scratch.borrow());
    let ct_b = encrypt_real(&params, module, &sk, &b, &mut scratch.borrow());

    // One store for every operation: the context's maps, the rotation by one matrix slot, the sum and both public
    // products of the diagonal and masked plans, and the tensor key.
    let (order, plan, diagonal) = (module.cyclotomic_order(), ctx.plan(), BlasPlan::new(log_d));
    let elements: BTreeSet<i64> = plan
        .galois_elements(order)
        .into_iter()
        .chain(plan.rotate_galois_elements(order, 1))
        .chain(sum_galois_elements(module.cyclotomic_order(), &ct_a))
        .chain(plan.matmul_pt_right_galois_elements(order))
        .chain(plan.matmul_pt_left_galois_elements(order))
        .chain(diagonal.matmul_pt_right_galois_elements(order))
        .chain(diagonal.matmul_pt_left_galois_elements(order))
        .collect();
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let (atk, tsk) = (params.atk_layout(), params.tsk_layout());
    let alloc = |rows: usize, cols: usize| alloc_grid(&params, module, log_d, rows, cols, k);
    let check =
        |label: &str, res: &BlasCiphertextOwned<BE>, want: &Grid, layout: BlasLayout, scratch: &mut ScratchArena<'_, BE>| {
            assert_real_grid(label, &params, module, &sk, res, want, &layout, scratch)
        };

    let mut res = alloc(2, 3);
    with_scratch::<BE, _>(module.blas_rotate_tmp_bytes(&res, &ct_a, &atk), |s| {
        module.blas_rotate(1, &mut res, &ct_a, &keys, s)
    })
    .unwrap();
    let layout = copied_layout(&ct_a, 2, 3, k);
    check("rotate", &res, &a.rotate_lanes(1), layout, &mut scratch.borrow());

    let mut res = alloc(2, 3);
    with_scratch::<BE, _>(module.blas_sum_tmp_bytes(&res, &ct_a, &atk), |s| {
        module.blas_sum(&mut res, &ct_a, &keys, s)
    })
    .unwrap();
    check("sum", &res, &a.sum_lanes(), layout, &mut scratch.borrow());

    let mut res = alloc(3, 2);
    with_scratch::<BE, _>(module.blas_transpose_tmp_bytes(&res, &ct_a, &ctx, &atk), |s| {
        module.blas_transpose(&mut res, &ct_a, &ctx, &keys, s)
    })
    .unwrap();
    let layout = mapped_layout(&ct_a, 3, 2, k, p);
    check("transpose", &res, &a.transpose(), layout, &mut scratch.borrow());

    // Shifts by one row or column: two half maps per block.
    let layout = mapped_layout(&ct_a, 2, 3, k, p);
    let mut res = alloc(2, 3);
    with_scratch::<BE, _>(module.blas_rotate_rows_tmp_bytes(&res, &ct_a, &ctx, &atk), |s| {
        module.blas_rotate_rows(1, &mut res, &ct_a, &ctx, &keys, s)
    })
    .unwrap();
    check("rotate_rows", &res, &a.rotate_rows(1), layout, &mut scratch.borrow());

    let mut res = alloc(2, 3);
    with_scratch::<BE, _>(module.blas_rotate_cols_tmp_bytes(&res, &ct_a, &ctx, &atk), |s| {
        module.blas_rotate_cols(1, &mut res, &ct_a, &ctx, &keys, s)
    })
    .unwrap();
    check("rotate_cols", &res, &a.rotate_cols(1), layout, &mut scratch.borrow());

    // Row and column sums, of `A` and of a grid one block high whose compact encryption proves it holds one matrix:
    // that grid sums its rows and shifts them by rotations, at no scale.
    let mut one_row = Grid::model(log_d, 1, 2, 1, 14);
    one_row.im.fill(0.0);
    let ct_one_row = encrypt_real(&params, module, &sk, &one_row, &mut scratch.borrow());
    assert!(replicated::<ConjugateInvariant>(&ct_one_row), "the fixture holds one matrix");
    let spread_row = Grid::from_fn(log_d, 1, 2, lanes, |_, r, c| one_row.at(0, r, c));
    for (name, ct_x, x) in [("", &ct_a, &a), (" of one matrix", &ct_one_row, &spread_row)] {
        let (rows, cols) = (ct_x.rows(), ct_x.cols());
        let mapped = mapped_layout(ct_x, rows, cols, k, p);
        let unmapped = if replicated::<ConjugateInvariant>(ct_x) {
            copied_layout(ct_x, rows, cols, k)
        } else {
            mapped
        };
        let mut res = alloc(rows, cols);
        with_scratch::<BE, _>(module.blas_sum_rows_tmp_bytes(&res, ct_x, &ctx, &atk), |s| {
            module.blas_sum_rows(&mut res, ct_x, &ctx, &keys, s)
        })
        .unwrap();
        check(
            &format!("sum_rows{name}"),
            &res,
            &x.sum_rows(),
            unmapped,
            &mut scratch.borrow(),
        );
        with_scratch::<BE, _>(module.blas_sum_cols_tmp_bytes(&res, ct_x, &ctx, &atk), |s| {
            module.blas_sum_cols(&mut res, ct_x, &ctx, &keys, s)
        })
        .unwrap();
        check(&format!("sum_cols{name}"), &res, &x.sum_cols(), mapped, &mut scratch.borrow());
        if rows == 1 {
            let mut res = alloc(rows, cols);
            with_scratch::<BE, _>(module.blas_rotate_rows_tmp_bytes(&res, ct_x, &ctx, &atk), |s| {
                module.blas_rotate_rows(1, &mut res, ct_x, &ctx, &keys, s)
            })
            .unwrap();
            check(
                &format!("rotate_rows{name}"),
                &res,
                &x.rotate_rows(1),
                unmapped,
                &mut scratch.borrow(),
            );
        }
    }

    // Both operands under either hint, hot first, then the prepared product of every mix.
    let lefts = HINTS.map(|hint| {
        let left = with_scratch::<BE, _>(module.blas_matmul_prepare_left_tmp_bytes(&ct_a, &ctx, &atk, hint), |s| {
            matmul_left_operand(module, &ct_a, &ctx, &keys, hint, s)
        })
        .unwrap();
        assert_eq!(left.block_layout().slots(), SlotsKind::Real, "prepare_left {hint:?}: slots");
        left
    });
    // A right factor holding one matrix, its label proving it, so that its psi^k are rotations at no scale.
    let mut one = Grid::model(log_d, 3, 2, 1, 13);
    one.im.fill(0.0);
    let ct_one = encrypt_real(&params, module, &sk, &one, &mut scratch.borrow());
    assert!(replicated::<ConjugateInvariant>(&ct_one), "the fixture holds one matrix");
    let spread_one = Grid::from_fn(log_d, 3, 2, lanes, |_, r, c| one.at(0, r, c));
    let side = |hint| usize::from(hint == PrepareHint::Reuse);
    for (name, ct_y, y) in [("", &ct_b, &b), (" of one matrix", &ct_one, &spread_one)] {
        let rights = HINTS.map(|hint| {
            let right = with_scratch::<BE, _>(module.blas_matmul_prepare_right_tmp_bytes(ct_y, &ctx, &atk, hint), |s| {
                matmul_right_operand(module, ct_y, &ctx, &keys, hint, s)
            })
            .unwrap();
            assert_eq!(
                right.block_layout().slots(),
                SlotsKind::Real,
                "prepare_right{name} {hint:?}: slots"
            );
            right
        });
        let (ab, layout) = (a.matmul(y), product_layout::<BE::Ring>(&ct_a, ct_y, 2, 2, k, p));
        for (hl, hr) in MIXES {
            let (left, right) = (&lefts[side(hl)], &rights[side(hr)]);
            let mut res = alloc(2, 2);
            with_scratch::<BE, _>(
                module.blas_matmul_prepared_tmp_bytes(&res, left, right, &ctx, &tsk, &atk),
                |s| module.blas_matmul_prepared(&mut res, left, right, &ctx, &keys, &keys, s),
            )
            .unwrap();
            check(
                &format!("matmul_prepared{name} {hl:?} x {hr:?}"),
                &res,
                &ab,
                layout,
                &mut scratch.borrow(),
            );
        }

        let mut res = alloc(2, 2);
        with_scratch::<BE, _>(module.blas_matmul_tmp_bytes(&res, &ct_a, ct_y, &ctx, &tsk, &atk), |s| {
            module.blas_matmul(&mut res, &ct_a, ct_y, &ctx, &keys, &keys, s)
        })
        .unwrap();
        check(&format!("matmul{name}"), &res, &ab, layout, &mut scratch.borrow());
    }

    // Products by two public matrices, matrix slot `s` reading matrix `s mod 2`; one on a single matrix slot, and on
    // the right of the masked paths.
    let count = lanes.min(2);
    for algorithm in PT_ALGORITHMS.into_iter().chain(MASKED_PT_ALGORITHMS) {
        let (scale, right_count) = match algorithm {
            BlasPtAlgorithm::Diagonal => (p, count),
            BlasPtAlgorithm::Jkls => (2 * p, count),
            BlasPtAlgorithm::JklsStored | BlasPtAlgorithm::JklsGalois => (3 * p, 1),
        };
        let pt = pt_operand::<BE, f64, Right>(
            module,
            BlasPlan::new(log_d).with_pt_algorithm(algorithm),
            3,
            2,
            right_count,
            public_entry,
            params.base2k.into(),
            params.prec().into(),
        );
        let mut res = alloc(2, 2);
        with_scratch::<BE, _>(module.blas_matmul_pt_right_tmp_bytes(&res, &ct_a, &pt, &ctx, &atk), |s| {
            module.blas_matmul_pt_right(&mut res, &ct_a, &pt, &ctx, &keys, s)
        })
        .unwrap();
        let (want, layout) = (
            a.matmul_public(2, right_count, public_entry),
            mapped_layout(&ct_a, 2, 2, k, scale),
        );
        check(
            &format!("matmul_pt_right {algorithm:?}"),
            &res,
            &want,
            layout,
            &mut scratch.borrow(),
        );

        let pt_left: BlasPtOperandPrepared<BE, Left> = pt_operand::<BE, f64, Left>(
            module,
            BlasPlan::new(log_d).with_pt_algorithm(algorithm),
            2,
            3,
            count,
            public_entry,
            params.base2k.into(),
            params.prec().into(),
        );
        let mut res = alloc(2, 2);
        with_scratch::<BE, _>(module.blas_matmul_pt_left_tmp_bytes(&res, &pt_left, &ct_b, &ctx, &atk), |s| {
            module.blas_matmul_pt_left(&mut res, &pt_left, &ct_b, &ctx, &keys, s)
        })
        .unwrap();
        let (want, layout) = (b.public_matmul(2, count, public_entry), mapped_layout(&ct_b, 2, 2, k, scale));
        check(
            &format!("matmul_pt_left {algorithm:?}"),
            &res,
            &want,
            layout,
            &mut scratch.borrow(),
        );
    }

    // A product of a real vector by a public matrix, several ciphertexts on each side, resident and streamed.
    let layout = matvec_layouts(&params, slots).pop().unwrap();
    let (streamed, resident) = matvec_operands::<BE, f64>(module, &layout);
    let keys = TestKeys::<BE> {
        atks: rotation_keys(
            &params,
            module,
            &sk_raw,
            matvec_pt_galois_elements(order, &layout),
            &mut scratch.borrow(),
        ),
        tsk: None,
    };
    let v: Vec<f64> = (0..layout.input.len).map(vector_entry).collect();
    let ct_v = encrypt_real_vector(
        &params,
        module,
        &sk,
        layout.input,
        layout.input_period,
        &v,
        &mut scratch.borrow(),
    );
    let want = matvec_model(layout.output.len, &v);
    let want_layout = blas_matvec_pt_output_layout(&alloc_vector(&params, module, layout.output, k), &layout, &ct_v);
    let mut res = alloc_vector(&params, module, layout.output, k);
    with_scratch::<BE, _>(module.blas_matvec_pt_tmp_bytes(&res, &resident, &ct_v, &atk), |s| {
        module.blas_matvec_pt(&mut res, &resident, &ct_v, &keys, s)
    })
    .unwrap();
    let period = layout.output_period;
    assert_real_vector(
        "matvec_pt resident",
        &params,
        module,
        &sk,
        &res,
        &want,
        period,
        &want_layout,
        &mut scratch.borrow(),
    );
    let mut res = alloc_vector(&params, module, layout.output, k);
    with_scratch::<BE, _>(module.blas_matvec_pt_tmp_bytes(&res, &streamed, &ct_v, &atk), |s| {
        module.blas_matvec_pt(&mut res, &streamed, &ct_v, &keys, s)
    })
    .unwrap();
    assert_real_vector(
        "matvec_pt streamed",
        &params,
        module,
        &sk,
        &res,
        &want,
        period,
        &want_layout,
        &mut scratch.borrow(),
    );

    // A diagonal ciphertext of real diagonals: a row and a column appended by real masks of one and two slots, prepared to
    // reuse and once, then its product by a real vector.
    let layout = diagonal_ciphertext_layout(&params, slots, k - p);
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
    let whole = BlasVectorShape::new(slots, slots, 0, 1);
    let real_mask = |set: &[usize], hint: PrepareHint, scratch: &mut ScratchArena<'_, BE>| {
        let mut re = vec![0.0; slots];
        set.iter().for_each(|&s| re[s] = 1.0);
        let mut pt = module.ckks_pt_vec_alloc(slots, params.base2k.into(), params.prec().k());
        pt.set_meta(params.prec().meta());
        module
            .ckks_encode_reim_into(&mut pt, &re, &vec![0.0; slots], scratch)
            .unwrap();
        let mask = module.ckks_prepare_right(&pt, hint, scratch).unwrap();
        (re, mask)
    };
    let masks = [
        real_mask(&[0], PrepareHint::Reuse, &mut scratch.borrow()),
        real_mask(&[0, slots / 2], PrepareHint::OneShot, &mut scratch.borrow()),
    ];
    let mut m = module.blas_diagonal_ciphertext_alloc_from_infos(&layout);
    let mut model = DiagonalCiphertextModel::new(slots, (0..8).collect());
    for (axis, at, (mask, mask_prepared)) in [(AppendAxis::Row, 1, &masks[0]), (AppendAxis::Col, 4, &masks[1])] {
        let v: Vec<f64> = (0..slots).map(|e| vector_entry(3 * e + at as usize)).collect();
        let ct_v = encrypt_real_vector(&params, module, &sk, whole, slots, &v, &mut scratch.borrow());
        with_scratch::<BE, _>(append_tmp_bytes(module, axis, &m, &ct_v, mask_prepared, &atk), |s| {
            append(module, axis, &mut m, at, &ct_v, mask_prepared, &keys, s)
        })
        .unwrap();
        model.append(axis, at, mask, &v);
    }
    let x: Vec<f64> = (0..slots).map(vector_entry).collect();
    let ct_x = encrypt_real_vector(&params, module, &sk, whole, slots, &x, &mut scratch.borrow());
    let mut res = alloc_vector(&params, module, whole, k);
    let want_layout = blas_matvec_diagonal_output_layout(&res, &m, &ct_x);
    with_scratch::<BE, _>(module.blas_matvec_diagonal_tmp_bytes(&res, &m, &ct_x, &tsk, &atk), |s| {
        module.blas_matvec_diagonal(&mut res, &m, &ct_x, &keys, s)
    })
    .unwrap();
    assert_real_vector(
        "streamed matvec of a diagonal ciphertext",
        &params,
        module,
        &sk,
        &res,
        &model.matvec(&x),
        slots,
        &want_layout,
        &mut scratch.borrow(),
    );
    let prepared = prepared_of(module, &m);
    let mut res = alloc_vector(&params, module, whole, k);
    with_scratch::<BE, _>(
        module.blas_matvec_diagonal_tmp_bytes(&res, &prepared, &ct_x, &tsk, &atk),
        |s| module.blas_matvec_diagonal(&mut res, &prepared, &ct_x, &keys, s),
    )
    .unwrap();
    assert_real_vector(
        "resident matvec of a diagonal ciphertext",
        &params,
        module,
        &sk,
        &res,
        &model.matvec(&x),
        slots,
        &want_layout,
        &mut scratch.borrow(),
    );
}

/// Encrypts the real `values` as a vector of `shape` repeated every `period` slots at width `params.k`, each ciphertext
/// encoded at `params.prec()` by the module and labelled with the sparsity of `period`.
fn encrypt_real_vector<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    shape: BlasVectorShape,
    period: usize,
    values: &[f64],
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasVectorOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, f64>,
{
    let slots = module.ckks_max_slots();
    let zeros = vec![0.0; slots];
    let cts = (0..shape.count())
        .map(|t| {
            let mut pt = module.ckks_pt_vec_alloc(slots, params.base2k.into(), params.prec().k());
            pt.set_meta(params.prec().meta());
            let re = vector_slots(shape, period, slots, t, values);
            module.ckks_encode_reim_into(&mut pt, &re, &zeros, scratch).unwrap();
            let mut ct = ckks_encrypt_pt(params, module, sk, params.k, &pt.to_host_owned::<BE>(), scratch);
            ct.set_log_sparsity((slots / period).ilog2() as usize);
            ct
        })
        .collect();
    BlasVector::from_cts(cts, shape).unwrap()
}

/// [`assert_vector_decrypt`](super::matvec::assert_vector_decrypt) on the conjugate-invariant ring: every ciphertext's
/// slots are `Real`, then the layout `layout` gives and empty noise tags, then `want`, repeated every `period` and zero
/// elsewhere, at `layout`'s `log_delta`, decoded by the module, with zero imaginary parts.
#[allow(clippy::too_many_arguments)]
fn assert_real_vector<BE>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    v: &BlasVectorOwned<BE>,
    want: &[f64],
    period: usize,
    layout: &BlasVectorLayout,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, f64>,
{
    assert_eq!(v.blas_vector_layout(), *layout, "{label}: layout");
    let (slots, shape, log_delta) = (module.ckks_max_slots(), v.vector_shape(), layout.ct.log_delta());
    for (t, ct) in v.cts().iter().enumerate() {
        assert_eq!(ct.slots(), SlotsKind::Real, "{label} ciphertext {t}: slots");
        assert!(ct.noise().is_none(), "{label} ciphertext {t}: kept a noise tag");
        let mut prec = precision_at(params, log_delta);
        prec.meta = ct.meta();
        let pt = ckks_decrypt_with_prec(module, ct, sk, prec, scratch).unwrap();
        let (mut re, mut im) = (vec![0.0; slots], vec![0.0; slots]);
        module.ckks_decode_reim_into(&pt, &mut re, &mut im, scratch).unwrap();
        let want = vector_slots(shape, period, slots, t, want);
        assert_precision(&format!("{label} ciphertext {t}"), &re, &want, log_delta, params.n);
        assert!(im.iter().all(|&x| x == 0.0), "{label} ciphertext {t}: imaginary parts");
    }
}

/// Encrypts every block of the real grid `grid` at width `params.k`, encoded at `params.prec()` by the module into a
/// plaintext of the grid's `lanes d^2` slots, so the grid records its lanes.
fn encrypt_real<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    grid: &Grid,
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasCiphertextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, f64>,
{
    let slots = grid.lanes() << (2 * grid.log_d);
    let zeros = vec![0.0; slots];
    let blocks = grid
        .blocks(slots)
        .iter()
        .map(|(re, _)| {
            let mut pt = module.ckks_pt_vec_alloc(slots, params.base2k.into(), params.prec().k());
            pt.set_meta(params.prec().meta());
            module.ckks_encode_reim_into(&mut pt, re, &zeros, scratch).unwrap();
            ckks_encrypt_pt(params, module, sk, params.k, &pt.to_host_owned::<BE>(), scratch)
        })
        .collect();
    BlasCiphertext::from_blocks(blocks, grid.log_d, grid.rows, grid.cols).unwrap()
}

/// [`assert_grid_decrypt`](super::helpers::assert_grid_decrypt) on the conjugate-invariant ring: every block's slots are
/// `Real`, then the shape, labels, layouts, empty noise tags and batch record `layout` gives, then the values `want`
/// holds at `layout`'s `log_delta`, decoded by the module, with zero imaginary parts.
fn assert_real_grid<BE>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    ct: &BlasCiphertextOwned<BE>,
    want: &Grid,
    layout: &BlasLayout,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, f64>,
{
    for (b, block) in ct.blocks().iter().enumerate() {
        assert_eq!(block.slots(), SlotsKind::Real, "{label} block {b}: slots");
    }
    let shape = (layout.log_d, layout.rows, layout.cols);
    assert_eq!((ct.log_d(), ct.rows(), ct.cols()), shape, "{label}: grid shape");
    assert_eq!((want.log_d, want.rows, want.cols), shape, "{label}: model shape");
    for (b, block) in ct.blocks().iter().enumerate() {
        assert_eq!(layout_of(block), layout.block, "{label} block {b}: layout");
        assert!(block.noise().is_none(), "{label} block {b}: kept a noise tag");
    }
    let slots = module.ckks_max_slots();
    let batch = (slots >> layout.block.log_sparsity()) >> (2 * layout.log_d);
    assert_eq!(ct.mat_slots(), Some(batch), "{label}: batch record");
    let log_delta = layout.block.log_delta();
    for (b, (block, (want_re, _))) in ct.blocks().iter().zip(want.blocks(slots)).enumerate() {
        let mut prec = precision_at(params, log_delta);
        prec.meta = block.meta();
        let pt = ckks_decrypt_with_prec(module, block, sk, prec, scratch).unwrap();
        let (mut re, mut im) = (vec![0.0; slots], vec![0.0; slots]);
        module.ckks_decode_reim_into(&pt, &mut re, &mut im, scratch).unwrap();
        assert_precision(&format!("{label} block {b}"), &re, &want_re, log_delta, params.n);
        assert!(im.iter().all(|&x| x == 0.0), "{label} block {b}: imaginary parts");
    }
}
