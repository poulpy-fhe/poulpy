//! JKLS matrix products of whole grids.

use std::collections::HashMap;

use poulpy_ckks::{
    CKKSInfos,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_decrypt_precision_at_log_delta, gen_atk,
            gen_sk_with_raw, gen_tsk,
        },
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{Base2K, DiagonalArithmetic};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, Standard},
};

use super::helpers::{
    D, Grid, LOG_D,
    Rejection::{InvalidPlan, MissingKey, MissingTensorKey},
    TestKeys, assert_grid_decrypt, assert_guard, assert_guard_on, assert_short_scratch, assert_short_scratch_on,
    context_and_keys, encrypt_grid, foreign_contexts, grid_snapshot, rejected_grids, relabel,
};
use crate::{
    api::BlasMatMulOps,
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasMatMulLeft, BlasMatMulRight, BlasModuleAlloc, BlasPtOperand,
        BlasPtOperandUnprepared,
    },
    oep::BlasMatMulImpl,
};

/// `dec(matmul(enc(A), enc(B)))` is `A B` for every matrix of the batch, for a `2 x 3` grid times a `3 x 2` grid, and
/// `blas_matmul` equals `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, then `blas_matmul_prepared`, bit for bit.
pub fn test_blas_matmul<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, atks) = context_and_keys::<BE, F>(&params, module, &sk_raw, &mut scratch.borrow());
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let keys = TestKeys::<BE> { atks, tsk: Some(tsk) };
    let (m, inner, n) = (2, 3, 2);
    let lanes = slots / (D * D);
    let (a, b) = (Grid::model(m, inner, lanes, 7), Grid::model(inner, n, lanes, 8));
    let ct_a = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &a,
        &mut scratch.borrow(),
    );
    let ct_b = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &b,
        &mut scratch.borrow(),
    );
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let mut res = module.blas_ciphertext_alloc(LOG_D, m, n, params.base2k.into(), params.k.into());
    let bytes = module.blas_matmul_tmp_bytes(&res, &ct_a, &ct_b, &tsk_layout, &atk_layout);
    module
        .blas_matmul(
            &mut res,
            &ct_a,
            &ct_b,
            &ctx,
            &keys,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        )
        .unwrap();
    assert_grid_decrypt::<BE, F, E>(
        "matmul",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &a.matmul(&b),
        &mut scratch.borrow(),
    );

    // The transformed operands are allocated like the inputs, so every step runs at the widths `blas_matmul` uses.
    let mut left = BlasMatMulLeft::from_grid(module.blas_ciphertext_alloc_from_infos(&ct_a));
    let mut right = BlasMatMulRight::from_grid(module.blas_ciphertext_alloc_from_infos(&ct_b));
    let bytes = module.blas_matmul_prepare_left_tmp_bytes(left.as_grid(), &ct_a, &atk_layout);
    module
        .blas_matmul_prepare_left(&mut left, &ct_a, &ctx, &keys, &mut ScratchOwned::<BE>::alloc(bytes).borrow())
        .unwrap();
    let bytes = module.blas_matmul_prepare_right_tmp_bytes(right.as_grid(), &ct_b, &atk_layout);
    module
        .blas_matmul_prepare_right(&mut right, &ct_b, &ctx, &keys, &mut ScratchOwned::<BE>::alloc(bytes).borrow())
        .unwrap();
    let mut prepared = module.blas_ciphertext_alloc(LOG_D, m, n, params.base2k.into(), params.k.into());
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
    assert_eq!(grid_snapshot::<BE, _>(&prepared), grid_snapshot::<BE, _>(&res));
}

/// The public matrices of a `rows x cols` operand grid, entries in `[-0.5, 0.5]`.
fn public_entry(m: usize, r: usize, c: usize) -> f64 {
    ((m * 7 + r * 5 + c * 3) % 9) as f64 / 8.0 - 0.5
}

/// `dec(matmul_pt(enc(A), B))` is `A B_(s mod count)` in matrix slot `s`, for a `2 x 3` grid times a `3 x 2` operand
/// with one, two, and one public matrix per matrix slot.
pub fn test_blas_matmul_pt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
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
    let (m, inner, n) = (2, 3, 2);
    let operands: Vec<(usize, BlasPtOperand<BE>)> = [1, 2, lanes]
        .into_iter()
        .map(|count| {
            let b = BlasPtOperandUnprepared::encode::<F>(
                module,
                LOG_D,
                inner,
                n,
                count,
                |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
                params.base2k.into(),
                params.prec().into(),
                &mut scratch.borrow(),
            )
            .unwrap()
            .prepare(module, &mut scratch.borrow());
            (count, b)
        })
        .collect();
    let order = module.cyclotomic_order();
    let mut elements: Vec<i64> = operands.iter().flat_map(|(_, b)| b.galois_elements(order)).collect();
    elements.sort_unstable();
    elements.dedup();
    let atks = elements
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
        .collect();
    let keys = TestKeys::<BE> { atks, tsk: None };
    let a = Grid::model(m, inner, lanes, 9);
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &a,
        &mut scratch.borrow(),
    );
    for (count, b) in &operands {
        // `res` one limb wider than `a` for one public matrix.
        let res_k = if *count == 1 { params.k + params.base2k } else { params.k };
        let mut res = module.blas_ciphertext_alloc(LOG_D, m, n, params.base2k.into(), res_k.into());
        let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_matmul_pt_tmp_bytes(&res, &ct, &params.atk_layout()));
        module
            .blas_matmul_pt(&mut res, &ct, b, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_grid_decrypt::<BE, F, E>(
            &format!("matmul_pt count={count} res_k={res_k}"),
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &a.matmul_public(n, *count, public_entry),
            &mut scratch.borrow(),
        );
    }
}

/// An inner-dimension or `res` shape mismatch, more public matrices than a sparse input records, a copy relabelled past
/// `log2` of the ring degree, an operand encoded on a module of another degree, at another radix or at
/// another dimension, every grid [`rejected_grids`] builds, as `a` or as `res`, a radix whose digits cannot take a second
/// inner index, and a missing rotation key are rejected before `res` changes; a product one scratch byte short of its
/// query panics before `res` changes; an operand whose block count overflows `usize` does not encode.
pub fn test_blas_matmul_pt_guards<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
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
    let encode = |module: &Module<BE>, log_d: usize, count: usize, base2k: usize, scratch: &mut ScratchArena<'_, BE>| {
        BlasPtOperandUnprepared::encode::<F>(
            module,
            log_d,
            2,
            1,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            base2k.into(),
            params.prec().into(),
            scratch,
        )
        .unwrap()
        .prepare(module, scratch)
    };
    let b = encode(module, LOG_D, lanes, params.base2k, &mut scratch.borrow());
    // One public matrix: `lanes` of them do not divide the matrix slots of a module of half the degree, nor those of a
    // grid read at the next dimension.
    let one = encode(module, LOG_D, 1, params.base2k, &mut scratch.borrow());
    // At the previous dimension, whose smaller matrices fit wherever a block does.
    let small = encode(module, LOG_D - 1, 1, params.base2k, &mut scratch.borrow());
    let half = Module::<BE>::new(params.n as u64 / 2);
    let foreign = [
        ("degree", encode(&half, LOG_D, 1, params.base2k, &mut scratch.borrow())),
        ("base2k", encode(module, LOG_D, 1, params.base2k - 1, &mut scratch.borrow())),
    ];
    // Keys for the other operands too, so no missing key stops `blas_matmul_pt` from evaluating them.
    let order = module.cyclotomic_order();
    let mut elements: Vec<i64> = [&b, &one, &small]
        .into_iter()
        .chain(foreign.iter().map(|(_, b)| b))
        .flat_map(|b| b.galois_elements(order))
        .collect();
    elements.sort_unstable();
    elements.dedup();
    let atks = elements
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
        .collect();
    let keys = TestKeys::<BE> { atks, tsk: None };
    let encrypt = |rows: usize, cols: usize, seed: u64, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            0,
            &Grid::model(rows, cols, lanes, seed),
            scratch,
        )
    };
    let ct = encrypt(1, 2, 11, &mut scratch.borrow());
    let narrow = encrypt(1, 3, 12, &mut scratch.borrow());
    let sparse = BlasCiphertext::from_blocks(relabel(&ct, |meta| meta.log_sparsity = 1).into_blocks(), LOG_D, 1, 2).unwrap();
    let sparse_64 = relabel(&ct, |meta| meta.log_sparsity = 64);
    let sparse_65 = relabel(&ct, |meta| meta.log_sparsity = 65);
    let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_matmul_pt_tmp_bytes(&ct, &ct, &params.atk_layout()));

    // `narrow` has 3 block columns against 2 block rows of `b`; `res` must be 1 x 1; `sparse` records 4 matrices;
    // `sparse_64` and `sparse_65` are labelled past `log2` of the ring degree.
    for (label, rows, cols, a) in [
        ("inner", 1, 1, &narrow),
        ("shape", 1, 2, &ct),
        ("count", 1, 1, &sparse),
        ("sparsity 64", 1, 1, &sparse_64),
        ("sparsity 65", 1, 1, &sparse_65),
    ] {
        let mut res = module.blas_ciphertext_alloc(LOG_D, rows, cols, params.base2k.into(), params.k.into());
        assert_guard::<BE, _>(label, "blas_matmul_pt", InvalidPlan, &mut res, |res| {
            module.blas_matmul_pt(res, a, &b, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, b) in &foreign {
        let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 1, params.base2k.into(), params.k.into());
        assert_guard::<BE, _>(label, "blas_matmul_pt", InvalidPlan, &mut res, |res| {
            module.blas_matmul_pt(res, &ct, b, &keys, &mut op_scratch.borrow())
        });
    }
    // An operand encoded at the previous dimension, which `ct` and `res` are not read at.
    let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 1, params.base2k.into(), params.k.into());
    assert_guard::<BE, _>("operand dimension", "blas_matmul_pt", InvalidPlan, &mut res, |res| {
        module.blas_matmul_pt(res, &ct, &small, &keys, &mut op_scratch.borrow())
    });

    // Each rejected grid as `a` or as `res` of a 2 x 2 grid times one public matrix.
    let square = encrypt(2, 2, 13, &mut scratch.borrow());
    let res_2x1 = module.blas_ciphertext_alloc(LOG_D, 2, 1, params.base2k.into(), params.k.into());
    for ((label, a), (_, mut bad)) in rejected_grids(module, &square)
        .into_iter()
        .zip(rejected_grids(module, &res_2x1))
    {
        let mut res = res_2x1.clone();
        assert_guard::<BE, _>(&format!("{label} a"), "blas_matmul_pt", InvalidPlan, &mut res, |res| {
            module.blas_matmul_pt(res, &a, &one, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&format!("{label} res"), "blas_matmul_pt", InvalidPlan, &mut bad, |res| {
            module.blas_matmul_pt(res, &square, &one, &keys, &mut op_scratch.borrow())
        });
    }

    // `res` at radix 63: its digits hold one lazy term, short of the two a second inner index brings to a normalized block.
    let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 1, Base2K(63), params.k.into());
    assert_guard::<BE, _>("lazy sums", "blas_matmul_pt", InvalidPlan, &mut res, |res| {
        module.blas_matmul_pt(res, &ct, &b, &keys, &mut op_scratch.borrow())
    });

    let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 1, params.base2k.into(), params.k.into());
    let no_keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    assert_guard::<BE, _>("no keys", "blas_matmul_pt", MissingKey, &mut res, |res| {
        module.blas_matmul_pt(res, &ct, &b, &no_keys, &mut op_scratch.borrow())
    });

    let bytes = module.blas_matmul_pt_tmp_bytes(&res, &ct, &params.atk_layout());
    assert_short_scratch::<BE, _>("blas_matmul_pt", &mut res, bytes, |res, s| {
        module.blas_matmul_pt(res, &ct, &b, &keys, s)
    });

    // (2^62 + 1) x 4 blocks overflow a usize on 64 bits.
    let overflow = BlasPtOperandUnprepared::encode::<F>(
        module,
        LOG_D,
        usize::MAX / 4 + 2,
        4,
        lanes,
        |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    );
    assert!(overflow.is_err());
}

/// Every grid [`rejected_grids`] builds, in place of an operand or of `res`, a `res` shape mismatch, an operand
/// relabelled past `log2` of the ring degree, a context encoded on a module of another degree or at another
/// radix, and a missing rotation key are rejected before `res` changes, for the product, its prepared form and the
/// preparations; so are an inner-dimension mismatch, one factor at a radix other than the context's, a radix whose
/// digits cannot take a second inner index, and a missing tensor key, for the product and its prepared form. Each of
/// these operations one scratch byte short of its query panics before `res` changes.
pub fn test_blas_matmul_guards<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
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
    let foreign = foreign_contexts::<BE, F>(&params, module, &sk_raw, &mut atks, &mut scratch.borrow());
    let keys = TestKeys::<BE> {
        atks,
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &Grid::model(1, 2, lanes, 10),
        &mut scratch.borrow(),
    );
    let column = BlasCiphertext::from_blocks(ct.clone().into_blocks(), LOG_D, 2, 1).unwrap();
    let alloc = |rows: usize, cols: usize| module.blas_ciphertext_alloc(LOG_D, rows, cols, params.base2k.into(), params.k.into());
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    // `column ct`, the widest product below.
    let bytes = module.blas_matmul_tmp_bytes(&alloc(2, 2), &column, &ct, &tsk_layout, &atk_layout);
    let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);

    // `ct` is 1 x 2 and `column` 2 x 1: `ct ct` has a mismatched inner dimension, `ct column` is 1 x 1, not 1 x 2.
    for (label, mut res, b) in [("inner", alloc(1, 2), &ct), ("shape", alloc(1, 2), &column)] {
        assert_guard::<BE, _>(label, "blas_matmul", InvalidPlan, &mut res, |res| {
            module.blas_matmul(res, &ct, b, &ctx, &keys, &mut op_scratch.borrow())
        });
    }

    let left = BlasMatMulLeft::from_grid(ct.clone());
    let right = BlasMatMulRight::from_grid(column.clone());
    let row_right = BlasMatMulRight::from_grid(ct.clone());
    let mut wide = alloc(1, 2);
    for (label, right) in [("inner", &row_right), ("shape", &right)] {
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidPlan, &mut wide, |res| {
            module.blas_matmul_prepared(res, &left, right, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    // Each prepared grid must have its input's shape.
    let mut left_wrong = BlasMatMulLeft::from_grid(alloc(2, 1));
    assert_guard_on::<BE, _, _>(
        "shape",
        "blas_matmul_prepare_left",
        InvalidPlan,
        &mut left_wrong,
        |left| left.as_grid(),
        |left| module.blas_matmul_prepare_left(left, &ct, &ctx, &keys, &mut op_scratch.borrow()),
    );
    let mut right_wrong = BlasMatMulRight::from_grid(alloc(1, 2));
    assert_guard_on::<BE, _, _>(
        "shape",
        "blas_matmul_prepare_right",
        InvalidPlan,
        &mut right_wrong,
        |right| right.as_grid(),
        |right| module.blas_matmul_prepare_right(right, &column, &ctx, &keys, &mut op_scratch.borrow()),
    );

    // Each rejected grid in place of an operand or of `res` of `column ct`, a 2 x 2 product, and of the preparations.
    let left_column = BlasMatMulLeft::from_grid(column.clone());
    for (label, bad) in rejected_grids(module, &column) {
        let label = &format!("{label} column");
        let mut res = alloc(2, 2);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidPlan, &mut res, |res| {
            module.blas_matmul(res, &bad, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_right",
            InvalidPlan,
            &mut BlasMatMulRight::from_grid(alloc(2, 1)),
            |right| right.as_grid(),
            |right| module.blas_matmul_prepare_right(right, &bad, &ctx, &keys, &mut op_scratch.borrow()),
        );
        let bad = BlasMatMulLeft::from_grid(bad);
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidPlan, &mut res, |res| {
            module.blas_matmul_prepared(res, &bad, &row_right, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, bad) in rejected_grids(module, &ct) {
        let label = &format!("{label} row");
        let mut res = alloc(2, 2);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidPlan, &mut res, |res| {
            module.blas_matmul(res, &column, &bad, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_left",
            InvalidPlan,
            &mut BlasMatMulLeft::from_grid(alloc(1, 2)),
            |left| left.as_grid(),
            |left| module.blas_matmul_prepare_left(left, &bad, &ctx, &keys, &mut op_scratch.borrow()),
        );
        let bad = BlasMatMulRight::from_grid(bad);
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidPlan, &mut res, |res| {
            module.blas_matmul_prepared(res, &left_column, &bad, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, mut bad) in rejected_grids(module, &alloc(2, 2)) {
        let label = &format!("{label} res");
        assert_guard::<BE, _>(label, "blas_matmul", InvalidPlan, &mut bad, |res| {
            module.blas_matmul(res, &column, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidPlan, &mut bad, |res| {
            module.blas_matmul_prepared(res, &left_column, &row_right, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    let prepared_res = rejected_grids(module, &alloc(1, 2))
        .into_iter()
        .zip(rejected_grids(module, &alloc(2, 1)));
    for ((label, left_res), (_, right_res)) in prepared_res {
        let label = &format!("{label} res");
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_left",
            InvalidPlan,
            &mut BlasMatMulLeft::from_grid(left_res),
            |left| left.as_grid(),
            |left| module.blas_matmul_prepare_left(left, &ct, &ctx, &keys, &mut op_scratch.borrow()),
        );
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_right",
            InvalidPlan,
            &mut BlasMatMulRight::from_grid(right_res),
            |right| right.as_grid(),
            |right| module.blas_matmul_prepare_right(right, &column, &ctx, &keys, &mut op_scratch.borrow()),
        );
    }

    // Copies relabelled past `log2` of the ring degree, which keep recording the dense batch.
    for log_sparsity in [64, 65] {
        let label = &format!("sparsity {log_sparsity}");
        let sparse_row = relabel(&ct, |meta| meta.log_sparsity = log_sparsity);
        let sparse_column = relabel(&column, |meta| meta.log_sparsity = log_sparsity);
        let mut res = alloc(1, 1);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidPlan, &mut res, |res| {
            module.blas_matmul(res, &sparse_row, &column, &ctx, &keys, &mut op_scratch.borrow())
        });
        let sparse_left = BlasMatMulLeft::from_grid(sparse_row.clone());
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidPlan, &mut res, |res| {
            module.blas_matmul_prepared(res, &sparse_left, &right, &ctx, &keys, &mut op_scratch.borrow())
        });
        let mut left_res = BlasMatMulLeft::from_grid(alloc(1, 2));
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_left",
            InvalidPlan,
            &mut left_res,
            |left| left.as_grid(),
            |left| module.blas_matmul_prepare_left(left, &sparse_row, &ctx, &keys, &mut op_scratch.borrow()),
        );
        let mut right_res = BlasMatMulRight::from_grid(alloc(2, 1));
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_right",
            InvalidPlan,
            &mut right_res,
            |right| right.as_grid(),
            |right| module.blas_matmul_prepare_right(right, &sparse_column, &ctx, &keys, &mut op_scratch.borrow()),
        );
    }

    for (label, ctx) in &foreign {
        let mut res = alloc(1, 1);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidPlan, &mut res, |res| {
            module.blas_matmul(res, &ct, &column, ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidPlan, &mut res, |res| {
            module.blas_matmul_prepared(res, &left, &right, ctx, &keys, &mut op_scratch.borrow())
        });
        let mut left_res = BlasMatMulLeft::from_grid(alloc(1, 2));
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_left",
            InvalidPlan,
            &mut left_res,
            |left| left.as_grid(),
            |left| module.blas_matmul_prepare_left(left, &ct, ctx, &keys, &mut op_scratch.borrow()),
        );
        let mut right_res = BlasMatMulRight::from_grid(alloc(2, 1));
        assert_guard_on::<BE, _, _>(
            label,
            "blas_matmul_prepare_right",
            InvalidPlan,
            &mut right_res,
            |right| right.as_grid(),
            |right| module.blas_matmul_prepare_right(right, &column, ctx, &keys, &mut op_scratch.borrow()),
        );
    }

    // One operand at a radix other than the context's, the other at the context's.
    let radix =
        |rows: usize, cols: usize| module.blas_ciphertext_alloc(LOG_D, rows, cols, (params.base2k - 1).into(), params.k.into());
    for (label, a, b) in [("radix a", radix(1, 2), column.clone()), ("radix b", ct.clone(), radix(2, 1))] {
        let mut res = alloc(1, 1);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidPlan, &mut res, |res| {
            module.blas_matmul(res, &a, &b, &ctx, &keys, &mut op_scratch.borrow())
        });
        let (a, b) = (BlasMatMulLeft::from_grid(a), BlasMatMulRight::from_grid(b));
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidPlan, &mut res, |res| {
            module.blas_matmul_prepared(res, &a, &b, &ctx, &keys, &mut op_scratch.borrow())
        });
    }

    // `res` at radix 61: its digits hold 4 lazy terms, short of the `d + 1` a second inner index brings to a normalized
    // block.
    let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 1, Base2K(61), params.k.into());
    assert_guard::<BE, _>("lazy sums", "blas_matmul", InvalidPlan, &mut res, |res| {
        module.blas_matmul(res, &ct, &column, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("lazy sums", "blas_matmul_prepared", InvalidPlan, &mut res, |res| {
        module.blas_matmul_prepared(res, &left, &right, &ctx, &keys, &mut op_scratch.borrow())
    });

    // Each operation one scratch byte short of its query.
    let mut res = alloc(1, 1);
    let bytes = module.blas_matmul_tmp_bytes(&res, &ct, &column, &tsk_layout, &atk_layout);
    assert_short_scratch::<BE, _>("blas_matmul", &mut res, bytes, |res, s| {
        module.blas_matmul(res, &ct, &column, &ctx, &keys, s)
    });
    let bytes = module.blas_matmul_prepared_tmp_bytes(&res, left.as_grid(), right.as_grid(), &tsk_layout, &atk_layout);
    assert_short_scratch::<BE, _>("blas_matmul_prepared", &mut res, bytes, |res, s| {
        module.blas_matmul_prepared(res, &left, &right, &ctx, &keys, s)
    });
    let mut left_res = BlasMatMulLeft::from_grid(alloc(1, 2));
    let bytes = module.blas_matmul_prepare_left_tmp_bytes(left_res.as_grid(), &ct, &atk_layout);
    assert_short_scratch_on::<BE, _, _>(
        "blas_matmul_prepare_left",
        &mut left_res,
        |left| left.as_grid(),
        bytes,
        |left, s| module.blas_matmul_prepare_left(left, &ct, &ctx, &keys, s),
    );
    let mut right_res = BlasMatMulRight::from_grid(alloc(2, 1));
    let bytes = module.blas_matmul_prepare_right_tmp_bytes(right_res.as_grid(), &column, &atk_layout);
    assert_short_scratch_on::<BE, _, _>(
        "blas_matmul_prepare_right",
        &mut right_res,
        |right| right.as_grid(),
        bytes,
        |right, s| module.blas_matmul_prepare_right(right, &column, &ctx, &keys, s),
    );

    // Valid grids, a key store without rotation keys or without a tensor key.
    let TestKeys { atks, tsk } = keys;
    let no_atks = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk,
    };
    let no_tsk = TestKeys::<BE> { atks, tsk: None };
    let mut res = alloc(1, 1);
    for (label, keys, rejection) in [
        ("no rotation keys", &no_atks, MissingKey),
        ("no tensor key", &no_tsk, MissingTensorKey),
    ] {
        assert_guard::<BE, _>(label, "blas_matmul", rejection, &mut res, |res| {
            module.blas_matmul(res, &ct, &column, &ctx, keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_matmul_prepared", rejection, &mut res, |res| {
            module.blas_matmul_prepared(res, &left, &right, &ctx, keys, &mut op_scratch.borrow())
        });
    }
    let mut left_res = BlasMatMulLeft::from_grid(alloc(1, 2));
    assert_guard_on::<BE, _, _>(
        "no rotation keys",
        "blas_matmul_prepare_left",
        MissingKey,
        &mut left_res,
        |left| left.as_grid(),
        |left| module.blas_matmul_prepare_left(left, &ct, &ctx, &no_atks, &mut op_scratch.borrow()),
    );
    let mut right_res = BlasMatMulRight::from_grid(alloc(2, 1));
    assert_guard_on::<BE, _, _>(
        "no rotation keys",
        "blas_matmul_prepare_right",
        MissingKey,
        &mut right_res,
        |right| right.as_grid(),
        |right| module.blas_matmul_prepare_right(right, &column, &ctx, &no_atks, &mut op_scratch.borrow()),
    );
}

/// Lazy sums three times longer than the `2^(63 - base2k)` terms an `i64` digit holds: `dec(matmul(enc(A), enc(B)))` is
/// `A B` for a `1 x K` grid times a `K x 1` grid, `blas_matmul_prepared` equals `blas_matmul` bit for bit, and
/// `dec(matmul_pt(enc(A), B))` is `A B`. Every block is a copy of one ciphertext, so the terms' digits add up coherently,
/// and so does their noise.
///
/// The bound is in reach only at a high radix: at `N = 32`, NTT4x30's `MaxBase2k` is 57, a bound of 64 terms, while
/// FFT64's stops near 23, a bound of `2^40`. Suites below radix 52 return.
pub fn test_blas_matmul_long_sums<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    if params.base2k < 52 {
        return;
    }
    let params = CKKSTestParams {
        n: 32,
        base2k: 57,
        hw: 16,
        ..params
    };
    let module = &Module::<BE>::new(params.n as u64);
    let host_module = &Module::<HostBytesBackend>::new(params.n as u64);
    let (slots, bound) = (params.n / 2, 1usize << (63 - params.base2k));
    let lanes = slots / (D * D);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, atks) = context_and_keys::<BE, F>(&params, module, &sk_raw, &mut scratch.borrow());
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let keys = TestKeys::<BE> { atks, tsk: Some(tsk) };
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let (a0, b0) = (Grid::model(1, 1, lanes, 7), Grid::model(1, 1, lanes, 8));
    let ct_a0 = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &a0,
        &mut scratch.borrow(),
    );
    let ct_b0 = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &b0,
        &mut scratch.borrow(),
    );
    let repeat = |g: &BlasCiphertextOwned<BE>, rows: usize, cols: usize| {
        BlasCiphertext::from_blocks(vec![g.block(0, 0).clone(); rows * cols], LOG_D, rows, cols).unwrap()
    };
    // `copies` equal terms repeat their noise as often: `log2(copies)` bits of precision go.
    let check = |label: &str, res: &BlasCiphertextOwned<BE>, want: &Grid, copies: usize, scratch: &mut ScratchArena<'_, BE>| {
        let (block, (re, im)) = (res.block(0, 0), &want.blocks(slots)[0]);
        let to_f = |v: &[f64]| v.iter().map(|&x| F::from_f64(x).unwrap()).collect::<Vec<F>>();
        let log_delta = block.log_delta() - copies.next_power_of_two().ilog2() as usize;
        assert_decrypt_precision_at_log_delta::<BE, F, E>(
            label,
            &params,
            module,
            &encoder,
            block,
            &sk,
            &to_f(re),
            &to_f(im),
            log_delta,
            scratch,
        );
    };

    // `d` terms per inner block: `K d` is three times the bound.
    let inner = 3 * bound / D;
    let a = Grid::from_fn(1, inner, lanes, |s, r, c| a0.at(s, r, c % D));
    let b = Grid::from_fn(inner, 1, lanes, |s, r, c| b0.at(s, r % D, c));
    let (ct_a, ct_b) = (repeat(&ct_a0, 1, inner), repeat(&ct_b0, inner, 1));
    let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 1, params.base2k.into(), params.k.into());
    let bytes = module.blas_matmul_tmp_bytes(&res, &ct_a, &ct_b, &tsk_layout, &atk_layout);
    module
        .blas_matmul(
            &mut res,
            &ct_a,
            &ct_b,
            &ctx,
            &keys,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        )
        .unwrap();
    check("matmul", &res, &a.matmul(&b), inner, &mut scratch.borrow());

    // Copies of the transformed blocks, allocated like the inputs as in `test_blas_matmul`.
    let mut left = BlasMatMulLeft::from_grid(module.blas_ciphertext_alloc_from_infos(&ct_a0));
    let mut right = BlasMatMulRight::from_grid(module.blas_ciphertext_alloc_from_infos(&ct_b0));
    let bytes = module.blas_matmul_prepare_left_tmp_bytes(left.as_grid(), &ct_a0, &atk_layout);
    module
        .blas_matmul_prepare_left(&mut left, &ct_a0, &ctx, &keys, &mut ScratchOwned::<BE>::alloc(bytes).borrow())
        .unwrap();
    let bytes = module.blas_matmul_prepare_right_tmp_bytes(right.as_grid(), &ct_b0, &atk_layout);
    module
        .blas_matmul_prepare_right(
            &mut right,
            &ct_b0,
            &ctx,
            &keys,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        )
        .unwrap();
    let left = BlasMatMulLeft::from_grid(repeat(left.as_grid(), 1, inner));
    let right = BlasMatMulRight::from_grid(repeat(right.as_grid(), inner, 1));
    let mut prepared = module.blas_ciphertext_alloc(LOG_D, 1, 1, params.base2k.into(), params.k.into());
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
    assert_eq!(grid_snapshot::<BE, _>(&prepared), grid_snapshot::<BE, _>(&res));

    // One term per inner block: `K` is three times the bound.
    let inner = 3 * bound;
    let entry = |m: usize, r: usize, c: usize| public_entry(m, r % D, c);
    let b = BlasPtOperandUnprepared::encode::<F>(
        module,
        LOG_D,
        inner,
        1,
        1,
        |m, r, c| F::from_f64(entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    )
    .unwrap()
    .prepare(module, &mut scratch.borrow());
    let atks = b
        .galois_elements(module.cyclotomic_order())
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
        .collect();
    let keys = TestKeys::<BE> { atks, tsk: None };
    let a = Grid::from_fn(1, inner, lanes, |s, r, c| a0.at(s, r, c % D));
    let ct = repeat(&ct_a0, 1, inner);
    let mut res = module.blas_ciphertext_alloc(LOG_D, 1, 1, params.base2k.into(), params.k.into());
    let bytes = module.blas_matmul_pt_tmp_bytes(&res, &ct, &atk_layout);
    module
        .blas_matmul_pt(&mut res, &ct, &b, &keys, &mut ScratchOwned::<BE>::alloc(bytes).borrow())
        .unwrap();
    check("matmul_pt", &res, &a.matmul_public(1, 1, entry), inner, &mut scratch.borrow());
}
