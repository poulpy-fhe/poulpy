//! Transpose and row and column rotation of whole grids.

use std::collections::HashMap;

use poulpy_ckks::{
    CKKSInfos,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_sk_with_raw},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{DiagonalArithmetic, LWEInfos};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
};

use super::helpers::{
    Grid,
    Rejection::{InvalidPlan, MissingKey},
    TestKeys, alloc_grid, assert_grid_decrypt, assert_guard, assert_short_scratch, blas_context, context_and_keys, copied_layout,
    encrypt_grid, foreign_contexts, mapped_layout, rejected_grids, relabel, rotation_keys,
};
use crate::{
    api::BlasMatIntraOps,
    layouts::{BlasCiphertextOwned, BlasGridInfos, BlasPlan},
    oep::BlasMatIntraImpl,
};

/// `dec(transpose(enc(A)))` is `A^T` for every matrix of the batch, on a `2 x 3` grid, with the keys of the plan's
/// transpose set alone.
pub fn test_blas_transpose<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatIntraImpl,
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
    let ctx = blas_context::<BE, F>(&params, module, log_d, &mut scratch.borrow());
    // The keys of the transpose's set alone.
    let elements = ctx.plan().transpose_galois_elements(module.cyclotomic_order());
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: None,
    };
    let (rows, cols) = (2, 3);
    let a = Grid::model(log_d, rows, cols, slots >> (2 * log_d), 4);
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
    // Once with `res` as wide as `a`, once one limb wider.
    for res_k in [params.k, params.k + params.base2k] {
        let mut res = alloc_grid(&params, module, log_d, cols, rows, res_k);
        let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_transpose_tmp_bytes(&res, &ct, &params.atk_layout()));
        module
            .blas_transpose(&mut res, &ct, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_grid_decrypt::<BE, F, E>(
            &format!("transpose res_k={res_k}"),
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &a.transpose(),
            &mapped_layout(&ct, cols, rows, res_k, params.prec().log_delta()),
            &mut scratch.borrow(),
        );
    }
}

/// `dec(rotate_rows(enc(A), k))` and `dec(rotate_cols(enc(A), k))` shift every matrix by `k` rows or columns: shifts
/// by whole blocks, shifts crossing a block boundary, shifts past the matrix height or width, and grids one block high
/// or wide, each shift with the keys of the plan's set for `k` alone.
pub fn test_blas_rotate_rows_cols<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatIntraImpl,
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
    let ctx = blas_context::<BE, F>(&params, module, log_d, &mut scratch.borrow());
    // Each shift runs on the keys of its own set alone: one store per row and per column shift below `d`.
    let (d, order) = (1 << log_d, module.cyclotomic_order());
    let mut stores = |elements: fn(&BlasPlan, i64, usize) -> Vec<i64>| -> Vec<TestKeys<BE>> {
        (0..d)
            .map(|k| TestKeys::<BE> {
                atks: rotation_keys(
                    &params,
                    module,
                    &sk_raw,
                    elements(ctx.plan(), order, k),
                    &mut scratch.borrow(),
                ),
                tsk: None,
            })
            .collect()
    };
    let (row_keys, col_keys) = (
        stores(BlasPlan::rotate_rows_galois_elements),
        stores(BlasPlan::rotate_cols_galois_elements),
    );
    // `4 d - 1`, `5 d + 1` and `d + 1` are `2 d - 1`, `2 d + 1` and `1` plus the matrix height or width.
    let shapes: [(usize, usize, &[usize], &[usize]); 3] = [
        (
            2,
            3,
            &[0, 1, d, 2 * d - 1, 4 * d - 1],
            &[0, 2, 2 * d, 2 * d + 1, 3 * d - 1, 5 * d + 1],
        ),
        (1, 2, &[1, d - 1, d + 1], &[d + 1]),
        (2, 1, &[d - 1], &[1, d - 1, d + 1]),
    ];
    let widths = [params.k, params.k + params.base2k];
    for (rows, cols, row_shifts, col_shifts) in shapes {
        // The 2 x 3 grid runs once with `res` as wide as `a`, once one limb wider.
        let res_widths = if (rows, cols) == (2, 3) { &widths[..] } else { &widths[..1] };
        let a = Grid::model(log_d, rows, cols, slots >> (2 * log_d), 5);
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
        // A shift by whole blocks copies them, any other one maps every block; both narrow to the width `res` has when
        // the shift writes it.
        let want_layout = |k: usize, res: &BlasCiphertextOwned<BE>| {
            let res_k = res.block_layout().k().as_usize();
            if k.is_multiple_of(d) {
                copied_layout(&ct, rows, cols, res_k)
            } else {
                mapped_layout(&ct, rows, cols, res_k, params.prec().log_delta())
            }
        };
        for &res_k in res_widths {
            let mut res = alloc_grid(&params, module, log_d, rows, cols, res_k);
            let bytes = module
                .blas_rotate_rows_tmp_bytes(&res, &ct, &params.atk_layout())
                .max(module.blas_rotate_cols_tmp_bytes(&res, &ct, &params.atk_layout()));
            let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
            for &k in row_shifts {
                let layout = want_layout(k, &res);
                module
                    .blas_rotate_rows(&mut res, &ct, k, &ctx, &row_keys[k % d], &mut op_scratch.borrow())
                    .unwrap();
                assert_grid_decrypt::<BE, F, E>(
                    &format!("rotate_rows {rows}x{cols} k={k} res_k={res_k}"),
                    &params,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.rotate_rows(k),
                    &layout,
                    &mut scratch.borrow(),
                );
            }
            for &k in col_shifts {
                let layout = want_layout(k, &res);
                module
                    .blas_rotate_cols(&mut res, &ct, k, &ctx, &col_keys[k % d], &mut op_scratch.borrow())
                    .unwrap();
                assert_grid_decrypt::<BE, F, E>(
                    &format!("rotate_cols {rows}x{cols} k={k} res_k={res_k}"),
                    &params,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.rotate_cols(k),
                    &layout,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}

/// Every grid [`rejected_grids`] builds, as `a` or as `res`, a shape mismatch, a copy relabelled past `log2` of the ring
/// degree, a context encoded on a module of another degree or at another radix, and a missing rotation key are
/// rejected before `res` changes; each operation one scratch byte short of its query, a row or column shift along one
/// block or across blocks, panics before `res` changes.
pub fn test_blas_intra_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatIntraImpl,
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
    let (ctx, mut atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let foreign = foreign_contexts::<BE, F>(&params, module, log_d, &sk_raw, &mut atks, &mut scratch.borrow());
    let keys = TestKeys::<BE> { atks, tsk: None };
    let (rows, cols) = (2, 3);
    let a = Grid::model(log_d, rows, cols, slots >> (2 * log_d), 6);
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
    let alloc = |rows: usize, cols: usize| alloc_grid(&params, module, log_d, rows, cols, params.k);
    let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_rotate_rows_tmp_bytes(&ct, &ct, &params.atk_layout()));

    // `res` shaped like `a` for the transpose, like its transpose for the rotations.
    let (mut wrong, mut wrong_t) = (alloc(cols, rows), alloc(rows, cols));
    assert_guard::<BE, _>("shape", "blas_transpose", InvalidPlan, &mut wrong_t, |res| {
        module.blas_transpose(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("shape", "blas_rotate_rows", InvalidPlan, &mut wrong, |res| {
        module.blas_rotate_rows(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("shape", "blas_rotate_cols", InvalidPlan, &mut wrong, |res| {
        module.blas_rotate_cols(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
    });

    // Each rejected grid as `a`, copies relabelled past `log2` of the ring degree, then each rejected grid as `res`.
    let (mut res, mut res_t) = (alloc(rows, cols), alloc(cols, rows));
    let sparse = [64, 65].map(|log_sparsity| {
        let a = relabel(&ct, |meta| meta.log_sparsity = log_sparsity);
        (format!("sparsity {log_sparsity}"), a)
    });
    let rejected = rejected_grids(module, &ct).map(|(label, a)| (format!("{label} a"), a));
    for (label, a) in rejected.into_iter().chain(sparse) {
        assert_guard::<BE, _>(&label, "blas_transpose", InvalidPlan, &mut res_t, |res| {
            module.blas_transpose(res, &a, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_rotate_rows", InvalidPlan, &mut res, |res| {
            module.blas_rotate_rows(res, &a, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_rotate_cols", InvalidPlan, &mut res, |res| {
            module.blas_rotate_cols(res, &a, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for ((label, mut bad), (_, mut bad_t)) in rejected_grids(module, &res).into_iter().zip(rejected_grids(module, &res_t)) {
        let label = &format!("{label} res");
        assert_guard::<BE, _>(label, "blas_transpose", InvalidPlan, &mut bad_t, |res| {
            module.blas_transpose(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_rows", InvalidPlan, &mut bad, |res| {
            module.blas_rotate_rows(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_cols", InvalidPlan, &mut bad, |res| {
            module.blas_rotate_cols(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, ctx) in &foreign {
        assert_guard::<BE, _>(label, "blas_transpose", InvalidPlan, &mut res_t, |res| {
            module.blas_transpose(res, &ct, ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_rows", InvalidPlan, &mut res, |res| {
            module.blas_rotate_rows(res, &ct, 1, ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_cols", InvalidPlan, &mut res, |res| {
            module.blas_rotate_cols(res, &ct, 1, ctx, &keys, &mut op_scratch.borrow())
        });
    }
    let no_keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    assert_guard::<BE, _>("no keys", "blas_rotate_rows", MissingKey, &mut res, |res| {
        module.blas_rotate_rows(res, &ct, 1, &ctx, &no_keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("no keys", "blas_rotate_cols", MissingKey, &mut res, |res| {
        module.blas_rotate_cols(res, &ct, 1, &ctx, &no_keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("no keys", "blas_transpose", MissingKey, &mut res_t, |res| {
        module.blas_transpose(res, &ct, &ctx, &no_keys, &mut op_scratch.borrow())
    });

    // A shift by 1 takes both half maps along two or three blocks, the whole map along one.
    let atk = params.atk_layout();
    let (one_row, one_column) = (ct.clone().split_rows(1).0, ct.clone().split_cols(1).0);
    for a in [&ct, &one_row, &one_column] {
        let mut res = alloc(a.rows(), a.cols());
        let bytes = module.blas_rotate_rows_tmp_bytes(&res, a, &atk);
        assert_short_scratch::<BE, _>("blas_rotate_rows", &mut res, bytes, |res, s| {
            module.blas_rotate_rows(res, a, 1, &ctx, &keys, s)
        });
        let bytes = module.blas_rotate_cols_tmp_bytes(&res, a, &atk);
        assert_short_scratch::<BE, _>("blas_rotate_cols", &mut res, bytes, |res, s| {
            module.blas_rotate_cols(res, a, 1, &ctx, &keys, s)
        });
    }
    let bytes = module.blas_transpose_tmp_bytes(&res_t, &ct, &atk);
    assert_short_scratch::<BE, _>("blas_transpose", &mut res_t, bytes, |res, s| {
        module.blas_transpose(res, &ct, &ctx, &keys, s)
    });
}
