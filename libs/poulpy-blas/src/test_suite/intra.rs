//! Transpose, row and column rotation and axis reductions of whole grids.

use std::collections::HashMap;

use poulpy_ckks::{
    CKKSInfos, CoeffsMeta,
    api::{CKKSAddOps, CKKSEncodingOps, CKKSLinearTransformationOps},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_sk_with_raw},
        parity::helpers::with_scratch,
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::{
    GLWENormalize,
    layouts::{DiagonalArithmetic, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, Standard},
};

use super::helpers::{
    Grid, Misshapen,
    Rejection::{Budget, InvalidOperand, MissingKey},
    TestKeys, alloc_grid, assert_grid_decrypt, assert_guard, assert_guard_on, assert_short_scratch, at_scale, blas_context,
    context_and_keys, copied_layout, corner, encoded_context, encrypt_grid, foreign_contexts, foreign_layouts, grid_snapshot,
    mapped_layout, other_scales, poisoned_scratch, rejected_grids, relabel, rotation_keys,
};
use crate::{
    api::BlasMatIntraOps,
    layouts::{BlasCiphertextOwned, BlasContext, BlasContextInfos, BlasGridInfos, BlasPlan, replicated},
    oep::{
        BlasMatIntraImpl,
        derived::intra::{sum_line, sum_line_tmp_bytes},
    },
};

/// `dec(transpose(enc(A)))` is `A^T` for every matrix of the batch, on a `2 x 3` grid, with the keys of the plan's
/// transpose set alone, through maps at `A`'s scale and at scales below and above it ([`other_scales`]), whose scale
/// the transpose consumes in place of `A`'s, each into `res` as wide as `A` and one limb wider.
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
    // The keys of the transpose's set alone, whatever the maps' scale.
    let elements = BlasPlan::new(log_d).transpose_galois_elements(module.cyclotomic_order());
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
    // Maps at `a`'s scale, then at scales below and above it, each into `res` as wide as `a` and one limb wider.
    for p in [params.prec().log_delta()].into_iter().chain(other_scales(&params)) {
        let maps = at_scale(&params, p);
        let ctx = blas_context::<BE, F>(&maps, module, log_d, &mut scratch.borrow());
        for res_k in [params.k, params.k + params.base2k] {
            let mut res = alloc_grid(&params, module, log_d, cols, rows, res_k);
            let bytes = module.blas_transpose_tmp_bytes(&res, &ct, &ctx, &params.atk_layout());
            module
                .blas_transpose(&mut res, &ct, &ctx, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
                .unwrap();
            assert_grid_decrypt::<BE, F, E>(
                &format!("transpose p={p} res_k={res_k}"),
                &maps,
                module,
                &encoder,
                &res,
                &sk,
                &a.transpose(),
                &mapped_layout(&ct, cols, rows, res_k, p),
                &mut scratch.borrow(),
            );
        }
    }
}

/// `dec(rotate_rows(enc(A), k))` and `dec(rotate_cols(enc(A), k))` shift every matrix by `k` rows or columns: shifts
/// by whole blocks, shifts crossing a block boundary, shifts past the matrix height or width, and grids one block high
/// or wide, each shift with the keys of the plan's set for `k` alone. A grid one block high holding one matrix repeated
/// over the slots shifts its rows by rotations, at no scale, and its columns through phi^k. On the `2 x 3` grid, shifts
/// by 1 through both half maps also run on maps at scales below and above `A`'s ([`other_scales`]), whose scale they
/// consume in place of `A`'s.
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
                    elements(&ctx.plan(), order, k),
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
    // `4 d - 1`, `5 d + 1` and `d + 1` are `2 d - 1`, `2 d + 1` and `1` plus the matrix height or width. The last grid
    // holds one matrix repeated over the slots.
    let shapes: [(usize, usize, &[usize], &[usize]); 4] = [
        (
            2,
            3,
            &[0, 1, d, 2 * d - 1, 4 * d - 1],
            &[0, 2, 2 * d, 2 * d + 1, 3 * d - 1, 5 * d + 1],
        ),
        (1, 2, &[1, d - 1, d + 1], &[d + 1]),
        (2, 1, &[d - 1], &[1, d - 1, d + 1]),
        (1, 2, &[1, d - 1, d + 1], &[1, d + 1]),
    ];
    let lanes = slots >> (2 * log_d);
    let widths = [params.k, params.k + params.base2k];
    for (t, (rows, cols, row_shifts, col_shifts)) in shapes.into_iter().enumerate() {
        let spread = t == shapes.len() - 1;
        // The 2 x 3 grid runs once with `res` as wide as `a`, once one limb wider.
        let res_widths = if (rows, cols) == (2, 3) { &widths[..] } else { &widths[..1] };
        let (a, log_sparsity) = if spread {
            let one = Grid::model(log_d, rows, cols, 1, 9);
            let a = Grid::from_fn(log_d, rows, cols, lanes, |_, r, c| one.at(0, r, c));
            (a, lanes.ilog2() as usize)
        } else {
            (Grid::model(log_d, rows, cols, lanes, 5), 0)
        };
        let ct = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            log_sparsity,
            &a,
            &mut scratch.borrow(),
        );
        // A shift by whole blocks copies them, a row shift of a replicated grid one block high rotates them, any other
        // one maps every block; all narrow to the width `res` has when the shift writes it.
        let rotates = rows == 1 && replicated::<Standard>(&ct);
        let want_layout = |k: usize, res: &BlasCiphertextOwned<BE>, unmapped: bool| {
            let res_k = res.block_layout().k().as_usize();
            if k.is_multiple_of(d) || unmapped {
                copied_layout(&ct, rows, cols, res_k)
            } else {
                mapped_layout(&ct, rows, cols, res_k, params.prec().log_delta())
            }
        };
        for &res_k in res_widths {
            let mut res = alloc_grid(&params, module, log_d, rows, cols, res_k);
            let bytes = module
                .blas_rotate_rows_tmp_bytes(&res, &ct, &ctx, &params.atk_layout())
                .max(module.blas_rotate_cols_tmp_bytes(&res, &ct, &ctx, &params.atk_layout()));
            let mut op_scratch = poisoned_scratch::<BE>(bytes);
            for &k in row_shifts {
                let layout = want_layout(k, &res, rotates);
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
                let layout = want_layout(k, &res, false);
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
        if (rows, cols) != (2, 3) {
            continue;
        }
        // Maps at scales below and above `a`'s: a shift by 1 along either axis takes both half maps, whose scale it
        // consumes in place of `a`'s.
        for p in other_scales(&params) {
            let maps = at_scale(&params, p);
            let ctx = blas_context::<BE, F>(&maps, module, log_d, &mut scratch.borrow());
            let mut res = alloc_grid(&params, module, log_d, rows, cols, params.k);
            let bytes = module
                .blas_rotate_rows_tmp_bytes(&res, &ct, &ctx, &params.atk_layout())
                .max(module.blas_rotate_cols_tmp_bytes(&res, &ct, &ctx, &params.atk_layout()));
            let mut op_scratch = poisoned_scratch::<BE>(bytes);
            let layout = mapped_layout(&ct, rows, cols, params.k, p);
            module
                .blas_rotate_rows(&mut res, &ct, 1, &ctx, &row_keys[1], &mut op_scratch.borrow())
                .unwrap();
            let label = format!("rotate_rows {rows}x{cols} k=1 p={p}");
            assert_grid_decrypt::<BE, F, E>(
                &label,
                &maps,
                module,
                &encoder,
                &res,
                &sk,
                &a.rotate_rows(1),
                &layout,
                &mut scratch.borrow(),
            );
            module
                .blas_rotate_cols(&mut res, &ct, 1, &ctx, &col_keys[1], &mut op_scratch.borrow())
                .unwrap();
            let label = format!("rotate_cols {rows}x{cols} k=1 p={p}");
            assert_grid_decrypt::<BE, F, E>(
                &label,
                &maps,
                module,
                &encoder,
                &res,
                &sk,
                &a.rotate_cols(1),
                &layout,
                &mut scratch.borrow(),
            );
        }
    }
}

/// `dec(sum_rows(enc(A)))` and `dec(sum_cols(enc(A)))` hold, in every entry of every matrix, its column's sum over the
/// matrix rows and its row's sum over the matrix columns: on a `2 x 3` grid and on grids one block high and one block
/// wide, of distinct matrices and of one matrix repeated over the slots, whose rows sum at no scale, each reduction with
/// the keys of the plan's set alone, the `2 x 3` grid into `res` as wide as `A` and one limb wider. On the `2 x 3` grid
/// of distinct matrices, both also run through a context at scales below and above `A`'s ([`other_scales`]), whose
/// scale the projections consume in place of `A`'s.
pub fn test_blas_sum_rows_cols<BE, F, E>(
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
    let lanes = slots >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let ctx = blas_context::<BE, F>(&params, module, log_d, &mut scratch.borrow());
    let (plan, order) = (ctx.plan(), module.cyclotomic_order());
    let mut store = |elements: Vec<i64>| TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: None,
    };
    let (row_keys, col_keys) = (
        store(plan.sum_rows_galois_elements(order)),
        store(plan.sum_cols_galois_elements(order)),
    );
    let (p, atk) = (params.prec().log_delta(), params.atk_layout());
    let widths = [params.k, params.k + params.base2k];
    for (rows, cols) in [(2, 3), (1, 2), (3, 1)] {
        let one = Grid::model(log_d, rows, cols, 1, 11);
        let spread = Grid::from_fn(log_d, rows, cols, lanes, |_, r, c| one.at(0, r, c));
        for (a, log_sparsity) in [
            (Grid::model(log_d, rows, cols, lanes, 10), 0),
            (spread, lanes.ilog2() as usize),
        ] {
            let ct = encrypt_grid::<BE, F, E>(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                log_sparsity,
                &a,
                &mut scratch.borrow(),
            );
            let rep = replicated::<Standard>(&ct);
            let res_widths = if (rows, cols) == (2, 3) { &widths[..] } else { &widths[..1] };
            for &res_k in res_widths {
                let mut res = alloc_grid(&params, module, log_d, rows, cols, res_k);
                let label = format!("{rows}x{cols} replicated={rep} res_k={res_k}");
                // A replicated grid sums its rows without the projection, so at `a`'s scale and width.
                let rows_layout = if rep {
                    copied_layout(&ct, rows, cols, res_k)
                } else {
                    mapped_layout(&ct, rows, cols, res_k, p)
                };
                let mut op_scratch = poisoned_scratch::<BE>(module.blas_sum_rows_tmp_bytes(&res, &ct, &ctx, &atk));
                module
                    .blas_sum_rows(&mut res, &ct, &ctx, &row_keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_grid_decrypt::<BE, F, E>(
                    &format!("sum_rows {label}"),
                    &params,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.sum_rows(),
                    &rows_layout,
                    &mut scratch.borrow(),
                );
                let mut op_scratch = poisoned_scratch::<BE>(module.blas_sum_cols_tmp_bytes(&res, &ct, &ctx, &atk));
                module
                    .blas_sum_cols(&mut res, &ct, &ctx, &col_keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_grid_decrypt::<BE, F, E>(
                    &format!("sum_cols {label}"),
                    &params,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.sum_cols(),
                    &mapped_layout(&ct, rows, cols, res_k, p),
                    &mut scratch.borrow(),
                );
            }
            if (rows, cols) != (2, 3) || log_sparsity != 0 {
                continue;
            }
            // Projections at scales below and above `a`'s, whose scale the reductions consume in place of `a`'s.
            for p in other_scales(&params) {
                let maps = at_scale(&params, p);
                let ctx = blas_context::<BE, F>(&maps, module, log_d, &mut scratch.borrow());
                let mut res = alloc_grid(&params, module, log_d, rows, cols, params.k);
                let layout = mapped_layout(&ct, rows, cols, params.k, p);
                let rows_layout = if rep {
                    copied_layout(&ct, rows, cols, params.k)
                } else {
                    layout
                };
                let mut op_scratch = poisoned_scratch::<BE>(module.blas_sum_rows_tmp_bytes(&res, &ct, &ctx, &atk));
                module
                    .blas_sum_rows(&mut res, &ct, &ctx, &row_keys, &mut op_scratch.borrow())
                    .unwrap();
                let rows_maps = if rep { params } else { maps };
                assert_grid_decrypt::<BE, F, E>(
                    &format!("sum_rows p={p}"),
                    &rows_maps,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.sum_rows(),
                    &rows_layout,
                    &mut scratch.borrow(),
                );
                let mut op_scratch = poisoned_scratch::<BE>(module.blas_sum_cols_tmp_bytes(&res, &ct, &ctx, &atk));
                module
                    .blas_sum_cols(&mut res, &ct, &ctx, &col_keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_grid_decrypt::<BE, F, E>(
                    &format!("sum_cols p={p}"),
                    &maps,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.sum_cols(),
                    &layout,
                    &mut scratch.borrow(),
                );
            }
        }
    }
}

/// The line sum of [`blas_sum_rows`](BlasMatIntraOps::blas_sum_rows) and
/// [`blas_sum_cols`](BlasMatIntraOps::blas_sum_cols), of canonical blocks and of blocks each the sum of two canonical
/// ones, holds the bound it is given in canonical digits per coefficient, given since the radixes where
/// `max_lazy_terms` binds, 61 and 62, take key switches no backend computes exactly: through `ckks_add_many` at 4 terms, a flag-clear block
/// counting two, and block by block at 3 and 2 terms, where two flag-clear blocks exceed them, and through
/// `ckks_add_many` at 2 on canonical blocks alone, each within exactly the scratch its query reports. Never
/// normalized, each line exceeds the largest bound it is tested at, and every sum of a line normalizes to its digits.
pub fn test_blas_sum_rows_cols_lazy<BE, F, E>(
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
    let lanes = (params.n / 2) >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(params.n / 2).unwrap();
    let (_, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let a = Grid::model(log_d, 4, 3, lanes, 12);
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
    let layout = ct.block_layout();
    let mut pairs = alloc_grid(&params, module, log_d, 4, 1, params.k);
    for (i, pair) in pairs.blocks_mut().iter_mut().enumerate() {
        module
            .ckks_add_into(pair, &ct.blocks()[3 * i + 1], &ct.blocks()[3 * i + 2], &mut scratch.borrow())
            .unwrap();
        assert!(
            !GLWEToBackendRef::<BE>::is_canonical(pair),
            "a sum of two blocks is flag-clear"
        );
    }
    let canonical: Vec<_> = (0..4).map(|i| &ct.blocks()[3 * i]).collect();
    let mixed: Vec<_> = (0..4).flat_map(|i| [&pairs.blocks()[i], canonical[i]]).collect();
    let digit = 1i64 << (params.base2k - 1);
    let mut sum = |line: &[&_], terms: usize| {
        let mut out = alloc_grid(&params, module, log_d, 1, 1, params.k);
        with_scratch::<BE, _>(sum_line_tmp_bytes(module, &layout, terms), |s| {
            sum_line(module, &mut out.blocks_mut()[0], line, terms, s)
        })
        .unwrap();
        let lazy = grid_snapshot::<BE, _>(&out).remove(0).digits;
        module.glwe_normalize_assign(&mut out.blocks_mut()[0], &mut scratch.borrow());
        (lazy, grid_snapshot::<BE, _>(&out).remove(0).digits)
    };
    for (label, line, bounds) in [("mixed", &mixed, &[4, 3, 2][..]), ("canonical", &canonical, &[2][..])] {
        let (never, normalized) = sum(line, usize::MAX);
        assert!(
            never.iter().any(|x| x.abs() > bounds[0] as i64 * digit),
            "{label}: the sum never normalized stays within {} digits, so the bounds below test nothing",
            bounds[0]
        );
        for &terms in bounds {
            let (lazy, again) = sum(line, terms);
            assert!(
                lazy.iter().all(|x| x.abs() <= terms as i64 * digit),
                "{label} terms={terms}: a coefficient exceeds the bound"
            );
            assert!(again == normalized, "{label} terms={terms}: normalizing changed the sum");
        }
    }
}

/// Every grid [`rejected_grids`] and [`foreign_layouts`] build, as `a` or as `res`, a shape mismatch, a copy relabelled
/// past `log2` of the ring degree, a [`Misshapen`] grid as `a` or as `res`, `a` and `res` both of another degree than
/// the module's or of another rank than the keys, a context encoded on a module of another degree or at another radix,
/// and a missing rotation key are rejected through the API before `res` changes, as is a `res` narrower than the scale
/// of `a`, by poulpy-ckks's budget check; each operation one scratch byte short of its query, a row or column shift or
/// reduction along one block or across blocks, panics before `res` changes. The paths that read no map, a shift by `d`
/// and the row shift and row sum of a grid one block high holding one matrix, check the context's slot count alone.
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
    let foreign = foreign_contexts::<BE, F>(
        &params,
        module,
        BlasPlan::new(log_d),
        &sk_raw,
        &mut atks,
        &mut scratch.borrow(),
    );
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
    let atk = params.atk_layout();
    let mut op_scratch = poisoned_scratch::<BE>(
        module
            .blas_transpose_tmp_bytes(&alloc(cols, rows), &ct, &ctx, &atk)
            .max(module.blas_rotate_rows_tmp_bytes(&ct, &ct, &ctx, &atk))
            .max(module.blas_rotate_cols_tmp_bytes(&ct, &ct, &ctx, &atk))
            .max(module.blas_sum_rows_tmp_bytes(&ct, &ct, &ctx, &atk))
            .max(module.blas_sum_cols_tmp_bytes(&ct, &ct, &ctx, &atk)),
    );

    // `res` shaped like `a` for the transpose, like its transpose for the rotations and the reductions.
    let (mut wrong, mut wrong_t) = (alloc(cols, rows), alloc(rows, cols));
    assert_guard::<BE, _>("shape", "blas_transpose", InvalidOperand, &mut wrong_t, |res| {
        module.blas_transpose(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("shape", "blas_rotate_rows", InvalidOperand, &mut wrong, |res| {
        module.blas_rotate_rows(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("shape", "blas_rotate_cols", InvalidOperand, &mut wrong, |res| {
        module.blas_rotate_cols(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("shape", "blas_sum_rows", InvalidOperand, &mut wrong, |res| {
        module.blas_sum_rows(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("shape", "blas_sum_cols", InvalidOperand, &mut wrong, |res| {
        module.blas_sum_cols(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
    });

    // Each rejected grid as `a`, copies relabelled past `log2` of the ring degree, then each rejected grid as `res`.
    let (mut res, mut res_t) = (alloc(rows, cols), alloc(cols, rows));
    let sparse = [64, 65].map(|log_sparsity| {
        let a = relabel(&ct, |meta| meta.log_sparsity = log_sparsity);
        (format!("sparsity {log_sparsity}"), a)
    });
    let rejected = rejected_grids(module, &ct)
        .into_iter()
        .chain(foreign_layouts(module, &ct))
        .map(|(label, a)| (format!("{label} a"), a));
    for (label, a) in rejected.chain(sparse) {
        assert_guard::<BE, _>(&label, "blas_transpose", InvalidOperand, &mut res_t, |res| {
            module.blas_transpose(res, &a, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_rotate_rows", InvalidOperand, &mut res, |res| {
            module.blas_rotate_rows(res, &a, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_rotate_cols", InvalidOperand, &mut res, |res| {
            module.blas_rotate_cols(res, &a, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_sum_rows", InvalidOperand, &mut res, |res| {
            module.blas_sum_rows(res, &a, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_sum_cols", InvalidOperand, &mut res, |res| {
            module.blas_sum_cols(res, &a, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    let bad_res = rejected_grids(module, &res).into_iter().chain(foreign_layouts(module, &res));
    let bad_res_t = rejected_grids(module, &res_t)
        .into_iter()
        .chain(foreign_layouts(module, &res_t));
    for ((label, mut bad), (_, mut bad_t)) in bad_res.zip(bad_res_t) {
        let label = &format!("{label} res");
        assert_guard::<BE, _>(label, "blas_transpose", InvalidOperand, &mut bad_t, |res| {
            module.blas_transpose(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_rows", InvalidOperand, &mut bad, |res| {
            module.blas_rotate_rows(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_cols", InvalidOperand, &mut bad, |res| {
            module.blas_rotate_cols(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_sum_rows", InvalidOperand, &mut bad, |res| {
            module.blas_sum_rows(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_sum_cols", InvalidOperand, &mut bad, |res| {
            module.blas_sum_cols(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    // `a` and `res` both of another degree than the module's, then both of another rank than the keys, which the first
    // key of the maps rejects.
    for (label, i) in [("degree a and res", 0), ("rank a and res", 2)] {
        let other = |grid: &BlasCiphertextOwned<BE>| foreign_layouts(module, grid).into_iter().nth(i).unwrap().1;
        let (other_a, mut other_res, mut other_res_t) = (other(&ct), other(&res), other(&res_t));
        assert_guard::<BE, _>(label, "blas_transpose", InvalidOperand, &mut other_res_t, |res| {
            module.blas_transpose(res, &other_a, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_rows", InvalidOperand, &mut other_res, |res| {
            module.blas_rotate_rows(res, &other_a, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_cols", InvalidOperand, &mut other_res, |res| {
            module.blas_rotate_cols(res, &other_a, 1, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_sum_rows", InvalidOperand, &mut other_res, |res| {
            module.blas_sum_rows(res, &other_a, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_sum_cols", InvalidOperand, &mut other_res, |res| {
            module.blas_sum_cols(res, &other_a, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, ctx) in &foreign {
        assert_guard::<BE, _>(label, "blas_transpose", InvalidOperand, &mut res_t, |res| {
            module.blas_transpose(res, &ct, ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_rows", InvalidOperand, &mut res, |res| {
            module.blas_rotate_rows(res, &ct, 1, ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_rotate_cols", InvalidOperand, &mut res, |res| {
            module.blas_rotate_cols(res, &ct, 1, ctx, &keys, &mut op_scratch.borrow())
        });
        // The row sum of a grid holding one matrix reads no map: only another slot count rejects it.
        if *label == "degree" || !replicated::<Standard>(&ct) {
            assert_guard::<BE, _>(label, "blas_sum_rows", InvalidOperand, &mut res, |res| {
                module.blas_sum_rows(res, &ct, ctx, &keys, &mut op_scratch.borrow())
            });
        }
        assert_guard::<BE, _>(label, "blas_sum_cols", InvalidOperand, &mut res, |res| {
            module.blas_sum_cols(res, &ct, ctx, &keys, &mut op_scratch.borrow())
        });
    }

    // A shift by `d`, and the row shift and the row sum of a grid one block high holding one matrix, read no map: a
    // context at another radix or wider than their input gives the bits `ctx` gives, where the column sum, which
    // projects, rejects it; one of another slot count is rejected on every path.
    let d = 1usize << log_d;
    let lanes = slots >> (2 * log_d);
    let one = Grid::model(log_d, 1, 2, 1, 7);
    let spread = Grid::from_fn(log_d, 1, 2, lanes, |_, r, c| one.at(0, r, c));
    let ct_one = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        lanes.ilog2() as usize,
        &spread,
        &mut scratch.borrow(),
    );
    assert!(replicated::<Standard>(&ct_one), "the fixture holds one matrix");
    let wide = CoeffsMeta {
        k: (params.k + 1).into(),
        ..params.prec().into()
    };
    let wide = encoded_context::<BE, F>(
        module,
        BlasPlan::new(log_d),
        params.base2k.into(),
        wide,
        &mut scratch.borrow(),
    );
    let mut res_one = alloc(1, 2);
    let mut unmapped_scratch = poisoned_scratch::<BE>(
        module
            .blas_rotate_rows_tmp_bytes(&res, &ct, &ctx, &atk)
            .max(module.blas_rotate_rows_tmp_bytes(&res_one, &ct_one, &ctx, &atk))
            .max(module.blas_sum_rows_tmp_bytes(&res_one, &ct_one, &ctx, &atk)),
    );
    let mut unmapped = |ctx: &BlasContext<BE>| {
        let (mut shifted, mut rotated, mut summed) = (alloc(rows, cols), alloc(1, 2), alloc(1, 2));
        let mut s = unmapped_scratch.borrow();
        module.blas_rotate_rows(&mut shifted, &ct, d, ctx, &keys, &mut s).unwrap();
        module.blas_rotate_rows(&mut rotated, &ct_one, 1, ctx, &keys, &mut s).unwrap();
        module.blas_sum_rows(&mut summed, &ct_one, ctx, &keys, &mut s).unwrap();
        [shifted, rotated, summed].map(|g| grid_snapshot::<BE, _>(&g))
    };
    let want = unmapped(&ctx);
    for (label, other) in foreign.iter().map(|(label, ctx)| (*label, ctx)).chain([("width", &wide)]) {
        if label == "degree" {
            assert_guard::<BE, _>(label, "blas_rotate_rows", InvalidOperand, &mut res, |res| {
                module.blas_rotate_rows(res, &ct, d, other, &keys, &mut op_scratch.borrow())
            });
            assert_guard::<BE, _>(label, "blas_rotate_rows", InvalidOperand, &mut res_one, |res| {
                module.blas_rotate_rows(res, &ct_one, 1, other, &keys, &mut op_scratch.borrow())
            });
            assert_guard::<BE, _>(label, "blas_sum_rows", InvalidOperand, &mut res_one, |res| {
                module.blas_sum_rows(res, &ct_one, other, &keys, &mut op_scratch.borrow())
            });
            continue;
        }
        assert!(unmapped(other) == want, "{label}: a path without a map read the context");
        assert_guard::<BE, _>(label, "blas_sum_cols", InvalidOperand, &mut res_one, |res| {
            module.blas_sum_cols(res, &ct_one, other, &keys, &mut op_scratch.borrow())
        });
    }
    // A grid one block row short of the shape it reports, as `a`, then as `res`: its shape alone would pass.
    let short = |grid: BlasCiphertextOwned<BE>| Misshapen {
        rows: grid.rows() + 1,
        cols: grid.cols(),
        grid,
    };
    let short_a = short(corner::<BE>(&ct, 1, ct.cols()));
    assert_guard::<BE, _>("misshapen a", "blas_transpose", InvalidOperand, &mut res_t, |res| {
        module.blas_transpose(res, &short_a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("misshapen a", "blas_rotate_rows", InvalidOperand, &mut res, |res| {
        module.blas_rotate_rows(res, &short_a, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("misshapen a", "blas_rotate_cols", InvalidOperand, &mut res, |res| {
        module.blas_rotate_cols(res, &short_a, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("misshapen a", "blas_sum_rows", InvalidOperand, &mut res, |res| {
        module.blas_sum_rows(res, &short_a, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("misshapen a", "blas_sum_cols", InvalidOperand, &mut res, |res| {
        module.blas_sum_cols(res, &short_a, &ctx, &keys, &mut op_scratch.borrow())
    });
    let (mut short_res, mut short_res_t) = (short(alloc(rows - 1, cols)), short(alloc(cols - 1, rows)));
    assert_guard_on::<BE, _, _>(
        "misshapen res",
        "blas_transpose",
        InvalidOperand,
        &mut short_res_t,
        |res| &res.grid,
        |res| module.blas_transpose(res, &ct, &ctx, &keys, &mut op_scratch.borrow()),
    );
    assert_guard_on::<BE, _, _>(
        "misshapen res",
        "blas_rotate_rows",
        InvalidOperand,
        &mut short_res,
        |res| &res.grid,
        |res| module.blas_rotate_rows(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow()),
    );
    assert_guard_on::<BE, _, _>(
        "misshapen res",
        "blas_rotate_cols",
        InvalidOperand,
        &mut short_res,
        |res| &res.grid,
        |res| module.blas_rotate_cols(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow()),
    );
    assert_guard_on::<BE, _, _>(
        "misshapen res",
        "blas_sum_rows",
        InvalidOperand,
        &mut short_res,
        |res| &res.grid,
        |res| module.blas_sum_rows(res, &ct, &ctx, &keys, &mut op_scratch.borrow()),
    );
    assert_guard_on::<BE, _, _>(
        "misshapen res",
        "blas_sum_cols",
        InvalidOperand,
        &mut short_res,
        |res| &res.grid,
        |res| module.blas_sum_cols(res, &ct, &ctx, &keys, &mut op_scratch.borrow()),
    );

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
    assert_guard::<BE, _>("no keys", "blas_sum_rows", MissingKey, &mut res, |res| {
        module.blas_sum_rows(res, &ct, &ctx, &no_keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("no keys", "blas_sum_cols", MissingKey, &mut res, |res| {
        module.blas_sum_cols(res, &ct, &ctx, &no_keys, &mut op_scratch.borrow())
    });

    // `res` narrower than the scale of `a`: poulpy-ckks rejects the budget of the first map's product inside the
    // implementation, before any write.
    let narrow = ct.block_layout().log_delta() - 1;
    let (mut narrow_res, mut narrow_res_t) = (
        alloc_grid(&params, module, log_d, rows, cols, narrow),
        alloc_grid(&params, module, log_d, cols, rows, narrow),
    );
    assert_guard::<BE, _>("budget", "mul", Budget, &mut narrow_res_t, |res| {
        module.blas_transpose(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("budget", "mul", Budget, &mut narrow_res, |res| {
        module.blas_rotate_rows(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("budget", "mul", Budget, &mut narrow_res, |res| {
        module.blas_rotate_cols(res, &ct, 1, &ctx, &keys, &mut op_scratch.borrow())
    });
    // A replicated grid sums its rows without the projection: the inner sum into `res` rejects the budget.
    let sum_rows_op = if replicated::<Standard>(&ct) { "inner_sum" } else { "mul" };
    assert_guard::<BE, _>("budget", sum_rows_op, Budget, &mut narrow_res, |res| {
        module.blas_sum_rows(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("budget", "mul", Budget, &mut narrow_res, |res| {
        module.blas_sum_cols(res, &ct, &ctx, &keys, &mut op_scratch.borrow())
    });

    // A shift by 1 takes both half maps along two or three blocks, the whole map along one.
    let (one_row, one_column) = (corner::<BE>(&ct, 1, ct.cols()), corner::<BE>(&ct, ct.rows(), 1));
    for a in [&ct, &one_row, &one_column] {
        let mut res = alloc(a.rows(), a.cols());
        let bytes = module.blas_rotate_rows_tmp_bytes(&res, a, &ctx, &atk);
        assert_short_scratch::<BE, _>("blas_rotate_rows", &mut res, bytes, |res, s| {
            module.blas_rotate_rows(res, a, 1, &ctx, &keys, s)
        });
        let bytes = module.blas_rotate_cols_tmp_bytes(&res, a, &ctx, &atk);
        assert_short_scratch::<BE, _>("blas_rotate_cols", &mut res, bytes, |res, s| {
            module.blas_rotate_cols(res, a, 1, &ctx, &keys, s)
        });
        let bytes = module.blas_sum_rows_tmp_bytes(&res, a, &ctx, &atk);
        assert_short_scratch::<BE, _>("blas_sum_rows", &mut res, bytes, |res, s| {
            module.blas_sum_rows(res, a, &ctx, &keys, s)
        });
        let bytes = module.blas_sum_cols_tmp_bytes(&res, a, &ctx, &atk);
        assert_short_scratch::<BE, _>("blas_sum_cols", &mut res, bytes, |res, s| {
            module.blas_sum_cols(res, a, &ctx, &keys, s)
        });
    }
    let bytes = module.blas_transpose_tmp_bytes(&res_t, &ct, &ctx, &atk);
    assert_short_scratch::<BE, _>("blas_transpose", &mut res_t, bytes, |res, s| {
        module.blas_transpose(res, &ct, &ctx, &keys, s)
    });
}
