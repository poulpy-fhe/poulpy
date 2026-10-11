//! Matrix-slot rotation and sum on every block of a grid.

use std::collections::HashMap;

use poulpy_ckks::{
    CKKSInfos,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_atk, gen_sk_with_raw},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
};

use super::helpers::{
    Grid, Misshapen,
    Rejection::{Budget, InvalidOperand, MissingKey},
    TestKeys, alloc_grid, assert_grid_decrypt, assert_guard, assert_guard_on, assert_short_scratch, copied_layout, encrypt_grid,
    foreign_layouts, rejected_grids, relabel, rotation_keys,
};
use crate::{
    api::BlasMatInterOps,
    layouts::{BlasCiphertext, BlasCiphertextOwned, BlasGridInfos, BlasLayout, BlasModuleAlloc, BlasPlan, sum_galois_elements},
    oep::BlasMatInterImpl,
};

/// `dec(rotate(enc(A), r))` holds matrix `s + r` of `A` in matrix slot `s`, in every block of a `2 x 2` grid, with the
/// keys of the plan's set for `r` alone.
pub fn test_blas_rotate<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatInterImpl,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let a = Grid::model(log_d, 2, 2, slots >> (2 * log_d), 1);
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
    let plan = BlasPlan::new(log_d);
    for r in [1i64, -1, 3] {
        // The keys of the rotation's set alone.
        let elements = plan.rotate_galois_elements(module.cyclotomic_order(), r);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
            tsk: None,
        };
        // `res` one limb wider than `a` for r = 3.
        let res_k = if r == 3 { params.k + params.base2k } else { params.k };
        let mut res = alloc_grid(&params, module, log_d, 2, 2, res_k);
        let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_rotate_tmp_bytes(&res, &ct, &params.atk_layout()));
        module.blas_rotate(&mut res, &ct, r, &keys, &mut op_scratch.borrow()).unwrap();
        assert_grid_decrypt::<BE, F, E>(
            &format!("rotate r={r} res_k={res_k}"),
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &a.rotate_lanes(r),
            &copied_layout(&ct, 2, 2, res_k),
            &mut scratch.borrow(),
        );
    }
}

/// `dec(sum(enc(A)))` holds the sum of every matrix of `A` in each matrix slot, in every block of a `2 x 2` grid.
pub fn test_blas_sum<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>, log_d: usize)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatInterImpl,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let a = Grid::model(log_d, 2, 2, slots >> (2 * log_d), 2);
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
    let atks = sum_galois_elements(module, &ct)
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
        .collect();
    let keys = TestKeys::<BE> { atks, tsk: None };
    // Once with `res` as wide as `a`, once one limb wider.
    for res_k in [params.k, params.k + params.base2k] {
        let mut res = alloc_grid(&params, module, log_d, 2, 2, res_k);
        let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_sum_tmp_bytes(&res, &ct, &params.atk_layout()));
        module.blas_sum(&mut res, &ct, &keys, &mut op_scratch.borrow()).unwrap();
        assert_grid_decrypt::<BE, F, E>(
            &format!("sum res_k={res_k}"),
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &a.sum_lanes(),
            &copied_layout(&ct, 2, 2, res_k),
            &mut scratch.borrow(),
        );
    }
}

/// Every grid [`rejected_grids`] and [`foreign_layouts`] build, as `a` or as `res`, a shape mismatch, a copy relabelled
/// past `log2` of the ring degree, a [`Misshapen`] grid as `a` or as `res`, a missing rotation key, blocks wider than the
/// slots, and a sum of more lazy terms than its radix holds are rejected through the API before `res` changes, as is a
/// `res` narrower than the scale of `a`, by poulpy-ckks's budget check; a rotation or a sum one scratch byte short of its
/// query panics before `res` changes; `from_blocks` rejects a block count overflowing `usize`. On one matrix slot, where
/// every rotation is the identity, the missing-key, lazy-term and scratch cases are skipped.
pub fn test_blas_inter_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatInterImpl,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let lanes = slots >> (2 * log_d);
    let a = Grid::model(log_d, 2, 2, lanes, 3);
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
    let keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    let alloc = |log_d: usize, rows: usize, cols: usize| alloc_grid(&params, module, log_d, rows, cols, params.k);
    let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_sum_tmp_bytes(&ct, &ct, &params.atk_layout()));

    let sparse_64 = relabel(&ct, |meta| meta.log_sparsity = 64);
    let sparse_65 = relabel(&ct, |meta| meta.log_sparsity = 65);
    let rejected = rejected_grids(module, &ct);
    // Each rejected grid as `a`, then as `res`; a wrong shape of `res`; copies relabelled past `log2` of the ring degree,
    // which keep recording the dense batch.
    let foreign = foreign_layouts(module, &ct);
    let as_a = rejected
        .iter()
        .chain(&foreign)
        .map(|(label, a)| (format!("{label} a"), alloc(log_d, 2, 2), a));
    let as_res = rejected_grids(module, &alloc(log_d, 2, 2))
        .into_iter()
        .chain(foreign_layouts(module, &ct))
        .map(|(label, res)| (format!("{label} res"), res, &ct));
    // `a` and `res` both of another degree than the module's.
    let wide = |grid: &BlasCiphertextOwned<BE>| foreign_layouts(module, grid).into_iter().next().unwrap().1;
    let wide_a = wide(&ct);
    let others = [
        ("shape", alloc(log_d, 2, 1), &ct),
        ("degree a and res", wide(&alloc(log_d, 2, 2)), &wide_a),
        ("sparsity 64", alloc(log_d, 2, 2), &sparse_64),
        ("sparsity 65", alloc(log_d, 2, 2), &sparse_65),
    ]
    .map(|(label, res, a)| (label.to_string(), res, a));
    for (label, mut res, a) in as_a.chain(as_res).chain(others) {
        assert_guard::<BE, _>(&label, "blas_rotate", InvalidOperand, &mut res, |res| {
            module.blas_rotate(res, a, 1, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_sum", InvalidOperand, &mut res, |res| {
            module.blas_sum(res, a, &keys, &mut op_scratch.borrow())
        });
    }

    // A grid one block row short of the shape it reports, as `a`, then as `res`.
    let short = |grid: BlasCiphertextOwned<BE>| Misshapen {
        rows: grid.rows() + 1,
        cols: grid.cols(),
        grid,
    };
    let (one_row, mut res) = (short(ct.clone().split_rows(1).0), alloc(log_d, 2, 2));
    assert_guard::<BE, _>("misshapen a", "blas_rotate", InvalidOperand, &mut res, |res| {
        module.blas_rotate(res, &one_row, 1, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("misshapen a", "blas_sum", InvalidOperand, &mut res, |res| {
        module.blas_sum(res, &one_row, &keys, &mut op_scratch.borrow())
    });
    let mut short_res = short(alloc(log_d, 1, 2));
    assert_guard_on::<BE, _, _>(
        "misshapen res",
        "blas_rotate",
        InvalidOperand,
        &mut short_res,
        |res| &res.grid,
        |res| module.blas_rotate(res, &ct, 1, &keys, &mut op_scratch.borrow()),
    );
    assert_guard_on::<BE, _, _>(
        "misshapen res",
        "blas_sum",
        InvalidOperand,
        &mut short_res,
        |res| &res.grid,
        |res| module.blas_sum(res, &ct, &keys, &mut op_scratch.borrow()),
    );

    // Valid grids, empty key store; on one matrix slot every rotation is the identity, which takes no key.
    let mut res = alloc(log_d, 2, 2);
    if lanes > 1 {
        assert_guard::<BE, _>("no keys", "blas_rotate", MissingKey, &mut res, |res| {
            module.blas_rotate(res, &ct, 1, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>("no keys", "blas_sum", MissingKey, &mut res, |res| {
            module.blas_sum(res, &ct, &keys, &mut op_scratch.borrow())
        });
    }

    // 2^(2 * log_d) above the slot count: no block fits.
    let log_d_wide = (params.n / 2).ilog2() as usize / 2 + 1;
    let wide = BlasCiphertext::from_blocks(ct.clone().into_blocks(), log_d_wide, 2, 2).unwrap();
    let mut res_wide = alloc(log_d_wide, 2, 2);
    assert_guard::<BE, _>("wide", "blas_rotate", InvalidOperand, &mut res_wide, |res| {
        module.blas_rotate(res, &wide, 1, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("wide", "blas_sum", InvalidOperand, &mut res_wide, |res| {
        module.blas_sum(res, &wide, &keys, &mut op_scratch.borrow())
    });

    // At radix 63 a digit holds one lazy term, fewer than the log2(mat_slots) + 1 the sum adds up.
    if lanes > 1 {
        let mut block = ct.block_layout();
        block.glwe_layout.base2k = 63usize.into();
        let digits_63 = || module.blas_ciphertext_alloc_from_infos(&BlasLayout::new(block, log_d, 2, 2));
        let (a_63, mut res_63) = (digits_63(), digits_63());
        assert_guard::<BE, _>("lazy terms", "blas_sum", InvalidOperand, &mut res_63, |res| {
            module.blas_sum(res, &a_63, &keys, &mut op_scratch.borrow())
        });
    }

    // The keys of the sum's rotations, the first by one matrix slot; none on one matrix slot.
    let sum_keys = TestKeys::<BE> {
        atks: sum_galois_elements(module, &ct)
            .into_iter()
            .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
            .collect(),
        tsk: None,
    };
    // `res` narrower than the scale of `a`: poulpy-ckks rejects its budget inside the implementation, before any write,
    // in the rotation, or in the copy that starts the sum and replaces the rotation on one matrix slot.
    let mut narrow = alloc_grid(&params, module, log_d, 2, 2, ct.block_layout().log_delta() - 1);
    let rotate = if lanes > 1 { "rotate" } else { "copy" };
    assert_guard::<BE, _>("budget", rotate, Budget, &mut narrow, |res| {
        module.blas_rotate(res, &ct, 1, &sum_keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("budget", "copy", Budget, &mut narrow, |res| {
        module.blas_sum(res, &ct, &sum_keys, &mut op_scratch.borrow())
    });

    // A call that looks up no key, on one matrix slot, asserts no scratch.
    if lanes > 1 {
        let bytes = module.blas_rotate_tmp_bytes(&res, &ct, &params.atk_layout());
        assert_short_scratch::<BE, _>("blas_rotate", &mut res, bytes, |res, s| {
            module.blas_rotate(res, &ct, 1, &sum_keys, s)
        });
        let bytes = module.blas_sum_tmp_bytes(&res, &ct, &params.atk_layout());
        assert_short_scratch::<BE, _>("blas_sum", &mut res, bytes, |res, s| module.blas_sum(res, &ct, &sum_keys, s));
    }

    // (2^62 + 1) x 4 wraps to the 4 blocks of `ct` on 64 bits.
    assert!(BlasCiphertext::from_blocks(ct.into_blocks(), log_d, usize::MAX / 4 + 2, 4).is_err());
}
