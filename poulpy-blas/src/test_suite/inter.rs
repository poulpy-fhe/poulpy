//! Matrix-slot rotation and sum on every block of a grid.

use std::collections::HashMap;

use poulpy_ckks::{
    api::CKKSModuleInfos,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_atk, gen_sk_with_raw},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
};

use super::helpers::{
    D, Grid, LOG_D,
    Rejection::{InvalidPlan, MissingKey},
    TestKeys, assert_grid_decrypt, assert_guard, assert_short_scratch, encrypt_grid, rejected_grids, relabel,
};
use crate::{
    api::BlasMatInterOps,
    layouts::{BlasCiphertext, BlasModuleAlloc, sum_galois_elements},
    oep::BlasMatInterImpl,
};

/// `dec(rotate(enc(A), r))` holds matrix `s + r` of `A` in matrix slot `s`, in every block of a `2 x 2` grid.
pub fn test_blas_rotate<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let a = Grid::model(2, 2, slots / (D * D), 1);
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a,
        &mut scratch.borrow(),
    );
    let d2 = (D * D) as i64;
    for r in [1i64, -1, 3] {
        let p = module.ckks_galois_element(r * d2);
        let keys = TestKeys::<BE> {
            atks: HashMap::from([(p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()))]),
            tsk: None,
        };
        // `res` one limb wider than `a` for r = 3.
        let res_k = if r == 3 { params.k + params.base2k } else { params.k };
        let mut res = module.blas_ciphertext_alloc(LOG_D, 2, 2, params.base2k.into(), res_k.into());
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
            &mut scratch.borrow(),
        );
    }
}

/// `dec(sum(enc(A)))` holds the sum of every matrix of `A` in each matrix slot, in every block of a `2 x 2` grid.
pub fn test_blas_sum<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let a = Grid::model(2, 2, slots / (D * D), 2);
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
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
        let mut res = module.blas_ciphertext_alloc(LOG_D, 2, 2, params.base2k.into(), res_k.into());
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
            &mut scratch.borrow(),
        );
    }
}

/// Every grid [`rejected_grids`] builds, as `a` or as `res`, a shape mismatch, a sparsity shifting the slot count past
/// the word width, a missing rotation key, and blocks wider than the slots are rejected before `res` changes; a rotation
/// or a sum one scratch byte short of its query panics before `res` changes; `from_blocks` rejects a block count
/// overflowing `usize`.
pub fn test_blas_inter_guards<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let a = Grid::model(2, 2, slots / (D * D), 3);
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a,
        &mut scratch.borrow(),
    );
    let keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    let alloc = |log_d: usize, rows: usize, cols: usize| {
        module.blas_ciphertext_alloc(log_d, rows, cols, params.base2k.into(), params.k.into())
    };
    let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_sum_tmp_bytes(&ct, &ct, &params.atk_layout()));

    let sparse_64 = relabel(&ct, |meta| meta.log_sparsity = 64);
    let sparse_65 = relabel(&ct, |meta| meta.log_sparsity = 65);
    let rejected = rejected_grids(module, &ct);
    // Each rejected grid as `a`, then as `res`; a wrong shape of `res`; a sparsity shifting the slot count past the word
    // width.
    let as_a = rejected
        .iter()
        .map(|(label, a)| (format!("{label} a"), alloc(LOG_D, 2, 2), a));
    let as_res = rejected_grids(module, &alloc(LOG_D, 2, 2)).map(|(label, res)| (format!("{label} res"), res, &ct));
    let others = [
        ("shape", alloc(LOG_D, 2, 1), &ct),
        ("sparsity 64", alloc(LOG_D, 2, 2), &sparse_64),
        ("sparsity 65", alloc(LOG_D, 2, 2), &sparse_65),
    ]
    .map(|(label, res, a)| (label.to_string(), res, a));
    for (label, mut res, a) in as_a.chain(as_res).chain(others) {
        assert_guard::<BE, _>(&label, "blas_rotate", InvalidPlan, &mut res, |res| {
            module.blas_rotate(res, a, 1, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&label, "blas_sum", InvalidPlan, &mut res, |res| {
            module.blas_sum(res, a, &keys, &mut op_scratch.borrow())
        });
    }

    // Valid grids, empty key store.
    let mut res = alloc(LOG_D, 2, 2);
    assert_guard::<BE, _>("no keys", "blas_rotate", MissingKey, &mut res, |res| {
        module.blas_rotate(res, &ct, 1, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("no keys", "blas_sum", MissingKey, &mut res, |res| {
        module.blas_sum(res, &ct, &keys, &mut op_scratch.borrow())
    });

    // 2^(2 * log_d) above the slot count: no block fits.
    let log_d_wide = (params.n / 2).ilog2() as usize / 2 + 1;
    let wide = BlasCiphertext::from_blocks(ct.clone().into_blocks(), log_d_wide, 2, 2).unwrap();
    let mut res_wide = alloc(log_d_wide, 2, 2);
    assert_guard::<BE, _>("wide", "blas_rotate", InvalidPlan, &mut res_wide, |res| {
        module.blas_rotate(res, &wide, 1, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("wide", "blas_sum", InvalidPlan, &mut res_wide, |res| {
        module.blas_sum(res, &wide, &keys, &mut op_scratch.borrow())
    });

    let sum_keys = TestKeys::<BE> {
        atks: sum_galois_elements(module, &ct)
            .into_iter()
            .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
            .collect(),
        tsk: None,
    };
    // The sum's first rotation is by one matrix slot.
    let bytes = module.blas_rotate_tmp_bytes(&res, &ct, &params.atk_layout());
    assert_short_scratch::<BE, _>("blas_rotate", &mut res, bytes, |res, s| {
        module.blas_rotate(res, &ct, 1, &sum_keys, s)
    });
    let bytes = module.blas_sum_tmp_bytes(&res, &ct, &params.atk_layout());
    assert_short_scratch::<BE, _>("blas_sum", &mut res, bytes, |res, s| module.blas_sum(res, &ct, &sum_keys, s));

    // (2^62 + 1) x 4 wraps to the 4 blocks of `ct` on 64 bits.
    assert!(BlasCiphertext::from_blocks(ct.into_blocks(), LOG_D, usize::MAX / 4 + 2, 4).is_err());
}
