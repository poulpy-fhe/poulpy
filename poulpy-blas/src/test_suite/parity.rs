//! Paired backends on identical fixtures, compared bit for bit through the extension points.

use poulpy_ckks::{
    CoeffsMeta, SlotsKind,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, CKKSModuleInfos},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch},
        parity::{
            helpers::{Snapshot, assert_untagged, fixture_ciphertext, with_scratch},
            keys::{key_layout, prepared_automorphism_key, prepared_tensor_key},
            layout,
        },
    },
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{DiagonalArithmetic, GGLWELayout, GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, Module, Standard},
};

use super::helpers::{LOG_D, TestKeys, grid_snapshot};
use crate::{
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContext, BlasContextUnprepared, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand,
        BlasPtOperandUnprepared, sum_galois_elements,
    },
    oep::BlasImpl,
};

/// Bounds of a backend taking part in blas parity.
pub trait BlasParityBackend: TestContextBackend<Ring = Standard> + BlasImpl {}

impl<B: TestContextBackend<Ring = Standard> + BlasImpl> BlasParityBackend for B {}

/// Operand width: two diagonal levels and one product at `log_delta = base2k - 1` fit with room to spare.
fn width(params: CKKSTestParams) -> usize {
    6 * params.base2k + 5
}

fn atk_layout(params: CKKSTestParams) -> GGLWELayout {
    key_layout(params.n, params.base2k, width(params), 1, 1, 1)
}

/// A `rows x cols` grid of fixtures, block `t` filled from `seed + t`.
fn operand<B: Backend<ZnxWord = i64>>(
    params: CKKSTestParams,
    module: &Module<B>,
    rows: usize,
    cols: usize,
    seed: u8,
) -> BlasCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B>,
{
    let la = layout(params, 1, width(params), params.base2k - 1, 0, SlotsKind::Complex);
    let blocks = (0..rows * cols)
        .map(|t| fixture_ciphertext(module, &la, seed + t as u8))
        .collect();
    BlasCiphertext::from_blocks(blocks, LOG_D, rows, cols).unwrap()
}

fn fixture_keys<B>(params: CKKSTestParams, module: &Module<B>, elements: &[i64], tensor: bool) -> TestKeys<B>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B> + GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    let key = atk_layout(params);
    TestKeys {
        atks: elements
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, prepared_automorphism_key(module, &key, p, 40 + i as u8)))
            .collect(),
        tsk: tensor.then(|| prepared_tensor_key(module, &key, 39)),
    }
}

fn diagonal_meta(params: CKKSTestParams) -> CoeffsMeta {
    CoeffsMeta::from_delta_budget(params.base2k - 1, params.base2k)
}

fn context<B, F>(params: CKKSTestParams, module: &Module<B>) -> BlasContext<B>
where
    B: BlasParityBackend,
    Module<B>: TestContextModule<B> + CKKSEncodingOps<B, F> + CKKSLinearTransformationOps<B> + CnvPVecAlloc<B>,
    F: TestScalar + DiagonalArithmetic,
{
    let mut scratch = alloc_scratch(&params, module);
    BlasContextUnprepared::encode::<F>(
        module,
        LOG_D,
        params.base2k.into(),
        diagonal_meta(params),
        &mut scratch.borrow(),
    )
    .unwrap()
    .prepare(module, &mut scratch.borrow())
}

fn inter<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Vec<Snapshot>)>
where
    B: BlasParityBackend,
    Module<B>: TestContextModule<B> + GLWEMaskFill<B>,
{
    let a = operand(params, module, 2, 2, 61);
    let mut elements = sum_galois_elements(module, &a);
    elements.push(module.ckks_galois_element(3 << (2 * LOG_D)));
    let keys = fixture_keys(params, module, &elements, false);
    let key = atk_layout(params);
    let mut out = Vec::new();
    let mut res = operand(params, module, 2, 2, 70);
    with_scratch::<B, _>(B::blas_rotate_tmp_bytes_impl(module, &res, &a, &key), |s| {
        B::blas_rotate_impl(module, &mut res, &a, 3, &keys, s)
    })
    .unwrap();
    out.push(("blas_rotate", grid_snapshot::<B, _>(&res)));
    with_scratch::<B, _>(B::blas_sum_tmp_bytes_impl(module, &res, &a, &key), |s| {
        B::blas_sum_impl(module, &mut res, &a, &keys, s)
    })
    .unwrap();
    out.push(("blas_sum", grid_snapshot::<B, _>(&res)));
    out
}

fn intra<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Vec<Snapshot>)>
where
    B: BlasParityBackend,
    Module<B>: TestContextModule<B> + CKKSEncodingOps<B, F> + CKKSLinearTransformationOps<B> + CnvPVecAlloc<B> + GLWEMaskFill<B>,
    F: TestScalar + DiagonalArithmetic,
{
    let ctx = context::<B, F>(params, module);
    let keys = fixture_keys(params, module, &ctx.galois_elements(module.cyclotomic_order()), false);
    let key = atk_layout(params);
    let d = 1usize << LOG_D;
    let mut out = Vec::new();
    let a = operand(params, module, 2, 3, 80);
    let mut res = operand(params, module, 3, 2, 90);
    with_scratch::<B, _>(B::blas_transpose_tmp_bytes_impl(module, &res, &a, &key), |s| {
        B::blas_transpose_impl(module, &mut res, &a, &ctx, &keys, s)
    })
    .unwrap();
    out.push(("blas_transpose", grid_snapshot::<B, _>(&res)));
    // k' = 0 is a copy, which the functional tests cover. A one-row grid applies psi^k'.
    let a_row = operand(params, module, 1, 2, 110);
    let mut res = operand(params, module, 2, 3, 100);
    let mut res_row = operand(params, module, 1, 2, 112);
    for k in [1, d + 1] {
        with_scratch::<B, _>(B::blas_rotate_rows_tmp_bytes_impl(module, &res, &a, &key), |s| {
            B::blas_rotate_rows_impl(module, &mut res, &a, k, &ctx, &keys, s)
        })
        .unwrap();
        out.push(("blas_rotate_rows", grid_snapshot::<B, _>(&res)));
        with_scratch::<B, _>(B::blas_rotate_cols_tmp_bytes_impl(module, &res, &a, &key), |s| {
            B::blas_rotate_cols_impl(module, &mut res, &a, 2 * k, &ctx, &keys, s)
        })
        .unwrap();
        out.push(("blas_rotate_cols", grid_snapshot::<B, _>(&res)));
    }
    with_scratch::<B, _>(B::blas_rotate_rows_tmp_bytes_impl(module, &res_row, &a_row, &key), |s| {
        B::blas_rotate_rows_impl(module, &mut res_row, &a_row, 1, &ctx, &keys, s)
    })
    .unwrap();
    out.push(("blas_rotate_rows one row", grid_snapshot::<B, _>(&res_row)));
    out
}

fn matmul<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Vec<Snapshot>)>
where
    B: BlasParityBackend,
    Module<B>: TestContextModule<B> + CKKSEncodingOps<B, F> + CKKSLinearTransformationOps<B> + CnvPVecAlloc<B> + GLWEMaskFill<B>,
    F: TestScalar + DiagonalArithmetic,
{
    let order = module.cyclotomic_order();
    let ctx = context::<B, F>(params, module);
    let mut scratch = alloc_scratch(&params, module);
    // One public matrix for every matrix slot, then one per matrix slot (the parity ring holds two).
    let pts: Vec<BlasPtOperand<B>> = [1, 2]
        .into_iter()
        .map(|count| {
            BlasPtOperandUnprepared::encode::<F>(
                module,
                LOG_D,
                2,
                1,
                count,
                |m, r, c| F::from_f64(((m + r * 4 + c) * 3 % 5) as f64 - 2.0).unwrap(),
                params.base2k.into(),
                diagonal_meta(params),
                &mut scratch.borrow(),
            )
            .unwrap()
            .prepare(module, &mut scratch.borrow())
        })
        .collect();
    let mut elements = ctx.galois_elements(order);
    elements.extend(pts.iter().flat_map(|pt| pt.galois_elements(order)));
    elements.sort_unstable();
    elements.dedup();
    let keys = fixture_keys(params, module, &elements, true);
    let key = atk_layout(params);
    let (a, b) = (operand(params, module, 2, 2, 120), operand(params, module, 2, 1, 124));
    let mut out = Vec::new();
    let mut res = operand(params, module, 2, 1, 126);
    with_scratch::<B, _>(B::blas_matmul_tmp_bytes_impl(module, &res, &a, &b, &key, &key), |s| {
        B::blas_matmul_impl(module, &mut res, &a, &b, &ctx, &keys, s)
    })
    .unwrap();
    out.push(("blas_matmul", grid_snapshot::<B, _>(&res)));
    let mut left = BlasMatMulLeft::from_grid(operand(params, module, 2, 2, 130));
    with_scratch::<B, _>(
        B::blas_matmul_prepare_left_tmp_bytes_impl(module, left.as_grid(), &a, &key),
        |s| B::blas_matmul_prepare_left_impl(module, &mut left, &a, &ctx, &keys, s),
    )
    .unwrap();
    out.push(("blas_matmul_prepare_left", grid_snapshot::<B, _>(left.as_grid())));
    let mut right = BlasMatMulRight::from_grid(operand(params, module, 2, 1, 134));
    with_scratch::<B, _>(
        B::blas_matmul_prepare_right_tmp_bytes_impl(module, right.as_grid(), &b, &key),
        |s| B::blas_matmul_prepare_right_impl(module, &mut right, &b, &ctx, &keys, s),
    )
    .unwrap();
    out.push(("blas_matmul_prepare_right", grid_snapshot::<B, _>(right.as_grid())));
    let mut res = operand(params, module, 2, 1, 136);
    let bytes = B::blas_matmul_prepared_tmp_bytes_impl(module, &res, left.as_grid(), right.as_grid(), &key, &key);
    with_scratch::<B, _>(bytes, |s| {
        B::blas_matmul_prepared_impl(module, &mut res, &left, &right, &ctx, &keys, s)
    })
    .unwrap();
    out.push(("blas_matmul_prepared", grid_snapshot::<B, _>(&res)));
    for pt in &pts {
        let mut res = operand(params, module, 2, 1, 138);
        with_scratch::<B, _>(B::blas_matmul_pt_tmp_bytes_impl(module, &res, &a, &key), |s| {
            B::blas_matmul_pt_impl(module, &mut res, &a, pt, &keys, s)
        })
        .unwrap();
        out.push(("blas_matmul_pt", grid_snapshot::<B, _>(&res)));
    }
    out
}

fn compare(label: &str, want: Vec<(&'static str, Vec<Snapshot>)>, got: Vec<(&'static str, Vec<Snapshot>)>) {
    assert_untagged(label, want.iter().flat_map(|(_, s)| s));
    assert_eq!(want, got, "{label}");
}

/// Matrix-slot rotation and sum agree bit for bit.
pub fn test_blas_inter_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>: TestContextModule<BR> + GLWEMaskFill<BR>,
    Module<BT>: TestContextModule<BT> + GLWEMaskFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    compare("inter", inter(params, r), inter(params, t));
}

/// Transpose and row and column rotations agree bit for bit.
pub fn test_blas_intra_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>:
        TestContextModule<BR> + CKKSEncodingOps<BR, F> + CKKSLinearTransformationOps<BR> + CnvPVecAlloc<BR> + GLWEMaskFill<BR>,
    Module<BT>:
        TestContextModule<BT> + CKKSEncodingOps<BT, F> + CKKSLinearTransformationOps<BT> + CnvPVecAlloc<BT> + GLWEMaskFill<BT>,
    F: TestScalar + DiagonalArithmetic,
{
    compare("intra", intra::<BR, F>(params, r), intra::<BT, F>(params, t));
}

/// The products, the operand preparations and the prepared product agree bit for bit.
pub fn test_blas_matmul_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: BlasParityBackend,
    BT: BlasParityBackend,
    Module<BR>:
        TestContextModule<BR> + CKKSEncodingOps<BR, F> + CKKSLinearTransformationOps<BR> + CnvPVecAlloc<BR> + GLWEMaskFill<BR>,
    Module<BT>:
        TestContextModule<BT> + CKKSEncodingOps<BT, F> + CKKSLinearTransformationOps<BT> + CnvPVecAlloc<BT> + GLWEMaskFill<BT>,
    F: TestScalar + DiagonalArithmetic,
{
    compare("matmul", matmul::<BR, F>(params, r), matmul::<BT, F>(params, t));
}
