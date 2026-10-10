//! The metadata operations leave on their results: the batch's sparsity, the noise tag and the slot kind.

use poulpy_ckks::{
    GLWEPlaintextInfos, SlotsKind,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_atk, gen_sk_with_raw},
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

use super::helpers::{D, Grid, LOG_D, TestKeys, assert_grid_decrypt, context_and_keys, encrypt_grid, relabel};
use crate::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight, BlasModuleAlloc, BlasPtOperandUnprepared,
        sum_galois_elements,
    },
    oep::BlasImpl,
};

/// Asserts every block of `g` has sparsity `log_sparsity` and no noise tag.
fn assert_meta<DG: Data, W: ZnxWord, R: Ring>(label: &str, g: &BlasCiphertext<DG, W, R>, log_sparsity: usize) {
    for block in g.blocks() {
        assert_eq!(block.log_sparsity(), log_sparsity, "{label}: log_sparsity");
        assert!(block.noise().is_none(), "{label}: kept a noise tag");
    }
}

/// Every per-matrix map gives its result the sparsity of its input, whatever `res` held: transpose, row and column
/// rotations across blocks, within one block and by whole blocks, the operand preparations and the product by public
/// matrices, from a half-full batch into a fresh grid and from a full batch into a grid labelled half full. Summing a
/// transposed half-full batch adds each of its matrices once. Copies, the shifts by whole blocks and the sum of one
/// matrix, leave no noise tag, and transpose keeps real slots real.
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
    // repeated over the slots.
    let encrypt = |grid: &Grid, log_sparsity: usize, scratch: &mut ScratchArena<'_, BE>| {
        let ct = encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, params.k, grid, scratch);
        relabel(&ct, |meta| meta.log_sparsity = log_sparsity)
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
    for p in pt
        .galois_elements(order)
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
        .max(module.blas_sum_tmp_bytes(&full_row, &full_row, &atk));
    let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
    let alloc = |rows: usize, cols: usize, log_sparsity: usize| {
        let grid = module.blas_ciphertext_alloc(LOG_D, rows, cols, params.base2k.into(), params.k.into());
        relabel(&grid, |meta| meta.log_sparsity = log_sparsity)
    };

    // A shift by 1 crosses blocks along the axis two blocks long and stays in the block along the other; a shift by `D`
    // copies whole blocks.
    for (row, column, want, stale) in [(&half_row, &half_column, 1, 0), (&full_row, &full_column, 0, 1)] {
        let label = |op: &str| format!("{op} from log_sparsity {want} into {stale}");
        let mut res = alloc(2, 1, stale);
        module
            .blas_transpose(&mut res, row, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("transpose"), &res, want);
        for k in [1, D] {
            for a in [row, column] {
                let shape = format!("{}x{} k={k}", a.rows(), a.cols());
                let mut res = alloc(a.rows(), a.cols(), stale);
                module
                    .blas_rotate_rows(&mut res, a, k, &ctx, &keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_meta(&label(&format!("rotate_rows {shape}")), &res, want);
                let mut res = alloc(a.rows(), a.cols(), stale);
                module
                    .blas_rotate_cols(&mut res, a, k, &ctx, &keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_meta(&label(&format!("rotate_cols {shape}")), &res, want);
            }
        }
        let mut left = BlasMatMulLeft::from_grid(alloc(1, 2, stale));
        module
            .blas_matmul_prepare_left(&mut left, row, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("prepare_left"), left.as_grid(), want);
        let mut right = BlasMatMulRight::from_grid(alloc(2, 1, stale));
        module
            .blas_matmul_prepare_right(&mut right, column, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("prepare_right"), right.as_grid(), want);
        let mut res = alloc(1, 1, stale);
        module
            .blas_matmul_pt(&mut res, row, &pt, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_meta(&label("matmul_pt"), &res, want);
    }

    let mut transposed = alloc(2, 1, 0);
    module
        .blas_transpose(&mut transposed, &half_row, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    let mut res = alloc(2, 1, 0);
    module
        .blas_sum(&mut res, &transposed, &keys, &mut op_scratch.borrow())
        .unwrap();
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
    assert_meta("sum of one matrix", &res, one_log_sparsity);

    let mut res = alloc(2, 1, 0);
    module
        .blas_transpose(&mut res, &real, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    assert!(
        res.blocks().iter().all(|block| block.slots() == SlotsKind::Real),
        "the transpose of a real batch has complex slots"
    );
}
