//! Every operation on the conjugate-invariant ring, whose slots are real.

use std::collections::BTreeSet;

use poulpy_ckks::{
    CKKSInfos, CKKSModuleInfos, GLWEPlaintextInfos, SetCKKSInfos, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSLinearTransformationOps},
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
        Grid, HINTS, MIXES, TestKeys, alloc_grid, blas_context, copied_layout, mapped_layout, product_layout, rotation_keys,
    },
    matmul::public_entry,
};
use crate::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasGridInfos, BlasLayout, BlasPtOperand, BlasPtOperandUnprepared, layout_of,
        sum_galois_elements,
    },
    oep::BlasImpl,
    reference::Left,
};

/// Every operation once on the conjugate-invariant ring, on real matrices of `4 x 4` blocks encoded with the module's
/// own encoder (the standard decoding helper reads complex slots): `blas_rotate`, `blas_sum`, `blas_transpose`,
/// `blas_rotate_rows` and `blas_rotate_cols` across blocks, both preparations under either hint then
/// `blas_matmul_prepared` of every mix,
/// `blas_matmul`, `blas_matmul_pt` and `blas_matmul_pt_left`, each on exactly the scratch its query reports.
/// Every output decrypts to its cleartext model at the metadata [`copied_layout`], [`mapped_layout`] or
/// [`product_layout`] gives, and its slots, like the prepared operands', are `Real`. On fewer than 16 slots, which hold
/// no `4 x 4` block, it is skipped with a message.
pub fn test_blas_conjugate_invariant<BE>(
    mut params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<ConjugateInvariant>>,
) where
    BE: TestContextBackend<Ring = ConjugateInvariant> + BlasImpl,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, f64> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
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
    let ctx = blas_context::<BE, f64>(&params, module, log_d, &mut scratch.borrow());
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
    // products, and the tensor key.
    let (order, plan) = (module.cyclotomic_order(), ctx.plan());
    let elements: BTreeSet<i64> = plan
        .galois_elements(order)
        .into_iter()
        .chain(plan.rotate_galois_elements(order, 1))
        .chain(sum_galois_elements(module, &ct_a))
        .chain(plan.matmul_pt_galois_elements(order))
        .chain(plan.matmul_pt_left_galois_elements(order))
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
        module.blas_rotate(&mut res, &ct_a, 1, &keys, s)
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
        module.blas_rotate_rows(&mut res, &ct_a, 1, &ctx, &keys, s)
    })
    .unwrap();
    check("rotate_rows", &res, &a.rotate_rows(1), layout, &mut scratch.borrow());

    let mut res = alloc(2, 3);
    with_scratch::<BE, _>(module.blas_rotate_cols_tmp_bytes(&res, &ct_a, &ctx, &atk), |s| {
        module.blas_rotate_cols(&mut res, &ct_a, 1, &ctx, &keys, s)
    })
    .unwrap();
    check("rotate_cols", &res, &a.rotate_cols(1), layout, &mut scratch.borrow());

    // Both operands under either hint, hot first, then the prepared product of every mix.
    let lefts = HINTS.map(|hint| {
        let left = with_scratch::<BE, _>(module.blas_matmul_prepare_left_tmp_bytes(&ct_a, &ctx, &atk, hint), |s| {
            module.blas_matmul_prepare_left(&ct_a, &ctx, &keys, hint, s)
        })
        .unwrap();
        assert_eq!(left.block_layout().slots(), SlotsKind::Real, "prepare_left {hint:?}: slots");
        left
    });
    let rights = HINTS.map(|hint| {
        let right = with_scratch::<BE, _>(module.blas_matmul_prepare_right_tmp_bytes(&ct_b, &ctx, &atk, hint), |s| {
            module.blas_matmul_prepare_right(&ct_b, &ctx, &keys, hint, s)
        })
        .unwrap();
        assert_eq!(right.block_layout().slots(), SlotsKind::Real, "prepare_right {hint:?}: slots");
        right
    });
    let (ab, layout) = (a.matmul(&b), product_layout(&ct_a, &ct_b, 2, 2, k, p));
    let side = |hint| usize::from(hint == PrepareHint::Reuse);
    for (hl, hr) in MIXES {
        let (left, right) = (&lefts[side(hl)], &rights[side(hr)]);
        let mut res = alloc(2, 2);
        with_scratch::<BE, _>(
            module.blas_matmul_prepared_tmp_bytes(&res, left, right, &ctx, &tsk, &atk),
            |s| module.blas_matmul_prepared(&mut res, left, right, &ctx, &keys, &keys, s),
        )
        .unwrap();
        check(
            &format!("matmul_prepared {hl:?} x {hr:?}"),
            &res,
            &ab,
            layout,
            &mut scratch.borrow(),
        );
    }

    let mut res = alloc(2, 2);
    with_scratch::<BE, _>(module.blas_matmul_tmp_bytes(&res, &ct_a, &ct_b, &ctx, &tsk, &atk), |s| {
        module.blas_matmul(&mut res, &ct_a, &ct_b, &ctx, &keys, &keys, s)
    })
    .unwrap();
    check("matmul", &res, &ab, layout, &mut scratch.borrow());

    // Products by two public matrices at scale `p`, matrix slot `s` reading matrix `s mod 2`; one on a single matrix slot.
    let count = lanes.min(2);
    let pt = BlasPtOperandUnprepared::encode::<f64>(
        module,
        log_d,
        3,
        2,
        count,
        public_entry,
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    )
    .unwrap()
    .prepare(module, &mut scratch.borrow());
    let mut res = alloc(2, 2);
    with_scratch::<BE, _>(module.blas_matmul_pt_tmp_bytes(&res, &ct_a, &pt, &atk), |s| {
        module.blas_matmul_pt(&mut res, &ct_a, &pt, &keys, s)
    })
    .unwrap();
    let (want, layout) = (a.matmul_public(2, count, public_entry), mapped_layout(&ct_a, 2, 2, k, p));
    check("matmul_pt", &res, &want, layout, &mut scratch.borrow());

    let pt_left: BlasPtOperand<BE, Left> = BlasPtOperandUnprepared::encode_left::<f64>(
        module,
        log_d,
        2,
        3,
        count,
        public_entry,
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    )
    .unwrap()
    .prepare(module, &mut scratch.borrow());
    let mut res = alloc(2, 2);
    with_scratch::<BE, _>(module.blas_matmul_pt_left_tmp_bytes(&res, &pt_left, &ct_b, &atk), |s| {
        module.blas_matmul_pt_left(&mut res, &pt_left, &ct_b, &keys, s)
    })
    .unwrap();
    let (want, layout) = (b.public_matmul(2, count, public_entry), mapped_layout(&ct_b, 2, 2, k, p));
    check("matmul_pt_left", &res, &want, layout, &mut scratch.borrow());
}

/// Encrypts every block of the real grid `grid` at width `params.k`, encoded at `params.prec()` by the module, recording
/// the dense batch.
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
    let slots = module.ckks_max_slots();
    let zeros = vec![0.0; slots];
    let blocks = grid
        .blocks(slots)
        .iter()
        .map(|(re, _)| {
            let mut pt = module.ckks_pt_vec_alloc(params.base2k.into(), params.prec().k());
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
