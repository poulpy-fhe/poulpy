//! The reference `blas_matmul` streams its larger side: a counting global allocator, in this binary of one test, shows
//! that the heap it holds grows neither with the block columns of its right factor nor with the block rows of its left.
use poulpy_bench::heap::{self, Counting};
use poulpy_blas::{
    api::BlasMatMulOps,
    layouts::{BlasPlan, blas_matmul_left_term_layout},
    test_suite::helpers::{Grid, TestKeys, alloc_grid, blas_context, encrypt_grid, rotation_keys},
};
use poulpy_ckks::test_suite::{
    BASE19_PARAMS_F64,
    helpers::{alloc_scratch, gen_sk_with_raw, gen_tsk},
    reference_encoder::ReferenceEncoder,
};
use poulpy_core::layouts::{GLWEInfos, LWEInfos};
use poulpy_cpu_portable::{FFT64Portable, ckks_encoding::EncodingFFTTable};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, Module, ScratchOwned},
};

#[global_allocator]
static HEAP: Counting = Counting;

/// `blas_matmul` of an `m x 2` grid by a `2 x n` grid of `4 x 4` blocks: the reference holds the terms of the smaller
/// side and makes those of the other one unit at a time, a block column of `B` or a block row of `A`, reading both
/// grids in place, so the heap the call holds above its entry level grows by less than one term from `1 x 1` blocks of
/// `res` to `1 x 4`, which holds `A`, and to `4 x 1`, which holds `B`. Holding every term of the streamed side, or
/// copying the grids, grows it by three of its units.
#[test]
fn blas_matmul_heap_does_not_grow_with_the_streamed_side() {
    assert!(heap::counting(), "the counting allocator is not the global allocator");
    let (params, log_d) = (BASE19_PARAMS_F64, 2);
    let module = Module::<FFT64Portable>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<EncodingFFTTable<f64>>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);
    let elements = BlasPlan::new(log_d).matmul_galois_elements(module.cyclotomic_order());
    let keys = TestKeys::<FFT64Portable> {
        atks: rotation_keys(&params, &module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: Some(gen_tsk(&params, &module, &sk_raw, &mut scratch.borrow())),
    };
    let ctx = blas_context::<FFT64Portable, f64>(&params, &module, log_d);
    let mut encrypt = |rows: usize, cols: usize, seed: u64| {
        let grid = Grid::model(log_d, rows, cols, slots >> (2 * log_d), seed);
        encrypt_grid::<FFT64Portable, f64, EncodingFFTTable<f64>>(
            &params,
            &module,
            &host_module,
            &encoder,
            &sk,
            params.k,
            0,
            &grid,
            &mut scratch.borrow(),
        )
    };
    // One term, in coefficients; FFT64 holds its transform-domain form in as many bytes.
    let term = blas_matmul_left_term_layout(&encrypt(1, 2, 7), &ctx).block;
    let term_bytes = params.n * (term.rank().as_usize() + 1) * term.size() * size_of::<i64>();
    let mut peak = |m: usize, n: usize| {
        let (a, b) = (encrypt(m, 2, 7), encrypt(2, n, 8));
        let mut res = alloc_grid(&params, &module, log_d, m, n, params.k);
        let bytes = module.blas_matmul_tmp_bytes(&res, &a, &b, &ctx, &params.tsk_layout(), &params.atk_layout());
        let mut op_scratch = ScratchOwned::<FFT64Portable>::alloc(bytes);
        let entry = heap::reset_peak();
        module
            .blas_matmul(&mut res, &a, &b, &ctx, &keys, &keys, &mut op_scratch.borrow())
            .unwrap();
        heap::peak() - entry
    };
    let one = peak(1, 1);
    for (m, n) in [(1, 4), (4, 1)] {
        let held = peak(m, n);
        assert!(
            held <= one + term_bytes,
            "blas_matmul held {held} heap bytes at {m} x {n} against {one} at 1 x 1: more than one term ({term_bytes} bytes)"
        );
    }
}
