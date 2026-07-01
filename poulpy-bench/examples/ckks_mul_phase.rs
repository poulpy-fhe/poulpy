//! Single-op harness for clean cache/profile measurement of the two CKKS-mul
//! phases (the Criterion `glwe_tensor` bench is fine for runtime but its other
//! benches' setups pollute whole-process `perf` counters).
//!
//! Usage: ckks_mul_phase <tensor|relin> <log_n> <iters>
//! Measure: taskset -c 4 perf stat -d -d -- target/release/examples/ckks_mul_phase relin 15 200
//! Build:   RUSTFLAGS="-C target-cpu=native" cargo build --release --example ckks_mul_phase \
//!            -p poulpy-bench --features "core-bench enable-avx512f enable-ifma"
use poulpy_core::{
    GLWETensoring,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        Rank, TorusPrecision,
    },
};
use poulpy_cpu_avx512::NTT3x42Ifma;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let op = args.get(1).map(|s| s.as_str()).unwrap_or("tensor");
    let log_n: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let iters: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let (n, base2k, k, dsize, rank) = (1u32 << log_n, 52u32, 728u32, 1u32, 1u32);

    let module = Module::<NTT3x42Ifma>::new(n as u64);
    let glwe = GLWELayout {
        n: Degree(n),
        base2k: Base2K(base2k),
        k: TorusPrecision(k),
        rank: Rank(rank),
    };
    let a = module.glwe_alloc_from_infos(&glwe);
    let b = module.glwe_alloc_from_infos(&glwe);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe);

    let t0 = Instant::now();
    if op == "relin" {
        let tsk_layout = GLWETensorKeyLayout {
            n: Degree(n),
            base2k: Base2K(base2k),
            k: TorusPrecision(k + dsize * base2k),
            rank: Rank(rank),
            dsize: Dsize(dsize),
            dnum: Dnum(k.div_ceil(dsize * base2k)),
        };
        let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);
        let mut res = module.glwe_alloc_from_infos(&glwe);
        let tsk_size = tensor.max_size() + dsize as usize;
        let mut scratch = ScratchOwned::<NTT3x42Ifma>::alloc(module.glwe_tensor_relinearize_tmp_bytes(&res, &tensor, &tsk));
        for _ in 0..iters {
            module.glwe_tensor_relinearize(&mut res, &tensor, &tsk, tsk_size, &mut scratch.borrow());
            black_box(&res);
        }
    } else {
        let mut scratch = ScratchOwned::<NTT3x42Ifma>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));
        for _ in 0..iters {
            module.glwe_tensor_apply(0, &mut tensor, &a, &b, &mut scratch.borrow());
            black_box(&tensor);
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("{op} logN={log_n} iters={iters} -> {ms:.3} ms/op");
    black_box(&tensor);
}
