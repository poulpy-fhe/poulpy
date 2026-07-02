//! VMP bandwidth probe at bootstrapping ring dimensions.
//!
//! Measures `vmp_apply_dft_to_dft` throughput as GB/s of prepared-matrix bytes
//! streamed, at `log n ∈ {14, 15, 16}` and bootstrap-shaped operands, against a
//! STREAM-triad DRAM reference in the same binary, single- and multi-threaded.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run -p poulpy-bench --release \
//!   --features enable-ifma --example vmp_bandwidth
//! ```

use std::{sync::Barrier, time::Instant};

use poulpy_bench::{random_backend_vec_znx_dft, random_backend_vmp_pmat, vec_znx_dft_backend_ref};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes},
    layouts::{Backend, Module, ScratchOwned, VecZnxDft, VecZnxDftToBackendMut, VmpPMatToBackendRef},
    source::Source,
};

const SECS: f64 = 1.5;

/// STREAM triad `c = a + s·b` over per-thread 128 MiB arrays; aggregate GB/s
/// (counting 3 × 8 bytes per element, the STREAM convention).
fn triad(threads: usize) -> f64 {
    const LEN: usize = 1 << 24;
    let barrier = Barrier::new(threads);
    let rates: Vec<f64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = &barrier;
                s.spawn(move || {
                    let a = vec![1.0f64; LEN];
                    let b = vec![2.0f64; LEN];
                    let mut c = vec![0.0f64; LEN];
                    barrier.wait();
                    let start = Instant::now();
                    let mut passes = 0usize;
                    while start.elapsed().as_secs_f64() < SECS {
                        for i in 0..LEN {
                            c[i] = a[i] + 2.5 * b[i];
                        }
                        std::hint::black_box(&mut c);
                        passes += 1;
                    }
                    passes as f64 * (3 * 8 * LEN) as f64 / start.elapsed().as_secs_f64()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    rates.iter().sum::<f64>() / 1e9
}

/// `threads` independent `vmp_apply_dft_to_dft` streams (own pmat each);
/// returns (aggregate GB/s of pmat bytes, ms per op).
fn vmp_probe<B: Backend>(log_n: usize, rows: usize, size: usize, threads: usize) -> (f64, f64)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let n = 1usize << log_n;
    let (cols_in, cols_out) = (1usize, 2usize);
    let pmat_bytes = B::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size);
    let barrier = Barrier::new(threads);
    let results: Vec<(f64, f64)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                s.spawn(move || {
                    let module = Module::<B>::new(n as u64);
                    let mut source = Source::new([t as u8 + 1; 32]);
                    let pmat = random_backend_vmp_pmat::<B>(n, rows, cols_in, cols_out, size, &mut source);
                    let a = random_backend_vec_znx_dft::<B>(n, cols_in, rows, &mut source);
                    let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols_out, size);
                    let mut scratch: ScratchOwned<B> =
                        ScratchOwned::alloc(module.vmp_apply_dft_to_dft_tmp_bytes(size, rows, rows, cols_in, cols_out, size));
                    barrier.wait();
                    let start = Instant::now();
                    let mut iters = 0usize;
                    while start.elapsed().as_secs_f64() < SECS {
                        let a_ref = vec_znx_dft_backend_ref::<B>(&a);
                        module.vmp_apply_dft_to_dft(
                            &mut res.to_backend_mut(),
                            &a_ref,
                            &pmat.to_backend_ref(),
                            0,
                            &mut scratch.borrow(),
                        );
                        std::hint::black_box(&mut res);
                        iters += 1;
                    }
                    let elapsed = start.elapsed().as_secs_f64();
                    (
                        iters as f64 * pmat_bytes as f64 / elapsed,
                        elapsed / iters as f64 * 1e3,
                    )
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let gbs = results.iter().map(|r| r.0).sum::<f64>() / 1e9;
    let ms = results.iter().map(|r| r.1).sum::<f64>() / threads as f64;
    (gbs, ms)
}

fn run_backend<B: Backend>(label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    // (rows, size): (4, 27) = one dsize=7 digit-group of a bootstrap key;
    // (18, 27) = a full dsize=1 key stream (memory-heavy: thread sweep capped).
    for &(rows, size) in &[(4usize, 27usize), (18, 27)] {
        for &log_n in &[14usize, 15, 16] {
            let threads: &[usize] = if log_n == 16 {
                if rows > 4 { &[1, 2, 4] } else { &[1, 2, 4, 8, 16] }
            } else {
                &[1]
            };
            for &t in threads {
                let n = 1usize << log_n;
                let mib = B::bytes_of_vmp_pmat(n, rows, 1, 2, size) as f64 / (1 << 20) as f64;
                let (gbs, ms) = vmp_probe::<B>(log_n, rows, size, t);
                println!("{label:12} log_n={log_n} rows={rows:2} size={size} T={t:2}: {gbs:6.1} GB/s  {ms:7.3} ms/op  (pmat {mib:.0} MiB)");
            }
        }
    }
}

fn main() {
    for &t in &[1usize, 2, 4, 8, 16] {
        println!("triad        T={t:2}: {:6.1} GB/s", triad(t));
    }

    #[cfg(all(feature = "enable-ifma", target_arch = "x86_64"))]
    run_backend::<poulpy_cpu_avx512::NTT126Ifma>("ntt126-ifma");
    #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
    run_backend::<poulpy_cpu_avx512::NTT120Avx512>("ntt120-avx512");
    #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
    run_backend::<poulpy_cpu_avx512::FFT64Avx512>("fft64-avx512");
}
