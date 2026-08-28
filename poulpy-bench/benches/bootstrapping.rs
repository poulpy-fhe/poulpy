use poulpy_bench::schemes::bootstrapping;

const DEFAULT_THREADS: usize = 16;

fn main() {
    let repeats = std::env::var("POULPY_BTS_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    assert!(repeats > 0, "POULPY_BTS_REPEATS must be nonzero");

    match std::env::args().nth(1).as_deref() {
        Some("single-thread") => bootstrapping::run::<poulpy_cpu_avx512::NTT3x42Ifma, poulpy_cpu_avx512::FFT64Avx512ReimTable>(
            "single_thread",
            1,
            repeats,
        ),
        Some("multi-thread") => {
            let threads = std::env::var("POULPY_BTS_THREADS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_THREADS);
            assert!(threads > 0, "POULPY_BTS_THREADS must be nonzero");
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .expect("initialize the Rayon pool");
            let name = format!("multi_thread_{threads}");
            bootstrapping::run::<poulpy_cpu_avx512::NTT3x42IfmaRayon, poulpy_cpu_avx512::FFT64Avx512ReimTable>(
                &name, threads, repeats,
            );
        }
        _ => panic!("usage: bootstrapping single-thread|multi-thread"),
    }
}
