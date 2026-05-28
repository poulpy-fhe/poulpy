//! Sanity + perf check for `poulpy-cpu-arm` vs `poulpy-cpu-ref`.
//!
//! Usage (native AArch64):
//!
//!   cargo run --release --example bench_neon_vs_ref --features enable-neon
//!
//! Writes results to `bench_neon_vs_ref.txt` (override via `POULPY_ARM_BENCH_OUT`).
//! Each kernel runs once for correctness, then is timed and printed in a speedup table.

#[cfg(not(all(feature = "enable-neon", target_arch = "aarch64")))]
fn main() {
    eprintln!("Skipping: bench_neon_vs_ref requires --features enable-neon on target_arch = \"aarch64\".");
}

#[cfg(all(feature = "enable-neon", target_arch = "aarch64"))]
fn main() {
    neon::run();
}

#[cfg(all(feature = "enable-neon", target_arch = "aarch64"))]
mod neon {
    use std::fs::File;
    use std::hint::black_box;
    use std::io::Write;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use poulpy_cpu_arm::{FFT64Neon, NTT120Neon};
    use poulpy_cpu_ref::{
        FFT64Ref, NTT120Ref,
        reference::{
            fft64::reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
            ntt120::{
                I128BigOps, I128NormalizeOps, NttDFTExecute, NttFromZnx64,
                ntt::{NttTable, NttTableInv},
                primes::{PrimeSet, Primes30},
            },
            znx::{ZnxAdd, ZnxAutomorphism, ZnxNormalizeMiddleStepAssign},
        },
    };

    const SEED: [u8; 32] = *b"poulpy-cpu-arm-bench-seed--01234";
    const SIZES: &[usize] = &[1024, 4096, 16384];
    const DEFAULT_OUT: &str = "bench_neon_vs_ref.txt";

    /// Stdout + file tee. Drop on `run()` exit flushes the file.
    struct Reporter {
        file: Option<File>,
    }

    impl Reporter {
        fn new(path: &str) -> Self {
            match File::create(path) {
                Ok(f) => {
                    eprintln!("Writing results to {path}");
                    Self { file: Some(f) }
                }
                Err(e) => {
                    eprintln!("WARNING: could not open {path}: {e}; stdout-only output.");
                    Self { file: None }
                }
            }
        }

        fn line(&mut self, s: &str) {
            println!("{s}");
            if let Some(f) = self.file.as_mut() {
                let _ = writeln!(f, "{s}");
            }
        }
    }

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::from_seed(SEED)
    }

    fn rand_i64s(n: usize) -> Vec<i64> {
        let mut r = rng();
        (0..n).map(|_| r.random::<i64>() >> 3).collect()
    }

    fn rand_i128s(n: usize) -> Vec<i128> {
        let mut r = rng();
        (0..n).map(|_| r.random::<i128>() >> 4).collect()
    }

    fn rand_f64s(n: usize) -> Vec<f64> {
        let mut r = rng();
        (0..n).map(|_| r.random::<f64>() * 2e6 - 1e6).collect()
    }

    fn rand_q120b(n: usize) -> Vec<u64> {
        let mut r = rng();
        let mut out = vec![0u64; 4 * n];
        for chunk in out.chunks_exact_mut(4) {
            for (lane, slot) in chunk.iter_mut().enumerate() {
                *slot = r.random::<u64>() % Primes30::Q[lane] as u64;
            }
        }
        out
    }

    fn iters_for(n: usize) -> u64 {
        match n {
            0..=1024 => 5_000,
            1025..=4096 => 1_000,
            4097..=16_384 => 200,
            _ => 50,
        }
    }

    fn time<F: FnMut()>(iters: u64, mut f: F) -> u128 {
        f(); // warm up
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let total = t.elapsed().as_nanos();
        total / iters as u128
    }

    fn row(rep: &mut Reporter, name: &str, n: usize, ref_ns: u128, neon_ns: u128, speedups: &mut Vec<f64>) {
        let s = ref_ns as f64 / neon_ns.max(1) as f64;
        speedups.push(s);
        rep.line(&format!(
            "  {:36} n={:>6}   ref={:>9} ns   neon={:>9} ns   {:5.2}×",
            name, n, ref_ns, neon_ns, s
        ));
    }

    fn close_enough(a: &[f64], b: &[f64], abs_tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= abs_tol)
    }

    // ─── Znx (i64) ─────────────────────────────────────────────────────────

    fn bench_znx_add(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let a = rand_i64s(n);
        let b = rand_i64s(n);

        let mut neon = vec![0i64; n];
        let mut refr = vec![0i64; n];
        <FFT64Neon as ZnxAdd>::znx_add(&mut neon, &a, &b);
        <FFT64Ref as ZnxAdd>::znx_add(&mut refr, &a, &b);
        assert_eq!(neon, refr, "znx_add (n={n})");

        let iters = iters_for(n);
        let mut r = vec![0i64; n];
        let ref_ns = time(iters, || {
            <FFT64Ref as ZnxAdd>::znx_add(&mut r, &a, &b);
            black_box(&r);
        });
        let neon_ns = time(iters, || {
            <FFT64Neon as ZnxAdd>::znx_add(&mut r, &a, &b);
            black_box(&r);
        });
        row(rep, "znx_add", n, ref_ns, neon_ns, sp);
    }

    fn bench_znx_normalize(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let base2k = 18usize;
        let x_init = rand_i64s(n);
        let c_init = rand_i64s(n);

        let mut x_neon = x_init.clone();
        let mut c_neon = c_init.clone();
        let mut x_ref = x_init.clone();
        let mut c_ref = c_init.clone();
        <FFT64Neon as ZnxNormalizeMiddleStepAssign>::znx_normalize_middle_step_assign(base2k, 0, &mut x_neon, &mut c_neon);
        <FFT64Ref as ZnxNormalizeMiddleStepAssign>::znx_normalize_middle_step_assign(base2k, 0, &mut x_ref, &mut c_ref);
        assert_eq!(x_neon, x_ref, "znx_normalize_middle_step_assign x (n={n})");
        assert_eq!(c_neon, c_ref, "znx_normalize_middle_step_assign c (n={n})");

        let iters = iters_for(n);
        let mut x = x_init.clone();
        let mut c = c_init.clone();
        let ref_ns = time(iters, || {
            x.copy_from_slice(&x_init);
            c.copy_from_slice(&c_init);
            <FFT64Ref as ZnxNormalizeMiddleStepAssign>::znx_normalize_middle_step_assign(base2k, 0, &mut x, &mut c);
            black_box((&x, &c));
        });
        let neon_ns = time(iters, || {
            x.copy_from_slice(&x_init);
            c.copy_from_slice(&c_init);
            <FFT64Neon as ZnxNormalizeMiddleStepAssign>::znx_normalize_middle_step_assign(base2k, 0, &mut x, &mut c);
            black_box((&x, &c));
        });
        row(rep, "znx_normalize_middle_step_assign", n, ref_ns, neon_ns, sp);
    }

    fn bench_znx_automorphism(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let a = rand_i64s(n);
        let p: i64 = 5;

        let mut neon = vec![0i64; n];
        let mut refr = vec![0i64; n];
        <FFT64Neon as ZnxAutomorphism>::znx_automorphism(p, &mut neon, &a);
        <FFT64Ref as ZnxAutomorphism>::znx_automorphism(p, &mut refr, &a);
        assert_eq!(neon, refr, "znx_automorphism (n={n})");

        let iters = iters_for(n);
        let mut r = vec![0i64; n];
        let ref_ns = time(iters, || {
            <FFT64Ref as ZnxAutomorphism>::znx_automorphism(p, &mut r, &a);
            black_box(&r);
        });
        let neon_ns = time(iters, || {
            <FFT64Neon as ZnxAutomorphism>::znx_automorphism(p, &mut r, &a);
            black_box(&r);
        });
        row(rep, "znx_automorphism", n, ref_ns, neon_ns, sp);
    }

    // ─── Reim (f64) ────────────────────────────────────────────────────────

    fn bench_reim_add(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let a = rand_f64s(n);
        let b = rand_f64s(n);

        let mut neon = vec![0f64; n];
        let mut refr = vec![0f64; n];
        <FFT64Neon as ReimArith>::reim_add(&mut neon, &a, &b);
        <FFT64Ref as ReimArith>::reim_add(&mut refr, &a, &b);
        assert!(close_enough(&neon, &refr, 0.0), "reim_add (n={n})");

        let iters = iters_for(n);
        let mut r = vec![0f64; n];
        let ref_ns = time(iters, || {
            <FFT64Ref as ReimArith>::reim_add(&mut r, &a, &b);
            black_box(&r);
        });
        let neon_ns = time(iters, || {
            <FFT64Neon as ReimArith>::reim_add(&mut r, &a, &b);
            black_box(&r);
        });
        row(rep, "reim_add", n, ref_ns, neon_ns, sp);
    }

    fn bench_reim_mul(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let a = rand_f64s(n);
        let b = rand_f64s(n);

        let mut neon = vec![0f64; n];
        let mut refr = vec![0f64; n];
        <FFT64Neon as ReimArith>::reim_mul(&mut neon, &a, &b);
        <FFT64Ref as ReimArith>::reim_mul(&mut refr, &a, &b);
        assert!(close_enough(&neon, &refr, 1e-3), "reim_mul (n={n})");

        let iters = iters_for(n);
        let mut r = vec![0f64; n];
        let ref_ns = time(iters, || {
            <FFT64Ref as ReimArith>::reim_mul(&mut r, &a, &b);
            black_box(&r);
        });
        let neon_ns = time(iters, || {
            <FFT64Neon as ReimArith>::reim_mul(&mut r, &a, &b);
            black_box(&r);
        });
        row(rep, "reim_mul", n, ref_ns, neon_ns, sp);
    }

    fn bench_reim_addmul(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let r0 = rand_f64s(n);
        let a = rand_f64s(n);
        let b = rand_f64s(n);

        let mut neon = r0.clone();
        let mut refr = r0.clone();
        <FFT64Neon as ReimArith>::reim_addmul(&mut neon, &a, &b);
        <FFT64Ref as ReimArith>::reim_addmul(&mut refr, &a, &b);
        assert!(close_enough(&neon, &refr, 1e-3), "reim_addmul (n={n})");

        let iters = iters_for(n);
        let mut r = r0.clone();
        let ref_ns = time(iters, || {
            r.copy_from_slice(&r0);
            <FFT64Ref as ReimArith>::reim_addmul(&mut r, &a, &b);
            black_box(&r);
        });
        let neon_ns = time(iters, || {
            r.copy_from_slice(&r0);
            <FFT64Neon as ReimArith>::reim_addmul(&mut r, &a, &b);
            black_box(&r);
        });
        row(rep, "reim_addmul", n, ref_ns, neon_ns, sp);
    }

    // ─── FFT / IFFT ────────────────────────────────────────────────────────

    fn bench_fft(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let m = n / 2;
        let table = ReimFFTTable::<f64>::new(m);
        let data0 = rand_f64s(n);

        let mut neon = data0.clone();
        let mut refr = data0.clone();
        <FFT64Neon as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut neon);
        <FFT64Ref as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut refr);
        let rel = neon
            .iter()
            .zip(&refr)
            .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0))
            .fold(0f64, f64::max);
        assert!(rel < 1e-10, "fft rel-err {rel:.2e} (n={n})");

        let iters = iters_for(n);
        let mut d = data0.clone();
        let ref_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <FFT64Ref as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut d);
            black_box(&d);
        });
        let neon_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <FFT64Neon as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut d);
            black_box(&d);
        });
        row(rep, "fft", n, ref_ns, neon_ns, sp);
    }

    fn bench_ifft(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let m = n / 2;
        let table = ReimIFFTTable::<f64>::new(m);
        let data0 = rand_f64s(n);

        let mut neon = data0.clone();
        let mut refr = data0.clone();
        <FFT64Neon as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut neon);
        <FFT64Ref as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut refr);
        let rel = neon
            .iter()
            .zip(&refr)
            .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0))
            .fold(0f64, f64::max);
        assert!(rel < 1e-10, "ifft rel-err {rel:.2e} (n={n})");

        let iters = iters_for(n);
        let mut d = data0.clone();
        let ref_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <FFT64Ref as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut d);
            black_box(&d);
        });
        let neon_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <FFT64Neon as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(&table, &mut d);
            black_box(&d);
        });
        row(rep, "ifft", n, ref_ns, neon_ns, sp);
    }

    // ─── NTT120 ────────────────────────────────────────────────────────────

    fn bench_ntt_from_znx64(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let a = rand_i64s(n);

        let mut neon = vec![0u64; 4 * n];
        let mut refr = vec![0u64; 4 * n];
        <NTT120Neon as NttFromZnx64>::ntt_from_znx64(&mut neon, &a);
        <NTT120Ref as NttFromZnx64>::ntt_from_znx64(&mut refr, &a);
        assert_eq!(neon, refr, "ntt_from_znx64 (n={n})");

        let iters = iters_for(n);
        let mut r = vec![0u64; 4 * n];
        let ref_ns = time(iters, || {
            <NTT120Ref as NttFromZnx64>::ntt_from_znx64(&mut r, &a);
            black_box(&r);
        });
        let neon_ns = time(iters, || {
            <NTT120Neon as NttFromZnx64>::ntt_from_znx64(&mut r, &a);
            black_box(&r);
        });
        row(rep, "ntt_from_znx64", n, ref_ns, neon_ns, sp);
    }

    fn bench_ntt(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let table = NttTable::<Primes30>::new(n);
        let data0 = rand_q120b(n);

        let mut neon = data0.clone();
        let mut refr = data0.clone();
        <NTT120Neon as NttDFTExecute<NttTable<Primes30>>>::ntt_dft_execute(&table, &mut neon);
        <NTT120Ref as NttDFTExecute<NttTable<Primes30>>>::ntt_dft_execute(&table, &mut refr);
        assert_eq!(neon, refr, "ntt (n={n})");

        let iters = iters_for(n);
        let mut d = data0.clone();
        let ref_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <NTT120Ref as NttDFTExecute<NttTable<Primes30>>>::ntt_dft_execute(&table, &mut d);
            black_box(&d);
        });
        let neon_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <NTT120Neon as NttDFTExecute<NttTable<Primes30>>>::ntt_dft_execute(&table, &mut d);
            black_box(&d);
        });
        row(rep, "ntt", n, ref_ns, neon_ns, sp);
    }

    fn bench_intt(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let table = NttTableInv::<Primes30>::new(n);
        let data0 = rand_q120b(n);

        let mut neon = data0.clone();
        let mut refr = data0.clone();
        <NTT120Neon as NttDFTExecute<NttTableInv<Primes30>>>::ntt_dft_execute(&table, &mut neon);
        <NTT120Ref as NttDFTExecute<NttTableInv<Primes30>>>::ntt_dft_execute(&table, &mut refr);
        assert_eq!(neon, refr, "intt (n={n})");

        let iters = iters_for(n);
        let mut d = data0.clone();
        let ref_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <NTT120Ref as NttDFTExecute<NttTableInv<Primes30>>>::ntt_dft_execute(&table, &mut d);
            black_box(&d);
        });
        let neon_ns = time(iters, || {
            d.copy_from_slice(&data0);
            <NTT120Neon as NttDFTExecute<NttTableInv<Primes30>>>::ntt_dft_execute(&table, &mut d);
            black_box(&d);
        });
        row(rep, "intt", n, ref_ns, neon_ns, sp);
    }

    // ─── VecZnxBig (i128) ──────────────────────────────────────────────────

    fn bench_i128_add(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let a = rand_i128s(n);
        let b = rand_i128s(n);

        let mut neon = vec![0i128; n];
        let mut refr = vec![0i128; n];
        <NTT120Neon as I128BigOps>::i128_add(&mut neon, &a, &b);
        <NTT120Ref as I128BigOps>::i128_add(&mut refr, &a, &b);
        assert_eq!(neon, refr, "i128_add (n={n})");

        let iters = iters_for(n);
        let mut r = vec![0i128; n];
        let ref_ns = time(iters, || {
            <NTT120Ref as I128BigOps>::i128_add(&mut r, &a, &b);
            black_box(&r);
        });
        let neon_ns = time(iters, || {
            <NTT120Neon as I128BigOps>::i128_add(&mut r, &a, &b);
            black_box(&r);
        });
        row(rep, "i128_add", n, ref_ns, neon_ns, sp);
    }

    fn bench_nfc_middle_step(rep: &mut Reporter, n: usize, sp: &mut Vec<f64>) {
        let base2k = 32usize;
        let a = rand_i128s(n);
        let c_init = rand_i128s(n);

        let mut r_neon = vec![0i64; n];
        let mut c_neon = c_init.clone();
        let mut r_ref = vec![0i64; n];
        let mut c_ref = c_init.clone();
        <NTT120Neon as I128NormalizeOps>::nfc_middle_step(base2k, 0, &mut r_neon, &a, &mut c_neon);
        <NTT120Ref as I128NormalizeOps>::nfc_middle_step(base2k, 0, &mut r_ref, &a, &mut c_ref);
        assert_eq!(r_neon, r_ref, "nfc_middle_step r (n={n})");
        assert_eq!(c_neon, c_ref, "nfc_middle_step c (n={n})");

        let iters = iters_for(n);
        let mut r = vec![0i64; n];
        let mut c = c_init.clone();
        let ref_ns = time(iters, || {
            c.copy_from_slice(&c_init);
            <NTT120Ref as I128NormalizeOps>::nfc_middle_step(base2k, 0, &mut r, &a, &mut c);
            black_box((&r, &c));
        });
        let neon_ns = time(iters, || {
            c.copy_from_slice(&c_init);
            <NTT120Neon as I128NormalizeOps>::nfc_middle_step(base2k, 0, &mut r, &a, &mut c);
            black_box((&r, &c));
        });
        row(rep, "nfc_middle_step (i128)", n, ref_ns, neon_ns, sp);
    }

    // ─── driver ────────────────────────────────────────────────────────────

    fn section(rep: &mut Reporter, title: &str) {
        rep.line("");
        rep.line(&format!("== {title} =="));
    }

    fn summarize(rep: &mut Reporter, label: &str, speedups: &[f64]) {
        if speedups.is_empty() {
            return;
        }
        let geomean = (speedups.iter().map(|s| s.ln()).sum::<f64>() / speedups.len() as f64).exp();
        let min = speedups.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        rep.line(&format!(
            "  {} ({} kernels): geomean {:.2}×, min {:.2}×, max {:.2}×",
            label,
            speedups.len(),
            geomean,
            min,
            max
        ));
    }

    fn unix_now() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    }

    pub(super) fn run() {
        let out_path = std::env::var("POULPY_ARM_BENCH_OUT").unwrap_or_else(|_| DEFAULT_OUT.to_string());
        let mut rep = Reporter::new(&out_path);

        rep.line("== poulpy-cpu-arm: NEON vs cpu-ref ==");
        rep.line(&format!(
            "Host arch: {}, profile: {}, target_os: {}, ts(unix): {}",
            std::env::consts::ARCH,
            if cfg!(debug_assertions) { "debug" } else { "release" },
            std::env::consts::OS,
            unix_now(),
        ));
        if cfg!(debug_assertions) {
            rep.line("  (run with --release for meaningful timings)");
        }

        let mut sp = Vec::new();

        section(&mut rep, "Znx (i64)");
        for &n in SIZES {
            bench_znx_add(&mut rep, n, &mut sp);
            bench_znx_normalize(&mut rep, n, &mut sp);
            bench_znx_automorphism(&mut rep, n, &mut sp);
        }

        section(&mut rep, "Reim (f64 pointwise)");
        for &n in SIZES {
            bench_reim_add(&mut rep, n, &mut sp);
            bench_reim_mul(&mut rep, n, &mut sp);
            bench_reim_addmul(&mut rep, n, &mut sp);
        }

        section(&mut rep, "FFT / IFFT (f64)");
        for &n in SIZES {
            bench_fft(&mut rep, n, &mut sp);
            bench_ifft(&mut rep, n, &mut sp);
        }

        section(&mut rep, "NTT120 (q120b)");
        for &n in SIZES {
            bench_ntt_from_znx64(&mut rep, n, &mut sp);
            bench_ntt(&mut rep, n, &mut sp);
            bench_intt(&mut rep, n, &mut sp);
        }

        section(&mut rep, "VecZnxBig (i128)");
        for &n in SIZES {
            bench_i128_add(&mut rep, n, &mut sp);
            bench_nfc_middle_step(&mut rep, n, &mut sp);
        }

        section(&mut rep, "Summary");
        summarize(&mut rep, "All kernels", &sp);
        rep.line("");
        rep.line("All correctness checks passed.");
    }
}
