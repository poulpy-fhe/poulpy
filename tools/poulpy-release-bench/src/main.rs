//! Release measurements of the CKKS bootstrapping presets, of `poulpy-libm`
//! functions and of `poulpy-blas` operations.
//!
//! `poulpy-release-bench list` prints, as JSON, the backends this build
//! contains, the suites each one runs, the presets of the release set and the
//! libm and blas cases. `poulpy-release-bench run` measures one preset or case
//! on one backend and prints one JSON object on stdout. `tools/release_bench.py`
//! builds this binary once per set of target flags, runs every measurement in
//! its own process and assembles the release record.
//!
//! ```text
//! poulpy-release-bench run --backend NTT4x30Portable --preset n16_d35_k733_p20_s2c
//!     [--pow2] [--replay] [--warmup 3] [--iters 20] [--device 0]
//! poulpy-release-bench run --backend NTT4x30Portable --libm rsqrt_r16 [--warmup 3] [--iters 20]
//! poulpy-release-bench run --backend NTT4x30Portable --blas matmul_d64 [--warmup 3] [--iters 20]
//! ```

mod blas;
mod libm;
mod runner;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use runner::{CaseArgs, RunArgs, direct, run};

/// The presets a release measures: C2S-first and S2C-first at 20 bits, and the
/// fastest preset.
const RELEASE_PRESETS: [&str; 3] = ["n16_d35_k636_p20_c2s", "n16_d35_k733_p20_s2c", "n16_d35_k685_p19_s2c_ffr_ci"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Parallelism {
    Serial,
    Rayon,
    #[cfg_attr(
        not(any(feature = "cuda", feature = "hip", feature = "metal", feature = "vulkan")),
        allow(dead_code, reason = "constructed by the GPU entries")
    )]
    Device,
}

/// A backend this build can measure.
struct Entry {
    name: &'static str,
    krate: &'static str,
    /// The features of `krate` this build enables.
    features: &'static [&'static str],
    transform: &'static str,
    parallelism: Parallelism,
    bootstrapping: fn(&RunArgs) -> Result<Value>,
    /// The libm and blas cases, on the backends that run them.
    libm: Option<fn(&CaseArgs) -> Result<Value>>,
    blas: Option<fn(&CaseArgs) -> Result<Value>>,
}

/// Rejects the replay mode on the backends that have none.
fn direct_only(args: &RunArgs) -> Result<()> {
    ensure!(!args.replay, "--replay is only available on the GPU backends");
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "hip", feature = "metal", feature = "vulkan"))]
fn select_device<RT: poulpy_gpu_portable::GpuRuntime>(args: &RunArgs) -> Result<()> {
    if let Some(index) = args.device {
        ensure!(
            RT::select_device(index),
            "device {index} of {} could not be selected",
            RT::NAME
        );
    }
    Ok(())
}

/// A GPU entry on runtime `$rt`, with `NTT4x30Portable` as the client.
#[cfg(any(feature = "cuda", feature = "hip", feature = "metal", feature = "vulkan"))]
macro_rules! gpu_entry {
    ($name:literal, $backend:ty, $rt:ty, $features:expr) => {
        Entry {
            name: $name,
            krate: "poulpy-gpu-portable",
            features: $features,
            transform: "NTT4x30",
            parallelism: Parallelism::Device,
            bootstrapping: |args| {
                select_device::<$rt>(args)?;
                let measure = if args.replay { runner::replay::<$rt> } else { direct };
                run::<$backend, poulpy_cpu_portable::NTT4x30Portable>(args, measure)
            },
            libm: None,
            blas: None,
        }
    };
}

/// A CPU entry on backend `$be`, which is its own client and runs every suite.
macro_rules! cpu_entry {
    ($name:literal, $krate:literal, $features:expr, $transform:literal, $parallelism:expr, $be:ty) => {
        Entry {
            name: $name,
            krate: $krate,
            features: $features,
            transform: $transform,
            parallelism: $parallelism,
            bootstrapping: |args| {
                direct_only(args)?;
                run::<$be, $be>(args, direct)
            },
            libm: Some(libm::run::<$be>),
            blas: Some(blas::run::<$be>),
        }
    };
}

fn entries() -> Vec<Entry> {
    use poulpy_cpu_portable::NTT4x30Portable;
    #[allow(unused_mut)]
    let mut entries = vec![cpu_entry!(
        "NTT4x30Portable",
        "poulpy-cpu-portable",
        &["enable-ckks", "enable-blas"],
        "NTT4x30",
        Parallelism::Serial,
        NTT4x30Portable
    )];
    #[cfg(feature = "portable-rayon")]
    {
        use poulpy_cpu_rayon::NTT4x30PortableRayon;
        entries.push(cpu_entry!(
            "NTT4x30PortableRayon",
            "poulpy-cpu-rayon",
            &["enable-ckks", "enable-blas"],
            "NTT4x30",
            Parallelism::Rayon,
            NTT4x30PortableRayon
        ));
    }
    #[cfg(feature = "avx")]
    {
        use poulpy_cpu_avx::{NTT4x30Avx, NTT4x30AvxRayon};
        const FEATURES: &[&str] = &["enable-avx", "enable-ckks", "enable-blas", "enable-rayon"];
        entries.push(cpu_entry!(
            "NTT4x30Avx",
            "poulpy-cpu-avx",
            FEATURES,
            "NTT4x30",
            Parallelism::Serial,
            NTT4x30Avx
        ));
        entries.push(cpu_entry!(
            "NTT4x30AvxRayon",
            "poulpy-cpu-avx",
            FEATURES,
            "NTT4x30",
            Parallelism::Rayon,
            NTT4x30AvxRayon
        ));
    }
    #[cfg(all(feature = "avx512", not(feature = "ifma")))]
    {
        use poulpy_cpu_avx512::{NTT4x30Avx512, NTT4x30Avx512Rayon};
        const FEATURES: &[&str] = &["enable-avx512f", "enable-ckks", "enable-blas", "enable-rayon"];
        entries.push(cpu_entry!(
            "NTT4x30Avx512",
            "poulpy-cpu-avx512",
            FEATURES,
            "NTT4x30",
            Parallelism::Serial,
            NTT4x30Avx512
        ));
        entries.push(cpu_entry!(
            "NTT4x30Avx512Rayon",
            "poulpy-cpu-avx512",
            FEATURES,
            "NTT4x30",
            Parallelism::Rayon,
            NTT4x30Avx512Rayon
        ));
    }
    #[cfg(feature = "ifma")]
    {
        use poulpy_cpu_avx512::{NTT3x42Ifma, NTT3x42IfmaRayon};
        const FEATURES: &[&str] = &["enable-ifma", "enable-ckks", "enable-blas", "enable-rayon"];
        entries.push(cpu_entry!(
            "NTT3x42Ifma",
            "poulpy-cpu-avx512",
            FEATURES,
            "NTT3x42",
            Parallelism::Serial,
            NTT3x42Ifma
        ));
        entries.push(cpu_entry!(
            "NTT3x42IfmaRayon",
            "poulpy-cpu-avx512",
            FEATURES,
            "NTT3x42",
            Parallelism::Rayon,
            NTT3x42IfmaRayon
        ));
    }
    #[cfg(feature = "neon")]
    {
        use poulpy_cpu_arm::{NTT4x30Neon, NTT4x30NeonRayon};
        const FEATURES: &[&str] = &["enable-neon", "enable-ckks", "enable-blas", "enable-rayon"];
        entries.push(cpu_entry!(
            "NTT4x30Neon",
            "poulpy-cpu-arm",
            FEATURES,
            "NTT4x30",
            Parallelism::Serial,
            NTT4x30Neon
        ));
        entries.push(cpu_entry!(
            "NTT4x30NeonRayon",
            "poulpy-cpu-arm",
            FEATURES,
            "NTT4x30",
            Parallelism::Rayon,
            NTT4x30NeonRayon
        ));
    }
    #[cfg(feature = "cuda")]
    entries.push(gpu_entry!(
        "NTT4x30Cuda",
        poulpy_gpu_portable::NTT4x30Cuda,
        poulpy_gpu_portable::Cuda,
        &["enable-cuda", "enable-ckks"]
    ));
    #[cfg(feature = "hip")]
    entries.push(gpu_entry!(
        "NTT4x30Hip",
        poulpy_gpu_portable::NTT4x30Hip,
        poulpy_gpu_portable::Hip,
        &["enable-hip", "enable-ckks"]
    ));
    #[cfg(feature = "metal")]
    entries.push(gpu_entry!(
        "NTT4x30Wgpu",
        poulpy_gpu_portable::NTT4x30Wgpu,
        poulpy_gpu_portable::Wgpu,
        &["enable-metal", "enable-ckks"]
    ));
    #[cfg(all(feature = "vulkan", not(feature = "metal")))]
    entries.push(gpu_entry!(
        "NTT4x30Wgpu",
        poulpy_gpu_portable::NTT4x30Wgpu,
        poulpy_gpu_portable::Wgpu,
        &["enable-vulkan", "enable-ckks"]
    ));
    entries
}

fn entry_json(entry: &Entry) -> Value {
    let suites: Vec<&str> = [Some("bootstrapping"), entry.libm.map(|_| "libm"), entry.blas.map(|_| "blas")]
        .into_iter()
        .flatten()
        .collect();
    json!({
        "backend": entry.name,
        "crate": entry.krate,
        "crate_features": entry.features,
        "transform": entry.transform,
        "parallelism": match entry.parallelism {
            Parallelism::Serial => "serial",
            Parallelism::Rayon => "rayon",
            Parallelism::Device => "device",
        },
        "suites": suites,
    })
}

fn list() -> Value {
    json!({
        "backends": entries().iter().map(entry_json).collect::<Vec<_>>(),
        "presets": RELEASE_PRESETS,
        "libm": libm::CASES,
        "blas": blas::cases(),
    })
}

/// What `run` measures: a bootstrapping preset, or a libm or blas case.
enum Job {
    Bootstrapping(RunArgs),
    Libm(CaseArgs),
    Blas(CaseArgs),
}

fn parse_run(args: &[String]) -> Result<(String, Job)> {
    let mut backend = None;
    let (mut libm, mut blas) = (None, None);
    let mut run = RunArgs {
        preset: String::new(),
        pow2: false,
        replay: false,
        warmup: 3,
        iters: 20,
        device: None,
    };
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let mut value = || args.next().with_context(|| format!("{flag} takes a value"));
        match flag.as_str() {
            "--backend" => backend = Some(value()?.clone()),
            "--preset" => run.preset = value()?.clone(),
            "--libm" => libm = Some(value()?.clone()),
            "--blas" => blas = Some(value()?.clone()),
            "--pow2" => run.pow2 = true,
            "--replay" => run.replay = true,
            "--warmup" => run.warmup = value()?.parse()?,
            "--iters" => run.iters = value()?.parse()?,
            "--device" => run.device = Some(value()?.parse()?),
            other => bail!("unknown argument {other}"),
        }
    }
    ensure!(run.iters > 0, "--iters must be positive");
    let backend = backend.context("--backend is required")?;
    let case = |case| {
        ensure!(
            !run.pow2 && !run.replay && run.device.is_none(),
            "--pow2, --replay and --device only apply to --preset"
        );
        Ok(CaseArgs {
            case,
            warmup: run.warmup,
            iters: run.iters,
        })
    };
    let job = match (run.preset.is_empty(), libm, blas) {
        (false, None, None) => Job::Bootstrapping(run),
        (true, Some(name), None) => Job::Libm(case(name)?),
        (true, None, Some(name)) => Job::Blas(case(name)?),
        _ => bail!("give exactly one of --preset, --libm and --blas"),
    };
    Ok((backend, job))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let output = match args.first().map(String::as_str) {
        Some("list") => list(),
        Some("run") => {
            let (backend, job) = parse_run(&args[1..])?;
            let entries = entries();
            let entry = entries
                .iter()
                .find(|entry| entry.name == backend)
                .with_context(|| format!("backend {backend} is not in this build"))?;
            let unsupported = |suite: &str| format!("backend {backend} does not run the {suite} suite");
            let (suite, mode, warmup, iters, mut result) = match &job {
                Job::Bootstrapping(run) => {
                    let mode = if run.replay { "replay" } else { "direct" };
                    ("bootstrapping", mode, run.warmup, run.iters, (entry.bootstrapping)(run)?)
                }
                Job::Libm(case) => {
                    let measure = entry.libm.with_context(|| unsupported("libm"))?;
                    ("libm", "direct", case.warmup, case.iters, measure(case)?)
                }
                Job::Blas(case) => {
                    let measure = entry.blas.with_context(|| unsupported("blas"))?;
                    ("blas", "direct", case.warmup, case.iters, measure(case)?)
                }
            };
            let threads = match entry.parallelism {
                Parallelism::Serial => Some(1),
                Parallelism::Rayon => Some(rayon::current_num_threads()),
                Parallelism::Device => None,
            };
            let record = result.as_object_mut().unwrap();
            record.insert("suite".into(), json!(suite));
            record.insert("backend".into(), entry_json(entry));
            record.insert("threads".into(), json!(threads));
            record.insert("mode".into(), json!(mode));
            record.insert("warmup".into(), json!(warmup));
            record.insert("iters".into(), json!(iters));
            result
        }
        _ => bail!(
            "usage: poulpy-release-bench list | run --backend NAME (--preset NAME [--pow2] [--replay] [--device N] | --libm CASE | --blas CASE) [--warmup N] [--iters N]"
        ),
    };
    println!("{output}");
    Ok(())
}
