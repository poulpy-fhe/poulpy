//! Release measurements of the CKKS bootstrapping presets.
//!
//! `poulpy-release-bench list` prints, as JSON, the backends this build
//! contains and the presets of the release set. `poulpy-release-bench run`
//! measures one preset on one backend and prints one JSON object on stdout.
//! `tools/release_bench.py` builds this binary once per set of target flags,
//! runs every measurement in its own process and assembles the release record.
//!
//! ```text
//! poulpy-release-bench run --backend NTT4x30Portable --preset n16_d35_k733_p20_s2c
//!     [--pow2] [--replay] [--warmup 3] [--iters 20] [--device 0]
//! ```

mod runner;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use runner::{RunArgs, direct, run};

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
    run: fn(&RunArgs) -> Result<Value>,
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
            run: |args| {
                select_device::<$rt>(args)?;
                let measure = if args.replay { runner::replay::<$rt> } else { direct };
                run::<$backend, poulpy_cpu_portable::NTT4x30Portable>(args, measure)
            },
        }
    };
}

fn entries() -> Vec<Entry> {
    use poulpy_cpu_portable::NTT4x30Portable;
    #[allow(unused_mut)]
    let mut entries = vec![Entry {
        name: "NTT4x30Portable",
        krate: "poulpy-cpu-portable",
        features: &["enable-ckks"],
        transform: "NTT4x30",
        parallelism: Parallelism::Serial,
        run: |args| {
            direct_only(args)?;
            run::<NTT4x30Portable, NTT4x30Portable>(args, direct)
        },
    }];
    #[cfg(feature = "portable-rayon")]
    {
        use poulpy_cpu_rayon::NTT4x30PortableRayon;
        entries.push(Entry {
            name: "NTT4x30PortableRayon",
            krate: "poulpy-cpu-rayon",
            features: &["enable-ckks"],
            transform: "NTT4x30",
            parallelism: Parallelism::Rayon,
            run: |args| {
                direct_only(args)?;
                run::<NTT4x30PortableRayon, NTT4x30PortableRayon>(args, direct)
            },
        });
    }
    #[cfg(feature = "avx")]
    {
        use poulpy_cpu_avx::{NTT4x30Avx, NTT4x30AvxRayon};
        const FEATURES: &[&str] = &["enable-avx", "enable-ckks", "enable-rayon"];
        entries.push(Entry {
            name: "NTT4x30Avx",
            krate: "poulpy-cpu-avx",
            features: FEATURES,
            transform: "NTT4x30",
            parallelism: Parallelism::Serial,
            run: |args| {
                direct_only(args)?;
                run::<NTT4x30Avx, NTT4x30Avx>(args, direct)
            },
        });
        entries.push(Entry {
            name: "NTT4x30AvxRayon",
            krate: "poulpy-cpu-avx",
            features: FEATURES,
            transform: "NTT4x30",
            parallelism: Parallelism::Rayon,
            run: |args| {
                direct_only(args)?;
                run::<NTT4x30AvxRayon, NTT4x30AvxRayon>(args, direct)
            },
        });
    }
    #[cfg(all(feature = "avx512", not(feature = "ifma")))]
    {
        use poulpy_cpu_avx512::{NTT4x30Avx512, NTT4x30Avx512Rayon};
        const FEATURES: &[&str] = &["enable-avx512f", "enable-ckks", "enable-rayon"];
        entries.push(Entry {
            name: "NTT4x30Avx512",
            krate: "poulpy-cpu-avx512",
            features: FEATURES,
            transform: "NTT4x30",
            parallelism: Parallelism::Serial,
            run: |args| {
                direct_only(args)?;
                run::<NTT4x30Avx512, NTT4x30Avx512>(args, direct)
            },
        });
        entries.push(Entry {
            name: "NTT4x30Avx512Rayon",
            krate: "poulpy-cpu-avx512",
            features: FEATURES,
            transform: "NTT4x30",
            parallelism: Parallelism::Rayon,
            run: |args| {
                direct_only(args)?;
                run::<NTT4x30Avx512Rayon, NTT4x30Avx512Rayon>(args, direct)
            },
        });
    }
    #[cfg(feature = "ifma")]
    {
        use poulpy_cpu_avx512::{NTT3x42Ifma, NTT3x42IfmaRayon};
        const FEATURES: &[&str] = &["enable-ifma", "enable-ckks", "enable-rayon"];
        entries.push(Entry {
            name: "NTT3x42Ifma",
            krate: "poulpy-cpu-avx512",
            features: FEATURES,
            transform: "NTT3x42",
            parallelism: Parallelism::Serial,
            run: |args| {
                direct_only(args)?;
                run::<NTT3x42Ifma, NTT3x42Ifma>(args, direct)
            },
        });
        entries.push(Entry {
            name: "NTT3x42IfmaRayon",
            krate: "poulpy-cpu-avx512",
            features: FEATURES,
            transform: "NTT3x42",
            parallelism: Parallelism::Rayon,
            run: |args| {
                direct_only(args)?;
                run::<NTT3x42IfmaRayon, NTT3x42IfmaRayon>(args, direct)
            },
        });
    }
    #[cfg(feature = "neon")]
    {
        use poulpy_cpu_arm::{NTT4x30Neon, NTT4x30NeonRayon};
        const FEATURES: &[&str] = &["enable-neon", "enable-ckks", "enable-rayon"];
        entries.push(Entry {
            name: "NTT4x30Neon",
            krate: "poulpy-cpu-arm",
            features: FEATURES,
            transform: "NTT4x30",
            parallelism: Parallelism::Serial,
            run: |args| {
                direct_only(args)?;
                run::<NTT4x30Neon, NTT4x30Neon>(args, direct)
            },
        });
        entries.push(Entry {
            name: "NTT4x30NeonRayon",
            krate: "poulpy-cpu-arm",
            features: FEATURES,
            transform: "NTT4x30",
            parallelism: Parallelism::Rayon,
            run: |args| {
                direct_only(args)?;
                run::<NTT4x30NeonRayon, NTT4x30NeonRayon>(args, direct)
            },
        });
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
    })
}

fn list() -> Value {
    json!({
        "backends": entries().iter().map(entry_json).collect::<Vec<_>>(),
        "presets": RELEASE_PRESETS,
    })
}

fn parse_run(args: &[String]) -> Result<(String, RunArgs)> {
    let mut backend = None;
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
            "--pow2" => run.pow2 = true,
            "--replay" => run.replay = true,
            "--warmup" => run.warmup = value()?.parse()?,
            "--iters" => run.iters = value()?.parse()?,
            "--device" => run.device = Some(value()?.parse()?),
            other => bail!("unknown argument {other}"),
        }
    }
    ensure!(!run.preset.is_empty(), "--preset is required");
    ensure!(run.iters > 0, "--iters must be positive");
    Ok((backend.context("--backend is required")?, run))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let output = match args.first().map(String::as_str) {
        Some("list") => list(),
        Some("run") => {
            let (backend, run) = parse_run(&args[1..])?;
            let entries = entries();
            let entry = entries
                .iter()
                .find(|entry| entry.name == backend)
                .with_context(|| format!("backend {backend} is not in this build"))?;
            let mut result = (entry.run)(&run)?;
            let threads = match entry.parallelism {
                Parallelism::Serial => Some(1),
                Parallelism::Rayon => Some(rayon::current_num_threads()),
                Parallelism::Device => None,
            };
            let record = result.as_object_mut().unwrap();
            record.insert("backend".into(), entry_json(entry));
            record.insert("threads".into(), json!(threads));
            record.insert("mode".into(), json!(if run.replay { "replay" } else { "direct" }));
            record.insert("warmup".into(), json!(run.warmup));
            record.insert("iters".into(), json!(run.iters));
            result
        }
        _ => bail!(
            "usage: poulpy-release-bench list | run --backend NAME --preset NAME [--pow2] [--replay] [--warmup N] [--iters N] [--device N]"
        ),
    };
    println!("{output}");
    Ok(())
}
