//! One measured preset: setup, warm-ups, timed bootstraps and the precision of
//! the last output, on the shared [`BootstrappingPresetRun`] driver.

use std::{cell::RefCell, time::Instant};

use anyhow::{Context, Result};
use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingCIOps, CKKSDFTOps, CKKSEncodingOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, EvalModRing},
    presets::bootstrapping::{BootstrappingPreset, all},
    test_suite::{
        helpers::{PrecisionStats, TestContextBackend, TestContextModule, TestContextSharedModule},
        presets::{BOOTSTRAP_FAILURE_BITS, BootstrappingPresetRun, CIBackend, PresetRunModule, preset_for_backend, preset_label},
    },
};
use poulpy_core::{
    GLWECIKeyEncryptSk,
    api::{GLWECIKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{
        GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc},
    layouts::{Backend, HostDataMut, HostDataRef, HostStaged, MaxBase2k, Module, ScratchOwned, Standard},
};
use serde_json::{Value, json};

/// What `run` was asked to measure.
pub struct RunArgs {
    pub preset: String,
    pub pow2: bool,
    pub replay: bool,
    pub warmup: usize,
    pub iters: usize,
    pub device: Option<usize>,
}

/// Wall-clock nanoseconds of every warm-up and every timed bootstrap.
pub struct Timings {
    pub warmup_ns: Vec<u64>,
    pub samples_ns: Vec<u64>,
    /// Facts a timing mode reports beside its times, such as the replay graph.
    pub extra: Option<Value>,
}

/// Times the bootstraps of a run. `queue` issues one bootstrap and `sync`
/// returns once the backend has completed it: a timed iteration covers both.
pub type Measure = fn(queue: &mut dyn FnMut(), sync: &dyn Fn(), warmup: usize, iters: usize) -> Timings;

fn timed(queue: &mut dyn FnMut(), sync: &dyn Fn()) -> u64 {
    let start = Instant::now();
    queue();
    sync();
    start.elapsed().as_nanos() as u64
}

/// Every iteration calls the bootstrap.
pub fn direct(queue: &mut dyn FnMut(), sync: &dyn Fn(), warmup: usize, iters: usize) -> Timings {
    sync();
    Timings {
        warmup_ns: (0..warmup).map(|_| timed(queue, sync)).collect(),
        samples_ns: (0..iters).map(|_| timed(queue, sync)).collect(),
        extra: None,
    }
}

/// The bootstrap is recorded once and every iteration replays the recording,
/// as an application that repeats a bootstrap on the same buffers runs it.
#[cfg(any(feature = "cuda", feature = "hip", feature = "metal", feature = "vulkan"))]
pub fn replay<RT: poulpy_gpu_portable::GpuRuntime>(
    queue: &mut dyn FnMut(),
    sync: &dyn Fn(),
    warmup: usize,
    iters: usize,
) -> Timings {
    sync();
    let mut replay = poulpy_gpu_portable::Replay::<RT>::record(queue);
    sync();
    let mut replayed = || replay.replay();
    let warmup_ns = (0..warmup).map(|_| timed(&mut replayed, sync)).collect();
    let samples_ns = (0..iters).map(|_| timed(&mut replayed, sync)).collect();
    Timings {
        warmup_ns,
        samples_ns,
        extra: Some(json!({
            "graph": replay.is_graph(),
            "launches": replay.launches(),
            "retained_bytes": replay.retained_bytes(),
        })),
    }
}

/// The preset of the release set named `name`, with power-of-two rotation keys
/// when `pow2`, at the radix [`preset_for_backend`] selects for `BE`.
fn preset_for<BE, BCI>(name: &str, pow2: bool) -> Result<(BootstrappingPreset, usize)>
where
    BE: MaxBase2k,
    BCI: MaxBase2k,
{
    let preset = all()?
        .into_iter()
        .find(|preset| preset.name() == name)
        .with_context(|| format!("no bootstrapping preset named {name}"))?;
    let preset = if pow2 { preset.with_power_of_two_keys()? } else { preset };
    let nominal_base2k = preset.base2k();
    Ok((preset_for_backend::<BE, BCI>(&preset)?, nominal_base2k))
}

fn precision_json(stats: &PrecisionStats) -> Value {
    json!({
        "avg_bits": stats.avg_log2_prec,
        "min_bits": stats.min_log2_prec,
        "max_bits": stats.max_log2_prec,
        "max_abs_err": stats.worst_err,
        "worst_idx": stats.worst_idx,
    })
}

fn preset_json(preset: &BootstrappingPreset, nominal_base2k: usize, pow2: bool) -> Value {
    let plan = preset.plan();
    json!({
        "name": preset.name(),
        "label": preset_label(preset),
        "pow2_keys": pow2,
        "pipeline": format!("{:?}", plan.pipeline()),
        "eval_mod_ring": match plan.eval_mod_ring() {
            EvalModRing::Standard => "Standard",
            _ => "ConjugateInvariant",
        },
        "log_n": preset.log_n(),
        "n": preset.n(),
        "base2k": preset.base2k(),
        "nominal_base2k": nominal_base2k,
        "failure_bits": BOOTSTRAP_FAILURE_BITS,
        "log_delta": preset.log_delta(),
        "input_k": preset.input_k(),
        "output_k": preset.output_k(),
        "bootstrap_k": preset.bootstrap_k(),
        "log_modulus": preset.log_modulus(),
        "advertised_bits": preset.log2_precision(),
        "key_dsize": preset.key_dsize(),
        "key_k_digit": preset.key_k_digit(),
        "dense_to_sparse_dsize": preset.dense_to_sparse_dsize(),
        "dense_secret_hamming_weight": preset.dense_secret_hamming_weight(),
        "sparse_secret_hamming_weight": preset.sparse_secret_hamming_weight(),
        "max_dense_modulus": preset.max_dense_modulus(),
        "max_sparse_modulus": preset.max_sparse_modulus(),
    })
}

/// Sets the preset up with `BE` evaluating and `BC` as the client, times its
/// bootstraps with `measure`, and measures the precision of the last output.
/// Setup covers context compilation, key generation and preparation, encryption
/// and the bootstrap the driver runs to check the output width. The timed region
/// is the bootstrap alone, synchronized.
pub fn run<BE, BC>(args: &RunArgs, measure: Measure) -> Result<Value>
where
    BE: Backend<Ring = Standard, ZnxWord = i64> + HostStaged + MaxBase2k + 'static,
    CIBackend<BE>: MaxBase2k,
    BC: TestContextBackend<Ring = Standard>,
    CIBackend<BC>: TestContextBackend,
    Module<BE>: PresetRunModule<BE> + CKKSDFTOps<BE> + CKKSBootstrappingCIOps<BE> + GLWECIKeyPreparedFactory<BE>,
    Module<CIBackend<BE>>: ModuleNew<CIBackend<BE>>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSModuleAlloc<CIBackend<BE>>
        + CKKSEncodingOps<CIBackend<BE>, f64>
        + GLWETensorKeyPreparedFactory<CIBackend<BE>>,
    Module<BC>: TestContextModule<BC> + CKKSEncodingOps<BC, f64> + CKKSBootstrappingCIOps<BC> + GLWECIKeyEncryptSk<BC>,
    Module<CIBackend<BC>>: TestContextSharedModule<CIBackend<BC>>,
    for<'a> <BC as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BC as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    ScratchOwned<CIBackend<BE>>: ScratchOwnedAlloc<CIBackend<BE>>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos + CKKSCtBounds + poulpy_core::layouts::DiagonalProd<BE>,
    CKKSCiphertextOwned<BC>: GLWEToBackendMut<BC> + GLWEToBackendRef<BC> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BC>: GLWEToBackendRef<BC> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let (preset, nominal_base2k) = preset_for::<BE, CIBackend<BE>>(&args.preset, args.pow2)?;
    let preset_info = preset_json(&preset, nominal_base2k, args.pow2);

    let start = Instant::now();
    let mut run = match preset.plan().eval_mod_ring() {
        EvalModRing::Standard => BootstrappingPresetRun::<BE, BC>::setup(preset),
        _ => BootstrappingPresetRun::<BE, BC>::setup_ci(preset),
    };
    run.synchronize();
    let setup_ns = start.elapsed().as_nanos() as u64;

    let timings = {
        let run = RefCell::new(&mut run);
        let mut queue = || run.borrow_mut().bootstrap_queued();
        let sync = || run.borrow().synchronize();
        measure(&mut queue, &sync, args.warmup, args.iters)
    };

    let (re, im) = run.precision();
    let advertised = run.preset().log2_precision() as f64;
    Ok(json!({
        "preset": preset_info,
        "setup_ns": setup_ns,
        "warmup_ns": timings.warmup_ns,
        "samples_ns": timings.samples_ns,
        "replay": timings.extra,
        "precision": {
            "re": precision_json(&re),
            "im": precision_json(&im),
            "advertised_bits": advertised,
            "meets_advertised": re.min_log2_prec >= advertised && im.min_log2_prec >= advertised,
        },
    }))
}
