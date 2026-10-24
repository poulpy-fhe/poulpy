//! The upstream `poulpy-ckks` suites, with `NTT4x30Portable` as the comparison backend.

type Gpu = crate::NTT4x30Gpu<super::Runtime, poulpy_hal::layouts::Standard>;
type GpuCI = crate::NTT4x30Gpu<super::Runtime, poulpy_hal::layouts::ConjugateInvariant>;

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_f64,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
        bootstrapping => poulpy_ckks::test_suite::parity::test_bootstrapping_parity,
        fold => poulpy_ckks::test_suite::parity::test_fold_parity,
    }
}

// The suite at the degree of the bootstrapping presets.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_n65536,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 1 << 16, hw: 192, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        #[ignore = "full degree; run in release with --ignored"]
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        #[ignore = "full degree; run in release with --ignored"]
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        #[ignore = "full degree; run in release with --ignored"]
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        #[ignore = "full degree; run in release with --ignored"]
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        #[ignore = "full degree; run in release with --ignored"]
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        #[ignore = "full degree; run in release with --ignored"]
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        #[ignore = "full degree; run in release with --ignored"]
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "full degree; run in release with --ignored"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        #[ignore = "full degree; run in release with --ignored"]
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        #[ignore = "full degree; run in release with --ignored"]
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        #[ignore = "full degree; run in release with --ignored"]
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        #[ignore = "full degree; run in release with --ignored"]
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
        #[ignore = "full degree; run in release with --ignored"]
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        #[ignore = "full degree; run in release with --ignored"]
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
        #[ignore = "full degree; run in release with --ignored"]
        bootstrapping => poulpy_ckks::test_suite::parity::test_bootstrapping_parity,
        #[ignore = "full degree; run in release with --ignored"]
        fold => poulpy_ckks::test_suite::parity::test_fold_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_quad,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
        bootstrapping => poulpy_ckks::test_suite::parity::test_bootstrapping_parity,
        fold => poulpy_ckks::test_suite::parity::test_fold_parity,
    }
}

// Explicit rank-2 contracts.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_rank2,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
        bootstrapping => poulpy_ckks::test_suite::parity::test_bootstrapping_parity,
        fold => poulpy_ckks::test_suite::parity::test_fold_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_encryption,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64Portable,
    backend_test = super::Gpu,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_encryption_rank2,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64Portable,
    backend_test = super::Gpu,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

// The families of the conjugate-invariant ring, as `NTT4x30CIPortable` registers them.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_f64,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_quad,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_encryption,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64CIPortable,
    backend_test = super::GpuCI,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

/// Serializes the full-size tests: each takes about 20 GB of device memory, and the
/// launch counters two of them assert are global to the process.
static FULL_SIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The bootstrapping presets at full size: keys generated and inputs
/// encrypted by the portable client, context compiled, keys prepared and
/// bootstrap evaluated on the device.
#[test]
#[ignore = "full-size presets: minutes and about 20 GB of device memory; run in release"]
fn bootstrapping_presets_meet_precision() {
    let _serial = FULL_SIZE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    poulpy_ckks::test_suite::presets::bootstrapping_presets_meet_precision_with_client::<Gpu, poulpy_cpu_portable::NTT4x30Portable>(
    );
}

/// One preset at full size: its measured precision, then the kernel launches
/// of one steady-state bootstrap (after a warm-up that compiles the kernels).
#[test]
#[ignore = "full-size preset: minutes and about 20 GB of device memory; run in release with --test-threads=1"]
fn bootstrapping_n16_d35_k733_p20_s2c_precision_and_launches() {
    let _serial = FULL_SIZE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    use poulpy_ckks::test_suite::presets::{BootstrappingPresetRun, preset_for_backend};
    let preset = poulpy_ckks::presets::bootstrapping::n16_d35_k733_p20_s2c().unwrap();
    let preset = preset_for_backend::<Gpu, GpuCI>(&preset).unwrap();
    let mut run = BootstrappingPresetRun::<Gpu, poulpy_cpu_portable::NTT4x30Portable>::setup(preset);
    // What the device holds once the keys are prepared, and what the pools reserve for it.
    {
        use crate::runtime::RuntimeDevice;
        let usage = super::Runtime::client().memory_report(cubecl::MemoryScope::Device).usage();
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        println!(
            "MEMORY in_use={:.2}GiB reserved={:.2}GiB",
            gib(usage.bytes_in_use),
            gib(usage.bytes_reserved)
        );
    }
    run.bootstrap();
    let (re, im) = run.precision();
    println!(
        "PRECISION preset={} base2k={} re_min={:.2}b re_avg={:.2}b im_min={:.2}b im_avg={:.2}b (advertised {}b)",
        run.preset().name(),
        run.preset().base2k(),
        re.min_log2_prec,
        re.avg_log2_prec,
        im.min_log2_prec,
        im.avg_log2_prec,
        run.preset().log2_precision(),
    );
    crate::reset_launch_counters();
    let start = std::time::Instant::now();
    run.bootstrap();
    let elapsed = start.elapsed();
    let counters = crate::launch_counters();
    println!(
        "LAUNCHES preset={} launches={} bytes={} allocations={} elapsed={:.1}ms",
        run.preset().name(),
        counters.launches,
        counters.bytes,
        counters.allocations,
        elapsed.as_secs_f64() * 1e3
    );
    for kernel in crate::launch_breakdown() {
        println!(
            "KERNEL {:<24} launches={:>5} bytes={:>13} share={:>5.1}%",
            kernel.label,
            kernel.launches,
            kernel.bytes,
            100.0 * kernel.bytes as f64 / counters.bytes as f64
        );
    }
    // The same bootstrap with a synchronization around every launch: the time of each kernel alone.
    crate::runtime::profile::set(true);
    run.bootstrap();
    let mut seconds = crate::runtime::profile::seconds();
    crate::runtime::profile::set(false);
    seconds.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = seconds.iter().map(|kernel| kernel.1).sum();
    println!("PROFILE total={:.1}ms", total * 1e3);
    for (label, kernel) in seconds {
        println!("PROFILE {label:<24} {:>7.1}ms {:>5.1}%", kernel * 1e3, 100.0 * kernel / total);
    }
    let advertised = run.preset().log2_precision() as f64;
    assert!(re.min_log2_prec >= advertised && im.min_log2_prec >= advertised);
}

/// The bootstrap of one preset at full size, recorded and replayed: the
/// recording runs the body twice, each replay issues no launch and no
/// allocation, and the output decrypted and measured is the one a graph
/// replay produced. The closure is then checked against its recording.
#[test]
#[ignore = "full-size preset: minutes and about 20 GB of device memory; run in release with --test-threads=1"]
fn bootstrapping_n16_d35_k733_p20_s2c_replay() {
    let _serial = FULL_SIZE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    use poulpy_ckks::test_suite::presets::{BootstrappingPresetRun, preset_for_backend};
    use std::cell::RefCell;

    let preset = poulpy_ckks::presets::bootstrapping::n16_d35_k733_p20_s2c().unwrap();
    let preset = preset_for_backend::<Gpu, GpuCI>(&preset).unwrap();
    let advertised = preset.log2_precision() as f64;
    let run = RefCell::new(BootstrappingPresetRun::<Gpu, poulpy_cpu_portable::NTT4x30Portable>::setup(
        preset,
    ));
    // Compiles the kernels, so that the counters below are those of a steady-state bootstrap.
    run.borrow_mut().bootstrap();
    let meets_precision = |what: &str| {
        let (re, im) = run.borrow_mut().precision();
        println!(
            "REPLAY {what:<10} re_min={:.2}b im_min={:.2}b",
            re.min_log2_prec, im.min_log2_prec
        );
        assert!(re.min_log2_prec >= advertised && im.min_log2_prec >= advertised, "{what}");
    };

    crate::reset_launch_counters();
    let mut replay = crate::Replay::<super::Runtime>::record(|| run.borrow_mut().bootstrap_queued());
    let recording = crate::launch_counters();
    println!(
        "REPLAY recording  launches={} graph={} retained_bytes={}",
        replay.launches(),
        replay.is_graph(),
        replay.retained_bytes()
    );
    assert!(replay.is_graph());
    assert_eq!(recording.launches, 2 * replay.launches() as u64);
    meets_precision("recording");
    for _ in 0..4 {
        crate::reset_launch_counters();
        let start = std::time::Instant::now();
        replay.replay();
        run.borrow().synchronize();
        let elapsed = start.elapsed();
        let counters = crate::launch_counters();
        println!("REPLAY replay     elapsed={:.1}ms", elapsed.as_secs_f64() * 1e3);
        // A graph replay issues no launch of its own: the counters report it as a replay.
        assert_eq!(
            (counters.launches, counters.bytes, counters.allocations, counters.replays),
            (0, 0, 0, 1)
        );
        meets_precision("replay");
    }
    replay.check();
    meets_precision("check");
}

/// Slot encoding on the conjugate-invariant ring, staged through the host:
/// the plaintexts are those of the portable backend, limb for limb, and
/// decode to the encoded values. (The upstream conjugate-invariant CKKS suite
/// reads backend allocations as host slices, which a device backend does not have.)
#[test]
fn conjugate_invariant_encoding_matches_portable() {
    use poulpy_ckks::{
        GLWEPlaintextInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
        api::{CKKSEncodingHostOps, CKKSEncodingOps},
        layouts::CKKSModuleAlloc,
    };
    use poulpy_core::layouts::GLWEToBackendRef;
    use poulpy_hal::{
        api::{ModuleN, ScratchOwnedAlloc},
        layouts::{Backend, ConjugateInvariant, HostBytesBackend, Module, ScratchOwned},
    };

    /// The limbs of the plaintext and the values it decodes to.
    type Encoded = (Vec<u8>, Vec<f64>);

    fn encode<BE>(module: &Module<BE>, slots: usize, re: &[f64]) -> Encoded
    where
        BE: Backend<Ring = ConjugateInvariant, ZnxWord = i64>,
        Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, f64> + CKKSEncodingHostOps<BE, f64>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    {
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(module.n()));
        let mut pt = module.ckks_pt_vec_alloc(slots, 19usize.into(), 76usize.into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(40),
            ..Default::default()
        });
        let im = vec![0.25; slots];
        module.ckks_encode_reim_into(&mut pt, re, &im, &mut scratch.arena()).unwrap();
        assert_eq!(pt.slots(), SlotsKind::Real);
        let (mut got_re, mut got_im) = (vec![0.0; slots], vec![1.0; slots]);
        module
            .ckks_decode_reim_into(&pt, &mut got_re, &mut got_im, &mut scratch.arena())
            .unwrap();
        assert!(got_im.iter().all(|&value| value == 0.0));
        let host = pt.to_host_owned::<BE>();
        let limbs = GLWEToBackendRef::<HostBytesBackend<ConjugateInvariant>>::to_backend_ref(&host);
        (limbs.data().data().to_vec(), got_re)
    }

    let n = 64;
    let device = Module::<crate::NTT4x30Gpu<super::Runtime, ConjugateInvariant>>::new(n);
    let portable = Module::<poulpy_cpu_portable::NTT4x30CIPortable>::new(n);
    // A vector plaintext has the degree of its slots on this ring, at least the backend's minimum.
    let min_log_slots = <crate::NTT4x30Gpu<super::Runtime, ConjugateInvariant> as Backend>::MIN_DEGREE.ilog2();
    for log_slots in min_log_slots..=n.ilog2() {
        let slots = 1usize << log_slots;
        let re: Vec<f64> = (0..slots).map(|i| ((17 * i + 3) % 31) as f64 / 37.0).collect();
        let (have, have_re) = encode(&device, slots, &re);
        let (want, want_re) = encode(&portable, slots, &re);
        assert!(have == want, "plaintexts differ at {slots} slots");
        assert_eq!(have_re, want_re, "decoded values differ at {slots} slots");
        for (got, expected) in have_re.iter().zip(&re) {
            assert!((got - expected).abs() < 1e-8);
        }
    }
}
