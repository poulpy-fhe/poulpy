//! The PaCo and SHIP drivers enter the selected coefficient encoding, slot trace,
//! masking and mux hooks, their queries take the hooks' workspaces one for one, and their outputs equal
//! the oracle's bit for bit on the same keys and inputs.
use super::OverrideBackend;
use super::carry::{assert_query_takes_extra, with_marked_bytes};
use poulpy_ckks::{
    CKKSCompositionError, CKKSCtBounds,
    api::{PaCoScalar, ShipScalar},
    layouts::{
        CKKSCiphertextOwned, CKKSPlaintextOwned, HMuxRotKeyPrepared, PaCoContext, PaCoKeys, PaCoWorker, ShipKeysPrepared,
        ShipPlan,
    },
    oep::CKKSEncodingImpl,
    reference::{paco::CKKSPaCoReference, ship::CKKSShipReference},
    test_suite::{
        BASE19_PARAMS_F64,
        helpers::{assert_ckks_error, assert_ckks_internal},
        parity::helpers::{poisoned_scratch, snapshot, with_scratch},
    },
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos},
};
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, CnvPVecL, HostBytesBackend, Module, ScratchArena, ScratchOwned, galois_element},
    source::Source,
};
use std::{
    collections::HashMap,
    sync::{
        Mutex, MutexGuard, PoisonError,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
};

// PaCo branches run on worker threads, so the probe state is static and its tests take turns.
static PROBE: Mutex<()> = Mutex::new(());
static PACO_ENCODING_CALLS: AtomicUsize = AtomicUsize::new(0);
static SLOT_TRACE_CALLS: AtomicUsize = AtomicUsize::new(0);
static SHIP_MASKING_CALLS: AtomicUsize = AtomicUsize::new(0);
static SHIP_MUX_CALLS: AtomicUsize = AtomicUsize::new(0);
const PACO_ENCODING_DEFAULT: usize = 4352;
const SLOT_TRACE_DEFAULT: usize = 5120;
const SHIP_MASKING_DEFAULT: usize = 4608;
const SHIP_MUX_DEFAULT: usize = 4864;
static PACO_ENCODING_EXTRA: AtomicUsize = AtomicUsize::new(PACO_ENCODING_DEFAULT);
static SLOT_TRACE_EXTRA: AtomicUsize = AtomicUsize::new(SLOT_TRACE_DEFAULT);
static SHIP_MASKING_EXTRA: AtomicUsize = AtomicUsize::new(SHIP_MASKING_DEFAULT);
static SHIP_MUX_EXTRA: AtomicUsize = AtomicUsize::new(SHIP_MUX_DEFAULT);

/// Takes the probe and restores its default workspaces.
fn probe() -> MutexGuard<'static, ()> {
    let guard = PROBE.lock().unwrap_or_else(PoisonError::into_inner);
    PACO_ENCODING_EXTRA.store(PACO_ENCODING_DEFAULT, Relaxed);
    SLOT_TRACE_EXTRA.store(SLOT_TRACE_DEFAULT, Relaxed);
    SHIP_MASKING_EXTRA.store(SHIP_MASKING_DEFAULT, Relaxed);
    SHIP_MUX_EXTRA.store(SHIP_MUX_DEFAULT, Relaxed);
    guard
}

/// Runs `call` with the hook counters zeroed and returns them, in the order given.
fn count<const N: usize, R>(counters: [&AtomicUsize; N], call: impl FnOnce() -> R) -> (R, [usize; N]) {
    counters.iter().for_each(|calls| calls.store(0, Relaxed));
    let result = call();
    (result, counters.map(|calls| calls.load(Relaxed)))
}

unsafe impl poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl for OverrideBackend {
    fn ckks_paco_coeff_encodings_tmp_bytes<F>(
        _module: &Module<Self>,
        _context: &PaCoContext<Self, F>,
    ) -> poulpy_ckks::CKKSResult<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
    {
        // The staged encoding it forwards to takes no scratch.
        Ok(PACO_ENCODING_EXTRA.load(Relaxed))
    }

    fn ckks_paco_coeff_encodings<F, Src>(
        module: &Module<Self>,
        ciphertext: &Src,
        context: &PaCoContext<Self, F>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<[CKKSPlaintextOwned<Self>; 4]>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        PACO_ENCODING_CALLS.fetch_add(1, Relaxed);
        with_marked_bytes(PACO_ENCODING_EXTRA.load(Relaxed), scratch, |_| {
            crate::ckks_paco::paco_coeff_encodings_staged::<Self, F, Src>(module, context.base2k(), ciphertext, context.plan())
                .map_err(poulpy_ckks::CKKSError::from)
        })
    }
}
// The slot trace override runs its own composition, one fused automorphism-add per fold level, behind its own extra
// workspace; the other methods forward to the references.
unsafe impl poulpy_ckks::oep::CKKSPaCoImpl for OverrideBackend {
    fn ckks_paco_bootstrap_direct_tmp_bytes<F, K, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
    ) -> poulpy_ckks::CKKSResult<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_paco_bootstrap_direct_tmp_bytes_reference(output, input, context, keys)
    }

    fn ckks_paco_bootstrap_tmp_bytes<F, K, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
    ) -> poulpy_ckks::CKKSResult<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_paco_bootstrap_tmp_bytes_reference(output, input, context, keys)
    }

    fn ckks_paco_bootstrap_direct_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_paco_bootstrap_direct_into_reference(output, input, context, keys, scratch)
    }

    fn ckks_paco_bootstrap_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_paco_bootstrap_into_reference(output, input, context, keys, scratch)
    }

    fn ckks_paco_bootstrap_parallel_direct_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        workers: &mut [PaCoWorker<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self> + Sync,
        ScratchOwned<Self>: ScratchOwnedBorrow<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
    {
        module.ckks_paco_bootstrap_parallel_direct_into_reference(output, input, context, keys, workers, scratch)
    }

    fn ckks_paco_bootstrap_parallel_into<F, K, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        context: &PaCoContext<Self, F>,
        keys: &K,
        workers: &mut [PaCoWorker<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        K: PaCoKeys<Self> + Sync,
        ScratchOwned<Self>: ScratchOwnedBorrow<Self>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_paco_bootstrap_parallel_into_reference(output, input, context, keys, workers, scratch)
    }

    fn ckks_slot_product_assign_tmp_bytes<C, K, T>(module: &Module<Self>, ct_infos: &C, key_infos: &K, tsk_infos: &T) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        module.ckks_slot_product_assign_tmp_bytes_reference(ct_infos, key_infos, tsk_infos)
    }

    fn ckks_slot_product_assign<Dst, H, TH>(
        module: &Module<Self>,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        tsk: &TH,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + poulpy_ckks::SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
        TH: GetTensorKey<Self>,
    {
        module.ckks_slot_product_assign_reference(ct, a, b, keys, tsk, scratch)
    }

    fn ckks_slot_trace_assign_tmp_bytes<C, K>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        SLOT_TRACE_EXTRA.load(Relaxed)
            + poulpy_core::GLWEAutomorphism::glwe_automorphism_tmp_bytes(module, ct_infos, ct_infos, key_infos)
    }

    fn ckks_slot_trace_assign<Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        a: usize,
        b: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + poulpy_ckks::SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        use poulpy_hal::layouts::CyclotomicOrder;
        SLOT_TRACE_CALLS.fetch_add(1, Relaxed);
        if !(a.is_power_of_two() && b.is_power_of_two() && b <= a && a <= module.n() / 2) {
            return Err(anyhow::anyhow!("invalid PaCo fold").into());
        }
        with_marked_bytes(SLOT_TRACE_EXTRA.load(Relaxed), scratch, |rest| {
            for level in 0..(a / b).trailing_zeros() {
                let rotation = (a >> (level + 1)) as i64;
                let key = keys
                    .get_automorphism_key(galois_element(rotation, module.cyclotomic_order()), ct.k())
                    .map_err(|source| poulpy_ckks::CKKSCompositionError::MissingAutomorphismKey {
                        op: "ckks_slot_trace_assign",
                        rotation,
                        k: ct.k().into(),
                        source,
                    })?;
                poulpy_core::GLWEAutomorphism::glwe_automorphism_add_assign(module, ct, &key, rest);
            }
            Ok(())
        })
    }
}
crate::impl_ckks_ship_coeff_encoding!(OverrideBackend);

unsafe impl poulpy_ckks::oep::CKKSShipImpl for OverrideBackend {
    fn ckks_ship_bootstrap_tmp_bytes<F, Src>(
        module: &Module<Self>,
        output: &CKKSCiphertextOwned<Self>,
        input: &Src,
        keys: &ShipKeysPrepared<<Self as Backend>::OwnedBuf, Self>,
    ) -> poulpy_ckks::CKKSResult<usize>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_ship_bootstrap_tmp_bytes_reference::<F, Src>(output, input, keys)
    }

    fn ckks_ship_bootstrap_into<F, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        keys: &ShipKeysPrepared<<Self as Backend>::OwnedBuf, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_ship_bootstrap_into_reference::<F, Src>(output, input, keys, scratch)
    }

    fn ckks_ship_bootstrap_complex_into<F, Src>(
        module: &Module<Self>,
        output: &mut CKKSCiphertextOwned<Self>,
        input: &Src,
        keys: &ShipKeysPrepared<<Self as Backend>::OwnedBuf, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        F: ShipScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        module.ckks_ship_bootstrap_complex_into_reference::<F, Src>(output, input, keys, scratch)
    }

    fn ckks_ship_masking_accumulate_tmp_bytes(module: &Module<Self>, plan: &ShipPlan, base2k: usize) -> usize {
        SHIP_MASKING_EXTRA.load(Relaxed) + module.ckks_ship_masking_accumulate_tmp_bytes_reference(plan, base2k)
    }

    fn ckks_ship_masking_accumulate(
        module: &Module<Self>,
        acc: &mut CKKSCiphertextOwned<Self>,
        plan: &ShipPlan,
        masks: &[CnvPVecL<<Self as Backend>::OwnedBuf, <Self as Backend>::DftWord, Self>],
        pis: &[CKKSPlaintextOwned<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()> {
        SHIP_MASKING_CALLS.fetch_add(1, Relaxed);
        with_marked_bytes(SHIP_MASKING_EXTRA.load(Relaxed), scratch, |rest| {
            module.ckks_ship_masking_accumulate_reference(acc, plan, masks, pis, rest)
        })
    }

    fn ckks_ship_mux_rotate_tmp_bytes<C, K>(module: &Module<Self>, ct: &C, key: &K, term_count: usize) -> usize
    where
        C: LWEInfos,
        K: GGLWEInfos,
    {
        SHIP_MUX_EXTRA.load(Relaxed) + module.ckks_ship_mux_rotate_tmp_bytes_reference(ct, key, term_count)
    }

    fn ckks_ship_mux_rotate(
        module: &Module<Self>,
        ct: &mut CKKSCiphertextOwned<Self>,
        keys: &[HMuxRotKeyPrepared<<Self as Backend>::OwnedBuf, Self>],
        plans: &HashMap<i64, <Self as poulpy_hal::oep::HalVecZnxDftImpl>::AutomorphismPlan>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Self: poulpy_hal::oep::HalVecZnxDftImpl,
    {
        SHIP_MUX_CALLS.fetch_add(1, Relaxed);
        with_marked_bytes(SHIP_MUX_EXTRA.load(Relaxed), scratch, |rest| {
            module.ckks_ship_mux_rotate_reference(ct, keys, plans, rest)
        })
    }
}

/// Key material and inputs are built once on the oracle and shared with the probe. Each PaCo driver, sequential and
/// parallel, direct and encapsulated, enters the coefficient encoding and the slot trace once per branch on its exact
/// poisoned query, a parallel worker on exactly the branch share, the queries take both hooks' workspaces one for one,
/// the outputs equal the oracle's sequential ones bit for bit, and a refused input, arena or worker enters no hook.
#[test]
fn paco_drivers_enter_the_selected_coefficient_encoding() {
    use poulpy_ckks::{
        CKKSInfos, SetCKKSInfos,
        api::CKKSPaCoOps,
        layouts::{CKKSModuleAlloc, PaCoDFTPlan, PaCoKeySet, PaCoPlan, PaCoSecretSpec, PaCoWorker},
        test_suite::{
            CKKSTestParams,
            helpers::{alloc_scratch, ckks_encrypt, ckks_encrypt_coeffs, ckks_spec, gen_sk_with_raw},
            parity::helpers::Snapshot,
            reference_encoder::ReferenceEncoder,
        },
    };
    use poulpy_core::{
        GLWEAutomorphismKeyEncryptSk, GLWEBytesOf, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, TransferInto,
        layouts::{ModuleCoreAlloc, SetGLWEPlaintextInfos},
    };
    type Out = CKKSCiphertextOwned<OverrideBackend>;
    type Run<'a> = &'a mut dyn FnMut(&mut Out, &mut ScratchArena<'_, OverrideBackend>) -> poulpy_ckks::CKKSResult<()>;
    let _probe = probe();

    let base = BASE19_PARAMS_F64;
    let log_delta = base.prec().log_delta();
    let k_in = (log_delta - 2).min(52);
    let log_msg = k_in - 10;
    let kappa = 2usize;
    let plan = PaCoPlan::new(base.n.trailing_zeros() as usize, 4, 8, k_in as u32)
        .unwrap()
        .with_evaluation(
            log_delta,
            16,
            PaCoDFTPlan::new(vec![4, 1], vec![2, 2], log_delta, 16, 1.0).unwrap(),
            PaCoDFTPlan::uniform(3, 2, 2, log_delta, 16).unwrap(),
        )
        .unwrap();
    let k_boot = plan.k_boot(base.base2k, log_msg + 8).unwrap();
    let params = CKKSTestParams { k: k_boot, ..base };

    let module_ref = Module::<FFT64Oracle>::new(params.n as u64);
    let module_test = Module::<OverrideBackend>::new(params.n as u64);
    let host = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder_full = ReferenceEncoder::<crate::ckks_encoding::EncodingFFTTable<f64>>::new::<f64>(params.n / 2).unwrap();
    let mut scratch = alloc_scratch(&params, &module_ref);

    let (sk_dense_raw, sk_dense) = gen_sk_with_raw(&params, &module_ref, &host, [21u8; 32]);
    let spec = PaCoSecretSpec::sample(&plan, &mut Source::new([22u8; 32])).unwrap();
    let glwe_infos = params.glwe_layout();
    let mut sk_paco_host = host.glwe_secret_alloc_from_infos(&glwe_infos);
    spec.fill_glwe_secret(&plan, &mut sk_paco_host).unwrap();
    let mut sk_paco_raw = module_ref.glwe_secret_alloc_from_infos(&sk_paco_host);
    sk_paco_host.transfer_into(&mut sk_paco_raw);

    let (mut xa, mut xe) = (Source::new([1u8; 32]), Source::new([2u8; 32]));
    let mut rotation_keys = HashMap::new();
    for element in plan.galois_elements() {
        let mut atk = module_ref.glwe_automorphism_key_alloc_from_infos(&params.atk_layout());
        module_ref.glwe_automorphism_key_encrypt_sk(&mut atk, element, &sk_dense_raw, &mut xe, &mut xa, &mut scratch.borrow());
        rotation_keys.insert(element, atk);
    }
    let mut tensor_key = module_ref.glwe_tensor_key_alloc_from_infos(&params.tsk_layout());
    module_ref.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk_dense_raw, &mut xe, &mut xa, &mut scratch.borrow());
    let mut dense_to_paco = module_ref.glwe_switching_key_alloc_from_infos(&params.ksk_layout(k_in));
    module_ref.glwe_switching_key_encrypt_sk(
        &mut dense_to_paco,
        &sk_dense_raw,
        &sk_paco_raw,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    );
    // Any four ciphertexts at the bootstrapping-key scale and width serve a bit-exact comparison.
    let bsk = std::array::from_fn(|t| {
        let re: Vec<f64> = (0..params.n / 2).map(|i| ((i + t) % 7) as f64 / 16.0).collect();
        let im = vec![0.0f64; params.n / 2];
        ckks_encrypt(
            &params,
            &module_ref,
            &host,
            &encoder_full,
            &sk_dense,
            k_boot,
            &re,
            &im,
            &mut scratch.borrow(),
        )
    });
    let keyset = PaCoKeySet::new(&plan, bsk, rotation_keys, tensor_key, Some(dense_to_paco)).unwrap();

    let coeffs: Vec<f64> = (0..params.n)
        .map(|i| 0.4 * (((i.wrapping_mul(2654435761) % 1024) as f64) / 512.0 - 1.0))
        .collect();
    let mut ct_in = ckks_encrypt_coeffs(
        &params,
        &module_ref,
        &host,
        &sk_dense,
        k_in,
        &coeffs,
        ckks_spec(params.n, params.base2k, log_msg, 10),
        &mut scratch.borrow(),
    );
    ct_in.set_log_sparsity((plan.n() / (kappa * plan.c())).trailing_zeros() as usize);

    // The oracle's sequential outputs, which every driver of the probe must reproduce.
    let keys_ref = keyset.prepare(&plan, &module_ref, &mut scratch.borrow()).unwrap();
    let ctx_ref =
        PaCoContext::<FFT64Oracle, f64>::compile(&module_ref, params.base2k.into(), plan.clone(), &mut scratch.borrow()).unwrap();
    let out_k = ctx_ref.max_output_k(&keys_ref).unwrap();
    let oracle = |direct: bool| -> Snapshot {
        let mut out = module_ref.ckks_ciphertext_alloc(ctx_ref.base2k(), out_k);
        let bytes = if direct {
            module_ref.ckks_paco_bootstrap_direct_tmp_bytes(&out, &ct_in, &ctx_ref, &keys_ref)
        } else {
            module_ref.ckks_paco_bootstrap_tmp_bytes(&out, &ct_in, &ctx_ref, &keys_ref)
        }
        .unwrap();
        with_scratch::<FFT64Oracle, _>(bytes, |s| {
            if direct {
                module_ref.ckks_paco_bootstrap_direct_into(&mut out, &ct_in, &ctx_ref, &keys_ref, s)
            } else {
                module_ref.ckks_paco_bootstrap_into(&mut out, &ct_in, &ctx_ref, &keys_ref, s)
            }
        })
        .unwrap();
        snapshot::<FFT64Oracle, _>(&out)
    };
    let (want_encapsulated, want_direct) = (oracle(false), oracle(true));

    let mut setup = alloc_scratch(&params, &module_test);
    let keys = keyset.prepare(&plan, &module_test, &mut setup.borrow()).unwrap();
    let ctx = PaCoContext::<OverrideBackend, f64>::compile(&module_test, params.base2k.into(), plan.clone(), &mut setup.borrow())
        .unwrap();
    assert_eq!(ctx.max_output_k(&keys).unwrap(), out_k);
    // A destination holding other digits, so a write or a refused call that writes shows.
    let dirty = || -> Out {
        let mut out = module_test.ckks_ciphertext_alloc(ctx.base2k(), out_k);
        module_test.fill_glwe_from_source(&mut out, &mut Source::new([5u8; 32]));
        out
    };
    let out = dirty();
    for (label, extra, default) in [
        ("coefficient encoding", &PACO_ENCODING_EXTRA, PACO_ENCODING_DEFAULT),
        ("slot trace", &SLOT_TRACE_EXTRA, SLOT_TRACE_DEFAULT),
    ] {
        let set = |bytes| extra.store(bytes, Relaxed);
        assert_query_takes_extra(&format!("ckks_paco_bootstrap_direct_tmp_bytes, {label}"), set, || {
            module_test
                .ckks_paco_bootstrap_direct_tmp_bytes(&out, &ct_in, &ctx, &keys)
                .unwrap()
        });
        assert_query_takes_extra(&format!("ckks_paco_bootstrap_tmp_bytes, {label}"), set, || {
            module_test.ckks_paco_bootstrap_tmp_bytes(&out, &ct_in, &ctx, &keys).unwrap()
        });
        extra.store(default, Relaxed);
    }
    let encapsulated_bytes = module_test.ckks_paco_bootstrap_tmp_bytes(&out, &ct_in, &ctx, &keys).unwrap();
    let direct_bytes = module_test
        .ckks_paco_bootstrap_direct_tmp_bytes(&out, &ct_in, &ctx, &keys)
        .unwrap();
    // A worker runs one branch, without the branch output the sequential driver carves.
    let branch_bytes = direct_bytes - OverrideBackend::scratch_aligned(module_test.glwe_bytes_of_from_infos(&out));
    let worker = |bytes| {
        [PaCoWorker::new(
            Module::<OverrideBackend>::new(params.n as u64),
            poisoned_scratch::<OverrideBackend>(bytes),
        )]
    };
    let hooks = [&PACO_ENCODING_CALLS, &SLOT_TRACE_CALLS];

    let mut drivers: [(&str, &Snapshot, usize, Run<'_>); 4] = [
        (
            "ckks_paco_bootstrap_into",
            &want_encapsulated,
            encapsulated_bytes,
            &mut |out, s| module_test.ckks_paco_bootstrap_into(out, &ct_in, &ctx, &keys, s),
        ),
        (
            "ckks_paco_bootstrap_direct_into",
            &want_direct,
            direct_bytes,
            &mut |out, s| module_test.ckks_paco_bootstrap_direct_into(out, &ct_in, &ctx, &keys, s),
        ),
        (
            "ckks_paco_bootstrap_parallel_into",
            &want_encapsulated,
            encapsulated_bytes,
            &mut |out, s| module_test.ckks_paco_bootstrap_parallel_into(out, &ct_in, &ctx, &keys, &mut worker(branch_bytes), s),
        ),
        (
            "ckks_paco_bootstrap_parallel_direct_into",
            &want_direct,
            direct_bytes,
            &mut |out, s| {
                module_test.ckks_paco_bootstrap_parallel_direct_into(out, &ct_in, &ctx, &keys, &mut worker(branch_bytes), s)
            },
        ),
    ];
    for (label, want, bytes, run) in drivers.iter_mut() {
        let mut out = dirty();
        let unchanged = snapshot::<OverrideBackend, _>(&out);
        let (result, calls) = count(hooks, || with_scratch::<OverrideBackend, _>(*bytes - 1, |s| run(&mut out, s)));
        assert_ckks_internal(
            &format!("{label}: one byte short"),
            &result.unwrap_err(),
            &format!(
                "PaCo bootstrap needs {bytes} scratch bytes, but only {} are available",
                *bytes - 1
            ),
        );
        assert_eq!(calls, [0, 0], "{label}: a refused arena entered a hook");
        assert_eq!(
            snapshot::<OverrideBackend, _>(&out),
            unchanged,
            "{label}: a refused arena wrote the output"
        );
        let (result, calls) = count(hooks, || with_scratch::<OverrideBackend, _>(*bytes, |s| run(&mut out, s)));
        result.unwrap();
        assert_eq!(calls, [kappa, kappa], "{label}: hook calls");
        assert_eq!(
            &snapshot::<OverrideBackend, _>(&out),
            *want,
            "{label}: output differs from the oracle's"
        );
    }

    // An input at another width than the plan's modulus is refused before any scratch, encoding or trace.
    let mut wide = module_test.ckks_ciphertext_alloc(params.base2k.into(), (k_in + 1).into());
    SetCKKSInfos::set_meta(&mut wide, ct_in.meta());
    let width = format!("PaCo input torus width {} does not match plan log_q {k_in}", k_in + 1);
    let mut refusals: [(&str, Run<'_>); 4] = [
        ("ckks_paco_bootstrap_into", &mut |out, s| {
            module_test.ckks_paco_bootstrap_into(out, &wide, &ctx, &keys, s)
        }),
        ("ckks_paco_bootstrap_direct_into", &mut |out, s| {
            module_test.ckks_paco_bootstrap_direct_into(out, &wide, &ctx, &keys, s)
        }),
        ("ckks_paco_bootstrap_parallel_into", &mut |out, s| {
            module_test.ckks_paco_bootstrap_parallel_into(out, &wide, &ctx, &keys, &mut worker(branch_bytes), s)
        }),
        ("ckks_paco_bootstrap_parallel_direct_into", &mut |out, s| {
            module_test.ckks_paco_bootstrap_parallel_direct_into(out, &wide, &ctx, &keys, &mut worker(branch_bytes), s)
        }),
    ];
    for (label, run) in refusals.iter_mut() {
        let mut out = dirty();
        let unchanged = snapshot::<OverrideBackend, _>(&out);
        let (result, calls) = count(hooks, || with_scratch::<OverrideBackend, _>(0, |s| run(&mut out, s)));
        assert_ckks_error(
            &format!("{label}: wide input"),
            &result.unwrap_err(),
            CKKSCompositionError::InvalidOperand {
                op: label,
                reason: width.clone(),
            },
        );
        assert_eq!(calls, [0, 0], "{label}: a refused input entered a hook");
        assert_eq!(
            snapshot::<OverrideBackend, _>(&out),
            unchanged,
            "{label}: a refused input wrote the output"
        );
    }

    // Owned arenas round up to the alignment, so the short worker lacks one alignment unit.
    let short = branch_bytes - OverrideBackend::SCRATCH_ALIGN;
    let mut refusals: [(&str, usize, Run<'_>); 2] = [
        ("ckks_paco_bootstrap_parallel_into", encapsulated_bytes, &mut |out, s| {
            module_test.ckks_paco_bootstrap_parallel_into(out, &ct_in, &ctx, &keys, &mut worker(short), s)
        }),
        ("ckks_paco_bootstrap_parallel_direct_into", direct_bytes, &mut |out, s| {
            module_test.ckks_paco_bootstrap_parallel_direct_into(out, &ct_in, &ctx, &keys, &mut worker(short), s)
        }),
    ];
    for (label, bytes, run) in refusals.iter_mut() {
        let mut out = dirty();
        let unchanged = snapshot::<OverrideBackend, _>(&out);
        let (result, calls) = count(hooks, || with_scratch::<OverrideBackend, _>(*bytes, |s| run(&mut out, s)));
        assert_ckks_internal(
            &format!("{label}: short worker"),
            &result.unwrap_err(),
            &format!("PaCo worker 1 needs {branch_bytes} scratch bytes, but only {short} are available"),
        );
        assert_eq!(calls, [0, 0], "{label}: a refused worker entered a hook");
        assert_eq!(
            snapshot::<OverrideBackend, _>(&out),
            unchanged,
            "{label}: a refused worker wrote the output"
        );
    }
}

/// Key material and inputs are built once on the probe and shared with the oracle. The real and complex SHIP drivers,
/// each on its own key layout, enter the masking hook once per index key and coefficient half and the mux hook once
/// per digit group, index key and half, on their exact poisoned query, which takes both hooks' workspaces one for
/// one, their outputs equal the oracle's bit for bit, and an input wider than one limb or an arena one byte short is
/// refused before either hook.
#[test]
fn ship_drivers_enter_the_selected_masking_and_mux() {
    use poulpy_ckks::layouts::CKKSPlaintextVecHostCodec;
    use poulpy_ckks::{
        CKKSInfos, CKKSModuleInfos, SetCKKSInfos,
        api::CKKSShipOps,
        layouts::{CKKSModuleAlloc, ShipKeySet, ShipKeysLayout, ShipSecretSpec},
        test_suite::{
            CKKSTestParams,
            helpers::{alloc_ct, alloc_scratch, ckks_encrypt_pt, ckks_spec, gen_sk_with_raw},
        },
    };
    use poulpy_core::{TransferInto, layouts::ModuleCoreAlloc};
    let _probe = probe();

    let params = BASE19_PARAMS_F64;
    let (n, base2k) = (params.n, params.base2k);
    let plan = ShipPlan::new(
        params.log_n(),
        6,
        params.prec().log_delta().min(40),
        2 * base2k,
        32,
        (n / 32) / 2,
        4,
        4,
    )
    .unwrap();
    let kk = plan.raised_k(base2k);
    let big = CKKSTestParams { k: kk, ..params };

    let module_ref = Module::<FFT64Oracle>::new(n as u64);
    let module_test = Module::<OverrideBackend>::new(n as u64);
    let host = Module::<HostBytesBackend>::new(n as u64);
    let mut scratch = alloc_scratch(&big, &module_test);
    let (sk_raw, sk) = gen_sk_with_raw(&big, &module_test, &host, [0u8; 32]);
    let mut sk_host = host.glwe_secret_alloc_from_infos(&big.glwe_layout());
    sk_raw.transfer_into(&mut sk_host);
    let spec = ShipSecretSpec::sample(&plan, &mut Source::new([9u8; 32]));
    let gamma = 2f64.powi(plan.log_gamma() as i32);
    let hooks = [&SHIP_MASKING_CALLS, &SHIP_MUX_CALLS];

    for complex in [false, true] {
        let op = if complex {
            "ckks_ship_bootstrap_complex_into"
        } else {
            "ckks_ship_bootstrap_into"
        };
        let keys_layout = ShipKeysLayout {
            mux_dsize: params.dsize,
            tensor_key: big.tsk_layout(),
            conjugation_key: big.atk_layout(),
            complex,
        };
        let key_set = ShipKeySet::generate::<OverrideBackend, f64>(
            &module_test,
            &host,
            &plan,
            base2k.into(),
            &spec,
            &sk_host,
            &keys_layout,
            &mut Source::new([11u8; 32]),
            &mut Source::new([12u8; 32]),
            &mut scratch.borrow(),
        )
        .unwrap();

        // Real cleartexts in the first half of the coefficients, their imaginary parts in the second when complex.
        let coeffs: Vec<f64> = (0..n)
            .map(|i| match (i < n / 2, complex) {
                (true, _) => ((i as f64 * 0.7639) % 1.0 - 0.5) * 0.9 / gamma,
                (false, true) => ((i as f64 * 0.4117 + 0.31) % 1.0 - 0.5) * 0.9 / gamma,
                (false, false) => 0.0,
            })
            .collect();
        let bottom = ckks_spec(n, base2k, base2k, 0);
        let mut host_pt = host.ckks_pt_vec_alloc(host.ckks_max_slots(), base2k.into(), bottom.k());
        host_pt.set_meta(bottom.meta());
        host_pt.encode_host_floats(&coeffs).unwrap();
        let ct0 = ckks_encrypt_pt(&big, &module_test, &sk, base2k, &host_pt, &mut scratch.borrow());

        macro_rules! bootstrap {
            ($module:expr, $be:ty, $keys:expr, $out:expr, $input:expr, $bytes:expr) => {
                with_scratch::<$be, _>($bytes, |s| {
                    if complex {
                        CKKSShipOps::<$be, f64>::ckks_ship_bootstrap_complex_into($module, $out, $input, $keys, s)
                    } else {
                        CKKSShipOps::<$be, f64>::ckks_ship_bootstrap_into($module, $out, $input, $keys, s)
                    }
                })
            };
        }
        let keys_ref = key_set
            .prepare(&module_ref, &mut alloc_scratch(&big, &module_ref).borrow())
            .unwrap();
        let mut want = alloc_ct(&big, &module_ref, kk);
        let bytes = CKKSShipOps::<FFT64Oracle, f64>::ckks_ship_bootstrap_tmp_bytes(&module_ref, &want, &ct0, &keys_ref).unwrap();
        bootstrap!(&module_ref, FFT64Oracle, &keys_ref, &mut want, &ct0, bytes).unwrap();

        let keys = key_set.prepare(&module_test, &mut scratch.borrow()).unwrap();
        let mut out = alloc_ct(&big, &module_test, kk);
        module_test.fill_glwe_from_source(&mut out, &mut Source::new([5u8; 32]));
        let unchanged = snapshot::<OverrideBackend, _>(&out);
        for (label, extra, default) in [
            ("masking", &SHIP_MASKING_EXTRA, SHIP_MASKING_DEFAULT),
            ("mux", &SHIP_MUX_EXTRA, SHIP_MUX_DEFAULT),
        ] {
            assert_query_takes_extra(
                &format!("{op}, {label}"),
                |bytes| extra.store(bytes, Relaxed),
                || CKKSShipOps::<OverrideBackend, f64>::ckks_ship_bootstrap_tmp_bytes(&module_test, &out, &ct0, &keys).unwrap(),
            );
            extra.store(default, Relaxed);
        }
        let bytes = CKKSShipOps::<OverrideBackend, f64>::ckks_ship_bootstrap_tmp_bytes(&module_test, &out, &ct0, &keys).unwrap();

        // An input wider than one limb is refused before any scratch, masking or mux.
        let wide = alloc_ct(&big, &module_test, 2 * base2k);
        let (result, calls) = count(hooks, || bootstrap!(&module_test, OverrideBackend, &keys, &mut out, &wide, 0));
        assert_ckks_error(
            &format!("{op}: wide input"),
            &result.unwrap_err(),
            CKKSCompositionError::InvalidOperand {
                op,
                reason: format!(
                    "{op}: input must span a single limb of base2k {base2k}, got base2k {base2k} and width {}",
                    2 * base2k
                ),
            },
        );
        assert_eq!(calls, [0, 0], "{op}: a refused input entered a hook");
        assert_eq!(
            snapshot::<OverrideBackend, _>(&out),
            unchanged,
            "{op}: a refused input wrote the output"
        );
        let (result, calls) = count(hooks, || {
            bootstrap!(&module_test, OverrideBackend, &keys, &mut out, &ct0, bytes - 1)
        });
        assert_ckks_internal(
            &format!("{op}: one byte short"),
            &result.unwrap_err(),
            &format!("{op}: scratch below its query"),
        );
        assert_eq!(calls, [0, 0], "{op}: a refused arena entered a hook");
        assert_eq!(
            snapshot::<OverrideBackend, _>(&out),
            unchanged,
            "{op}: a refused arena wrote the output"
        );

        let (result, calls) = count(hooks, || {
            bootstrap!(&module_test, OverrideBackend, &keys, &mut out, &ct0, bytes)
        });
        result.unwrap();
        let halves = if complex { 2 } else { 1 };
        let groups: usize = keys.index_keys().iter().map(|index| index.mux_keys().len()).sum();
        assert_eq!(calls, [halves * keys.index_keys().len(), halves * groups], "{op}: hook calls");
        assert_eq!(
            snapshot::<OverrideBackend, _>(&out),
            snapshot::<FFT64Oracle, _>(&want),
            "{op}: output differs from the oracle's"
        );
    }
}
