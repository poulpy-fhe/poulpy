//! The PaCo and SHIP drivers enter the selected coefficient encoding, masking and
//! mux hooks, their queries carry the hooks' workspaces, and their outputs equal
//! the portable backend's bit for bit on the same keys and inputs.
use super::OverrideBackend;
use super::carry::with_marked_region;
use crate::{FFT64Portable, ckks_encoding::EncodingFFTTable};
use poulpy_ckks::{
    CKKSCtBounds,
    api::PaCoScalar,
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, HMuxRotKeyPrepared, PaCoContext, ShipPlan},
    oep::CKKSEncodingImpl,
    reference::ship::CKKSShipReference,
    test_suite::{
        BASE19_PARAMS_F64,
        parity::helpers::{snapshot, with_scratch},
    },
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, CnvPVecL, HostBytesBackend, Module, ScratchArena},
    source::Source,
};
use std::{cell::Cell, collections::HashMap};

thread_local! {
    static PACO_ENCODING_CALLS: Cell<usize> = const { Cell::new(0) };
    static SHIP_MASKING_CALLS: Cell<usize> = const { Cell::new(0) };
    static SHIP_MUX_CALLS: Cell<usize> = const { Cell::new(0) };
}
const PACO_ENCODING_EXTRA: usize = 4352;
const SHIP_MASKING_EXTRA: usize = 4608;
const SHIP_MUX_EXTRA: usize = 4864;

unsafe impl poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl for OverrideBackend {
    fn ckks_paco_coeff_encodings_tmp_bytes<F>(
        _module: &Module<Self>,
        _context: &PaCoContext<Self, F>,
    ) -> poulpy_ckks::CKKSResult<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
    {
        Ok(PACO_ENCODING_EXTRA)
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
        PACO_ENCODING_CALLS.set(PACO_ENCODING_CALLS.get() + 1);
        with_marked_region::<PACO_ENCODING_EXTRA, _>(scratch, |rest| {
            crate::ckks_paco::paco_coeff_encodings_staged::<Self, F, Src>(
                module,
                ciphertext,
                context.plan(),
                context.base2k(),
                rest,
            )
            .map_err(poulpy_ckks::CKKSError::from)
        })
    }
}
poulpy_ckks::impl_ckks_paco_derived_full!(OverrideBackend);
crate::impl_ckks_ship_coeff_encoding!(OverrideBackend);

unsafe impl poulpy_ckks::oep::CKKSShipImpl for OverrideBackend {
    fn ckks_ship_masking_accumulate_tmp_bytes(module: &Module<Self>, plan: &ShipPlan, base2k: usize) -> usize {
        SHIP_MASKING_EXTRA + module.ckks_ship_masking_accumulate_tmp_bytes_reference(plan, base2k)
    }

    fn ckks_ship_masking_accumulate(
        module: &Module<Self>,
        acc: &mut CKKSCiphertextOwned<Self>,
        plan: &ShipPlan,
        masks: &[CnvPVecL<<Self as Backend>::OwnedBuf, <Self as Backend>::DftWord, Self>],
        pis: &[CKKSPlaintextOwned<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()> {
        SHIP_MASKING_CALLS.set(SHIP_MASKING_CALLS.get() + 1);
        with_marked_region::<SHIP_MASKING_EXTRA, _>(scratch, |rest| {
            module.ckks_ship_masking_accumulate_reference(acc, plan, masks, pis, rest)
        })
    }

    fn ckks_ship_mux_rotate_tmp_bytes<C, K>(module: &Module<Self>, ct: &C, key: &K, term_count: usize) -> usize
    where
        C: LWEInfos,
        K: GGLWEInfos,
    {
        SHIP_MUX_EXTRA + module.ckks_ship_mux_rotate_tmp_bytes_reference(ct, key, term_count)
    }

    fn ckks_ship_mux_rotate(
        module: &Module<Self>,
        ct: &mut CKKSCiphertextOwned<Self>,
        keys: &[HMuxRotKeyPrepared<<Self as Backend>::OwnedBuf, Self>],
        plans: &HashMap<i64, <Self as poulpy_hal::oep::HalVecZnxDftImpl>::AutomorphismPlan>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()> {
        SHIP_MUX_CALLS.set(SHIP_MUX_CALLS.get() + 1);
        with_marked_region::<SHIP_MUX_EXTRA, _>(scratch, |rest| module.ckks_ship_mux_rotate_reference(ct, keys, plans, rest))
    }
}

/// Key material and inputs are built once on the portable backend and shared with
/// the probe, so the two outputs must agree bit for bit.
#[test]
fn paco_drivers_enter_the_selected_coefficient_encoding() {
    use poulpy_ckks::{
        CKKSInfos,
        api::CKKSPaCoOps,
        layouts::{CKKSModuleAlloc, PaCoDFTPlan, PaCoKeySet, PaCoPlan, PaCoSecretSpec},
        test_suite::{
            CKKSTestParams,
            helpers::{alloc_scratch, ckks_encrypt, ckks_encrypt_coeffs, ckks_spec, gen_sk_with_raw},
            reference_encoder::ReferenceEncoder,
        },
    };
    use poulpy_core::{
        GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, TransferInto,
        layouts::{ModuleCoreAlloc, SetGLWEPlaintextInfos},
    };

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

    let reference = Module::<FFT64Portable>::new(params.n as u64);
    let module = Module::<OverrideBackend>::new(params.n as u64);
    let host = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder_full = ReferenceEncoder::<EncodingFFTTable<f64>>::new::<f64>(params.n / 2).unwrap();
    let mut scratch = alloc_scratch(&params, &reference);

    let (sk_dense_raw, sk_dense) = gen_sk_with_raw(&params, &reference, &host, [21u8; 32]);
    let spec = PaCoSecretSpec::sample(&plan, &mut Source::new([22u8; 32])).unwrap();
    let glwe_infos = params.glwe_layout();
    let mut sk_paco_host = host.glwe_secret_alloc_from_infos(&glwe_infos);
    spec.fill_glwe_secret(&plan, &mut sk_paco_host).unwrap();
    let mut sk_paco_raw = reference.glwe_secret_alloc_from_infos(&sk_paco_host);
    sk_paco_host.transfer_into(&mut sk_paco_raw);

    let (mut xa, mut xe) = (Source::new([1u8; 32]), Source::new([2u8; 32]));
    let mut rotation_keys = HashMap::new();
    for element in plan.galois_elements() {
        let mut atk = reference.glwe_automorphism_key_alloc_from_infos(&params.atk_layout());
        reference.glwe_automorphism_key_encrypt_sk(&mut atk, element, &sk_dense_raw, &mut xe, &mut xa, &mut scratch.borrow());
        rotation_keys.insert(element, atk);
    }
    let mut tensor_key = reference.glwe_tensor_key_alloc_from_infos(&params.tsk_layout());
    reference.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk_dense_raw, &mut xe, &mut xa, &mut scratch.borrow());
    let mut dense_to_paco = reference.glwe_switching_key_alloc_from_infos(&params.ksk_layout(k_in));
    reference.glwe_switching_key_encrypt_sk(
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
            &reference,
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
        &reference,
        &host,
        &sk_dense,
        k_in,
        &coeffs,
        ckks_spec(params.n, params.base2k, log_msg, 10),
        &mut scratch.borrow(),
    );
    ct_in.set_log_sparsity((plan.n() / (kappa * plan.c())).trailing_zeros() as usize);

    macro_rules! bootstrap {
        ($module:expr, $be:ty) => {{
            let module = $module;
            let mut setup = alloc_scratch(&params, module);
            let keys = keyset.prepare(&plan, module, &mut setup.borrow()).unwrap();
            let ctx = PaCoContext::<$be, f64>::compile(module, params.base2k.into(), plan.clone(), &mut setup.borrow()).unwrap();
            let mut out = module.ckks_ciphertext_alloc(ctx.base2k(), ctx.max_output_k(&keys).unwrap());
            let bytes = module.ckks_paco_bootstrap_tmp_bytes(&out, &ct_in, &ctx, &keys).unwrap();
            with_scratch::<$be, _>(bytes, |s| module.ckks_paco_bootstrap_into(&mut out, &ct_in, &ctx, &keys, s)).unwrap();
            (snapshot::<$be, _>(&out), bytes, keys, ctx)
        }};
    }
    let (want, reference_bytes, _, _) = bootstrap!(&reference, FFT64Portable);
    PACO_ENCODING_CALLS.set(0);
    let (got, bytes, keys, ctx) = bootstrap!(&module, OverrideBackend);
    assert_eq!(PACO_ENCODING_CALLS.get(), kappa);
    assert!(bytes >= reference_bytes && bytes >= PACO_ENCODING_EXTRA);
    assert_eq!(got, want);

    // An input at another width than the plan's modulus is refused before any encoding.
    let wide = module.ckks_ciphertext_alloc(params.base2k.into(), (k_in + 1).into());
    let mut out = module.ckks_ciphertext_alloc(ctx.base2k(), ctx.max_output_k(&keys).unwrap());
    let unchanged = snapshot::<OverrideBackend, _>(&out);
    PACO_ENCODING_CALLS.set(0);
    assert!(
        with_scratch::<OverrideBackend, _>(bytes, |s| module.ckks_paco_bootstrap_into(&mut out, &wide, &ctx, &keys, s)).is_err()
    );
    assert_eq!(PACO_ENCODING_CALLS.get(), 0);
    assert_eq!(snapshot::<OverrideBackend, _>(&out), unchanged);
}

/// Key material and inputs are built once on the portable backend and shared with
/// the probe, so the two outputs must agree bit for bit.
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

    let reference = Module::<FFT64Portable>::new(n as u64);
    let module = Module::<OverrideBackend>::new(n as u64);
    let host = Module::<HostBytesBackend>::new(n as u64);
    let mut scratch = alloc_scratch(&big, &reference);
    let (sk_raw, sk) = gen_sk_with_raw(&big, &reference, &host, [0u8; 32]);
    let mut sk_host = host.glwe_secret_alloc_from_infos(&big.glwe_layout());
    sk_raw.transfer_into(&mut sk_host);
    let spec = ShipSecretSpec::sample(&plan, &mut Source::new([9u8; 32]));
    let keys_layout = ShipKeysLayout {
        mux_dsize: params.dsize,
        tensor_key: big.tsk_layout(),
        conjugation_key: big.atk_layout(),
        complex: false,
    };
    let key_set = ShipKeySet::generate::<FFT64Portable, f64>(
        &reference,
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

    let gamma = 2f64.powi(plan.log_gamma() as i32);
    let coeffs: Vec<f64> = (0..n)
        .map(|i| {
            if i < n / 2 {
                ((i as f64 * 0.7639) % 1.0 - 0.5) * 0.9 / gamma
            } else {
                0.0
            }
        })
        .collect();
    let bottom = ckks_spec(n, base2k, base2k, 0);
    let mut host_pt = host.ckks_pt_vec_alloc(host.ckks_max_slots(), base2k.into(), bottom.k());
    host_pt.set_meta(bottom.meta());
    host_pt.encode_host_floats(&coeffs).unwrap();
    let ct0 = ckks_encrypt_pt(&big, &reference, &sk, base2k, &host_pt, &mut scratch.borrow());

    macro_rules! bootstrap {
        ($module:expr, $be:ty) => {{
            let module = $module;
            let keys = key_set.prepare(module, &mut alloc_scratch(&big, module).borrow()).unwrap();
            let mut out = alloc_ct(&big, module, kk);
            let bytes = CKKSShipOps::<$be, f64>::ckks_ship_bootstrap_tmp_bytes(module, &out, &ct0, &keys).unwrap();
            with_scratch::<$be, _>(bytes, |s| {
                CKKSShipOps::<$be, f64>::ckks_ship_bootstrap_into(module, &mut out, &ct0, &keys, s)
            })
            .unwrap();
            (snapshot::<$be, _>(&out), bytes, keys)
        }};
    }
    let (want, reference_bytes, _) = bootstrap!(&reference, FFT64Portable);
    SHIP_MASKING_CALLS.set(0);
    SHIP_MUX_CALLS.set(0);
    let (got, bytes, keys) = bootstrap!(&module, OverrideBackend);
    assert!(SHIP_MASKING_CALLS.get() > 0);
    assert!(SHIP_MUX_CALLS.get() > 0);
    assert!(bytes >= reference_bytes);
    assert!(bytes >= SHIP_MASKING_EXTRA + module.ckks_ship_masking_accumulate_tmp_bytes_reference(&plan, base2k));
    assert_eq!(got, want);

    // An input wider than one limb is refused before the masking and mux hooks.
    let wide = alloc_ct(&big, &module, 2 * base2k);
    let mut out = alloc_ct(&big, &module, kk);
    let unchanged = snapshot::<OverrideBackend, _>(&out);
    SHIP_MASKING_CALLS.set(0);
    SHIP_MUX_CALLS.set(0);
    assert!(
        with_scratch::<OverrideBackend, _>(bytes, |s| CKKSShipOps::<OverrideBackend, f64>::ckks_ship_bootstrap_into(
            &module, &mut out, &wide, &keys, s
        ))
        .is_err()
    );
    assert_eq!((SHIP_MASKING_CALLS.get(), SHIP_MUX_CALLS.get()), (0, 0));
    assert_eq!(snapshot::<OverrideBackend, _>(&out), unchanged);
}
