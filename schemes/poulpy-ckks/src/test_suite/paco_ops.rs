//! Tests for the PaCo slot-fold composites and conjugate rotation.
//!
//! - **`Tr_{a→b}`** (`CKKSPaCoImpl::ckks_slot_trace_assign`): fold by
//!   addition. Checked against the cleartext recurrence (`trace_slots`) for several
//!   `(a, b)` pairs; consumes **zero** budget.
//! - **`Pr_{a→b}`** (`CKKSPaCoImpl::ckks_slot_product_assign`): fold by
//!   multiplication with tensor-key relinearization — seqPaCo's EvalMod
//!   replacement. Inputs are unit-circle slot values (as in the real
//!   pipeline). Checked against `product_slots`; consumes
//!   exactly `log(a/b) · log_delta` budget bits.
//! - **conj-rotate** (seqPaCo line 8): an automorphism key generated for the
//!   signed Galois element `−5^k` applied through the
//!   plain conjugation op must equal `conj(rotate(·, k))`, in one keyswitch.

use super::parity::helpers::{snapshot, with_scratch};
use crate::CKKSModuleInfos;
use crate::Scale;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWELayout, LWEInfos, Rank};
use poulpy_hal::{
    api::{ReimFFT, ReimIFFT, ReimNew, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, Module, Standard, galois_element, galois_elements_from_rotations},
};
use std::collections::HashMap;

use crate::SlotsKind;
use crate::{
    CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos,
    api::CKKSConjugateOps,
    encoding::paco::cpx::Cpx,
    layouts::slots::fold_rotations,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    oep::CKKSPaCoImpl,
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_internal, ckks_encrypt, gen_atk,
            gen_sk_with_raw, gen_tsk, test_vector_1,
        },
        paco_reference_model::{conjugate, product_slots, rotate_left, trace_slots},
    },
};

/// Structural noise bound (as in the DFT/LT suites).
fn noise_bound(log_delta: usize) -> f64 {
    -(log_delta as f64) + 16.0
}

/// Re-sizes `k` for `levels` chained ct×ct multiplies plus headroom.
fn leveled_params(base: &CKKSTestParams, levels: usize) -> CKKSTestParams {
    let log_delta = base.prec().log_delta();
    CKKSTestParams {
        k: (log_delta * (levels + 3)).next_multiple_of(base.base2k),
        ..*base
    }
}

/// Deterministic unit-circle slot vector (the value domain of the PaCo
/// product fold).
fn unit_circle_vector<F: TestScalar>(m: usize) -> (Vec<F>, Vec<F>) {
    let phase = |j: usize| 2.0 * std::f64::consts::PI * ((j.wrapping_mul(2654435761) % (1 << 20)) as f64 / (1 << 20) as f64);
    let re = (0..m).map(|j| F::from_f64(phase(j).cos()).unwrap()).collect();
    let im = (0..m).map(|j| F::from_f64(phase(j).sin()).unwrap()).collect();
    (re, im)
}

fn to_cpx<F: TestScalar>(re: &[F], im: &[F]) -> Vec<Cpx> {
    re.iter()
        .zip(im)
        .map(|(r, i)| Cpx::new(r.to_f64().unwrap(), i.to_f64().unwrap()))
        .collect()
}

/// Decrypts `ct` on its backend at full precision, downloads the plaintext
/// to the host, encodes the expected slots host-side at the same layout, and
/// bounds the max coefficient error (log2, value domain at the ciphertext's
/// scale). Shared by the PaCo suite files. No host-readability is assumed of
/// `BE`: device data is downloaded before any host-side processing.
///
/// The oracle (`want`) is f64 data, so errors below ~2^-46 are not
/// distinguishable from oracle rounding; the bound is clamped accordingly
/// (structural failures sit at signal level, far above the clamp).
#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_slots<BE, F, E>(
    label: &str,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &ReferenceEncoder<E>,
    ct: &CKKSCiphertextOwned<BE>,
    sk: &poulpy_core::layouts::prepared::GLWESecretPrepared<BE::OwnedBuf, BE>,
    want: &[Cpx],
    bound: f64,
    scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimFFT<F> + ReimIFFT<F>,
{
    use crate::test_suite::helpers::ckks_decrypt_with_prec;
    use poulpy_hal::layouts::ZnxView;

    let (log_delta, base2k) = (ct.log_delta(), ct.base2k());
    let prec = crate::CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct.n(),
            base2k,
            k: ct.k(),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
    };
    // Full-precision decrypt on the backend, then download to host bytes.
    let got_pt = ckks_decrypt_with_prec(module, ct, sk, prec, scratch).unwrap();

    // Expected plaintext, encoded host-side at the same layout.
    let want_re: Vec<F> = want.iter().map(|x| F::from_f64(x.re).unwrap()).collect();
    let want_im: Vec<F> = want.iter().map(|x| F::from_f64(x.im).unwrap()).collect();
    let mut want_pt = host_module.ckks_pt_vec_alloc(host_module.ckks_max_slots(), base2k, ct.k());
    want_pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(log_delta),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
    });
    encoder.encode_reim(&mut want_pt, &want_re, &want_im).unwrap();

    // Max coefficient error in the value domain (limb-wise, no float decode:
    // value weight of limb j is 2^{k - delta - (j+1)*base2k}).
    let (a, b) = (got_pt.data(), want_pt.data());
    let n = ct.n().as_usize();
    let size = a.size().min(b.size());
    // Digits align to the plaintexts' STORAGE width (max_k), not effective k.
    let (max_k, b2k) = (got_pt.encoded_k().as_usize() as i32, base2k.as_usize() as i32);
    let mut max_err = 0.0f64;
    for i in 0..n {
        let mut e = 0.0f64;
        for j in 0..size {
            let d = (a.at(0, j)[i] - b.at(0, j)[i]) as f64;
            e += d * 2.0f64.powi(max_k - log_delta as i32 - (j as i32 + 1) * b2k);
        }
        max_err = max_err.max(e.abs());
    }
    let bound = bound.max(-46.0);
    assert!(
        max_err.log2() < bound,
        "{label}: max value error log2={:.1} (bound {bound:.1})",
        max_err.log2()
    );
}

/// `Tr_{a→b}` matches the cleartext fold and consumes no budget.
pub fn test_paco_slot_trace<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + CKKSPaCoImpl,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let order = module.cyclotomic_order();
    let (a_re, a_im) = test_vector_1::<F>(m);
    let v = to_cpx(&a_re, &a_im);

    for (a, b) in [(m, m / 2), (m, m / 8), (m / 2, m / 16)] {
        let mut atks = HashMap::new();
        for p_el in galois_elements_from_rotations(fold_rotations(a, b), order) {
            atks.entry(p_el)
                .or_insert_with(|| gen_atk(&params, module, p_el, &sk_raw, &mut scratch.borrow()));
        }
        let mut ct = ckks_encrypt(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            &a_re,
            &a_im,
            &mut scratch.borrow(),
        );
        let budget_before = ct.log_budget();
        with_scratch::<BE, _>(BE::ckks_slot_trace_assign_tmp_bytes(module, &ct, &params.atk_layout()), |s| {
            BE::ckks_slot_trace_assign(module, &mut ct, a, b, &atks, s)
        })
        .unwrap();
        assert_eq!(ct.log_budget(), budget_before, "trace({a}->{b}) must consume no budget");
        assert_slots::<BE, F, E>(
            &format!("paco_slot_trace({a}->{b})"),
            module,
            host_module,
            &encoder,
            &ct,
            &sk,
            &trace_slots(&v, a, b),
            noise_bound(params.prec().log_delta()),
            &mut scratch.borrow(),
        );
    }
}

/// `Pr_{a→b}` matches the cleartext fold on unit-circle inputs and consumes
/// exactly `log(a/b) · log_delta` budget bits.
pub fn test_paco_slot_product<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + CKKSPaCoImpl,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    for (a_div_m, b) in [(1usize, 16usize), (2, 8)] {
        // (a, b) pairs relative to the slot count: (m, m/8)-style folds of
        // depth 3, exercising multi-level products.
        let base = params;
        let m = base.n / 2;
        let a = m / a_div_m;
        let levels = (a / b).trailing_zeros() as usize;
        let params = leveled_params(&base, levels);
        let log_delta = params.prec().log_delta();

        let module = Module::<BE>::new(params.n as u64);
        let host_module = Module::<HostBytesBackend>::new(params.n as u64);
        let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();
        let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
        let mut scratch = alloc_scratch(&params, &module);
        let order = module.cyclotomic_order();
        let tsk = gen_tsk(&params, &module, &sk_raw, &mut scratch.borrow());

        let mut atks = HashMap::new();
        for p_el in galois_elements_from_rotations(fold_rotations(a, b), order) {
            atks.entry(p_el)
                .or_insert_with(|| gen_atk(&params, &module, p_el, &sk_raw, &mut scratch.borrow()));
        }

        let (a_re, a_im) = unit_circle_vector::<F>(m);
        let v = to_cpx(&a_re, &a_im);
        let mut ct = ckks_encrypt(
            &params,
            &module,
            &host_module,
            &encoder,
            &sk,
            params.k,
            &a_re,
            &a_im,
            &mut scratch.borrow(),
        );
        let budget_before = ct.log_budget();
        let bytes = BE::ckks_slot_product_assign_tmp_bytes(&module, &ct, &params.atk_layout(), &params.tsk_layout());
        let before = snapshot::<BE, _>(&ct);
        let err = with_scratch::<BE, _>(bytes - 1, |s| {
            BE::ckks_slot_product_assign(&module, &mut ct, a, b, &atks, &tsk, s)
        })
        .unwrap_err();
        assert_ckks_internal(
            &format!("paco_slot_product({a}->{b}): one byte short"),
            &err,
            "ckks_slot_product_assign: scratch below its query",
        );
        assert_eq!(
            snapshot::<BE, _>(&ct),
            before,
            "paco_slot_product({a}->{b}): a refused fold wrote ct"
        );
        with_scratch::<BE, _>(bytes, |s| {
            BE::ckks_slot_product_assign(&module, &mut ct, a, b, &atks, &tsk, s)
        })
        .unwrap();
        assert_eq!(
            budget_before - ct.log_budget(),
            levels * log_delta,
            "product({a}->{b}) must consume log(a/b) × log_delta budget bits"
        );
        assert_slots::<BE, F, E>(
            &format!("paco_slot_product({a}->{b})"),
            &module,
            &host_module,
            &encoder,
            &ct,
            &sk,
            &product_slots(&v, a, b),
            noise_bound(log_delta),
            &mut scratch.borrow(),
        );
    }
}

/// A single automorphism keyed at `−5^k` equals `conj ∘ rotate_k`.
pub fn test_paco_conj_rotate<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let order = module.cyclotomic_order();
    let (a_re, a_im) = test_vector_1::<F>(m);
    let v = to_cpx(&a_re, &a_im);

    for k in [8i64, 1, m as i64 / 2] {
        let key = gen_atk(&params, module, -galois_element(k, order), &sk_raw, &mut scratch.borrow());
        let ct = ckks_encrypt(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            &a_re,
            &a_im,
            &mut scratch.borrow(),
        );
        let mut out = module.ckks_ciphertext_alloc_from_infos(&ct);
        with_scratch::<BE, _>(module.ckks_conjugate_tmp_bytes(&ct, &params.atk_layout()), |s| {
            module.ckks_conjugate_rotate_into(k, &mut out, &ct, &key, s)
        })
        .unwrap();
        assert_eq!(out.log_budget(), ct.log_budget(), "conj_rotate must consume no budget");
        assert_slots::<BE, F, E>(
            &format!("paco_conj_rotate(k={k})"),
            module,
            host_module,
            &encoder,
            &out,
            &sk,
            &conjugate(&rotate_left(&v, k as usize)),
            noise_bound(params.prec().log_delta()),
            &mut scratch.borrow(),
        );
    }
}

/// A fold whose second rotation has no key is `MissingAutomorphismKey` of that rotation before the first product writes
/// `ct`, on an empty arena and on the query.
pub fn test_slot_product_assign_rejects_missing_later_key<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard> + CKKSPaCoImpl,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    use super::parity::{
        helpers::{fixture_ciphertext, snapshot, with_scratch},
        layout,
    };
    use crate::CKKSCompositionError;
    let m = params.n / 2;
    let rotations = fold_rotations(m, m / 4);
    let params = leveled_params(&params, rotations.len());
    let mut scratch = alloc_scratch(&params, module);
    let (sk_raw, _) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let first = galois_element(rotations[0], module.cyclotomic_order());
    let keys = HashMap::from([(first, gen_atk(&params, module, first, &sk_raw, &mut scratch.borrow()))]);
    let log_delta = params.prec().log_delta();
    let mut ct = fixture_ciphertext(
        module,
        &layout(params, params.rank, params.k, log_delta, 0, SlotsKind::Complex),
        41,
    );
    let query = BE::ckks_slot_product_assign_tmp_bytes(module, &ct, &params.atk_layout(), &params.tsk_layout());
    for bytes in [0, query] {
        let before = snapshot::<BE, _>(&ct);
        let err = with_scratch::<BE, _>(bytes, |s| {
            BE::ckks_slot_product_assign(module, &mut ct, m, m / 4, &keys, &tsk, s)
        })
        .unwrap_err();
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::MissingAutomorphismKey { op: "ckks_slot_product_assign", rotation, .. })
                    if *rotation == rotations[1]
            ),
            "slot_product_assign: missing second key: {err}"
        );
        assert_eq!(snapshot::<BE, _>(&ct), before, "a refused product fold wrote ct");
    }
}

/// A fold whose second rotation has no key is `MissingAutomorphismKey` of that rotation before the first
/// automorphism-add writes `ct`, on an empty arena and on the query.
pub fn test_slot_trace_assign_rejects_missing_later_key<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard> + CKKSPaCoImpl,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    use super::parity::{
        helpers::{fixture_ciphertext, snapshot, with_scratch},
        layout,
    };
    use crate::CKKSCompositionError;
    let m = params.n / 2;
    let rotations = fold_rotations(m, m / 4);
    let mut scratch = alloc_scratch(&params, module);
    let (sk_raw, _) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let first = galois_element(rotations[0], module.cyclotomic_order());
    let keys = HashMap::from([(first, gen_atk(&params, module, first, &sk_raw, &mut scratch.borrow()))]);
    let mut ct = fixture_ciphertext(
        module,
        &layout(
            params,
            params.rank,
            params.k,
            params.prec().log_delta(),
            0,
            SlotsKind::Complex,
        ),
        41,
    );
    for bytes in [0, BE::ckks_slot_trace_assign_tmp_bytes(module, &ct, &params.atk_layout())] {
        let before = snapshot::<BE, _>(&ct);
        let err = with_scratch::<BE, _>(bytes, |s| BE::ckks_slot_trace_assign(module, &mut ct, m, m / 4, &keys, s)).unwrap_err();
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::MissingAutomorphismKey { op: "ckks_slot_trace_assign", rotation, .. })
                    if *rotation == rotations[1]
            ),
            "slot_trace_assign: missing second key: {err}"
        );
        assert_eq!(snapshot::<BE, _>(&ct), before, "a refused trace fold wrote ct");
    }
}
