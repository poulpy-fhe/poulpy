//! SHIP bootstrapping tests (host replica, HMuxRot primitive, end-to-end).

use std::f64::consts::TAU;

use poulpy_core::{
    GLWEZero, ModuleTransfer,
    layouts::{
        BackendGLWESecret, GLWESecretPreparedFactory, GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftApply,
        VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApplyTmpA, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{HostBytesBackend, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSInfos, SetCKKSInfos, SlotsKind,
    api::{CKKSCopyOps, CKKSEncodingOps, CKKSShipOps, ShipScalar},
    encoding::ship::masks::{ship_pre_rotated_masks, ship_pre_rotated_masks_omega2},
    layouts::CKKSPlaintextVecHostCodec,
    layouts::{
        CKKSModuleAlloc, ShipKeySet, ShipKeysLayout, ShipPlan, ShipSecretSpec,
        ship::keyset::{HMuxRotKeyPrepared, hmux_rot_key_encrypt_sk},
    },
    oep::{CKKSEncodingImpl, CKKSShipCoeffEncodingImpl},
    test_suite::CKKSTestParams,
};

use super::helpers::{
    TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision,
    ckks_decrypt_decode, ckks_encrypt, ckks_encrypt_pt, ckks_spec, test_vector_1, want_rotate,
};
use poulpy_core::layouts::GLWESecretSampling;

/// SHIP suite plan on the test ring: sparse weight 32, full offset window,
/// base-4 mux with the low digit absorbed into masking.
fn ship_suite_plan(params: &CKKSTestParams) -> ShipPlan {
    const SPARSE_HW: usize = 32;
    let window = (params.n / SPARSE_HW) / 2;
    ShipPlan::new(
        params.log_n(),
        6,
        // ~14 bits of CKKS noise accumulate below log_delta_work; 30+ keeps
        // the gap model (~12.7 bits for gamma = 2^6) dominant.
        params.prec_meta.log_delta.min(40),
        2 * params.base2k,
        SPARSE_HW,
        window,
        4,
        4,
    )
    .unwrap()
}

/// Like `gen_sk_with_raw` but also returns the host-side secret.
#[allow(clippy::type_complexity)]
fn gen_sk_with_host<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    seed: [u8; 32],
) -> (
    BackendGLWESecret<HostBytesBackend>,
    BackendGLWESecret<BE>,
    GLWESecretPrepared<BE::OwnedBuf, BE>,
)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
{
    let glwe_infos = params.glwe_layout();
    let mut source = Source::new(seed);
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_infos);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, params.hw, &mut source);
    let sk_host = host_module.download_glwe_secret(&sk_raw);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    (sk_host, sk_raw, sk)
}

/// Paper-convention slot rotation: `out[i] = in[(i - r) mod m]`.
fn rot_paper(v: &[(f64, f64)], r: usize) -> Vec<(f64, f64)> {
    let m = v.len();
    (0..m).map(|i| v[(i + m - (r % m)) % m]).collect()
}

fn cmul((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// Cleartext replica of SHIP Algorithm 1 extended to the full/complex case
/// (Lemma 4 omega_2 masks): settles the phase, conjugation, mask and
/// recombination conventions before any homomorphic code runs.
pub fn test_ship_host_replica<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
{
    let plan = ShipPlan::new(params.log_n(), 6, 40, 0, 32, (params.n / 32) / 2, 4, 4).unwrap();
    let n = plan.n();
    let m = plan.half_n();
    let theta = plan.theta();
    let w = plan.window();
    let base2k = params.base2k;
    let q0f = 2f64.powi(base2k as i32);
    let gamma = 2f64.powi(plan.log_gamma() as i32);
    let bases = plan.mux_bases();

    let mut source = Source::new([42u8; 32]);
    let spec = ShipSecretSpec::sample(&plan, &mut source);
    let mut s = vec![0i64; n];
    for &(idx, sign) in spec.support() {
        s[idx] = sign;
    }

    // Complex cleartext payload mu with gap gamma: Re in the first m
    // coefficients, Im in the last m.
    let mu: Vec<(f64, f64)> = (0..m)
        .map(|_| (source.next_f64(-0.45, 0.45), source.next_f64(-0.45, 0.45)))
        .collect();
    let mut m_int = vec![0i64; n];
    for i in 0..m {
        m_int[i] = (q0f * mu[i].0 / gamma).round() as i64;
        m_int[i + m] = (q0f * mu[i].1 / gamma).round() as i64;
    }
    let mu_quant: Vec<(f64, f64)> = (0..m)
        .map(|i| (gamma * m_int[i] as f64 / q0f, gamma * m_int[i + m] as f64 / q0f))
        .collect();

    // Bottom ciphertext: b = m - a*s mod (X^n + 1, q0).
    let q0i = 1i64 << base2k;
    let a: Vec<i64> = (0..n)
        .map(|_| source.next_u64n(q0i as u64, (q0i - 1) as u64) as i64)
        .collect();
    let mut b = m_int.clone();
    for (j, &sj) in s.iter().enumerate() {
        if sj == 0 {
            continue;
        }
        for (i, &ai) in a.iter().enumerate() {
            let e = i + j;
            let (idx, sign) = if e >= n { (e - n, -sj) } else { (e, sj) };
            b[idx] = (b[idx] - sign * ai).rem_euclid(q0i);
        }
    }

    // Exponential cleartexts. pt0 of each half carries gamma/(4*i*pi) and is
    // seeded with the matching half of b.
    let omega = |x: i64| -> (f64, f64) {
        let phi = TAU * (x.rem_euclid(q0i) as f64) / q0f;
        (phi.cos(), phi.sin())
    };
    let pt0 = |off: usize| -> Vec<(f64, f64)> {
        (0..m)
            .map(|i| {
                let (re, im) = omega(b[i + off]);
                (gamma / (2.0 * TAU) * im, -(gamma / (2.0 * TAU)) * re)
            })
            .collect()
    };
    let pi: [Vec<(f64, f64)>; 4] = [
        (0..m).map(|i| omega(a[i])).collect(),
        (0..m).map(|i| omega(-a[i])).collect(),
        (0..m).map(|i| omega(a[i + m])).collect(),
        (0..m).map(|i| omega(-a[i + m])).collect(),
    ];

    // Per support index and coefficient half: theta-column masking then the
    // mux digit rotations, with a per-index correctness check.
    let mut v = [pt0(0), pt0(m)];
    for (k, &(j, sj)) in spec.support().iter().enumerate() {
        let p = plan.mask_rotation(k);
        let u = spec.offset(&plan, k);
        assert!(u <= 2 * w, "ship_host_replica: support index outside window");

        let mask_sets = [ship_pre_rotated_masks(j, sj, m), ship_pre_rotated_masks_omega2(j, sj, m)];
        for (half, masks) in mask_sets.iter().enumerate() {
            // Column/mux hybrid (§4.4, low-digit-column variant): the low
            // digit u0 = u mod theta is absorbed into the masking via
            // selector masks 1_{c = u0} x Rot_{p+c}(mask), paired with pi
            // rotated by p + c.
            let u0 = u % theta;
            let ct_j: Vec<(f64, f64)> = (0..m)
                .map(|i| {
                    let mut acc = (0.0, 0.0);
                    for c in 0..theta {
                        if c != u0 {
                            continue;
                        }
                        let src = (i + 2 * m - (p + c)) % m;
                        for k4 in 0..4 {
                            acc.0 += pi[k4][src].0 * masks[k4][src];
                            acc.1 += pi[k4][src].1 * masks[k4][src];
                        }
                    }
                    acc
                })
                .collect();

            // Mux blind rotation over the mixed-radix digits of u / theta.
            let mut rotated = ct_j;
            let mut weight = theta;
            for &base in &bases {
                let digit = (u / weight) % base;
                rotated = rot_paper(&rotated, digit * weight);
                weight *= base;
            }
            assert!(weight > 2 * w, "ship_host_replica: mux chain does not cover the window");
            assert_eq!((p + u) % m, j % m, "ship_host_replica: rotation split mismatch");

            // rotated must equal (omega^{(s_j * a * X^j)_{i + half*m}})_{i < m}.
            for (i, got) in rotated.iter().enumerate() {
                let e = i + half * m + n - j;
                let (idx, sign) = if e >= n { (e - n, sj) } else { (e, -sj) };
                let want = omega(sign * a[idx % n]);
                let err = ((got.0 - want.0).abs()).max((got.1 - want.1).abs());
                assert!(
                    err < 1e-9,
                    "ship_host_replica: blind rotation half={half} j={j} s_j={sj} slot {i}: err={err:.3e}"
                );
            }

            v[half] = (0..m).map(|i| cmul(v[half][i], rotated[i])).collect();
        }
    }

    // z_h = v_h + conj(v_h) extracts each coefficient half into real slots;
    // out = z_1 + i*z_2 ~ mu. With one conjugation:
    // out = (v_1 + i*v_2) + conj(v_1 - i*v_2).
    let mut max_err: f64 = 0.0;
    for i in 0..m {
        let (w_re, w_im) = (v[0][i].0 - v[1][i].1, v[0][i].1 + v[1][i].0); // v1 + i*v2
        let (c_re, c_im) = (v[0][i].0 + v[1][i].1, -(v[0][i].1 - v[1][i].0)); // conj(v1 - i*v2)
        let out = (w_re + c_re, w_im + c_im);
        assert!(
            (out.0 - 2.0 * v[0][i].0).abs() < 1e-12 && (out.1 - 2.0 * v[1][i].0).abs() < 1e-12,
            "ship_host_replica: one-conjugation recombination mismatch"
        );
        max_err = max_err.max((out.0 - mu_quant[i].0).abs()).max((out.1 - mu_quant[i].1).abs());
    }
    // Model: (2*pi)^2 * mu^3 / (6 * gamma^2) ~ 1.5e-4 for |mu| <= 0.45.
    assert!(
        max_err < 3e-4,
        "ship_host_replica: final error {max_err:.3e} exceeds model bound"
    );
}

/// Hoisted B-to-1 mux-rotate: a digit-position key group with `beta` on digit
/// `d` yields the paper-convention rotation `out[i] = in[i - d*weight]`; an
/// all-zero group yields (an encryption of) zero.
pub fn test_ship_mux_rotate<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
{
    let m = params.n / 2;
    let k = params.k;
    let encoder = super::reference_encoder::ReferenceEncoder::<E>::new::<F>(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_host, _, sk) = gen_sk_with_host(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut xe = Source::new([21u8; 32]);
    let mut xa = Source::new([22u8; 32]);
    let mut gen_key = |beta: bool, rot: usize, scratch: &mut ScratchOwned<BE>| {
        let key = hmux_rot_key_encrypt_sk(
            module,
            host_module,
            &sk_host,
            beta,
            rot,
            k,
            params.base2k.into(),
            params.dsize,
            &mut xe,
            &mut xa,
            &mut scratch.borrow(),
        )
        .unwrap();
        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key.key());
        module.glwe_switching_key_prepare(&mut prepared, key.key(), &mut scratch.borrow());
        HMuxRotKeyPrepared {
            key: prepared,
            gal_el: key.gal_el(),
        }
    };
    // Base-3 digit position of weight 5 with digit value 1 selected.
    let group_rot = vec![
        gen_key(false, 0, &mut scratch),
        gen_key(true, 5, &mut scratch),
        gen_key(false, 10, &mut scratch),
    ];
    let group_id = vec![gen_key(true, 0, &mut scratch), gen_key(false, 5, &mut scratch)];
    let group_zero = vec![gen_key(false, 0, &mut scratch), gen_key(false, 5, &mut scratch)];

    let mux_bytes =
        crate::default::ship::mux::ship_mux_rotate_tmp_bytes(module, &alloc_ct(&params, module, k), &group_rot[0].key);
    let plans = crate::default::ship::mux::ship_mux_plans(
        module,
        [group_rot.as_slice(), group_id.as_slice(), group_zero.as_slice()].into_iter(),
    );
    let mut scratch = ScratchOwned::<BE>::alloc(mux_bytes.max(scratch.borrow().available()));

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );

    // digit = 1, weight = 5: paper Rot_5 = poulpy rotation by -5.
    let (want_re, want_im) = want_rotate(&re1, &im1, -5, m);
    let mut out = alloc_ct(&params, module, k);
    module.ckks_copy(&mut out, &ct, &mut scratch.borrow()).unwrap();
    crate::default::ship::mux::ship_mux_rotate(module, &mut out, &group_rot, &plans, &mut scratch.borrow()).unwrap();
    assert_eq!(out.log_delta(), ct.log_delta());
    assert_eq!(out.log_budget(), ct.log_budget());
    assert_decrypt_precision(
        "mux_rotate(digit=1,weight=5)",
        &params,
        module,
        &encoder,
        &out,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    // digit = 0: identity.
    let mut out_id = alloc_ct(&params, module, k);
    module.ckks_copy(&mut out_id, &ct, &mut scratch.borrow()).unwrap();
    crate::default::ship::mux::ship_mux_rotate(module, &mut out_id, &group_id, &plans, &mut scratch.borrow()).unwrap();
    assert_decrypt_precision(
        "mux_rotate(digit=0)",
        &params,
        module,
        &encoder,
        &out_id,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );

    // No digit selected: all-zero cleartext.
    let zeros = vec![F::zero(); m];
    let mut out_zero = alloc_ct(&params, module, k);
    module.ckks_copy(&mut out_zero, &ct, &mut scratch.borrow()).unwrap();
    crate::default::ship::mux::ship_mux_rotate(module, &mut out_zero, &group_zero, &plans, &mut scratch.borrow()).unwrap();
    assert_decrypt_precision(
        "mux_rotate(empty)",
        &params,
        module,
        &encoder,
        &out_zero,
        &sk,
        &zeros,
        &zeros,
        &mut scratch.borrow(),
    );
}

/// End-to-end SHIP bootstrap driver: a one-limb bottom ciphertext under the
/// dense key, holding cleartexts with gap `gamma` in its coefficients (real:
/// first `N/2` only; complex: `Re`/`Im` halves), is lifted to a slots-domain
/// ciphertext at the raised modulus.
fn ship_bootstrap_case<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    complex: bool,
) where
    BE: TestContextBackend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: TestContextModule<BE>
        + CKKSShipOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + ShipScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let n = params.n;
    let m = n / 2;
    let base2k = params.base2k;
    let encoder = super::reference_encoder::ReferenceEncoder::<E>::new::<F>(m).unwrap();

    let plan = ship_suite_plan(&params);
    let kk = plan.raised_k(base2k);
    let big = CKKSTestParams { k: kk, ..params };

    let (sk_dense_host, _, sk_dense) = gen_sk_with_host(&big, module, host_module, [0u8; 32]);
    let mut source = Source::new([9u8; 32]);
    let spec = ShipSecretSpec::sample(&plan, &mut source);

    let keys_layout = ShipKeysLayout {
        mux_dsize: params.dsize,
        tensor_key: big.tsk_layout().layout,
        conjugation_key: big.atk_layout().layout,
        complex,
    };
    let mut scratch = alloc_scratch(&big, module);
    let mut xe = Source::new([11u8; 32]);
    let mut xa = Source::new([12u8; 32]);
    let key_set = ShipKeySet::generate::<BE, F>(
        module,
        host_module,
        &plan,
        base2k.into(),
        &spec,
        &sk_dense_host,
        &keys_layout,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    )
    .unwrap();
    let keys = key_set.prepare(module, &mut scratch.borrow()).unwrap();

    // Bottom ciphertext: torus content sum round(q0 * mu_i / gamma) X^i, i.e.
    // value mu_i / gamma at log_delta = base2k with no budget; the complex
    // case also carries Im(mu_i) in coefficient i + m.
    let gamma = 2f64.powi(plan.log_gamma() as i32);
    let mu_re: Vec<f64> = (0..m).map(|i| ((i as f64 * 0.7639) % 1.0 - 0.5) * 0.9).collect();
    let mu_im: Vec<f64> = (0..m)
        .map(|i| {
            if complex {
                ((i as f64 * 0.4117 + 0.31) % 1.0 - 0.5) * 0.9
            } else {
                0.0
            }
        })
        .collect();
    let coeffs: Vec<F> = (0..n)
        .map(|i| {
            if i < m {
                F::from_f64(mu_re[i] / gamma).unwrap()
            } else {
                F::from_f64(mu_im[i - m] / gamma).unwrap()
            }
        })
        .collect();
    let bottom_prec = ckks_spec(n, base2k, base2k, 0);
    let mut host_pt = host_module.ckks_pt_vec_alloc(base2k.into(), bottom_prec.k());
    host_pt.set_meta(bottom_prec.meta());
    host_pt.encode_host_floats(&coeffs).unwrap();
    let mut coeffs_quant = vec![F::zero(); n];
    host_pt.decode_host_floats(&mut coeffs_quant).unwrap();
    let mu_quant_re: Vec<f64> = (0..m).map(|i| gamma * coeffs_quant[i].to_f64().unwrap()).collect();
    let mu_quant_im: Vec<f64> = (0..m).map(|i| gamma * coeffs_quant[i + m].to_f64().unwrap()).collect();
    let ct0 = ckks_encrypt_pt(&big, module, &sk_dense, base2k, &host_pt, &mut scratch.borrow());

    let mut out = alloc_ct(&big, module, kk);
    let ship_bytes = CKKSShipOps::<BE, F>::ckks_ship_bootstrap_tmp_bytes(module, &out, &ct0, &keys).unwrap();
    let mut ship_scratch = ScratchOwned::<BE>::alloc(ship_bytes);
    if complex {
        CKKSShipOps::<BE, F>::ckks_ship_bootstrap_complex_into(module, &mut out, &ct0, &keys, &mut ship_scratch.borrow())
            .unwrap();
    } else {
        CKKSShipOps::<BE, F>::ckks_ship_bootstrap_into(module, &mut out, &ct0, &keys, &mut ship_scratch.borrow()).unwrap();
    }
    assert_eq!(
        out.slots(),
        if complex { SlotsKind::Complex } else { SlotsKind::Real },
        "ship_bootstrap: output slot kind"
    );
    assert!(
        out.log_budget() >= plan.log_budget_out(),
        "ship_bootstrap: no budget regained (log_budget={} < {})",
        out.log_budget(),
        plan.log_budget_out()
    );

    let (got_re, got_im) = ckks_decrypt_decode::<BE, F, E>(&big, module, &encoder, &out, &sk_dense, &mut scratch.borrow());
    let mut max_err: f64 = 0.0;
    for i in 0..m {
        let er = (got_re[i].to_f64().unwrap() - mu_quant_re[i]).abs();
        let ei = (got_im[i].to_f64().unwrap() - mu_quant_im[i]).abs();
        max_err = max_err.max(er).max(ei);
    }
    let measured_bits = -max_err.log2();
    // Gap model: error ~ (2*pi)^2 * mu^3 / (6 * gamma^2), 12.74 bits for
    // gamma = 2^6 and |mu| <= 0.45; exact backends measure exactly that.
    // FFT64 (f64 arithmetic) loses ~3 bits over the deep keyswitch chain and
    // ~2 more to the encapsulation switch at its toy base2k bottom modulus.
    let required_bits = if plan.log_delta_work() < 40 { 7.0 } else { 12.0 };
    assert!(
        measured_bits >= required_bits,
        "ship_bootstrap: precision {measured_bits:.2} bits < required {required_bits:.2}"
    );
}

/// End-to-end SHIP half bootstrap over real cleartexts.
pub fn test_ship_bootstrap<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: TestContextModule<BE>
        + CKKSShipOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + ShipScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    ship_bootstrap_case::<BE, F, E>(params, module, host_module, false);
}

/// End-to-end complex SHIP bootstrap: `Re`/`Im` coefficient halves refreshed
/// into complex slots through the shared mux keys and the `omega_2` masks.
pub fn test_ship_bootstrap_complex<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: TestContextModule<BE>
        + CKKSShipOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + ShipScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    ship_bootstrap_case::<BE, F, E>(params, module, host_module, true);
}
