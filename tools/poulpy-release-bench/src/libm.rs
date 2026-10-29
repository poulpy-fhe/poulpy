//! Release measurements of `poulpy-libm` functions on one ciphertext at `N = 2^16`.
//!
//! A case builds its point outside the timed region: its plan on a host module, transferred to the backend, the secret,
//! the tensor key, the rotation keys of softmax, and the encryption of values spread over its domain. It calls the
//! function once, decrypts the result and measures its error against `f64`, then times that same call. The ciphertext
//! is as wide as what the function consumes plus `2 log_delta` of result headroom and two limbs of guard, rounded up to
//! a limb, as in the libm benchmarks of poulpy-bench.

use std::{collections::HashMap, time::Instant};

use anyhow::{Result, bail};
use poulpy_ckks::{
    CKKSInfos, CKKSLayout, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{CKKSAllOpsTmpBytes, CKKSCopyOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    polynomial::SplitStrategy,
    test_suite::helpers::{TestContextSharedModule, upload_pt},
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPreparedFactory},
    layouts::{
        BackendGLWESecret, Base2K, Degree, Dsize, GGLWELayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout,
        LWEInfos, ModuleCoreAlloc, Rank, SetK, TorusPrecision, k_digit_of,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared, GLWETensorKeyPrepared},
    },
};
use poulpy_hal::{
    api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostStaged, Module, ScratchOwned, Standard},
    source::Source,
};
use poulpy_libm::{
    approximation::{CKKSApproximationOps, DegreeChoice, Parity, PolynomialApproximation, degree_for_precision, error_bits},
    reduce::CKKSReductionOps,
    roots::{CKKSInverseDomainOps, CKKSInverseOps},
    softmax::{CKKSSoftmaxOps, SoftmaxOptions, SoftmaxPlan},
};
use serde_json::{Value, json};

use crate::runner::{CaseArgs, direct};

/// The cases of the release set.
pub const CASES: [&str; 4] = ["rsqrt_r16", "rsqrt_r256", "softmax_b20", "inv_r256"];

const LOG_N: usize = 16;
const BASE2K: usize = 52;
/// The scale: at `2^40` the noise caps the results near 21 bits relative to the largest output, below the targets of
/// `rsqrt_r256` and `inv_r256`.
const LOG_DELTA: usize = 50;
/// Limbs per key digit: four keep a softmax point, with its tensor key and 15 rotation keys, under 6 GB.
const DSIZE: usize = 4;
/// Secret-key Hamming weight.
const HW: usize = 192;
/// Absolute-error bits the rsqrt and inverse cases aim for, as libm's own options count them.
const TARGET_BITS: f64 = 20.0;
/// Integer bits kept when decrypting, for results up to `1/min`.
const DECODE_HEADROOM: usize = 24;
/// Bits above the scale an encoded input keeps for a magnitude of at most one.
const HEADROOM: usize = 4;
/// Largest degree of the rsqrt seed.
const MAX_SEED_DEGREE: usize = 63;
/// Domain of the softmax inputs, the one of libm's own softmax test.
const SOFTMAX_DOMAIN: (f64, f64) = (-1.0, 1.0);

enum Case {
    /// `1/sqrt(x)` on `[min, 1]`: a polynomial seed refined by Newton steps.
    Rsqrt { min: f64 },
    /// Softmax over every slot at `target_bits` of absolute error.
    Softmax { target_bits: f64 },
    /// `1/x` on `[min, 1]` by Goldschmidt iterations.
    Inverse { min: f64 },
}

fn case(name: &str) -> Result<Case> {
    Ok(match name {
        "rsqrt_r16" => Case::Rsqrt { min: 1.0 / 16.0 },
        "rsqrt_r256" => Case::Rsqrt { min: 1.0 / 256.0 },
        "softmax_b20" => Case::Softmax { target_bits: 20.0 },
        "inv_r256" => Case::Inverse { min: 1.0 / 256.0 },
        _ => bail!("no libm case named {name}"),
    })
}

/// What a case asks of a module: key generation, encryption, encoding and decryption, the functions it times, and
/// synchronization.
pub trait LibmModule<BE: Backend>:
    TestContextSharedModule<BE>
    + CKKSEncodingOps<BE, f64>
    + CKKSApproximationOps<BE>
    + CKKSInverseOps<BE>
    + CKKSInverseDomainOps<BE>
    + CKKSSoftmaxOps<BE>
    + CKKSReductionOps<BE>
    + ModuleSynchronize<BE>
{
}

impl<BE: Backend, M> LibmModule<BE> for M where
    M: TestContextSharedModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSApproximationOps<BE>
        + CKKSInverseOps<BE>
        + CKKSInverseDomainOps<BE>
        + CKKSSoftmaxOps<BE>
        + CKKSReductionOps<BE>
        + ModuleSynchronize<BE>
{
}

fn n() -> usize {
    1 << LOG_N
}

fn slots() -> usize {
    n() / 2
}

fn base2k() -> Base2K {
    Base2K(BASE2K as u32)
}

/// The layout of a ciphertext of width `k` at scale `2^LOG_DELTA`, every slot used.
fn layout(k: usize) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: Degree(n() as u32),
            base2k: base2k(),
            k: TorusPrecision(k as u32),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(LOG_DELTA),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
    }
}

/// The plans' coefficient layout: one limb of budget above the scale.
fn coeff_meta() -> CKKSLayout {
    layout(LOG_DELTA + BASE2K)
}

/// The ciphertext width for a function consuming `consumed` bits.
fn ct_k(consumed: usize) -> usize {
    (consumed + 2 * LOG_DELTA + 2 * BASE2K).next_multiple_of(BASE2K)
}

/// Digits of [`DSIZE`] limbs covering a ciphertext of width `k`, with a guard of one digit and `log2 n` bits.
fn key_shape(k: usize) -> (TorusPrecision, poulpy_core::layouts::Dnum, TorusPrecision) {
    let k_digit = k_digit_of(base2k(), Dsize(DSIZE as u32));
    let dnum = GGLWELayout::dnum_for_input(TorusPrecision(k as u32), k_digit);
    (k_digit, dnum, TorusPrecision(k_digit.0 + LOG_N as u32))
}

fn tsk_layout(k: usize) -> GLWETensorKeyLayout {
    let (k_digit, dnum, k_aux) = key_shape(k);
    GLWETensorKeyLayout {
        n: Degree(n() as u32),
        base2k: base2k(),
        k_aux,
        rank: Rank(1),
        k_digit,
        dnum,
    }
}

fn atk_layout(k: usize) -> GLWEAutomorphismKeyLayout {
    let (k_digit, dnum, k_aux) = key_shape(k);
    GLWEAutomorphismKeyLayout {
        n: Degree(n() as u32),
        base2k: base2k(),
        k_aux,
        rank: Rank(1),
        k_digit,
        dnum,
    }
}

/// `slots()` values spread evenly over `[a, b]`, both ends included.
fn spread(a: f64, b: f64) -> Vec<f64> {
    let last = (slots() - 1) as f64;
    (0..slots()).map(|i| a + (b - a) * i as f64 / last).collect()
}

/// The seed of `1/sqrt(x)` on `[min, 1]` and the Newton steps that take it to [`TARGET_BITS`] of absolute error: the
/// pair of least depth, a step costing two levels, and of least degree among those. It takes at least one step, so that
/// the case measures `ckks_rsqrt` rather than a polynomial alone.
///
/// On the domain `1/sqrt(x)` lies in `[1, 1/sqrt(min)]`, so a seed of absolute error `e` has a relative error of at most
/// `e`, a step maps a relative error `e` to at most `1.5 e^2 + 0.5 e^3`, and the result needs a relative error of
/// `2^-TARGET_BITS sqrt(min)`. Seeds of error above one half are left out, well inside the basin `x y^2 < 3`.
fn rsqrt_seed(min: f64) -> Result<(DegreeChoice<f64>, usize)> {
    let need = TARGET_BITS + 0.5 * (1.0 / min).log2();
    let mut best: Option<(DegreeChoice<f64>, usize)> = None;
    for seed_bits in 1..=need.ceil() as usize {
        let Ok(choice) = degree_for_precision(
            |x: f64| 1.0 / x.sqrt(),
            min,
            1.0,
            Parity::Full,
            seed_bits as f64,
            MAX_SEED_DEGREE,
            SplitStrategy::MinDepth,
        ) else {
            continue;
        };
        let mut err = choice.minimax.error;
        if err > 0.5 {
            continue;
        }
        let mut steps = 0;
        while steps == 0 || error_bits(err) < need {
            err = 1.5 * err * err + 0.5 * err * err * err;
            steps += 1;
        }
        let cost = (choice.depth + 2 * steps, choice.degree);
        if best.as_ref().is_none_or(|(c, s)| cost < (c.depth + 2 * s, c.degree)) {
            best = Some((choice, steps));
        }
    }
    best.ok_or_else(|| anyhow::anyhow!("rsqrt: no seed of degree at most {MAX_SEED_DEGREE} on [{min}, 1]"))
}

/// Goldschmidt iterations for [`TARGET_BITS`] of absolute error on `[min, 1]`: `i` iterations leave a relative error of
/// `(1 - x)^(2^i)`, largest at `min`, where the result is `1/min`.
fn inverse_iters(min: f64) -> usize {
    let need = TARGET_BITS + (1.0 / min).log2();
    (need / -(1.0 - min).log2()).log2().ceil() as usize
}

/// The client side of a point: the module, the secret, and one arena for key generation, encoding, encryption and
/// decryption.
struct Client<BE: Backend> {
    module: Module<BE>,
    k: usize,
    sk_raw: BackendGLWESecret<BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    arena: ScratchOwned<BE>,
    xe: Source,
    xa: Source,
}

impl<BE> Client<BE>
where
    BE: Backend<Ring = Standard> + 'static,
    Module<BE>: LibmModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    fn new(k: usize) -> Self {
        let module = Module::<BE>::new(n() as u64);
        let layout = layout(k);
        let mut xs = Source::new([0u8; 32]);
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, HW, &mut xs);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let (tsk, atk) = (tsk_layout(k), atk_layout(k));
        let bytes = module
            .ckks_all_ops_with_atk_tmp_bytes(&layout, &layout, &tsk, &atk)
            .max(<Module<BE> as CKKSEncodingHostOps<BE, f64>>::ckks_reim_tmp_bytes(
                &module,
                slots(),
            ))
            .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk))
            .max(module.glwe_tensor_key_prepare_tmp_bytes(&tsk))
            .max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk))
            .max(module.glwe_automorphism_key_prepare_tmp_bytes(&atk));
        Self {
            module,
            k,
            sk_raw,
            sk,
            arena: ScratchOwned::alloc(bytes),
            xe: Source::new([2u8; 32]),
            xa: Source::new([1u8; 32]),
        }
    }

    fn tensor_key(&mut self) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE> {
        let layout = tsk_layout(self.k);
        let mut tsk = self.module.glwe_tensor_key_alloc_from_infos(&layout);
        self.module
            .glwe_tensor_key_encrypt_sk(&mut tsk, &self.sk_raw, &mut self.xe, &mut self.xa, &mut self.arena.borrow());
        let mut prepared = self.module.alloc_tensor_key_prepared_from_infos(&layout);
        self.module
            .glwe_tensor_key_prepare(&mut prepared, &tsk, &mut self.arena.borrow());
        prepared
    }

    /// One prepared rotation key for each Galois element.
    fn rotation_keys(&mut self, elements: Vec<i64>) -> HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> {
        let layout = atk_layout(self.k);
        elements
            .into_iter()
            .map(|p| {
                let mut atk = self.module.glwe_automorphism_key_alloc_from_infos(&layout);
                self.module.glwe_automorphism_key_encrypt_sk(
                    &mut atk,
                    p,
                    &self.sk_raw,
                    &mut self.xe,
                    &mut self.xa,
                    &mut self.arena.borrow(),
                );
                let mut prepared = self.module.glwe_automorphism_key_prepared_alloc_from_infos(&layout);
                self.module
                    .glwe_automorphism_key_prepare(&mut prepared, &atk, &mut self.arena.borrow());
                (p, prepared)
            })
            .collect()
    }

    /// A fresh ciphertext of width `k` holding `values` in its real parts.
    fn encrypt(&mut self, values: &[f64]) -> CKKSCiphertextOwned<BE> {
        let mut pt = self
            .module
            .ckks_pt_vec_alloc(slots(), base2k(), TorusPrecision((LOG_DELTA + HEADROOM) as u32));
        pt.set_meta(layout(self.k).meta);
        self.module
            .ckks_encode_reim_into(&mut pt, values, &vec![0.0; slots()], &mut self.arena.borrow())
            .unwrap();
        let mut ct = self.alloc();
        self.module
            .ckks_encrypt_sk(&mut ct, &pt, &self.sk, &mut self.xe, &mut self.xa, &mut self.arena.borrow())
            .unwrap();
        ct
    }

    /// A destination of width `k`.
    fn alloc(&self) -> CKKSCiphertextOwned<BE> {
        let mut ct = self.module.ckks_ciphertext_alloc(base2k(), TorusPrecision(self.k as u32));
        ct.set_meta(layout(self.k).meta);
        ct
    }

    /// The real parts of the slots of `ct`.
    fn decrypt(&mut self, ct: &CKKSCiphertextOwned<BE>) -> Vec<f64> {
        let log_delta = ct.log_delta();
        let mut pt = self
            .module
            .ckks_pt_vec_alloc(slots(), base2k(), TorusPrecision((log_delta + DECODE_HEADROOM) as u32));
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        });
        self.module
            .ckks_decrypt(&mut pt, ct, &self.sk, &mut self.arena.borrow())
            .unwrap();
        let (mut re, mut im) = (vec![0.0; slots()], vec![0.0; slots()]);
        self.module
            .ckks_decode_reim_into(&pt, &mut re, &mut im, &mut self.arena.borrow())
            .unwrap();
        re
    }
}

/// A built point: one call of its function, what its first call returned against the cleartext, and what it consumed.
struct Point {
    call: Box<dyn FnMut()>,
    have: Vec<f64>,
    want: Vec<f64>,
    consumed_bits: usize,
    output_log_budget: usize,
    rotation_keys: usize,
    details: Value,
}

fn rsqrt<BE>(min: f64) -> Result<Point>
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (seed, steps) = rsqrt_seed(min)?;
    let host = Module::<HostBytesBackend>::new(n() as u64);
    let approximation =
        PolynomialApproximation::from_polynomial(&seed.minimax.poly, base2k(), coeff_meta(), SplitStrategy::MinDepth, &host)?;
    let consumed = approximation.consumed_bits(LOG_DELTA) + 2 * steps * LOG_DELTA;
    let mut client = Client::<BE>::new(ct_k(consumed));
    let approximation = approximation.map_plaintexts(|pt| upload_pt(&client.module, pt));
    let tsk = client.tensor_key();
    let x = spread(min, 1.0);
    let x_half: Vec<f64> = x.iter().map(|v| v / 2.0).collect();
    let (ct_x, ct_half) = (client.encrypt(&x), client.encrypt(&x_half));
    let mut y = client.alloc();
    let tsk_infos = tsk_layout(client.k);
    let module = &client.module;
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .ckks_approximation_tmp_bytes(&y, &ct_x, &approximation, &tsk_infos)
            .max(module.ckks_rsqrt_tmp_bytes(&y, &ct_half, &tsk_infos)),
    );
    // Each call widens the destination back, a change of label, so that every call runs as the first, checked one.
    let k = TorusPrecision(client.k as u32);
    let op = move |module: &Module<BE>, y: &mut CKKSCiphertextOwned<BE>, scratch: &mut ScratchOwned<BE>| {
        y.set_k(k);
        module
            .ckks_eval_approximation(y, &ct_x, &approximation, &tsk, &mut scratch.borrow())
            .unwrap();
        module.ckks_rsqrt(y, &ct_half, steps, &tsk, &mut scratch.borrow()).unwrap();
        module.synchronize();
    };
    op(&client.module, &mut y, &mut scratch);
    let have = client.decrypt(&y);
    let output_log_budget = y.log_budget();
    let module = client.module;
    Ok(Point {
        call: Box::new(move || op(&module, &mut y, &mut scratch)),
        have,
        want: x.iter().map(|v| 1.0 / v.sqrt()).collect(),
        consumed_bits: consumed,
        output_log_budget,
        rotation_keys: 0,
        details: json!({
            "function": "ckks_eval_approximation + ckks_rsqrt",
            "domain": [min, 1.0],
            "target_bits": TARGET_BITS,
            "seed_degree": seed.degree,
            "seed_depth": seed.depth,
            "seed_bits": seed.bits(),
            "newton_steps": steps,
        }),
    })
}

fn softmax<BE>(target_bits: f64) -> Result<Point>
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let host = Module::<HostBytesBackend>::new(n() as u64);
    let options = SoftmaxOptions {
        target_bits,
        ..SoftmaxOptions::default()
    };
    let (a, b) = SOFTMAX_DOMAIN;
    let plan = SoftmaxPlan::from_precision(a, b, base2k(), coeff_meta(), options, &host)?;
    let consumed = plan.consumed_bits(LOG_DELTA);
    let (depth, reciprocal_iters, exp_degree) = (plan.depth(), plan.reciprocal_iters(), plan.exp().degree());
    let mut client = Client::<BE>::new(ct_k(consumed));
    let plan = plan.map_plaintexts(|pt| upload_pt(&client.module, pt));
    let tsk = client.tensor_key();
    let x = spread(a, b);
    let ct_x = client.encrypt(&x);
    let atks = client.rotation_keys(client.module.ckks_sum_slots_galois_elements(&ct_x));
    let rotation_keys = atks.len();
    let mut res = client.alloc();
    let mut scratch = ScratchOwned::<BE>::alloc(client.module.ckks_softmax_tmp_bytes(
        &res,
        &ct_x,
        &tsk_layout(client.k),
        &atk_layout(client.k),
        &plan,
    ));
    // Each call widens the destination back, a change of label, so that every call runs as the first, checked one.
    let k = TorusPrecision(client.k as u32);
    let op = move |module: &Module<BE>, res: &mut CKKSCiphertextOwned<BE>, scratch: &mut ScratchOwned<BE>| {
        res.set_k(k);
        module
            .ckks_softmax(res, &ct_x, &plan, &tsk, &atks, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
    };
    op(&client.module, &mut res, &mut scratch);
    let have = client.decrypt(&res);
    let output_log_budget = res.log_budget();
    let denominator: f64 = x.iter().map(|v| v.exp()).sum();
    let module = client.module;
    Ok(Point {
        call: Box::new(move || op(&module, &mut res, &mut scratch)),
        have,
        want: x.iter().map(|v| v.exp() / denominator).collect(),
        consumed_bits: consumed,
        output_log_budget,
        rotation_keys,
        details: json!({
            "function": "ckks_softmax",
            "domain": [a, b],
            "target_bits": target_bits,
            "exp_degree": exp_degree,
            "reciprocal_iters": reciprocal_iters,
            "depth": depth,
        }),
    })
}

fn inverse<BE>(min: f64) -> Result<Point>
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let iters = inverse_iters(min);
    let consumed = (iters + 1) * LOG_DELTA;
    let mut client = Client::<BE>::new(ct_k(consumed));
    let tsk = client.tensor_key();
    let x = spread(min, 1.0);
    let ct_x = client.encrypt(&x);
    let mut res = client.alloc();
    let module = &client.module;
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .ckks_copy_tmp_bytes(&res, &ct_x)
            .max(module.ckks_goldschmidt_division_tmp_bytes(&res, &tsk_layout(client.k))),
    );
    // The inversion runs in place and narrows its operand, so each call widens it back, a change of label, and starts
    // from a copy of the input, which the timed region includes.
    let k = TorusPrecision(client.k as u32);
    let op = move |module: &Module<BE>, res: &mut CKKSCiphertextOwned<BE>, scratch: &mut ScratchOwned<BE>| {
        res.set_k(k);
        module.ckks_copy(res, &ct_x, &mut scratch.borrow()).unwrap();
        module
            .ckks_inverse_positive_domain(res, iters, &tsk, &mut scratch.borrow())
            .unwrap();
        module.synchronize();
    };
    op(&client.module, &mut res, &mut scratch);
    let have = client.decrypt(&res);
    let output_log_budget = res.log_budget();
    let module = client.module;
    Ok(Point {
        call: Box::new(move || op(&module, &mut res, &mut scratch)),
        have,
        want: x.iter().map(|v| 1.0 / v).collect(),
        consumed_bits: consumed,
        output_log_budget,
        rotation_keys: 0,
        details: json!({
            "function": "ckks_copy + ckks_inverse_positive_domain",
            "domain": [min, 1.0],
            "target_bits": TARGET_BITS,
            "iters": iters,
        }),
    })
}

/// Builds the case `args.case` on `BE`, times its calls, and reports its parameters, modulus and precision.
pub fn run<BE>(args: &CaseArgs) -> Result<Value>
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let start = Instant::now();
    let mut point = match case(&args.case)? {
        Case::Rsqrt { min } => rsqrt::<BE>(min)?,
        Case::Softmax { target_bits } => softmax::<BE>(target_bits)?,
        Case::Inverse { min } => inverse::<BE>(min)?,
    };
    let setup_ns = start.elapsed().as_nanos() as u64;
    let timings = direct(&mut point.call, &|| {}, args.warmup, args.iters)?.expect("direct timing is always available");

    let (mut abs, mut rel) = (0.0f64, 0.0f64);
    for (have, want) in point.have.iter().zip(&point.want) {
        abs = abs.max((have - want).abs());
        rel = rel.max(((have - want) / want).abs());
    }
    let target_bits = point.details["target_bits"].as_f64().unwrap();
    let k = ct_k(point.consumed_bits);
    let key = tsk_layout(k);
    Ok(json!({
        "libm": {
            "case": args.case,
            "log_n": LOG_N,
            "n": n(),
            "slots": slots(),
            "base2k": BASE2K,
            "log_delta": LOG_DELTA,
            "hamming_weight": HW,
            "rotation_keys": point.rotation_keys,
            "details": point.details,
        },
        "modulus": {
            "log_q": k,
            "consumed_bits": point.consumed_bits,
            "output_log_budget": point.output_log_budget,
            "key_log_qp": key.k().as_usize(),
            "key_dsize": DSIZE,
            "key_dnum": key.dnum.as_usize(),
            "key_k_aux": key.k_aux.as_usize(),
        },
        "setup_ns": setup_ns,
        "warmup_ns": timings.warmup_ns,
        "samples_ns": timings.samples_ns,
        "precision": {
            "max_abs_err": abs,
            "abs_bits": error_bits(abs),
            "max_rel_err": rel,
            "rel_bits": error_bits(rel),
            "target_bits": target_bits,
            "meets_target": error_bits(abs) >= target_bits,
        },
    }))
}
