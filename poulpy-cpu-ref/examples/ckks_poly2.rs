//! End-to-end CKKS example: degree-DEGREE Chebyshev sine approximation over [−1, 1]
//!
//! ```bash
//! cargo run -p poulpy-cpu-ref --features enable-ckks --example ckks_poly2
//! ```
//!
//! This example evaluates
//!
//! `sin(x) ≈ Σ cᵢ·Tᵢ(x)`
//!
//! using a degree-DEGREE Chebyshev interpolation, encoded in BSGS form and
//! evaluated homomorphically via `ckks_eval_poly_real_const_coeffs_from_power_basis`. The example
//! follows the standard six-phase CKKS workflow:
//!
//! 1. `setup`
//! 2. `encoding`
//! 3. `encryption`
//! 4. `evaluation`
//! 5. `decryption`
//! 6. `verification`

use anyhow::Result;
use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos,
    encoding::Encoder,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    leveled::api::{CKKSAllOpsTmpBytes, CKKSDecrypt, CKKSEncrypt, PolynomialEvaluation},
    polynomial::{BSGSPolynomial, Basis, EncodeBSGS, Polynomial},
    power_basis::{PowerBasis, PowerBasisGen},
};
use poulpy_core::{
    EncryptionLayout, GLWETensorKeyEncryptSk,
    layouts::{
        Base2K, Degree, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank,
        TorusPrecision,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory, GLWETensorKeyPrepared},
    },
};
use poulpy_cpu_ref::{FFT64ReimTable, NTT120Ref};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchOwned},
    source::Source,
};

type BackendImpl = NTT120Ref;
type SecretKeyPrepared = GLWESecretPrepared<<BackendImpl as Backend>::OwnedBuf, BackendImpl>;
type TensorKeyPrepared = GLWETensorKeyPrepared<<BackendImpl as Backend>::OwnedBuf, BackendImpl>;

const DEGREE: usize = 31;
const N: usize = 1024;
const M: usize = N / 2;
const BASE2K: usize = 52;
/// Ciphertext capacity for the degree-31 Chebyshev BSGS path at log_delta=45.
/// This stores six 52-bit limbs and uses 300 of the available 312 bits.
const CT_K: usize = 300;
const HW: usize = 192;
const DSIZE: usize = 1;

/// Encoding precision for the input slot vector.
const PREC_CT: CKKSLayout = CKKSLayout {
    glwe_layout: GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(45 + 5),
        rank: Rank(1),
    },
    meta: CKKSMeta {
        log_sparsity: 0,
        log_delta: 45,
    },
};

/// Encoding precision for the BSGS polynomial coefficients.
const COEFF_META: CKKSLayout = CKKSLayout {
    glwe_layout: GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(45 + 1),
        rank: Rank(1),
    },
    meta: CKKSMeta {
        log_sparsity: 0,
        log_delta: 45,
    },
};
const ABS_ERROR_TOLERANCE: f64 = 5e-5;

/// Long-lived objects prepared during setup.
struct SetupArtifacts {
    module: Module<BackendImpl>,
    encoder: Encoder<FFT64ReimTable<f64>>,
    sk: SecretKeyPrepared,
    tsk_prepared: TensorKeyPrepared,
    scratch: ScratchOwned<BackendImpl>,
}

/// Host-side artifacts produced by the encoding phase.
struct EncodingArtifacts {
    x_re: Vec<f64>,
    poly: Polynomial<f64>,
    bsgs: BSGSPolynomial<CKKSPlaintext<Vec<u8>>>,
    pt_znx: CKKSPlaintext<Vec<u8>>,
}

/// Ciphertext produced by the encryption phase.
struct EncryptionArtifacts {
    ct_x: CKKSCiphertext<Vec<u8>>,
}

/// Ciphertext produced by the homomorphic evaluation phase.
struct EvaluationArtifacts {
    ct_sin: CKKSCiphertext<Vec<u8>>,
}

/// Decoded values recovered after decryption.
struct DecryptionArtifacts {
    have_re: Vec<f64>,
}

fn glwe_layout() -> EncryptionLayout<GLWELayout> {
    EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: CT_K.into(),
        rank: Rank(1),
    })
    .unwrap()
}

fn tsk_layout() -> EncryptionLayout<GLWETensorKeyLayout> {
    let k = CT_K + DSIZE * BASE2K;
    let dnum = CT_K.div_ceil(DSIZE * BASE2K);
    EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        k: k.into(),
        rank: Rank(1),
        dsize: DSIZE.into(),
        dnum: dnum.into(),
    })
    .unwrap()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

fn avg_err(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());

    if n == 0 {
        return 0.0;
    }

    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / n as f64
}

fn print_phase(name: &str) {
    println!("\n== {name} ==");
}

fn print_ct_meta(label: &str, ct: &CKKSCiphertext<Vec<u8>>) {
    println!(
        "  {label:<28} log_delta={:>2} log_budget={:>3} k={:>3} limbs={:>2} max_k={:>3}",
        ct.log_delta(),
        ct.log_budget(),
        ct.k(),
        ct.size(),
        ct.max_k().as_usize()
    );
}

fn print_pt_meta(label: &str, pt: &CKKSPlaintext<Vec<u8>>) {
    println!(
        "  {label:<28} log_delta={:>2} log_budget={:>3} k={:>3} limbs={:>2} max_k={:>3}",
        pt.log_delta(),
        pt.log_budget(),
        pt.k(),
        pt.size(),
        pt.max_k().as_usize()
    );
}

/// Phase 1: setup.
fn setup() -> Result<SetupArtifacts> {
    print_phase("setup");
    println!("  polynomial: sin(x) via degree-{DEGREE} Chebyshev interpolation on [-1, 1]");
    println!(
        "  params: n={N}, slots={M}, base2k={BASE2K}, ct_k={CT_K}, prec_ct=({}, {}), coeff=({}, {})",
        PREC_CT.log_delta(),
        PREC_CT.log_budget(),
        COEFF_META.log_delta(),
        COEFF_META.log_budget()
    );

    let module = Module::<BackendImpl>::new(N as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(M)?;

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_layout());
    sk_raw.fill_ternary_hw(HW, &mut source_xs);

    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout());
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    println!("  prepared secret key");

    let ct_infos = module.ckks_ciphertext_alloc_from_infos(&glwe_layout());
    let scratch_bytes = module.ckks_all_ops_tmp_bytes(&ct_infos, &tsk_layout(), &COEFF_META);
    let mut scratch = ScratchOwned::<BackendImpl>::alloc(scratch_bytes);
    println!("  scratch bytes: {scratch_bytes}");

    let mut tsk = module.glwe_tensor_key_alloc_from_infos(&tsk_layout());
    {
        let mut scratch_local = scratch.borrow();
        module.glwe_tensor_key_encrypt_sk(
            &mut tsk,
            &sk_raw,
            &tsk_layout(),
            &mut source_xa,
            &mut source_xe,
            &mut scratch_local,
        );
    }

    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_layout());
    {
        let mut scratch_local = scratch.borrow();
        module.prepare_tensor_key(&mut tsk_prepared, &tsk, &mut scratch_local);
    }
    println!("  prepared tensor key");

    Ok(SetupArtifacts {
        module,
        encoder,
        sk,
        tsk_prepared,
        scratch,
    })
}

/// Phase 2: encoding.
///
/// Computes the degree-DEGREE Chebyshev interpolation of sin, decomposes it in
/// BSGS form, and encodes the input slot vector.
fn encoding(setup: &SetupArtifacts) -> Result<EncodingArtifacts> {
    print_phase("encoding");

    let x_re: Vec<f64> = (0..M).map(|i| 2.0 * i as f64 / (M as f64 - 1.0) - 1.0).collect();
    let x_im = vec![0.0f64; M];

    let poly = Polynomial::chebyshev_interpolate(DEGREE, -1.0_f64, 1.0_f64, f64::sin)?;
    println!("  degree={}, parity={:?}", poly.degree(), poly.parity);

    let host_module = Module::<HostBytesBackend>::new(N as u64);
    let bsgs = poly.encode_bsgs(&host_module, BASE2K.into(), COEFF_META)?;
    println!("  BSGS baby steps: {}, parity={:?}", bsgs.baby_steps().len(), bsgs.parity());

    let mut pt_znx = setup.module.ckks_pt_vec_alloc(BASE2K.into(), PREC_CT.k());
    pt_znx.set_meta(PREC_CT.meta());
    setup.encoder.encode_reim(&mut pt_znx, &x_re, &x_im)?;
    print_pt_meta("encoded plaintext x", &pt_znx);
    println!("  x[0] = {:.6}", x_re[0]);

    Ok(EncodingArtifacts {
        x_re,
        poly,
        bsgs,
        pt_znx,
    })
}

/// Phase 3: encryption.
fn encryption(setup: &mut SetupArtifacts, encoding: &EncodingArtifacts) -> Result<EncryptionArtifacts> {
    print_phase("encryption");

    let mut ct_x = setup.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
    let mut source_xa = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);
    {
        let mut scratch = setup.scratch.borrow();
        setup.module.ckks_encrypt_sk(
            &mut ct_x,
            &encoding.pt_znx,
            &setup.sk,
            &glwe_layout(),
            &mut source_xa,
            &mut source_xe,
            &mut scratch,
        )?;
    }
    print_ct_meta("ciphertext x", &ct_x);

    Ok(EncryptionArtifacts { ct_x })
}

/// Phase 4: evaluation.
///
/// Populates the Chebyshev power basis up to degree DEGREE, then runs the BSGS
/// polynomial evaluation.
fn evaluation(
    setup: &mut SetupArtifacts,
    encoding: &EncodingArtifacts,
    encryption: EncryptionArtifacts,
) -> Result<EvaluationArtifacts> {
    print_phase("evaluation");
    print_ct_meta("input x", &encryption.ct_x);

    println!("  -> populate Chebyshev power basis (degree {DEGREE})");
    let mut pb = PowerBasis::new(Basis::Chebyshev, encryption.ct_x);
    {
        let mut scratch = setup.scratch.borrow();
        let log_split = encoding.bsgs.log_split();
        pb.populate(
            DEGREE,
            log_split,
            encoding.bsgs.parity(),
            &setup.module,
            &setup.tsk_prepared,
            &mut scratch,
        )?;
    }

    println!("  -> BSGS polynomial evaluation");
    let ct_sin = {
        let mut ct_sin = setup.module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
        {
            let mut scratch = setup.scratch.borrow();
            setup
                .module
                .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
                    &mut ct_sin,
                    &encoding.bsgs,
                    &pb,
                    &setup.tsk_prepared,
                    &mut scratch,
                )?;
        }
        let mut scratch = setup.scratch.borrow();
        ct_sin.compact(&setup.module, &mut scratch)?
    };
    print_ct_meta("sin(x)", &ct_sin);

    Ok(EvaluationArtifacts { ct_sin })
}

/// Phase 5: decryption.
fn decryption(setup: &mut SetupArtifacts, evaluation: &EvaluationArtifacts) -> Result<DecryptionArtifacts> {
    print_phase("decryption");

    let mut pt_znx = setup.module.ckks_pt_vec_alloc_from_infos(&evaluation.ct_sin);
    {
        let mut scratch = setup.scratch.borrow();
        setup
            .module
            .ckks_decrypt(&mut pt_znx, &evaluation.ct_sin, &setup.sk, &mut scratch)?;
    }
    print_pt_meta("decrypted plaintext", &pt_znx);

    let mut have_re = vec![0.0; M];
    let mut have_im = vec![0.0; M];
    setup.encoder.decode_reim(&pt_znx, &mut have_re, &mut have_im)?;
    println!("  sin(x[0]) decrypted = {:.15}", have_re[0]);

    Ok(DecryptionArtifacts { have_re })
}

/// Phase 6: verification.
///
/// Compares the decrypted result against the reference f64::sin evaluation.
fn verification(encoding: &EncodingArtifacts, evaluation: &EvaluationArtifacts, decryption: &DecryptionArtifacts) -> Result<()> {
    print_phase("verification");

    let want_re: Vec<f64> = encoding.x_re.iter().map(|&x| encoding.poly.evaluate_on_interval(x)).collect();
    let avg_err_re = avg_err(&decryption.have_re, &want_re);
    let max_err_re = max_err(&decryption.have_re, &want_re);

    print_ct_meta("verified sin(x)", &evaluation.ct_sin);
    println!("  max |have − sin(x)| = {max_err_re:.15}");
    println!("  avg |have − sin(x)| = {avg_err_re:.15}");
    println!("  x[0]   = {:.15}", encoding.x_re[0]);
    println!("  want   = {:.15}", want_re[0]);
    println!("  have   = {:.15}", decryption.have_re[0]);

    assert!(
        max_err_re <= ABS_ERROR_TOLERANCE,
        "max error {max_err_re:.15} exceeds {ABS_ERROR_TOLERANCE:.15}"
    );
    println!("  status: PASS");

    Ok(())
}

fn main() -> Result<()> {
    let mut setup_artifacts = setup()?;
    let encoding_artifacts = encoding(&setup_artifacts)?;
    let encryption_artifacts = encryption(&mut setup_artifacts, &encoding_artifacts)?;
    let evaluation_artifacts = evaluation(&mut setup_artifacts, &encoding_artifacts, encryption_artifacts)?;
    let decryption_artifacts = decryption(&mut setup_artifacts, &evaluation_artifacts)?;
    verification(&encoding_artifacts, &evaluation_artifacts, &decryption_artifacts)
}
