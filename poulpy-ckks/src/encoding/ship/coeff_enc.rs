//! Cleartext coefficient encodings of the SHIP bottom ciphertext.
//!
//! The bottom ciphertext `(body, mask)` at the one-limb modulus `q0 =
//! 2^base2k` is consumed in the clear: its public coefficients yield the
//! hoisted plaintext material of Algorithm 1 — `pt0 = Ecd((gamma/(4*i*pi)) *
//! w^{b_i})` from the body phases, and every rotated `Ecd(pi_k(a))` vector
//! gathered from the mask phases without further trigonometry.

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{Base2K, GLWEInfos, LWEInfos};
use poulpy_hal::layouts::{Backend, HostDataRef, Module};

use crate::{
    CKKSMeta,
    api::{CKKSEncodingOps, ShipScalar},
    encoding::paco::coeff_enc::glwe_column_residues,
    layouts::{CKKSCiphertext, CKKSEncodingBuffer, CKKSModuleAlloc, CKKSPlaintext, ShipCoeffEncodings, ShipPlan},
};

/// Phases `(cos, sin)(2*pi*x/q0)` of a residue vector.
fn phases<F: ShipScalar>(residues: &[i64], base2k: usize) -> Result<(Vec<F>, Vec<F>)> {
    let q0 = 1i64 << base2k;
    let q0_f = F::from_i64(q0).context("SHIP bottom modulus is not representable by the working scalar")?;
    let mut cos = Vec::with_capacity(residues.len());
    let mut sin = Vec::with_capacity(residues.len());
    for &x in residues {
        let phase = F::TAU() * F::from_i64(x.rem_euclid(q0)).expect("residues fit the scalar") / q0_f;
        cos.push(phase.cos());
        sin.push(phase.sin());
    }
    Ok((cos, sin))
}

fn encode_reim_pt<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    k_pt: usize,
    log_delta: usize,
    re: &[F],
    im: &[F],
) -> Result<CKKSPlaintext<BE::OwnedBuf>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: ShipScalar,
{
    let mut values = vec![F::zero(); 2 * re.len()];
    values[..re.len()].copy_from_slice(re);
    values[re.len()..].copy_from_slice(im);
    let mut buffer = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&values);
    let mut pt = module.ckks_pt_vec_alloc(base2k, k_pt.into());
    pt.set_meta_checked(CKKSMeta {
        log_delta,
        log_sparsity: 0,
    })?;
    module.ckks_encode_slots_assign_into(&mut pt, &mut buffer)?;
    Ok(pt)
}

/// Scheme-defining SHIP coefficient encoding: reads the bottom ciphertext's
/// public residues at `q0 = 2^base2k` and builds the hoisted plaintext
/// material consumed by the SHIP pipeline — `pt0` (and `pt0_2` when
/// `complex`) at the raised width, plus the `4*theta` rotated `pi` vectors per
/// support slot at the working width. Device backends bypass this host
/// reference path through
/// [`CKKSShipCoeffEncodingImpl`](crate::oep::CKKSShipCoeffEncodingImpl).
pub fn ship_coeff_encodings_host<BE, D, F>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<D>,
    plan: &ShipPlan,
    base2k: Base2K,
    complex: bool,
) -> Result<ShipCoeffEncodings<BE::OwnedBuf>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    D: HostDataRef,
    F: ShipScalar,
{
    let n = plan.n();
    let m = plan.half_n();
    let b2k = base2k.as_usize();
    ensure!(
        (1..63).contains(&b2k) && (b2k as u32) < F::MANTISSA_BITS,
        "SHIP base2k {b2k} must be in [1, 63) and below the scalar's {} mantissa bits",
        F::MANTISSA_BITS
    );
    ensure!(
        ct.n().as_usize() == n,
        "SHIP bottom ciphertext degree {} does not match plan degree {n}",
        ct.n()
    );
    ensure!(
        ct.rank().as_usize() == 1,
        "SHIP bottom ciphertext must have rank 1, got {}",
        ct.rank()
    );
    ensure!(
        ct.base2k() == base2k && ct.k().as_usize() == b2k,
        "SHIP bottom ciphertext must span a single limb of base2k {base2k}, got base2k {} and width {}",
        ct.base2k(),
        ct.k()
    );

    let body = glwe_column_residues(ct.data(), 0, b2k, b2k)?;
    let mask = glwe_column_residues(ct.data(), 1, b2k, b2k)?;

    let kk = plan.raised_k(b2k);
    let ld = plan.log_delta_work();
    let gamma_4pi = F::from_i64(1i64 << plan.log_gamma()).context("SHIP gamma is not representable by the working scalar")?
        / (F::PI() * F::from_f64(4.0).expect("4 is exact"));

    // pt0 carries gamma/(4*i*pi): (gamma/4pi) * (sin(phi_b), -cos(phi_b)).
    let (body_cos, body_sin) = phases::<F>(&body, b2k)?;
    let encode_pt0 = |off: usize| -> Result<CKKSPlaintext<BE::OwnedBuf>> {
        let re: Vec<F> = (0..m).map(|i| gamma_4pi * body_sin[i + off]).collect();
        let im: Vec<F> = (0..m).map(|i| -gamma_4pi * body_cos[i + off]).collect();
        encode_reim_pt(module, base2k, kk, ld, &re, &im)
    };
    let pt0 = encode_pt0(0)?;
    let pt0_2 = if complex { Some(encode_pt0(m)?) } else { None };

    // The four `Rot_{p+c}(Ecd(pi_k(a)))` vectors per candidate and support
    // slot, gathered from the hoisted mask phases.
    let (cos, sin) = phases::<F>(&mask, b2k)?;
    let theta = plan.theta();
    let k_pi = ld + b2k;
    let mut pi = Vec::with_capacity(plan.sparse_hamming_weight());
    for slot in 0..plan.sparse_hamming_weight() {
        let p = plan.mask_rotation(slot);
        let mut slot_pi = Vec::with_capacity(4 * theta);
        for c in 0..theta {
            let rot = (p + c) % m;
            for (half, conj) in [(0, false), (0, true), (m, false), (m, true)] {
                let mut re = Vec::with_capacity(m);
                let mut im = Vec::with_capacity(m);
                for i in 0..m {
                    let src = half + (i + m - rot) % m;
                    re.push(cos[src]);
                    im.push(if conj { -sin[src] } else { sin[src] });
                }
                slot_pi.push(encode_reim_pt(module, base2k, k_pi, ld, &re, &im)?);
            }
        }
        pi.push(slot_pi);
    }

    Ok(ShipCoeffEncodings { pt0, pt0_2, pi })
}
