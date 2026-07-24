//! Cleartext coefficient encodings of the exhausted input ciphertext (paper
//! Algorithm 3, getCoeffEnc).
//!
//! The input ciphertext `ct = (ct0, ct1)` at the base modulus `q = 2^log_q`
//! is consumed **in the clear**: its coefficients are public, and this module
//! turns them into the four packed slot vectors β_t that the homomorphic
//! pipeline multiplies against the bootstrapping keys `bsk_t`. No FHE
//! arithmetic is involved; in the real bootstrap these vectors are simply
//! CKKS-encoded as plaintexts.
//!
//! Decryption is rewritten as `m = Σ_v d_v · X^{4h·u_v} · a_v mod q` with the
//! public polynomials `a_0 = ct0 + ct1` and `a_v = X^v · ct1` (`v ≥ 1`). Only
//! coefficients of `a_v` at indices `4h·j` matter; [`a_tilde`] reads them.
//! Their circle-group embeddings `ψ(ã_{v,jk+r})` fill the degree-`< C`
//! polynomials `b̃_v^{(r)}` packed (per chunk `r`, one `2C`-wide block per
//! `v`) exactly like the σ_t vectors of [`key`](super::key).

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{Base2K, GLWEInfos, LWEInfos};
use poulpy_hal::layouts::{Backend, HostDataRef, Module, VecZnx};

use crate::api::{CKKSEncodingOps, PaCoScalar};
use crate::{
    CKKSMeta,
    layouts::{CKKSEncodingBuffer, CKKSModuleAlloc},
};

use crate::layouts::{CKKSCiphertext, CKKSPlaintext};

use std::fmt::Debug;

use super::cpx::Cpx;
use crate::default::dft::DftScalar;
use crate::layouts::paco::{
    plan::PaCoPlan,
    secret::{PaCoPackingWorkspace, pack_chunk_into_with},
};

/// The circle-group embedding `ψ(a) = exp(2πi·a / 2^log_q)`, computed at the
/// working precision `F`.
///
/// ψ depends only on `a mod 2^log_q`: any signed representative maps to the
/// same point on the circle, so callers never need to reduce.
pub(crate) fn psi<F: DftScalar>(a: i64, log_q: u32) -> Result<Cpx<F>> {
    ensure!(log_q > 0, "psi modulus exponent must be positive");
    let modulus = 1u64
        .checked_shl(log_q)
        .with_context(|| format!("psi modulus 2^{log_q} does not fit u64"))?;
    let a = F::from(a).context("psi input cannot be represented by the working scalar")?;
    let modulus = F::from(modulus).context("psi modulus cannot be represented by the working scalar")?;
    let two = F::one() + F::one();
    let theta = two * F::PI() * a / modulus;
    Ok(Cpx::new(theta.cos(), theta.sin()))
}

fn validate_coeff_inputs(ct0: &[i64], ct1: &[i64], p: &PaCoPlan) -> Result<usize> {
    p.check().context("invalid PaCo plan for coefficient extraction")?;
    ensure!(
        ct0.len() == p.n(),
        "PaCo body coefficient vector has length {}, expected degree N = {}",
        ct0.len(),
        p.n(),
    );
    ensure!(
        ct1.len() == p.n(),
        "PaCo mask coefficient vector has length {}, expected degree N = {}",
        ct1.len(),
        p.n(),
    );
    4usize.checked_mul(p.h()).context("4*h overflows usize")
}

fn a_tilde_validated(ct0: &[i64], ct1: &[i64], p: &PaCoPlan, h4: usize, v: usize, j: usize) -> Result<i64> {
    ensure!(v < h4, "PaCo coefficient class v must be in [0, 4*h = {h4}), got {v}");
    ensure!(j < p.b(), "PaCo coefficient row j must be in [0, B = {}), got {j}", p.b());
    let idx = h4.checked_mul(j).context("PaCo coefficient index overflows usize")?;
    if v == 0 {
        let body = ct0
            .get(idx)
            .copied()
            .with_context(|| format!("PaCo body coefficient index {idx} exceeds length {}", ct0.len()))?;
        let mask = ct1
            .get(idx)
            .copied()
            .with_context(|| format!("PaCo mask coefficient index {idx} exceeds length {}", ct1.len()))?;
        Ok(body.wrapping_add(mask))
    } else if j == 0 {
        let idx = p.n().checked_sub(v).context("PaCo wrapped coefficient index underflows")?;
        Ok(ct1
            .get(idx)
            .copied()
            .with_context(|| format!("PaCo mask coefficient index {idx} exceeds length {}", ct1.len()))?
            .wrapping_neg())
    } else {
        let idx = idx.checked_sub(v).context("PaCo shifted coefficient index underflows")?;
        ct1.get(idx)
            .copied()
            .with_context(|| format!("PaCo mask coefficient index {idx} exceeds length {}", ct1.len()))
    }
}

/// `ã_{v,j}`: coefficient `4h·j` of the public polynomial `a_v`, as a signed
/// representative modulo `q`. No reduction is applied: [`psi`] and centered
/// representative conversion depend only on the residue class, and wrapping
/// `i64` arithmetic preserves it because `2^64 ≡ 0 (mod q)`. Here
/// `a_0 = ct0 + ct1`; for `v ∈ [1, 4h)`, `a_v = X^v · ct1`, whose coefficient
/// `0` wraps negacyclically to `−ct1[N−v]`.
#[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
pub(crate) fn a_tilde(ct0: &[i64], ct1: &[i64], p: &PaCoPlan, v: usize, j: usize) -> Result<i64> {
    let h4 = validate_coeff_inputs(ct0, ct1, p)?;
    a_tilde_validated(ct0, ct1, p, h4, v, j)
}

/// Fills one raw (pre-packing) β_t chunk after the outer operation has
/// validated the coefficient vectors and loop bounds.
fn beta_raw_chunk_into_validated<F: DftScalar>(
    ct0: &[i64],
    ct1: &[i64],
    p: &PaCoPlan,
    h4: usize,
    t: usize,
    r: usize,
    z: &mut [Cpx<F>],
) -> Result<()> {
    let (h, c, k) = (p.h(), p.c(), p.k());
    let two_c = 2usize.checked_mul(c).context("2*C overflows usize")?;
    let group_start = t.checked_mul(h).context("PaCo beta group offset overflows usize")?;
    z.fill(Cpx::zero());
    for v in 0..h {
        let class = group_start.checked_add(v).context("PaCo beta class index overflows usize")?;
        let block_start = v.checked_mul(two_c).context("PaCo beta block offset overflows usize")?;
        for i in 0..c {
            let row = i
                .checked_mul(k)
                .and_then(|offset| offset.checked_add(r))
                .context("PaCo beta coefficient row overflows usize")?;
            let slot = block_start.checked_add(i).context("PaCo beta slot index overflows usize")?;
            let coefficient = a_tilde_validated(ct0, ct1, p, h4, class, row)?;
            z[slot] = psi(coefficient, p.log_q())?;
        }
    }
    Ok(())
}

/// Transform-parameterized β core shared by reference tests and the backend
/// operation path
/// ([`ckks_coeffs_to_slots_assign`](crate::oep::CKKSEncodingImpl::ckks_coeffs_to_slots_assign)).
pub(crate) fn coeff_encodings_with<F, T>(ct0: &[i64], ct1: &[i64], p: &PaCoPlan, transform: &mut T) -> Result<[Vec<Cpx<F>>; 4]>
where
    F: DftScalar + Debug,
    T: FnMut(&[F], &mut [F], &mut [F]) -> Result<()>,
{
    let h4 = validate_coeff_inputs(ct0, ct1, p)?;
    let mut encodings: [Vec<Cpx<F>>; 4] = std::array::from_fn(|_| Vec::with_capacity(p.half_n()));
    let mut raw = vec![Cpx::zero(); p.slots()];
    let mut workspace = PaCoPackingWorkspace::new(p)?;
    for (t, packed) in encodings.iter_mut().enumerate() {
        for r in 0..p.k() {
            beta_raw_chunk_into_validated(ct0, ct1, p, h4, t, r, &mut raw)?;
            let start = packed.len();
            packed.resize(start + p.slots(), Cpx::zero());
            pack_chunk_into_with(p, transform, &raw, &mut packed[start..], &mut workspace)?;
        }
        ensure!(
            packed.len() == p.half_n(),
            "packed PaCo beta {t} has length {}, expected {}",
            packed.len(),
            p.half_n(),
        );
    }
    Ok(encodings)
}

/// Backend-agnostic PaCo coefficient encoding (Algorithm 3, getCoeffEnc):
/// reads the exhausted rank-one ciphertext's residues at `plan.log_q()`,
/// builds the four β_t slot vectors, and CKKS-encodes them at the
/// bootstrapping-key scale into four backend-owned plaintexts — the
/// input-dependent counterpart of
/// [`PaCoSecretSpec::sigma_slots_reim`](crate::layouts::PaCoSecretSpec::sigma_slots_reim) at key
/// generation, and the complete scheme definition of
/// [`CKKSPaCoOps::ckks_paco_coeff_encodings`](crate::api::CKKSPaCoOps::ckks_paco_coeff_encodings).
///
/// This scheme-defining reference routine reads a host-accessible ciphertext
/// and explicitly stages its scalar blocks into backend buffers. The block FFT
/// and final slot encoding dispatch through the module's
/// [`CKKSEncodingOps`](crate::api::CKKSEncodingOps). Device backends bypass
/// this host reference path and implement the fused, arena-backed operation in
/// [`CKKSPaCoCoeffEncodingImpl`](crate::oep::CKKSPaCoCoeffEncodingImpl).
///
/// Returns an error for incompatible degree/rank/radix metadata,
/// non-representable coefficients, or a failed encoding.
pub fn paco_coeff_encodings_host<BE, D, F>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<D>,
    plan: &PaCoPlan,
    base2k: Base2K,
) -> Result<[CKKSPlaintext<BE::OwnedBuf>; 4]>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    D: HostDataRef,
    F: PaCoScalar,
{
    let log_delta = plan.log_delta_bsk();
    let k_pt = log_delta
        .checked_add(plan.log_beta_budget())
        .context("PaCo coefficient-encoding torus width overflows usize")?;
    ensure!(
        (1..=63).contains(&base2k.as_usize()),
        "PaCo coefficient-encoding base2k must be in [1, 63], got {base2k}",
    );
    ensure!(
        (1..=63).contains(&ct.base2k().as_usize()),
        "PaCo ciphertext base2k must be in [1, 63], got {}",
        ct.base2k(),
    );
    ensure!(
        log_delta <= k_pt,
        "PaCo plaintext scale {log_delta} exceeds plaintext width {k_pt}"
    );
    ensure!(
        u32::try_from(k_pt).is_ok(),
        "PaCo plaintext width {k_pt} exceeds the layout type's u32 range",
    );
    let (ct0, ct1) = exhausted_ciphertext_residues(ct, plan)?;
    let block_coeff_count = 4usize.checked_mul(plan.c()).context("4*C overflows usize")?;
    let mut block_buffer = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&vec![F::zero(); block_coeff_count]);
    let beta = coeff_encodings_with::<F, _>(&ct0, &ct1, plan, &mut |coeffs, re, im| {
        block_buffer.copy_from_host::<BE>(coeffs);
        module.ckks_coeffs_to_slots_assign(&mut block_buffer)?;
        let values = block_buffer.to_host::<BE>();
        re.copy_from_slice(&values[..re.len()]);
        im.copy_from_slice(&values[re.len()..]);
        Ok(())
    })?;

    let mut out = Vec::with_capacity(4);
    let mut re = vec![F::zero(); plan.half_n()];
    let mut im = vec![F::zero(); plan.half_n()];
    let mut encoding_values = vec![F::zero(); plan.n()];
    let mut encoding_buffer = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&encoding_values);
    for (index, b) in beta.iter().enumerate() {
        ensure!(
            b.len() == re.len(),
            "PaCo coefficient encoding {index} has {} slots, expected {}",
            b.len(),
            re.len(),
        );
        for ((re_slot, im_slot), value) in re.iter_mut().zip(im.iter_mut()).zip(b) {
            *re_slot = value.re;
            *im_slot = value.im;
        }
        let mut pt = module.ckks_pt_vec_alloc(base2k, k_pt.into());
        pt.set_meta_checked(CKKSMeta {
            log_sparsity: 0,
            log_delta,
        })?;
        encoding_values[..re.len()].copy_from_slice(&re);
        encoding_values[re.len()..].copy_from_slice(&im);
        encoding_buffer.copy_from_host::<BE>(&encoding_values);
        module.ckks_encode_slots_assign_into(&mut pt, &mut encoding_buffer)?;
        out.push(pt);
    }
    out.try_into().map_err(|values: Vec<_>| {
        anyhow::anyhow!(
            "PaCo coefficient encoding produced {} plaintexts instead of four",
            values.len()
        )
    })
}

/// Reads column `col` of a host-accessible GLWE data buffer as balanced
/// (signed) representatives modulo `2^k` (`k` the ciphertext's torus
/// precision), recomposed from the limbs by [`VecZnx::decode_vec_i64`]. No
/// reduction is applied — every consumer, including [`psi`] and centered
/// representative conversion in the test model, depends only on the residue
/// class.
pub(crate) fn glwe_column_residues<D: HostDataRef>(data: &VecZnx<D>, col: usize, k: usize, base2k: usize) -> Result<Vec<i64>> {
    ensure!(k > 0, "residue modulus exponent k must be positive");
    ensure!(k <= 63, "residues must fit an i64 modulus, got k = {k}");
    ensure!(base2k > 0, "residue limb width base2k must be positive");
    ensure!(base2k <= 63, "residue limb width base2k must be at most 63, got {base2k}");
    ensure!(data.n() > 0, "residue buffer degree must be positive");
    ensure!(
        col < data.cols(),
        "residue column {col} exceeds buffer column count {}",
        data.cols(),
    );
    let limbs = (k - 1)
        .checked_div(base2k)
        .and_then(|q| q.checked_add(1))
        .context("required residue limb count overflows usize")?;
    ensure!(
        limbs <= data.size(),
        "residue modulus k = {k} at base2k = {base2k} needs {limbs} limbs, but the buffer has {}",
        data.size(),
    );
    let mut out = vec![0i64; data.n()];
    data.decode_vec_i64(base2k, col, k, &mut out);
    Ok(out)
}

/// Algorithm 3 (getCoeffEnc) on a **real** exhausted ciphertext: reads the
/// body (`ct0`) and mask (`ct1`) residues modulo `q = 2^{log_q}` off the
/// base2k limbs and builds the four packed β_t slot vectors. Poulpy decrypts
/// as `body + mask·s`, matching the paper's `m = ct0 + s·ct1` — no sign
/// adjustment is needed.
pub(crate) fn exhausted_ciphertext_residues<D: HostDataRef>(
    ct: &CKKSCiphertext<D>,
    p: &PaCoPlan,
) -> Result<(Vec<i64>, Vec<i64>)> {
    ensure!(
        ct.n().as_usize() == p.n(),
        "exhausted ciphertext degree is {}, expected PaCo degree {}",
        ct.n().as_usize(),
        p.n(),
    );
    let k = ct.k().as_usize();
    ensure!(
        k == p.log_q() as usize,
        "exhausted ciphertext modulus (k = {k}) must equal the PaCo base modulus (log_q = {})",
        p.log_q(),
    );
    ensure!(
        ct.rank().as_usize() == 1,
        "PaCo expects rank-1 ciphertexts, got rank {}",
        ct.rank().as_usize()
    );
    ensure!(
        ct.data().n() == p.n(),
        "ciphertext data degree is {}, expected PaCo degree {}",
        ct.data().n(),
        p.n(),
    );
    ensure!(
        ct.data().cols() == 2,
        "rank-1 PaCo ciphertext data must have exactly 2 columns, got {}",
        ct.data().cols(),
    );
    let base2k = ct.base2k().as_usize();
    let ct0 = glwe_column_residues(ct.data(), 0, k, base2k)?;
    let ct1 = glwe_column_residues(ct.data(), 1, k, base2k)?;
    Ok((ct0, ct1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psi_is_a_group_homomorphism() {
        let log_q = 29u32;
        let q = 1i64 << log_q;
        let (a, b) = (123456789i64, q - 987654321 % q);
        let lhs = psi::<f64>(a + b, log_q).unwrap();
        let rhs = psi::<f64>(a, log_q).unwrap() * psi::<f64>(b, log_q).unwrap();
        assert!((lhs - rhs).abs() < 1e-12);
        assert!(
            (psi::<f64>(-a, log_q).unwrap() - psi::<f64>(a, log_q).unwrap().conj()).abs() < 1e-12,
            "ψ(−a) = conj(ψ(a))"
        );
        assert!(
            (psi::<f64>(a - q, log_q).unwrap() - psi::<f64>(a, log_q).unwrap()).abs() < 1e-12,
            "ψ ignores the representative"
        );
    }

    #[test]
    fn a_tilde_matches_negacyclic_monomial_shift() {
        let p = PaCoPlan::new(9, 8, 4, 20).unwrap();
        let n = p.n();
        let mask = p.q_mask() as i64;
        // Deterministic pseudo-random ct1.
        let ct1: Vec<i64> = (0..n)
            .map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) as i64 & mask)
            .collect();
        let ct0: Vec<i64> = (0..n)
            .map(|i| (i as u64).wrapping_mul(0xC2B2AE3D27D4EB4F) as i64 & mask)
            .collect();
        for v in 1..4 * p.h() {
            // Full negacyclic X^v · ct1.
            let mut shifted = vec![0i64; n];
            for (j, s) in shifted.iter_mut().enumerate() {
                *s = if j >= v { ct1[j - v] } else { -ct1[n + j - v] };
            }
            for j in 0..n / (4 * p.h()) {
                assert_eq!(a_tilde(&ct0, &ct1, &p, v, j).unwrap(), shifted[4 * p.h() * j], "v={v} j={j}");
            }
        }
    }
}
