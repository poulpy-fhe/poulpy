//! LUT evaluation via trigonometric Hermite interpolation
//! (<https://eprint.iacr.org/2024/1623>).
//!
//! The interpolation of a `p`-to-`p` LUT `f` is `R(x) = Re(T(x))` for
//! `T(x) = Σ α_k·E(x)^k` in `E(x) = exp(2πi·x)`.

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive};
use poulpy_core::layouts::GetAutomorphismKey;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::{GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos, SlotsKind,
    api::{
        Basis, CKKSAddOps, CKKSAffineOps, CKKSConjugateOps, CKKSCopyOps, CKKSEvalModOps, CKKSMulOps, CKKSPolynomialEvaluationOps,
        CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, EncodedLut, ScratchArenaTakeCKKS, eval_mod::EvalMod},
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial, ComplexPolynomial, Polynomial},
    power_basis::{PowerBasis, PowerBasisGen},
};

/// Builds the LUT power series `Σ (α_k/2)·E^k`; `T + conj(T)` recovers the
/// interpolation of `f`.
pub(crate) fn trig_hermite_lut<F>(f: &[F]) -> Result<ComplexPolynomial<F>>
where
    F: Float + FloatConst + FromPrimitive,
{
    ensure!(!f.is_empty(), "trig_hermite_lut: table must not be empty");
    let p = f.len();
    let two = F::one() + F::one();
    let half = F::one() / two;
    let pf = F::from_usize(p).ok_or_else(|| anyhow!("cannot represent LUT length {p}"))?;
    let two_pi = two * F::PI();

    let mut re = vec![F::zero(); p];
    let mut im = vec![F::zero(); p];

    let sum: F = f.iter().fold(F::zero(), |acc, &v| acc + v);
    re[0] = half * sum / pf;

    for k in 1..p {
        let scale =
            two * F::from_usize(p - k).ok_or_else(|| anyhow!("cannot represent LUT coefficient index {}", p - k))? / (pf * pf);
        let (mut sr, mut si) = (F::zero(), F::zero());
        for (l, &fl) in f.iter().enumerate() {
            let index = (k * l) % p;
            let angle = two_pi * F::from_usize(index).ok_or_else(|| anyhow!("cannot represent LUT sample index {index}"))? / pf;
            sr = sr + fl * angle.cos();
            si = si - fl * angle.sin();
        }
        re[k] = half * scale * sr;
        im[k] = half * scale * si;
    }

    Ok(ComplexPolynomial {
        basis: Basis::Monomial,
        re,
        im,
        a: -F::one(),
        b: F::one(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_eval_lut<BE, F, K, C, R, H>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    eval_exp: &EvalMod<F, CKKSPlaintextOwned<BE>>,
    lut: &ComplexBSGSPolynomial<CKKSPlaintextOwned<BE>>,
    conj_key: &K,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSEvalModOps<BE> + CKKSPolynomialEvaluationOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
    K: GetAutomorphismKey<BE>,
    C: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    R: GLWEToBackendMut<BE, State = Normalized>
        + GLWEToBackendRef<BE, State = Normalized>
        + CKKSCtBounds
        + SetCKKSInfos
        + SetBSGSMeta,
    H: GetTensorKey<BE>,
{
    // Work at the destination's effective width. `res` may reuse a wider
    // allocation after being relabelled with a smaller `k`; using `max_k()`
    // here would make the runtime scratch requirement exceed the public
    // functional-bootstrap scratch query, which is intentionally based on
    // the effective width.
    let layout = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.k(),
        rank: res.rank(),
    };
    scratch.scope(|scratch| {
        let (mut e_x, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout, res.meta());
        module.ckks_eval_mod(&mut e_x, ct, eval_exp, tsk, &mut scratch)?;
        module.ckks_eval_poly_complex_const_coeffs(res, &e_x, lut, tsk, &mut scratch)
    })?;
    scratch.scope(|scratch| {
        let (mut conj, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout, res.meta());
        module.ckks_conjugate_into(&mut conj, &*res, conj_key, &mut scratch)?;
        module.ckks_add_assign(res, &conj, &mut scratch)
    })?;
    // `T + conj(T) = 2·Re(T)` interpolates the (real) table.
    res.set_slots(SlotsKind::Real);

    Ok(())
}

/// Builds the shared `E(x) = exp(2πi·x)` power basis for a batch of general
/// LUTs: one EvalMod, then the union of every LUT's baby-step schedule.
///
/// Equal-arity LUTs have the same message ratio, but their coefficient parity
/// (and, for some split strategies, their BSGS split) can still differ, so the
/// populated schedule must not depend on which LUT happens to be first.
pub(crate) fn ckks_lut_power_basis<BE, F, C, H>(
    module: &Module<BE>,
    ct: &C,
    layout: &GLWELayout,
    eval_exp: &EvalMod<F, CKKSPlaintextOwned<BE>>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<PowerBasis<CKKSCiphertextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + CKKSModuleAlloc<BE>,
    C: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    H: GetTensorKey<BE>,
{
    let series = luts
        .iter()
        .map(|lut| {
            lut.general_series()
                .ok_or_else(|| anyhow!("ckks_lut_power_basis: shared evaluation supports general LUTs only"))
        })
        .collect::<Result<Vec<_>>>()?;
    let head = &series
        .first()
        .ok_or_else(|| anyhow!("ckks_lut_power_basis: at least one LUT is required"))?
        .re;

    let mut x1 = module.ckks_ciphertext_alloc(layout.base2k, layout.k);
    module.ckks_eval_mod(&mut x1, ct, eval_exp, tsk, scratch)?;
    let mut power_basis = PowerBasis::new(head.basis(), x1);
    for lut in &series {
        ensure!(
            lut.re.basis() == head.basis(),
            "ckks_lut_power_basis: all LUTs must use the same polynomial basis"
        );
        power_basis.populate(lut.re.degree(), lut.re.log_split(), lut.re.parity(), module, tsk, scratch)?;
    }
    Ok(power_basis)
}

/// Evaluates one general LUT against a [`ckks_lut_power_basis`] result.
pub(crate) fn ckks_eval_lut_from_basis<BE, K, R, H>(
    module: &Module<BE>,
    res: &mut R,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    power_basis: &PowerBasis<CKKSCiphertextOwned<BE>>,
    conj_key: &K,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSPolynomialEvaluationOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
    K: GetAutomorphismKey<BE>,
    R: GLWEToBackendMut<BE, State = Normalized>
        + GLWEToBackendRef<BE, State = Normalized>
        + CKKSCtBounds
        + SetCKKSInfos
        + SetBSGSMeta,
    H: GetTensorKey<BE>,
{
    let series = lut
        .general_series()
        .ok_or_else(|| anyhow!("ckks_eval_lut_from_basis: shared evaluation supports general LUTs only"))?;
    module.ckks_eval_poly_complex_const_coeffs_from_power_basis::<_, _, CKKSCiphertextOwned<BE>, _, _>(
        res,
        series,
        power_basis,
        tsk,
        scratch,
    )?;
    let layout = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.k(),
        rank: res.rank(),
    };
    scratch.scope(|scratch| {
        let (mut conj, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout, res.meta());
        module.ckks_conjugate_into(&mut conj, &*res, conj_key, &mut scratch)?;
        module.ckks_add_assign(res, &conj, &mut scratch)
    })?;
    // `T + conj(T) = 2·Re(T)` interpolates the (real) table.
    res.set_slots(SlotsKind::Real);
    Ok(())
}

pub(crate) fn cos_hermite_binary<F>(
    f0: F,
    f1: F,
    degree: usize,
    k_interval: usize,
    log_interval_reduction: usize,
) -> Result<(Polynomial<F>, [F; 2])>
where
    F: Float + FloatConst + FromPrimitive + std::fmt::Debug,
{
    let two = F::one() + F::one();
    let two_pi = two * F::PI();
    ensure!(
        log_interval_reduction < usize::BITS as usize,
        "log_interval_reduction must be < {}",
        usize::BITS
    );
    let k_eff = F::from_usize(k_interval).ok_or_else(|| anyhow!("cannot represent interval bound {k_interval}"))?
        / F::from_usize(1usize << log_interval_reduction)
            .ok_or_else(|| anyhow!("cannot represent interval reduction 2^{log_interval_reduction}"))?;
    let cos = Polynomial::chebyshev_interpolate(degree, -k_eff, k_eff, |x| (two_pi * x).cos())?;
    Ok((cos, [(f0 + f1) / two, (f0 - f1) / two]))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_eval_lut_binary<BE, C, R, H>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    cos_bsgs: &BSGSPolynomial<CKKSPlaintextOwned<BE>>,
    log_interval_reduction: usize,
    affine: &CKKSPlaintextOwned<BE>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSPolynomialEvaluationOps<BE> + CKKSMulOps<BE> + CKKSPow2Ops<BE> + CKKSSubOps<BE> + CKKSAffineOps<BE>,
    C: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
    R: GLWEToBackendMut<BE, State = Normalized>
        + GLWEToBackendRef<BE, State = Normalized>
        + CKKSCtBounds
        + SetCKKSInfos
        + SetBSGSMeta,
    H: GetTensorKey<BE>,
{
    module.ckks_eval_poly_real_const_coeffs(res, ct, cos_bsgs, tsk, scratch)?;

    for _ in 0..log_interval_reduction {
        module.ckks_square_assign(res, tsk, scratch)?;
        module.ckks_mul_pow2_assign(res, 1, scratch)?;
        module.ckks_sub_one_assign(res, scratch)?;
    }

    module.ckks_affine_pt_const_assign(res, affine, 0, 1, scratch)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cos_hermite_binary, trig_hermite_lut};

    #[test]
    fn trig_lut_rejects_empty_table() {
        assert!(trig_hermite_lut::<f64>(&[]).is_err());
    }

    #[test]
    fn binary_lut_rejects_oversized_interval_reduction() {
        let error = cos_hermite_binary::<f64>(0.0, 1.0, 3, 1, usize::BITS as usize)
            .err()
            .expect("oversized reduction must fail");
        assert!(error.to_string().contains("log_interval_reduction"));
    }
}
