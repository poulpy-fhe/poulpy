//! LUT evaluation via trigonometric Hermite interpolation
//! (<https://eprint.iacr.org/2024/1623>).
//!
//! The interpolation of a `p`-to-`p` LUT `f` is `R(x) = Re(T(x))` for
//! `T(x) = Σ α_k·E(x)^k` in `E(x) = exp(2πi·x)`.

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
    SetBSGSMeta,
    prepared::{GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{
        Basis, CKKSAddOps, CKKSAffineOps, CKKSConjugateOps, CKKSCopyOps, CKKSEvalModOps, CKKSMulOps, CKKSPolynomialEvaluationOps,
        CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, eval_mod::EvalMod},
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial, ComplexPolynomial, Polynomial},
    power_basis::{PowerBasis, PowerBasisGen},
};

/// Builds the LUT power series `Σ (α_k/2)·E^k`; `T + conj(T)` recovers the
/// interpolation of `f`.
pub fn trig_hermite_lut<F>(f: &[F]) -> ComplexPolynomial<F>
where
    F: CKKSScalar + Float + FloatConst,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
{
    let p = f.len();
    let two = F::one() + F::one();
    let half = F::one() / two;
    let pf = F::from_usize(p).expect("p representable");
    let two_pi = two * F::PI();

    let mut re = vec![F::zero(); p];
    let mut im = vec![F::zero(); p];

    let sum: F = f.iter().fold(F::zero(), |acc, &v| acc + v);
    re[0] = half * sum / pf;

    for k in 1..p {
        let scale = two * F::from_usize(p - k).expect("p-k representable") / (pf * pf);
        let (mut sr, mut si) = (F::zero(), F::zero());
        for (l, &fl) in f.iter().enumerate() {
            let angle = two_pi * F::from_usize((k * l) % p).expect("index representable") / pf;
            sr = sr + fl * angle.cos();
            si = si - fl * angle.sin();
        }
        re[k] = half * scale * sr;
        im[k] = half * scale * si;
    }

    ComplexPolynomial::new(Basis::Monomial, re, im)
}

#[allow(clippy::too_many_arguments)]
pub fn ckks_eval_lut<BE, F, K, C, R>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    eval_exp: &EvalMod<F, CKKSPlaintext<BE::OwnedBuf>>,
    lut: &ComplexBSGSPolynomial<CKKSPlaintext<BE::OwnedBuf>>,
    conj_key: &K,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>:
        CKKSEvalModOps<BE> + CKKSPolynomialEvaluationOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE> + CKKSModuleAlloc<BE>,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let (base2k, k) = (res.base2k(), res.max_k());

    let mut e_x = module.ckks_ciphertext_alloc(base2k, k);
    module.ckks_eval_mod(&mut e_x, ct, eval_exp, tsk, scratch)?;

    module.ckks_eval_poly_complex_const_coeffs(res, &e_x, lut, tsk, scratch)?;

    let mut conj = module.ckks_ciphertext_alloc(base2k, k);
    module.ckks_conjugate_into(&mut conj, &*res, conj_key, scratch)?;
    module.ckks_add_assign(res, &conj, scratch)?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn ckks_eval_lut_multi<BE, F, K, C>(
    module: &Module<BE>,
    res: &mut [CKKSCiphertext<BE::OwnedBuf>],
    ct: &C,
    eval_exp: &EvalMod<F, CKKSPlaintext<BE::OwnedBuf>>,
    luts: &[&ComplexBSGSPolynomial<CKKSPlaintext<BE::OwnedBuf>>],
    conj_key: &K,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + CKKSModuleAlloc<BE>,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    ensure!(!luts.is_empty(), "ckks_eval_lut_multi: at least one LUT is required");
    ensure!(
        res.len() == luts.len(),
        "ckks_eval_lut_multi: res/luts length mismatch ({} vs {})",
        res.len(),
        luts.len()
    );
    let (base2k, k) = (res[0].base2k(), res[0].max_k());

    let mut e_x = module.ckks_ciphertext_alloc(base2k, k);
    module.ckks_eval_mod(&mut e_x, ct, eval_exp, tsk, scratch)?;

    let mut x1 = module.ckks_ciphertext_alloc_from_infos(&e_x);
    module.ckks_copy(&mut x1, &e_x, scratch)?;
    let head = &luts[0].re;
    let mut power_basis = PowerBasis::new(head.basis(), x1);
    power_basis.populate(head.degree(), head.log_split(), head.parity(), module, tsk, scratch)?;

    let mut conj = module.ckks_ciphertext_alloc(base2k, k);
    for (res_i, lut) in res.iter_mut().zip(luts) {
        module.ckks_eval_poly_complex_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            res_i,
            lut,
            &power_basis,
            tsk,
            scratch,
        )?;
        module.ckks_conjugate_into(&mut conj, &*res_i, conj_key, scratch)?;
        module.ckks_add_assign(res_i, &conj, scratch)?;
    }

    Ok(())
}

pub fn cos_hermite_binary<F>(
    f0: F,
    f1: F,
    degree: usize,
    k_interval: usize,
    log_interval_reduction: usize,
) -> Result<(Polynomial<F>, [F; 2])>
where
    F: CKKSScalar + Float + FloatConst,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
{
    let two = F::one() + F::one();
    let two_pi = two * F::PI();
    let k_eff = F::from_usize(k_interval).expect("K representable")
        / F::from_usize(1usize << log_interval_reduction).expect("2^r representable");
    let cos = Polynomial::chebyshev_interpolate(degree, -k_eff, k_eff, |x| (two_pi * x).cos())?;
    Ok((cos, [(f0 + f1) / two, (f0 - f1) / two]))
}

#[allow(clippy::too_many_arguments)]
pub fn ckks_eval_lut_binary<BE, C, R>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    cos_bsgs: &BSGSPolynomial<CKKSPlaintext<BE::OwnedBuf>>,
    log_interval_reduction: usize,
    affine: &CKKSPlaintext<BE::OwnedBuf>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSPolynomialEvaluationOps<BE> + CKKSMulOps<BE> + CKKSPow2Ops<BE> + CKKSSubOps<BE> + CKKSAffineOps<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
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
