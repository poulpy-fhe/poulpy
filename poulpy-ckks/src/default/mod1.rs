use anyhow::{Result, anyhow, ensure};
use poulpy_core::layouts::{Base2K, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_core::{GLWENormalize, GLWEZero, ScratchArenaTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, HostBytesBackend, Module, ScratchArena},
};

use rand_distr::num_traits::{Float, FloatConst};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    api::{Basis, CKKSAddOps, CKKSCopyOps, CKKSMulAddOps, CKKSMulOps, CKKSSubOps, Parity},
    cosine,
    default::polynomial_evaluation::PolynomialEvaluationDefault,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, ScratchArenaTakeCKKS},
    polynomial::{BSGSPolynomial, Polynomial},
    power_basis::PowerBasis,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mod1Type {
    CosDiscrete,
    SinContinuous,
    CosContinuous,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mod1ParametersLiteral {
    pub mod1_type: Mod1Type,
    pub log_message_ratio: usize,
    pub mod1_degree: usize,
    pub mod1_interval: usize,
    pub double_angle: usize,
    pub mod1_inv_degree: usize,
    pub scaling: f64,
}

pub struct Mod1Parameters<P> {
    pub mod1_type: Mod1Type,
    pub log_message_ratio: usize,
    pub double_angle: usize,
    pub chebyshev_offset_pt: Option<P>,
    pub double_angle_consts: Vec<P>,
    pub mod1_bsgs: BSGSPolynomial<P>,
    pub mod1_inv_bsgs: Option<BSGSPolynomial<P>>,
}

impl Mod1Parameters<CKKSPlaintext<Vec<u8>>> {
    pub fn from_literal<F>(
        coeff_meta: CKKSMeta,
        base2k: Base2K,
        lit: Mod1ParametersLiteral,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst,
    {
        ensure!(
            !(lit.mod1_type == Mod1Type::SinContinuous && lit.double_angle != 0),
            "SinContinuous requires double_angle = 0"
        );
        ensure!(
            !(lit.mod1_type == Mod1Type::CosDiscrete && lit.mod1_degree < 2 * (lit.mod1_interval - 1)),
            "CosDiscrete requires mod1_degree >= 2*(K-1)"
        );
        ensure!(lit.double_angle < 31, "double_angle must be < 31");

        let double_angle = match lit.mod1_type {
            Mod1Type::SinContinuous => 0,
            _ => lit.double_angle,
        };

        let scaling_f64 = if lit.scaling == 0.0 { 1.0 } else { lit.scaling };
        let scaling = F::from_f64(scaling_f64).expect("scaling must be a finite scalar");
        let sc_fac = F::from_u64(1u64 << double_angle).expect("2^double_angle must fit");
        let k_eff = F::from_usize(lit.mod1_interval).expect("mod1_interval must fit") / sc_fac;

        let two = F::one() + F::one();
        let two_pi = two * F::PI();
        let inv_two_pi = F::one() / two_pi;

        let mut mod1_inv_poly_opt: Option<Polynomial<F>> = None;
        let s: F = if lit.mod1_inv_degree > 0 {
            let n = lit.mod1_inv_degree;
            ensure!(!n.is_multiple_of(2), "mod1_inv_degree must be odd");
            let mut coeffs = vec![F::zero(); n + 1];
            coeffs[1] = inv_two_pi * scaling;
            let mut i = 1usize;
            while i + 2 <= n {
                let next = i + 2;
                let num = F::from_i64((next as i64 - 2) * (next as i64 - 2)).expect("num fits");
                let den = F::from_i64(next as i64 * (next as i64 - 1)).expect("den fits");
                coeffs[next] = coeffs[i] * num / den;
                i = next;
            }
            mod1_inv_poly_opt = Some(Polynomial::new_with_parity(Basis::Monomial, coeffs, Parity::Odd));
            F::one()
        } else {
            (inv_two_pi * scaling).powf(F::one() / sc_fac)
        };

        let mut mod1_poly: Polynomial<F> = match lit.mod1_type {
            Mod1Type::SinContinuous => Polynomial::chebyshev_interpolate(lit.mod1_degree, -k_eff, k_eff, |x| (two_pi * x).sin())?,
            Mod1Type::CosContinuous => Polynomial::chebyshev_interpolate(lit.mod1_degree, -k_eff, k_eff, |x| (two_pi * x).cos())?,
            Mod1Type::CosDiscrete => {
                let coeffs = cosine::approximate_cos::<F>(
                    lit.mod1_interval,
                    lit.mod1_degree,
                    (1u64 << lit.log_message_ratio) as f64,
                    double_angle,
                );
                // cos(2π·(x-1/4)/2^r) is not even in x; Parity::Full preserves
                // the odd-degree Chebyshev coefficients in BSGS evaluation.
                Polynomial::new_with_parity(Basis::Chebyshev, coeffs, Parity::Full)
            }
        };
        match lit.mod1_type {
            Mod1Type::SinContinuous => mod1_poly.parity = Parity::Odd,
            Mod1Type::CosContinuous => mod1_poly.parity = Parity::Even,
            Mod1Type::CosDiscrete => {}
        }

        for c in mod1_poly.coeffs.iter_mut() {
            *c = *c * s;
        }

        let mod1_bsgs = mod1_poly.encode_bsgs(module, base2k, coeff_meta)?;
        let mod1_inv_bsgs = match mod1_inv_poly_opt {
            Some(p) => Some(p.encode_bsgs(module, base2k, coeff_meta)?),
            None => None,
        };

        let mut double_angle_consts: Vec<CKKSPlaintext<Vec<u8>>> = Vec::with_capacity(double_angle);
        for i in 0..double_angle {
            let exp = 1i32 << (i + 1);
            let val = s.powi(exp);
            double_angle_consts.push(encode_scalar(module, base2k, coeff_meta, val)?);
        }

        // CosContinuous polynomial approximates cos(2π·x); the −1/4 phase
        // shift needed by the double-angle composition is added externally.
        // CosDiscrete bakes that shift into cosine_approx's target function.
        let chebyshev_offset_pt = match lit.mod1_type {
            Mod1Type::SinContinuous | Mod1Type::CosDiscrete => None,
            Mod1Type::CosContinuous => {
                let neg_quarter = F::from_f64(-0.25).expect("-0.25 must fit");
                let off = neg_quarter / F::from_usize(lit.mod1_interval).expect("mod1_interval must fit");
                Some(encode_scalar(module, base2k, coeff_meta, off)?)
            }
        };

        Ok(Self {
            mod1_type: lit.mod1_type,
            log_message_ratio: lit.log_message_ratio,
            double_angle,
            chebyshev_offset_pt,
            double_angle_consts,
            mod1_bsgs,
            mod1_inv_bsgs,
        })
    }
}

impl<P> Mod1Parameters<P> {
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(P) -> Q) -> Mod1Parameters<Q> {
        let Self {
            mod1_type,
            log_message_ratio,
            double_angle,
            chebyshev_offset_pt,
            double_angle_consts,
            mod1_bsgs,
            mod1_inv_bsgs,
        } = self;
        Mod1Parameters {
            mod1_type,
            log_message_ratio,
            double_angle,
            chebyshev_offset_pt: chebyshev_offset_pt.map(&mut f),
            double_angle_consts: double_angle_consts.into_iter().map(&mut f).collect(),
            mod1_bsgs: mod1_bsgs.map_baby_steps(&mut f),
            mod1_inv_bsgs: mod1_inv_bsgs.map(|p| p.map_baby_steps(&mut f)),
        }
    }

    pub fn map_plaintexts_ref<Q>(&self, mut f: impl FnMut(&P) -> Q) -> Mod1Parameters<Q> {
        Mod1Parameters {
            mod1_type: self.mod1_type,
            log_message_ratio: self.log_message_ratio,
            double_angle: self.double_angle,
            chebyshev_offset_pt: self.chebyshev_offset_pt.as_ref().map(&mut f),
            double_angle_consts: self.double_angle_consts.iter().map(&mut f).collect(),
            mod1_bsgs: self.mod1_bsgs.map_baby_steps_ref(&mut f),
            mod1_inv_bsgs: self.mod1_inv_bsgs.as_ref().map(|p| p.map_baby_steps_ref(&mut f)),
        }
    }
}

fn encode_scalar<F: CKKSScalar>(
    module: &Module<HostBytesBackend>,
    base2k: Base2K,
    coeff_meta: CKKSMeta,
    value: F,
) -> Result<CKKSPlaintext<Vec<u8>>> {
    let mut pt = module.ckks_pt_coeffs_alloc(1, base2k, coeff_meta);
    pt.encode_host_floats(&[value]).map_err(|e| anyhow!("encode_scalar: {e}"))?;
    Ok(pt)
}

pub trait CKKSMod1OpsDefault<BE: Backend> {
    fn ckks_eval_mod1_default<R, C, P, T>(
        &self,
        res: &mut R,
        ct: &C,
        params: &Mod1Parameters<P>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: PolynomialEvaluationDefault<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSCopyOps<BE>
            + CKKSModuleAlloc<BE>
            + GLWENormalize<BE>
            + GLWEZero<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>;
}

impl<BE: Backend> CKKSMod1OpsDefault<BE> for Module<BE>
where
    Module<BE>: PolynomialEvaluationDefault<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWENormalize<BE>
        + GLWEZero<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
{
    fn ckks_eval_mod1_default<R, C, P, T>(
        &self,
        res: &mut R,
        ct: &C,
        params: &Mod1Parameters<P>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        eval_mod1(self, res, ct, params, tsk, scratch)
    }
}

fn eval_mod1<R, C, P, T, BE: Backend>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    params: &Mod1Parameters<P>,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: PolynomialEvaluationDefault<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWENormalize<BE>
        + GLWEZero<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
{
    let mut t1 = module.ckks_ciphertext_alloc_from_infos(ct);
    module.ckks_copy(&mut t1, ct, scratch)?;
    if let Some(off) = params.chebyshev_offset_pt.as_ref() {
        module.ckks_add_pt_const_assign(&mut t1, 0, off, 0, scratch)?;
    }

    let mut power_basis = PowerBasis::new(Basis::Chebyshev, t1);
    power_basis.populate(
        params.mod1_bsgs.degree(),
        params.mod1_bsgs.log_split(),
        params.mod1_bsgs.parity(),
        module,
        tsk,
        scratch,
    )?;

    module.ckks_eval_poly_real_const_coeffs_from_power_basis_default::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
        res,
        &params.mod1_bsgs,
        &power_basis,
        tsk,
        scratch,
    )?;

    for i in 0..params.double_angle {
        let dac = &params.double_angle_consts[i];
        scratch.scope(|local| -> Result<()> {
            let (mut work, local) = local.take_compact_ckks_ciphertext_scratch(&*res);
            let (mut snapshot, mut local) = local.take_compact_ckks_ciphertext_scratch(&*res);
            module.ckks_copy(&mut work, &*res, &mut local)?;
            module.ckks_square_assign(&mut work, tsk, &mut local)?;
            module.ckks_copy(&mut snapshot, &work, &mut local)?;
            module.ckks_add_assign(&mut work, &snapshot, &mut local)?;
            module.ckks_sub_pt_const_assign(&mut work, 0, dac, 0, &mut local)?;
            module.ckks_copy(res, &work, &mut local)?;
            Ok(())
        })?;
    }

    if let Some(inv) = params.mod1_inv_bsgs.as_ref() {
        let compact_k = res.effective_k();
        let mut t1_inv = module.ckks_ciphertext_alloc(res.base2k(), compact_k.into());
        t1_inv.set_meta(res.meta());
        module.ckks_copy(&mut t1_inv, &*res, scratch)?;
        let mut pb = PowerBasis::new(Basis::Monomial, t1_inv);
        pb.populate(inv.degree(), inv.log_split(), inv.parity(), module, tsk, scratch)?;
        module.ckks_eval_poly_real_const_coeffs_from_power_basis_default::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            res, inv, &pb, tsk, scratch,
        )?;
    }

    Ok(())
}
