use poulpy_core::{
    EncryptionLayout,
    layouts::{
        GGLWEInfos, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        Rank, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSMod1Ops},
    default::mod1::{Mod1Parameters, Mod1ParametersLiteral, Mod1Type},
    encoding::reim::Encoder,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode, ckks_encrypt_with_prec, gen_sk_with_raw, gen_tsk,
    precision_stats, upload_pt,
};

#[derive(Clone, Copy, Debug)]
struct Mod1TestParams {
    pub n: usize,
    pub base2k: usize,
    pub k: usize,
    pub prec: CKKSMeta,
    pub hw: usize,
    pub dsize: usize,
}

impl Mod1TestParams {
    fn glwe_layout(&self) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: self.k.into(),
            rank: Rank(1),
        })
        .unwrap()
    }

    fn tsk_layout(&self) -> EncryptionLayout<GLWETensorKeyLayout> {
        let k = self.k + self.dsize * self.base2k;
        let dnum = k.div_ceil(self.dsize * self.base2k);
        EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: k.into(),
            rank: Rank(1),
            dsize: self.dsize.into(),
            dnum: dnum.into(),
        })
        .unwrap()
    }

    fn as_test_params(&self) -> super::CKKSTestParams {
        super::CKKSTestParams {
            n: self.n,
            base2k: self.base2k,
            k: self.k,
            prec: self.prec,
            hw: self.hw,
            dsize: self.dsize,
        }
    }
}

fn mod1_params(n: usize, base2k: usize, log_delta: usize, depth: usize) -> Mod1TestParams {
    let log_budget = depth * log_delta + base2k + log_delta;
    let k = (log_delta + log_budget).next_multiple_of(base2k);
    Mod1TestParams {
        n,
        base2k,
        k,
        prec: CKKSMeta {
            log_delta,
            log_budget,
        },
        hw: 192,
        dsize: 1,
    }
}

fn alloc_scratch_mod1<BE>(params: &Mod1TestParams, module: &Module<BE>) -> ScratchOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut ct = module.ckks_ciphertext_alloc_from_infos(&params.glwe_layout());
    ct.set_meta(params.prec);
    let pt_prec = CKKSMeta {
        log_delta: 8,
        log_budget: 10,
    };
    let scratch_size = module.ckks_all_ops_tmp_bytes(&ct, &params.tsk_layout(), &pt_prec);
    ScratchOwned::<BE>::alloc(scratch_size)
}

fn upload_params<BE>(
    module: &Module<BE>,
    host: Mod1Parameters<CKKSPlaintext<Vec<u8>>>,
) -> Mod1Parameters<CKKSPlaintext<BE::OwnedBuf>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    host.map_plaintexts(|pt| upload_pt(module, &pt))
}

fn depth_of(lit: &Mod1ParametersLiteral) -> usize {
    let d = lit.mod1_degree.next_power_of_two().trailing_zeros() as usize;
    let r = match lit.mod1_type {
        Mod1Type::SinContinuous => 0,
        _ => lit.double_angle,
    };
    let inv = if lit.mod1_inv_degree > 0 {
        lit.mod1_inv_degree.next_power_of_two().trailing_zeros() as usize
    } else {
        0
    };
    d + r + inv
}

fn oracle_sin(x: f64, inv: bool) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut y = (two_pi * x).sin();
    if inv {
        y = y.asin();
    }
    y / two_pi
}

fn oracle_pipeline(x: f64, lit: &Mod1ParametersLiteral) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let inv_two_pi = 1.0 / two_pi;
    let scaling = if lit.scaling == 0.0 { 1.0 } else { lit.scaling };
    let double_angle = match lit.mod1_type {
        Mod1Type::SinContinuous => 0,
        _ => lit.double_angle,
    };
    let sc_fac = (1u64 << double_angle) as f64;
    let k_eff = lit.mod1_interval as f64 / sc_fac;

    let s: f64 = if lit.mod1_inv_degree > 0 {
        1.0
    } else {
        (inv_two_pi * scaling).powf(1.0 / sc_fac)
    };

    let mut v = x;
    if matches!(lit.mod1_type, Mod1Type::CosContinuous) {
        v += -0.25 / (lit.mod1_interval as f64);
    }

    let k = lit.mod1_interval as f64;
    let mut p_val: f64 = match lit.mod1_type {
        Mod1Type::SinContinuous => s * (two_pi * k_eff * v).sin(),
        Mod1Type::CosContinuous => s * (two_pi * k_eff * v).cos(),
        Mod1Type::CosDiscrete => s * (two_pi * (k * v - 0.25) / sc_fac).cos(),
    };

    for i in 0..double_angle {
        let exp = 1u32 << (i + 1);
        let dac = s.powi(exp as i32);
        p_val = 2.0 * p_val * p_val - dac;
    }

    if lit.mod1_inv_degree > 0 {
        let n = lit.mod1_inv_degree;
        let mut coeffs = vec![0f64; n + 1];
        coeffs[1] = inv_two_pi * scaling;
        let mut i = 1usize;
        while i + 2 <= n {
            let next = i + 2;
            let num = ((next as i64 - 2) * (next as i64 - 2)) as f64;
            let den = (next as i64 * (next as i64 - 1)) as f64;
            coeffs[next] = coeffs[i] * num / den;
            i = next;
        }
        let mut acc = 0f64;
        for c in coeffs.iter().rev() {
            acc = acc * p_val + c;
        }
        p_val = acc;
    }

    p_val
}

#[derive(Clone, Copy)]
enum Oracle {
    Sin,
    Pipeline,
}

fn run_mod1_case<BE, F, E>(
    label: &str,
    base2k: usize,
    log_delta: usize,
    lit: Mod1ParametersLiteral,
    domain: f64,
    required_log2_prec: f64,
    oracle: Oracle,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSMod1Ops<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let depth = depth_of(&lit);
    let params = mod1_params(256, base2k, log_delta, depth);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let module = Module::<BE>::new(params.n as u64);
    let test_params = params.as_test_params();

    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();

    let x_re_raw: Vec<F> = (0..m)
        .map(|i| {
            let t = ((i as f64 + 0.5) / m as f64 - 0.5) * (2.0 * domain);
            F::from_f64(t).unwrap()
        })
        .collect();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    let coeff_meta = CKKSMeta {
        log_delta,
        log_budget: base2k,
    };
    let host_params = Mod1Parameters::from_literal(coeff_meta, params.base2k.into(), lit, &host_module)
        .expect("Mod1Parameters::from_literal");
    let params_be = upload_params(&module, host_params);

    let (sk_raw, sk) = gen_sk_with_raw(&test_params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch_mod1(&params, &module);
    let tsk = gen_tsk(&test_params, &module, &sk_raw, &mut scratch.borrow());

    let ct_input = ckks_encrypt_with_prec(
        &test_params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        params.prec,
        &mut scratch.borrow(),
    );

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_eval_mod1(&mut res, &ct_input, &params_be, &tsk, &mut scratch.borrow())
        .expect("ckks_eval_mod1");

    let (re_out, _im_out) = ckks_decrypt_decode::<BE, F, E>(&test_params, &module, &encoder, &res, &sk, &mut scratch.borrow());

    let inv = lit.mod1_inv_degree > 0;
    let want_re: Vec<F> = x_re_raw
        .iter()
        .map(|&x| {
            let xf = x.to_f64().unwrap();
            let want = match oracle {
                Oracle::Sin => oracle_sin(xf, inv),
                Oracle::Pipeline => oracle_pipeline(xf, &lit),
            };
            F::from_f64(want).unwrap()
        })
        .collect();

    let stats = precision_stats(&re_out, &want_re, log_delta);
    assert!(
        stats.avg_log2_prec >= required_log2_prec,
        "{label}: avg precision {:.1} bits < {required_log2_prec:.1} (worst_err={}, worst_idx={}, got={}, want={})",
        stats.avg_log2_prec,
        stats.worst_err,
        stats.worst_idx,
        stats.worst_got,
        stats.worst_want
    );
}

pub fn test_mod1_sin_continuous_minimal<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSMod1Ops<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = Mod1ParametersLiteral {
        mod1_type: Mod1Type::SinContinuous,
        log_message_ratio: 4,
        mod1_degree: 15,
        mod1_interval: 1,
        double_angle: 0,
        mod1_inv_degree: 0,
        scaling: 1.0,
    };
    run_mod1_case::<BE, F, E>("mod1_sin_continuous_minimal", 19, 30, lit, 0.5, 5.0, Oracle::Sin);
}

pub fn test_mod1_sin_continuous_with_arcsine<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSMod1Ops<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = Mod1ParametersLiteral {
        mod1_type: Mod1Type::SinContinuous,
        log_message_ratio: 4,
        mod1_degree: 31,
        mod1_interval: 1,
        double_angle: 0,
        mod1_inv_degree: 7,
        scaling: 1.0,
    };
    run_mod1_case::<BE, F, E>("mod1_sin_continuous_arcsine", 19, 30, lit, 0.25, 4.0, Oracle::Sin);
}

pub fn test_mod1_cos_discrete<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSMod1Ops<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = Mod1ParametersLiteral {
        mod1_type: Mod1Type::CosDiscrete,
        log_message_ratio: 8,
        mod1_degree: 30,
        mod1_interval: 12,
        double_angle: 3,
        mod1_inv_degree: 0,
        scaling: 1.0,
    };
    run_mod1_case::<BE, F, E>("mod1_cos_discrete", 19, 30, lit, 0.25, 4.0, Oracle::Pipeline);
}

pub fn test_mod1_cos_continuous<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSMod1Ops<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = Mod1ParametersLiteral {
        mod1_type: Mod1Type::CosContinuous,
        log_message_ratio: 4,
        mod1_degree: 31,
        mod1_interval: 4,
        double_angle: 2,
        mod1_inv_degree: 0,
        scaling: 1.0,
    };
    run_mod1_case::<BE, F, E>("mod1_cos_continuous", 19, 30, lit, 0.25, 4.0, Oracle::Pipeline);
}
