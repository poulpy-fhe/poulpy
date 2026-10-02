//! Encapsulated modulus raising and the bootstrapping pipeline, including
//! optimized and fallback shapes.
use std::collections::HashMap;

use super::{
    helpers::{Snapshot, fixture_ciphertext, fixture_operand, snapshot, with_scratch},
    keys::{fixture_gglwe, key_layout, prepared_automorphism_key, prepared_gglwe, prepared_tensor_key},
};
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, CoeffsMeta, SlotsKind,
    api::{CKKSAllOpsTmpBytes, CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan,
        BootstrappingTechniques, CKKSModuleAlloc, CKKSPlaintextOwned, DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout,
        EncodedLut, EvalModPlan, EvalModType, EvalRoundPlus, SparseSecretEncapsulation,
    },
    oep::{CKKSBootstrappingImpl, CKKSEncapsulatedModUpImpl},
    polynomial::SplitStrategy,
    test_suite::CKKSTestParams,
};
use poulpy_core::{
    GLWEAdd, GLWEMaskFill,
    layouts::{
        GGLWEInfos, GGLWELayout, GGLWEPrepared, GGLWEPreparedFactory, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared,
        GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWESwitchingKeyLayout, GLWETensorKeyLayout, GLWETensorKeyPrepared,
        GLWETensorKeyPreparedFactory, GLWEToBackendRef, LWEInfos,
        prepared::{GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
    reference::linear_transformation::DiagonalProd,
};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, HostBytesBackend, HostStaged, Module};

fn run<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(Result<(), String>, Snapshot, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    Module<B>: GGLWEPreparedFactory<B> + GLWEMaskFill<B> + GLWEAdd<B>,
{
    let b = params.base2k;
    let small = 3 * b + 1;
    let mut results = Vec::new();
    // Matching radix and multi-limb digits permit the optimized path. Other
    // cases require the general composition; widths also cover partial limbs,
    // no zero-prefix limbs, several full zero-prefix limbs, and rejected raises.
    for (dsize, key_b, extra, scale) in [
        (2, b, 4 * b + 3, 0),
        (2, b, 3 * b + 5, b + 1),
        (2, b, 5, 4),
        (1, b, 2 * b + 1, 1),
        (2, b - 1, 2 * b + 3, 0),
        (2, b, 0, 1),
    ] {
        let src_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: module.n().into(),
                base2k: b.into(),
                k: small.into(),
                rank: params.rank.into(),
            },
            meta: CKKSMeta {
                log_delta: b,
                log_sparsity: 1,
                slots: SlotsKind::Complex,
            },
        };
        let dst_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                k: (small + extra).into(),
                ..src_layout.glwe_layout
            },
            ..src_layout
        };
        let d2s_layout = key_layout(module.n(), b, small, 1, params.rank, params.rank);
        let s2d_layout = key_layout(module.n(), key_b, small + extra, dsize, params.rank, params.rank);
        let mut d2s = module.gglwe_prepared_alloc_from_infos(&d2s_layout);
        let mut s2d = module.gglwe_prepared_alloc_from_infos(&s2d_layout);
        for (prepared, layout, seed) in [(&mut d2s, &d2s_layout, 31), (&mut s2d, &s2d_layout, 37)] {
            let coefficients = fixture_gglwe(module, layout, seed);
            with_scratch::<B, _>(module.gglwe_prepare_tmp_bytes(layout), |scratch| {
                module.gglwe_prepare(prepared, &coefficients, scratch)
            });
        }
        let bytes = B::ckks_encapsulated_mod_up_tmp_bytes(module, &dst_layout, &src_layout, &d2s_layout, &s2d_layout);
        for (seed, lazy) in [(41, false), (43, true)] {
            let mut src = fixture_operand(module, &src_layout, seed, lazy);
            let mut dst = fixture_ciphertext(module, &dst_layout, 47);
            let result = with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_encapsulated_mod_up(
                    module,
                    &mut dst,
                    &mut src,
                    scale,
                    &d2s.to_backend_ref(),
                    &s2d.to_backend_ref(),
                    scratch,
                )
            });
            assert_eq!(result.is_ok(), extra >= scale, "modulus raise acceptance changed");
            if result.is_ok() {
                assert_eq!(
                    dst.meta(),
                    CKKSMeta {
                        log_delta: src_layout.meta.log_delta + scale,
                        ..src_layout.meta
                    }
                );
                assert_eq!(dst.k(), dst_layout.glwe_layout.k);
            }
            results.push((
                result.map_err(|error| error.to_string()),
                snapshot::<B, _>(&src),
                snapshot::<B, _>(&dst),
            ));
        }
    }
    results
}

/// Compare selected implementations using the same canonical inputs and key
/// coefficients. Source mutations, destination metadata, errors, and each
/// implementation's own exact guarded scratch budget are part of the contract.
pub fn test_encapsulated_mod_up_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    Module<BR>: GGLWEPreparedFactory<BR> + GLWEMaskFill<BR> + GLWEAdd<BR>,
    Module<BT>: GGLWEPreparedFactory<BT> + GLWEMaskFill<BT> + GLWEAdd<BT>,
{
    assert_eq!(reference.n(), tested.n());
    assert_eq!(run(params, reference), run(params, tested), "encapsulated ModUp differs");
}

/// Fixture key store of the bootstrapping parity tests: identical coefficients
/// on every backend, each preparing its own representation.
pub struct FixtureKeys<B: Backend> {
    rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<B::OwnedBuf, B>>,
    tensor_key: GLWETensorKeyPrepared<B::OwnedBuf, B>,
    encapsulation_keys: Option<(FixtureSwitchingKey<B>, FixtureSwitchingKey<B>)>,
}

type FixtureSwitchingKey<B> = GGLWEPrepared<<B as Backend>::OwnedBuf, B>;

impl<B: Backend> BootstrappingKeys<B> for FixtureKeys<B>
where
    GLWEAutomorphismKeyPrepared<B::OwnedBuf, B>: GLWEAutomorphismKeyPreparedToBackendRef<B>,
    GLWETensorKeyPrepared<B::OwnedBuf, B>: GLWETensorKeyPreparedToBackendRef<B>,
    GGLWEPrepared<B::OwnedBuf, B>: GGLWEPreparedToBackendRef<B> + GGLWEInfos,
{
    type RotationKeys = HashMap<i64, GLWEAutomorphismKeyPrepared<B::OwnedBuf, B>>;
    type TensorKey = GLWETensorKeyPrepared<B::OwnedBuf, B>;
    type SwitchingKey = GGLWEPrepared<B::OwnedBuf, B>;

    fn rotation_keys(&self) -> &Self::RotationKeys {
        &self.rotation_keys
    }

    fn tensor_key(&self) -> &Self::TensorKey {
        &self.tensor_key
    }

    fn encapsulation_keys(&self) -> Option<(&Self::SwitchingKey, &Self::SwitchingKey)> {
        self.encapsulation_keys.as_ref().map(|(d2s, s2d)| (d2s, s2d))
    }
}

const LOG_MSG_RATIO: usize = 4;

/// A small full-slot recipe: one radix-2 layer per factor and a degree-8 EvalMod.
fn plan(pipeline: BootstrappingPipeline, log_slots: usize, encapsulate: bool, eval_round: bool) -> BootstrappingPlan {
    let meta = CoeffsMeta::from_delta_budget;
    let dft = |kind, log_delta| {
        DFTPlan::new(
            kind,
            vec![(1, 2); log_slots],
            DFTOutputFormat::SplitRealAndImag,
            meta(log_delta, 2),
        )
        .unwrap()
    };
    let s2c_first = pipeline == BootstrappingPipeline::S2CFirst;
    let plan = BootstrappingPlan::new(
        pipeline,
        BootstrappingTechniques {
            sparse_secret_encapsulation: encapsulate.then_some(SparseSecretEncapsulation { hamming_weight: 8 }),
            eval_round_plus: eval_round.then(|| EvalRoundPlus {
                coeffs_to_slots_bypass: dft(DFTType::Encode, 20),
            }),
        },
        dft(DFTType::Encode, 14),
        EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: LOG_MSG_RATIO,
            f_mod_degree: 8,
            f_mod_interval: 4,
            f_mod_log_interval_reduction: 1,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: meta(16, 2),
            f_mod_log_delta: 24,
        },
        dft(DFTType::Decode, 14)
            .with_scaling(if s2c_first { 0.5 } else { (LOG_MSG_RATIO as f64).exp2() })
            .unwrap(),
    )
    .unwrap();
    if s2c_first {
        plan.with_c2s_guard_bits(2).unwrap()
    } else {
        plan
    }
}

fn ct_layout(n: usize, base2k: usize, k: usize, log_delta: usize, slots: SlotsKind) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: 1usize.into(),
        },
        meta: CKKSMeta {
            log_delta,
            log_sparsity: 0,
            slots,
        },
    }
}

fn keys_layout(atk: &GGLWELayout, d2s: &GGLWELayout, s2d: &GGLWELayout, encapsulate: bool) -> BootstrappingKeysLayout {
    BootstrappingKeysLayout {
        automorphism_key: GLWEAutomorphismKeyLayout {
            n: atk.n,
            base2k: atk.base2k,
            dnum: atk.dnum,
            k_aux: atk.k_aux,
            rank: atk.rank_out,
            dsize: atk.dsize,
        },
        tensor_key: GLWETensorKeyLayout {
            n: atk.n,
            base2k: atk.base2k,
            dnum: atk.dnum,
            k_aux: atk.k_aux,
            rank: atk.rank_out,
            dsize: atk.dsize,
        },
        encapsulation: encapsulate.then(|| EncapsulationKeysLayout {
            dense_to_sparse: switching_layout(d2s),
            sparse_to_dense: switching_layout(s2d),
        }),
    }
}

fn switching_layout(key: &GGLWELayout) -> GLWESwitchingKeyLayout {
    GLWESwitchingKeyLayout {
        n: key.n,
        base2k: key.base2k,
        dnum: key.dnum,
        k_aux: key.k_aux,
        rank_in: key.rank_in,
        rank_out: key.rank_out,
        dsize: key.dsize,
    }
}

fn fixture_keys<B>(
    module: &Module<B>,
    plan: &BootstrappingPlan,
    atk: &GGLWELayout,
    d2s: &GGLWELayout,
    s2d: &GGLWELayout,
) -> FixtureKeys<B>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B>
        + GLWETensorKeyPreparedFactory<B>
        + GGLWEPreparedFactory<B>
        + GLWEMaskFill<B>
        + CyclotomicOrder,
{
    let rotation_keys = plan
        .galois_elements(module.n().ilog2() as usize, module.cyclotomic_order())
        .into_iter()
        .chain([-1])
        .map(|p| (p, prepared_automorphism_key(module, atk, p, (p as u8).wrapping_add(71))))
        .collect();
    FixtureKeys {
        rotation_keys,
        tensor_key: prepared_tensor_key(module, atk, 73),
        encapsulation_keys: plan
            .sparse_secret_hamming_weight()
            .map(|_| (prepared_gglwe(module, d2s, 79), prepared_gglwe(module, s2d, 83))),
    }
}

type Outcome = (Result<(), String>, Vec<Snapshot>);

fn run_bootstrap<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<Outcome>
where
    B: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar,
    Module<B>: CKKSDFTOps<B>
        + CKKSDFTMatrixOps<B, F>
        + CKKSEncodingOps<B, F>
        + CKKSEncodingHostOps<B, F>
        + CKKSModuleAlloc<B>
        + CKKSAllOpsTmpBytes<B>
        + GLWEAutomorphismKeyPreparedFactory<B>
        + GLWETensorKeyPreparedFactory<B>
        + GGLWEPreparedFactory<B>
        + GLWEMaskFill<B>
        + CyclotomicOrder,
    CKKSPlaintextOwned<B>: GLWEToBackendRef<B> + CKKSCtBounds + DiagonalProd<B>,
    FixtureKeys<B>: BootstrappingKeys<B, TensorKey = GLWETensorKeyPrepared<B::OwnedBuf, B>> + Sync,
{
    let (n, b) = (module.n(), params.base2k);
    let log_slots = n.ilog2() as usize - 1;
    let log_delta = 12;
    let mut results = Vec::new();
    let host = Module::<HostBytesBackend>::new(n as u64);
    let lut = EncodedLut::binary(
        &host,
        F::from_f64(3.0).unwrap(),
        F::from_f64(1.0).unwrap(),
        8,
        4,
        1,
        b.into(),
        CoeffsMeta::from_delta_budget(16, b),
        SplitStrategy::MinDepth,
    )
    .unwrap();
    for (pipeline, encapsulate, eval_round, slots, functional) in [
        (BootstrappingPipeline::C2SFirst, true, false, SlotsKind::Complex, false),
        (BootstrappingPipeline::C2SFirst, false, true, SlotsKind::Complex, false),
        (BootstrappingPipeline::S2CFirst, false, false, SlotsKind::Complex, false),
        (BootstrappingPipeline::S2CFirst, true, false, SlotsKind::Real, false),
        (BootstrappingPipeline::S2CFirst, false, true, SlotsKind::Real, false),
        (BootstrappingPipeline::S2CFirst, false, false, SlotsKind::Complex, true),
        (BootstrappingPipeline::S2CFirst, false, false, SlotsKind::Real, true),
    ] {
        let mut plan = plan(pipeline, log_slots, encapsulate, eval_round);
        if functional {
            plan = plan.with_functional_bootstrap(&lut).unwrap();
        }
        let log_modulus_in = log_delta + if functional { lut.log_msg_ratio() } else { LOG_MSG_RATIO };
        let k_in = plan.input_k(log_modulus_in);
        let output_k = log_modulus_in + 2 * b;
        let k_boot = if functional {
            plan.functional_bootstrap_k(output_k, log_delta, &lut).unwrap()
        } else {
            plan.bootstrap_k(output_k, log_delta)
        }
        .next_multiple_of(b);
        let in_layout = ct_layout(n, b, k_in, log_delta, slots);
        let out_layout = ct_layout(n, b, k_boot, log_delta, SlotsKind::Complex);
        let atk = key_layout(n, b, k_boot, 2, 1, 1);
        let d2s = key_layout(n, b, k_in, 1, 1, 1);
        let s2d = key_layout(n, b, k_boot, 2, 1, 1);
        let layout = keys_layout(&atk, &d2s, &s2d, encapsulate);
        let pt = ct_layout(n, b, 16 + 2 * b, 16, SlotsKind::Complex);
        let compile = module
            .ckks_all_ops_with_atk_tmp_bytes(&out_layout, &atk, &atk, &pt)
            .max(<Module<B> as CKKSEncodingHostOps<B, F>>::ckks_reim_tmp_bytes(module, n / 2));
        let ctx = with_scratch::<B, _>(compile, |scratch| {
            BootstrappingContext::<B, F>::compile(module, b.into(), &plan, scratch)
        })
        .unwrap();
        let keys = fixture_keys(module, &plan, &atk, &d2s, &s2d);
        let ct_in = fixture_ciphertext(module, &in_layout, 89);
        if functional {
            let luts = [lut.transfer_to(module)];
            let mut outs = vec![fixture_ciphertext(module, &out_layout, 97)];
            let bytes = B::ckks_functional_bootstrap_tmp_bytes_impl(module, &outs[0], &ct_in, &ctx, &luts, &layout);
            let result = with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_functional_bootstrap_impl(module, &mut outs, &ct_in, &ctx, &luts, &keys, scratch)
            });
            assert!(result.is_ok(), "{pipeline:?}: {result:?}");
            results.push((result.map_err(|e| e.to_string()), vec![snapshot::<B, _>(&outs[0])]));
            continue;
        }
        let bytes = B::ckks_bootstrap_tmp_bytes_impl(module, &out_layout, &in_layout, &ctx, &layout);
        let mut out = fixture_ciphertext(module, &out_layout, 97);
        let result = with_scratch::<B, _>(bytes, |scratch| {
            B::ckks_bootstrap_impl(module, &mut out, &ct_in, &ctx, &keys, scratch)
        });
        assert!(result.is_ok(), "{pipeline:?}: {result:?}");
        results.push((result.map_err(|e| e.to_string()), vec![snapshot::<B, _>(&out)]));
        if pipeline == BootstrappingPipeline::C2SFirst {
            let mut raised = fixture_ciphertext(module, &out_layout, 101);
            let result = with_scratch::<B, _>(B::ckks_mod_up_tmp_bytes_impl(module, raised.size()), |scratch| {
                B::ckks_mod_up_into_impl(module, &mut raised, &ct_in, plan.eval_mod(), scratch)
            });
            assert!(result.is_ok(), "{pipeline:?}: {result:?}");
            results.push((result.map_err(|e| e.to_string()), vec![snapshot::<B, _>(&raised)]));
            let mut raised = fixture_ciphertext(module, &out_layout, 103);
            let result = with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_bootstrap_mod_up_impl(module, &mut raised, &ct_in, plan.eval_mod(), &keys, scratch)
            });
            assert!(result.is_ok(), "{pipeline:?}: {result:?}");
            results.push((result.map_err(|e| e.to_string()), vec![snapshot::<B, _>(&raised)]));
        }
    }
    results
}

/// Compare selected bootstrapping implementations on the same fixture inputs
/// and key coefficients: C2S-first, S2C-first, real slots, EvalRound+,
/// encapsulation, functional bootstrapping and the ModUp stages, each within its
/// own exact guarded scratch budget.
pub fn test_bootstrapping_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    BT: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar,
    Module<BR>: CKKSDFTOps<BR>
        + CKKSDFTMatrixOps<BR, F>
        + CKKSEncodingOps<BR, F>
        + CKKSEncodingHostOps<BR, F>
        + CKKSModuleAlloc<BR>
        + CKKSAllOpsTmpBytes<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + GGLWEPreparedFactory<BR>
        + GLWEMaskFill<BR>
        + CyclotomicOrder,
    Module<BT>: CKKSDFTOps<BT>
        + CKKSDFTMatrixOps<BT, F>
        + CKKSEncodingOps<BT, F>
        + CKKSEncodingHostOps<BT, F>
        + CKKSModuleAlloc<BT>
        + CKKSAllOpsTmpBytes<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWETensorKeyPreparedFactory<BT>
        + GGLWEPreparedFactory<BT>
        + GLWEMaskFill<BT>
        + CyclotomicOrder,
    CKKSPlaintextOwned<BR>: GLWEToBackendRef<BR> + CKKSCtBounds + DiagonalProd<BR>,
    CKKSPlaintextOwned<BT>: GLWEToBackendRef<BT> + CKKSCtBounds + DiagonalProd<BT>,
    FixtureKeys<BR>: BootstrappingKeys<BR, TensorKey = GLWETensorKeyPrepared<BR::OwnedBuf, BR>> + Sync,
    FixtureKeys<BT>: BootstrappingKeys<BT, TensorKey = GLWETensorKeyPrepared<BT::OwnedBuf, BT>> + Sync,
{
    assert_eq!(reference.n(), tested.n());
    assert_eq!(
        run_bootstrap::<BR, F>(params, reference),
        run_bootstrap::<BT, F>(params, tested),
        "bootstrapping differs"
    );
}
