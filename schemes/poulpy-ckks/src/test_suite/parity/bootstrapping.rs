//! Encapsulated modulus raising and the bootstrapping pipeline, including
//! optimized and fallback shapes.
use crate::{Scale, ckks_log_delta};
use std::collections::HashMap;

use super::{
    helpers::{Snapshot, assert_untagged, fixture_ciphertext, fixture_operand, snapshot, with_scratch},
    keys::{
        fixture_gglwe, key_layout, key_layout_bits, key_layout_offset, prepared_automorphism_key, prepared_gglwe,
        prepared_tensor_key,
    },
};
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, SlotLayout, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSCopyOps, CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory, CKKSDFTOps,
        CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSPow2Ops,
    },
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, BootstrappingPipeline,
        BootstrappingPlan, BootstrappingTechniques, CKKSCiphertextOwned, CKKSPlaintextOwned, DFTOutputFormat, DFTPlan, DFTType,
        Decode, DftScalar, EncapsulationKeysLayout, Encode, EncodedLut, EvalModPlan, EvalModType, EvalRoundPlus,
        SparseSecretEncapsulation,
    },
    oep::{CKKSBootstrappingImpl, CKKSEncapsulatedModUpImpl},
    polynomial::SplitStrategy,
    test_suite::{CKKSTestParams, presets::compile_context},
};
use poulpy_core::{
    GLWEAdd, GLWEDiagonalProd, GLWEFill, GLWELinearTransformationBabyStepsBytesOf,
    ScratchArenaTakeGLWELinearTransformationBabySteps,
    api::{GGLWEPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{
        GGLWEInfos, GGLWELayout, GGLWEPrepared, GLWEAutomorphismKeyPrepared, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendRef,
        GetAutomorphismKey, LWEInfos, TorusPrecision,
        prepared::{
            GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
            GLWETensorKeyPreparedToBackendRef,
        },
    },
};
use poulpy_hal::layouts::{
    Backend, ConjugateInvariant, CyclotomicOrder, HostBytesBackend, HostStaged, Module, Ring, ScratchArena, Standard,
};

fn run<BE>(params: CKKSTestParams, module: &Module<BE>) -> Vec<(Result<(), String>, Snapshot, Snapshot)>
where
    BE: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    Module<BE>: GGLWEPreparedFactory<BE> + GLWEFill<BE> + GLWEAdd<BE>,
{
    let b = params.base2k;
    let small = 3 * b + 1;
    let mut results = Vec::new();
    // Matching radix and multi-limb digits permit the optimized path. Other
    // cases require the general composition; widths also cover partial limbs,
    // no zero-prefix limbs, several full zero-prefix limbs, rejected raises,
    // and a key stored at one width read through a view at a multiple of it.
    // Digit widths in bits: whole limbs first, then widths off a limb boundary
    // with no, one and several zero-prefix digits, a shift ending exactly on a
    // digit boundary (where the gadget rounding may carry into that digit), a
    // sub-limb width, and a view of an unaligned key.
    for (k_digit, view_k_digit, key_b, extra, scale) in [
        (2 * b, 2 * b, b, 4 * b + 3, 0),
        (2 * b, 2 * b, b, 3 * b + 5, b + 1),
        (2 * b, 2 * b, b, 5, 4),
        (b, b, b, 2 * b + 1, 1),
        (2 * (b - 1), 2 * (b - 1), b - 1, 2 * b + 3, 0),
        (2 * b, 2 * b, b, 0, 1),
        (b, 2 * b, b, 4 * b + 3, 0),
        (2 * b + 3, 2 * b + 3, b, 4 * b + 3, 0),
        (2 * b - 1, 2 * b - 1, b, 3 * b + 5, b + 1),
        (b + 1, b + 1, b, 5, 4),
        (2 * b - 1, 2 * b - 1, b, 6 * b + 1, 0),
        (2 * b + 3, 2 * b + 3, b, 4 * b + 6, 0),
        (b / 2 + 1, b / 2 + 1, b, 2 * b + 1, 1),
        (b + 1, 2 * b + 2, b, 4 * b + 3, 0),
    ] {
        let src_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: module.n().into(),
                base2k: b.into(),
                k: small.into(),
                rank: params.rank.into(),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(b),
                slots: SlotsKind::Complex,
                log_sparsity: 1,
                slot_layout: SlotLayout::Full,
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
        // Covers `dst` through the view too: a read at a coarser digit keeps one
        // such digit as guard.
        let s2d_k = if view_k_digit == k_digit {
            small + extra
        } else {
            ((small + extra).div_ceil(view_k_digit) + 1) * view_k_digit
        };
        let s2d_layout = key_layout_bits(module.n(), key_b, s2d_k, k_digit, params.rank, params.rank);
        let mut d2s = module.gglwe_prepared_alloc_from_infos(&d2s_layout);
        let mut s2d = module.gglwe_prepared_alloc_from_infos(&s2d_layout);
        for (prepared, layout, seed) in [(&mut d2s, &d2s_layout, 31), (&mut s2d, &s2d_layout, 37)] {
            let coefficients = fixture_gglwe(module, layout, seed);
            with_scratch::<BE, _>(module.gglwe_prepare_tmp_bytes(layout), |scratch| {
                module.gglwe_prepare(prepared, &coefficients, scratch)
            });
        }
        let s2d = s2d.with_k_digit(TorusPrecision(view_k_digit as u32)).unwrap();
        let bytes = BE::ckks_encapsulated_mod_up_tmp_bytes(module, &dst_layout, &src_layout, &d2s_layout, &s2d);
        for (seed, lazy) in [(41, false), (43, true)] {
            let mut src = fixture_operand(module, &src_layout, seed, lazy);
            let mut dst = fixture_ciphertext(module, &dst_layout, 47);
            let result = with_scratch::<BE, _>(bytes, |scratch| {
                BE::ckks_encapsulated_mod_up(module, &mut dst, &mut src, scale, &d2s.to_backend_ref(), &s2d, scratch)
            });
            assert_eq!(result.is_ok(), extra >= scale, "modulus raise acceptance changed");
            if result.is_ok() {
                assert_eq!(
                    dst.meta(),
                    GLWEPlaintextMeta {
                        scale: Scale::Log(ckks_log_delta(&src_layout.meta) + scale),
                        ..src_layout.meta
                    }
                );
                assert_eq!(dst.k(), dst_layout.glwe_layout.k);
            }
            results.push((
                result.map_err(|error| error.to_string()),
                snapshot::<BE, _, Standard>(&src),
                snapshot::<BE, _, Standard>(&dst),
            ));
        }
    }
    // Sparse-to-dense keys whose rows sit lower: the raise is their switch. An
    // offset equal to the raise's shift switches `src` as it is, a smaller one
    // first shifts it inside the key's window, a larger one is refused. Digits
    // on and off a limb boundary, one row or several, and an input at a radix
    // other than the key's.
    for (k_digit, extra, scale, k_offset, src_b) in [
        (2 * b, 4 * b + 3, 0, 4 * b + 3, b),
        (b, 3 * b + 5, b + 1, 2 * b + 4, b),
        (4 * b, 4 * b + 3, 0, 4 * b + 3, b),
        (2 * b, 4 * b + 3, 0, 2 * b, b),
        (b + 1, 4 * b + 3, 1, 3 * b, b),
        (2 * b - 1, 3 * b + 5, b + 1, b + 3, b),
        (2 * b, 2 * b + 1, 0, 2 * b + 2, b),
        (2 * b, 4 * b + 3, 1, 2 * b, b - 1),
    ] {
        let k_large = small + extra;
        let src_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: module.n().into(),
                base2k: src_b.into(),
                k: small.into(),
                rank: params.rank.into(),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(b),
                slots: SlotsKind::Complex,
                log_sparsity: 1,
                slot_layout: SlotLayout::Full,
            },
        };
        let dst_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                base2k: b.into(),
                k: k_large.into(),
                ..src_layout.glwe_layout
            },
            ..src_layout
        };
        let d2s_layout = key_layout(module.n(), src_b, small, 1, params.rank, params.rank);
        let s2d_layout = key_layout_offset(module.n(), b, k_large - k_offset, k_digit, k_offset, params.rank, params.rank);
        let mut d2s = module.gglwe_prepared_alloc_from_infos(&d2s_layout);
        let mut s2d = module.gglwe_prepared_alloc_from_infos(&s2d_layout);
        for (prepared, layout, seed) in [(&mut d2s, &d2s_layout, 53), (&mut s2d, &s2d_layout, 59)] {
            let coefficients = fixture_gglwe(module, layout, seed);
            with_scratch::<BE, _>(module.gglwe_prepare_tmp_bytes(layout), |scratch| {
                module.gglwe_prepare(prepared, &coefficients, scratch)
            });
        }
        let bytes = BE::ckks_encapsulated_mod_up_tmp_bytes(module, &dst_layout, &src_layout, &d2s_layout, &s2d_layout);
        for (seed, lazy) in [(61, false), (67, true)] {
            let mut src = fixture_operand(module, &src_layout, seed, lazy);
            let mut dst = fixture_ciphertext(module, &dst_layout, 71);
            let result = with_scratch::<BE, _>(bytes, |scratch| {
                BE::ckks_encapsulated_mod_up(
                    module,
                    &mut dst,
                    &mut src,
                    scale,
                    &d2s.to_backend_ref(),
                    &s2d.to_backend_ref(),
                    scratch,
                )
            });
            assert_eq!(
                result.is_ok(),
                k_offset <= extra - scale,
                "offset modulus raise acceptance changed: k_digit={k_digit} extra={extra} scale={scale} k_offset={k_offset}"
            );
            if result.is_ok() {
                assert_eq!(
                    dst.meta(),
                    GLWEPlaintextMeta {
                        scale: Scale::Log(ckks_log_delta(&src_layout.meta) + scale),
                        ..src_layout.meta
                    }
                );
                assert_eq!(dst.k(), dst_layout.glwe_layout.k);
            }
            results.push((
                result.map_err(|error| error.to_string()),
                snapshot::<BE, _, Standard>(&src),
                snapshot::<BE, _, Standard>(&dst),
            ));
        }
    }
    results
}

/// Compare selected implementations using the same canonical inputs and key
/// coefficients. Source mutations, destination metadata, errors, and each
/// implementation's own exact guarded scratch budget are part of the contract.
pub fn test_encapsulated_mod_up_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    Module<BR>: GGLWEPreparedFactory<BR> + GLWEFill<BR> + GLWEAdd<BR>,
    Module<BT>: GGLWEPreparedFactory<BT> + GLWEFill<BT> + GLWEAdd<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let want = run(params, module_ref);
    assert_untagged(
        "encapsulated ModUp",
        want.iter().filter(|(result, _, _)| result.is_ok()).map(|(_, _, dst)| dst),
    );
    assert_eq!(want, run(params, module_test), "encapsulated ModUp differs");
}

/// The dense-to-sparse key and, unless the raise applies the first factor, the sparse-to-dense key.
type FixtureEncapsulationKeys<BE, Rg> = (FixtureSwitchingKey<BE, Rg>, Option<FixtureSwitchingKey<BE, Rg>>);

/// Fixture key store of the bootstrapping parity tests: identical coefficients
/// on every backend, each preparing its own representation.
pub struct FixtureKeys<BE: Backend, Rg: Ring = Standard> {
    rotation_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE, Rg>>,
    tensor_key: GLWETensorKeyPrepared<BE::OwnedBuf, BE, Rg>,
    encapsulation_keys: Option<FixtureEncapsulationKeys<BE, Rg>>,
    first_factor_keys: HashMap<i64, FixtureSwitchingKey<BE, Rg>>,
}

type FixtureSwitchingKey<BE, Rg = Standard> = GGLWEPrepared<<BE as Backend>::OwnedBuf, BE, Rg>;

impl<BE: Backend, Rg: Ring> GetAutomorphismKey<BE, Rg> for FixtureKeys<BE, Rg>
where
    GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE, Rg>: GLWEAutomorphismKeyPreparedToBackendRef<BE, Rg>,
{
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>> {
        self.rotation_keys.lookup_automorphism_key(p, k)
    }
}

impl<BE: Backend> BootstrappingLinearKeys<BE> for FixtureKeys<BE>
where
    GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GGLWEPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type SwitchingKey = GGLWEPrepared<BE::OwnedBuf, BE>;

    fn encapsulation_keys(&self) -> Option<(&Self::SwitchingKey, Option<&Self::SwitchingKey>)> {
        self.encapsulation_keys.as_ref().map(|(d2s, s2d)| (d2s, s2d.as_ref()))
    }

    fn first_factor_key(&self, rotation: i64) -> Option<&Self::SwitchingKey> {
        self.first_factor_keys.get(&rotation)
    }
}

impl<BE: Backend> BootstrappingKeys<BE> for FixtureKeys<BE>
where
    GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE, ConjugateInvariant>: GLWETensorKeyPreparedToBackendRef<BE, ConjugateInvariant>,
    GGLWEPrepared<BE::OwnedBuf, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>;
    type CITensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE, ConjugateInvariant>;
    type TraceKey = GGLWEPrepared<BE::OwnedBuf, BE>;
    type EmbedKey = GGLWEPrepared<BE::OwnedBuf, BE>;

    fn tensor_key(&self) -> Option<&Self::TensorKey> {
        Some(&self.tensor_key)
    }

    fn ci_tensor_key(&self) -> Option<&Self::CITensorKey> {
        None
    }

    fn conversion_keys(&self) -> Option<(&Self::TraceKey, &Self::EmbedKey)> {
        None
    }
}

const LOG_MSG_RATIO: usize = 4;

/// A small full-slot recipe: one radix-2 layer per factor and a degree-8 EvalMod.
fn plan(
    pipeline: BootstrappingPipeline,
    log_slots: usize,
    encapsulate: bool,
    eval_round: bool,
    first_factor_on_raise: bool,
) -> BootstrappingPlan {
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
        if first_factor_on_raise {
            dft(DFTType::Encode, 14).with_first_factor_on_raise()
        } else {
            dft(DFTType::Encode, 14)
        },
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
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        },
    }
}

/// Every rotation key of `elements` of layout `atk`.
fn keys_layout(
    elements: impl Iterator<Item = i64>,
    atk: &GGLWELayout,
    d2s: &GGLWELayout,
    s2d: &GGLWELayout,
    first: Option<&GGLWELayout>,
    encapsulate: bool,
) -> BootstrappingKeysLayout {
    BootstrappingKeysLayout {
        rotation_keys: elements.map(|p| (p, (*atk).into())).collect(),
        first_factor_key: first.map(|key| (*key).into()),
        conversion_keys: None,
        tensor_key: (*atk).into(),
        encapsulation: encapsulate.then(|| EncapsulationKeysLayout {
            dense_to_sparse: (*d2s).into(),
            sparse_to_dense: first.is_none().then(|| (*s2d).into()),
        }),
    }
}

fn fixture_keys<BE, Rg: Ring>(
    module: &Module<BE>,
    plan: &BootstrappingPlan,
    atk: &GGLWELayout,
    d2s: &GGLWELayout,
    s2d: &GGLWELayout,
    first: Option<&GGLWELayout>,
) -> FixtureKeys<BE, Rg>
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE, Rg>
        + GLWETensorKeyPreparedFactory<BE, Rg>
        + GGLWEPreparedFactory<BE, Rg>
        + GLWEFill<BE, Rg>
        + crate::api::CKKSRightOperandPreparedFactory<BE, Rg>
        + CyclotomicOrder,
{
    let log_n = module.n().ilog2() as usize;
    let rotation_keys = plan
        .galois_elements(log_n, module.cyclotomic_order::<Rg>())
        .into_iter()
        .chain([-1])
        .map(|p| (p, prepared_automorphism_key(module, atk, p, (p as u8).wrapping_add(71))))
        .collect();
    FixtureKeys {
        rotation_keys,
        tensor_key: prepared_tensor_key(module, atk, 73),
        encapsulation_keys: plan.sparse_secret_hamming_weight().map(|_| {
            (
                prepared_gglwe(module, d2s, 79),
                first.is_none().then(|| prepared_gglwe(module, s2d, 83)),
            )
        }),
        first_factor_keys: first.map_or_else(HashMap::new, |key| {
            plan.coeffs_to_slots()
                .first_factor_rotations(log_n)
                .into_iter()
                .map(|rotation| (rotation, prepared_gglwe(module, key, (rotation as u8).wrapping_add(113))))
                .collect()
        }),
    }
}

type Outcome = (Result<(), String>, Vec<Snapshot>);

/// Snapshots of the hand composition of one case: the raise, the EvalRound+ bypass when the plan has one,
/// CoeffsToSlots and the EvalMod or LUT output.
type Steps = Vec<Snapshot>;

/// Pipeline, encapsulation, EvalRound+, input slot kind and functional bootstrap of each fixture case.
type CaseShape = (BootstrappingPipeline, bool, bool, SlotsKind, bool);

const CASES: [CaseShape; 7] = [
    (BootstrappingPipeline::C2SFirst, true, false, SlotsKind::Complex, false),
    (BootstrappingPipeline::C2SFirst, false, true, SlotsKind::Complex, false),
    (BootstrappingPipeline::S2CFirst, false, false, SlotsKind::Complex, false),
    (BootstrappingPipeline::S2CFirst, true, false, SlotsKind::Real, false),
    (BootstrappingPipeline::S2CFirst, false, true, SlotsKind::Real, false),
    (BootstrappingPipeline::S2CFirst, false, false, SlotsKind::Complex, true),
    (BootstrappingPipeline::S2CFirst, false, false, SlotsKind::Real, true),
];

/// One fixture case: its compiled context, keys and key layout, layouts, input and, functional, its LUT.
struct Case<BE: Backend, F> {
    ctx: BootstrappingContextPrepared<BE, F>,
    keys: FixtureKeys<BE>,
    keys_layout: BootstrappingKeysLayout,
    in_layout: CKKSLayout,
    out_layout: CKKSLayout,
    ct_in: CKKSCiphertextOwned<BE>,
    luts: Option<[EncodedLut<CKKSPlaintextOwned<BE>>; 1]>,
}

fn fixture_case<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    (pipeline, encapsulate, eval_round, slots, functional): CaseShape,
) -> Case<BE, F>
where
    BE: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + CKKSDFTOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + CKKSEncodingHostOps<BE, F>
        + CKKSAllOpsTmpBytes<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GGLWEPreparedFactory<BE>
        + GLWEFill<BE>
        + crate::api::CKKSRightOperandPreparedFactory<BE>
        + CyclotomicOrder,
    Module<BE>: CKKSEncodingOps<BE, F, ConjugateInvariant>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
{
    let (n, b) = (module.n(), params.base2k);
    let log_slots = n.ilog2() as usize - 1;
    let log_delta = 12;
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
    let mut plan = plan(pipeline, log_slots, encapsulate, eval_round, false);
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
    let elements = plan
        .key_widths(n.ilog2() as usize, k_boot, k_in, log_delta)
        .rotation_keys
        .into_keys();
    let keys_layout = keys_layout(elements, &atk, &d2s, &s2d, None, encapsulate);
    let pt = ct_layout(n, b, 16 + 2 * b, 16, SlotsKind::Complex);
    let compile =
        module.ckks_all_ops_with_atk_tmp_bytes(&out_layout, &pt, &atk, &atk).max(
            <Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_encoding_host_tmp_bytes(module, n / 2),
        );
    // Staged compilation: preparation runs with exactly its advertised scratch.
    let unprepared = with_scratch::<BE, _>(compile, |scratch| {
        BootstrappingContextPrepared::<BE, F>::compile_unprepared(module, b.into(), &plan, scratch)
    })
    .unwrap();
    let ctx = with_scratch::<BE, _>(unprepared.prepare_tmp_bytes(module), |scratch| {
        unprepared.prepare(module, scratch)
    })
    .unwrap();
    Case {
        keys: fixture_keys(module, &plan, &atk, &d2s, &s2d, None),
        ctx,
        keys_layout,
        in_layout,
        out_layout,
        ct_in: fixture_ciphertext(module, &in_layout, 89),
        luts: functional.then(|| [lut.transfer_to(module)]),
    }
}

/// `ckks_bootstrap`, or `ckks_functional_bootstrap` with the LUT of a functional case, through the module on exactly
/// its query.
fn whole_bootstrap<BE, F>(module: &Module<BE>, case: &Case<BE, F>) -> Outcome
where
    BE: Backend<ZnxWord = i64> + CKKSBootstrappingImpl,
    F: Sync,
    Module<BE>: GLWEFill<BE>,
    FixtureKeys<BE>: BootstrappingKeys<BE> + Sync,
    <FixtureKeys<BE> as BootstrappingKeys<BE>>::TensorKey: Sync,
{
    let mut out = fixture_ciphertext(module, &case.out_layout, 97);
    let result = match &case.luts {
        Some(luts) => {
            let bytes = module.ckks_functional_bootstrap_tmp_bytes(&out, &case.ct_in, &case.ctx, luts, &case.keys_layout);
            with_scratch::<BE, _>(bytes, |scratch| {
                module.ckks_functional_bootstrap(
                    std::slice::from_mut(&mut out),
                    &case.ct_in,
                    &case.ctx,
                    luts,
                    &case.keys,
                    scratch,
                )
            })
        }
        None => {
            let bytes = module.ckks_bootstrap_tmp_bytes(&out, &case.ct_in, &case.ctx, &case.keys_layout);
            with_scratch::<BE, _>(bytes, |scratch| {
                module.ckks_bootstrap(&mut out, &case.ct_in, &case.ctx, &case.keys, scratch)
            })
        }
    };
    assert!(result.is_ok(), "{:?}: {result:?}", case.ctx.pipeline());
    (result.map_err(|e| e.to_string()), vec![snapshot::<BE, _, Standard>(&out)])
}

/// The outcome of every whole bootstrap of the fixture cases: `ckks_bootstrap` and `ckks_functional_bootstrap` alone,
/// through the module, each on exactly its query.
pub fn run_bootstrap<BE, F>(params: CKKSTestParams, module: &Module<BE>) -> Vec<Outcome>
where
    BE: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + CKKSDFTOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + CKKSEncodingHostOps<BE, F>
        + CKKSAllOpsTmpBytes<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GGLWEPreparedFactory<BE>
        + GLWEFill<BE>
        + crate::api::CKKSRightOperandPreparedFactory<BE>
        + CyclotomicOrder,
    Module<BE>: CKKSEncodingOps<BE, F, ConjugateInvariant>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    FixtureKeys<BE>: BootstrappingKeys<BE> + Sync,
    <FixtureKeys<BE> as BootstrappingKeys<BE>>::TensorKey: Sync,
{
    CASES
        .into_iter()
        .map(|shape| whole_bootstrap(module, &fixture_case::<BE, F>(params, module, shape)))
        .collect()
}

/// `ckks_bootstrap` on the first identity case and `ckks_functional_bootstrap` on the first functional case, of a
/// rank-2 input on a zero-byte arena: the refusals of their entry checks, each output left unchanged.
pub fn run_refused_bootstraps<BE, F>(params: CKKSTestParams, module: &Module<BE>) -> [crate::CKKSError; 2]
where
    BE: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + CKKSDFTOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + CKKSEncodingHostOps<BE, F>
        + CKKSAllOpsTmpBytes<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GGLWEPreparedFactory<BE>
        + GLWEFill<BE>
        + crate::api::CKKSRightOperandPreparedFactory<BE>
        + CyclotomicOrder,
    Module<BE>: CKKSEncodingOps<BE, F, ConjugateInvariant>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    FixtureKeys<BE>: BootstrappingKeys<BE> + Sync,
    <FixtureKeys<BE> as BootstrappingKeys<BE>>::TensorKey: Sync,
{
    [false, true].map(|functional| {
        let shape = CASES.into_iter().find(|shape| shape.4 == functional).unwrap();
        let case = fixture_case::<BE, F>(params, module, shape);
        let mut rank2 = case.in_layout;
        rank2.glwe_layout.rank = 2usize.into();
        let ct_rank2 = fixture_ciphertext(module, &rank2, 127);
        let mut out = fixture_ciphertext(module, &case.out_layout, 131);
        let before = snapshot::<BE, _, Standard>(&out);
        let error = with_scratch::<BE, _>(0, |scratch| match &case.luts {
            Some(luts) => module.ckks_functional_bootstrap(
                std::slice::from_mut(&mut out),
                &ct_rank2,
                &case.ctx,
                luts,
                &case.keys,
                scratch,
            ),
            None => module.ckks_bootstrap(&mut out, &ct_rank2, &case.ctx, &case.keys, scratch),
        })
        .expect_err("a bootstrap of a rank-2 input was accepted");
        assert_eq!(
            snapshot::<BE, _, Standard>(&out),
            before,
            "a refused bootstrap wrote its output"
        );
        error
    })
}

fn run_bootstrap_cases<BE, F>(params: CKKSTestParams, module: &Module<BE>) -> (Vec<Outcome>, Vec<Steps>)
where
    BE: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + CKKSDFTOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + CKKSEncodingHostOps<BE, F>
        + CKKSAllOpsTmpBytes<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GGLWEPreparedFactory<BE>
        + GLWEFill<BE>
        + crate::api::CKKSRightOperandPreparedFactory<BE>
        + CyclotomicOrder,
    Module<BE>: CKKSEncodingOps<BE, F, ConjugateInvariant>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    FixtureKeys<BE>: BootstrappingKeys<BE> + Sync,
    <FixtureKeys<BE> as BootstrappingKeys<BE>>::TensorKey: Sync,
{
    let mut results = Vec::new();
    let mut steps = Vec::new();
    for shape in CASES {
        let case = fixture_case::<BE, F>(params, module, shape);
        results.push(whole_bootstrap(module, &case));
        if case.ctx.pipeline() == BootstrappingPipeline::C2SFirst {
            let mut raised = fixture_ciphertext(module, &case.out_layout, 103);
            let mut raise_in = fixture_ciphertext(module, &case.in_layout, 89);
            let bytes = module.ckks_mod_raise_tmp_bytes(&raised, &raise_in, &case.ctx, &case.keys_layout);
            let result = with_scratch::<BE, _>(bytes, |scratch| {
                module.ckks_mod_raise(&mut raised, &mut raise_in, &case.ctx, &case.keys, scratch)
            });
            assert!(result.is_ok(), "{:?}: {result:?}", case.ctx.pipeline());
            results.push((result.map_err(|e| e.to_string()), vec![snapshot::<BE, _, Standard>(&raised)]));
        }
        steps.push(compose_steps(module, &case));
    }
    (results, steps)
}

/// Runs one step on exactly `bytes` of poisoned scratch.
fn run_step<BE: Backend>(label: &str, bytes: usize, step: impl FnOnce(&mut ScratchArena<'_, BE>) -> crate::CKKSResult<()>) {
    if let Err(error) = with_scratch::<BE, _>(bytes, step) {
        panic!("{label}: {error}");
    }
}

/// The steps of `ckks_bootstrap`, or of `ckks_functional_bootstrap` on a functional case, composed by hand on the
/// case's input, each on exactly its own query, the outputs snapshotted for the cross-backend comparison.
fn compose_steps<BE, F>(module: &Module<BE>, case: &Case<BE, F>) -> Steps
where
    BE: Backend<ZnxWord = i64> + CKKSBootstrappingImpl,
    F: Sync,
    Module<BE>: GLWEFill<BE>,
    FixtureKeys<BE>: BootstrappingKeys<BE> + Sync,
    <FixtureKeys<BE> as BootstrappingKeys<BE>>::TensorKey: Sync,
{
    let (ctx, keys, keys_layout, ct_in) = (&case.ctx, &case.keys, &case.keys_layout, &case.ct_in);
    let mut raised = fixture_ciphertext(module, &case.out_layout, 103);
    let mut z_hp = fixture_ciphertext(module, &case.out_layout, 107);
    let mut stepped = fixture_ciphertext(module, &case.out_layout, 109);
    let mut coeffs = fixture_ciphertext(module, &case.in_layout, 113);
    let run = ctx.run(ct_in.meta());
    let mut steps = Vec::new();
    if ctx.pipeline() == BootstrappingPipeline::S2CFirst {
        run_step::<BE>("ckks_double_into", module.ckks_double_tmp_bytes(coeffs.size()), |s| {
            module.ckks_double_into(&mut coeffs, ct_in, s)
        });
        let bytes = keys_layout
            .rotation_key_layouts()
            .map(|atk| module.ckks_dft_tmp_bytes::<Decode, crate::layouts::Standard, _, _, _, _>(&coeffs, &coeffs, &run, atk))
            .max()
            .unwrap();
        run_step::<BE>("ckks_slots_to_coeffs_assign", bytes, |s| {
            module.ckks_slots_to_coeffs_assign(&mut coeffs, &run, keys, s)
        });
    } else {
        run_step::<BE>("ckks_copy", module.ckks_copy_tmp_bytes(&coeffs, ct_in), |s| {
            module.ckks_copy(&mut coeffs, ct_in, s)
        });
    }
    run_step::<BE>(
        "ckks_mod_raise",
        module.ckks_mod_raise_tmp_bytes(&raised, &coeffs, &run, keys_layout),
        |s| module.ckks_mod_raise(&mut raised, &mut coeffs, &run, keys, s),
    );
    steps.push(snapshot::<BE, _, Standard>(&raised));
    if let Some(bypass) = run.bypass() {
        run_step::<BE>("ckks_copy", module.ckks_copy_tmp_bytes(&z_hp, &raised), |s| {
            module.ckks_copy(&mut z_hp, &raised, s)
        });
        let bytes = keys_layout
            .rotation_key_layouts()
            .map(|atk| module.ckks_dft_tmp_bytes::<Encode, crate::layouts::Standard, _, _, _, _>(&z_hp, &z_hp, &bypass, atk))
            .max()
            .unwrap();
        run_step::<BE>("ckks_coeffs_to_slots_assign", bytes, |s| {
            module.ckks_coeffs_to_slots_assign(&mut z_hp, &bypass, keys, s)
        });
        steps.push(snapshot::<BE, _, Standard>(&z_hp));
    }
    let bytes = keys_layout
        .rotation_key_layouts()
        .map(|atk| module.ckks_dft_tmp_bytes::<Encode, crate::layouts::Standard, _, _, _, _>(&raised, &raised, &run, atk))
        .max()
        .unwrap();
    run_step::<BE>("ckks_coeffs_to_slots_assign", bytes, |s| {
        module.ckks_coeffs_to_slots_assign(&mut raised, &run, keys, s)
    });
    steps.push(snapshot::<BE, _, Standard>(&raised));
    match &case.luts {
        Some(luts) => run_step::<BE>(
            "ckks_eval_lut",
            module.ckks_eval_lut_tmp_bytes(&stepped, &raised, &run, luts, keys_layout),
            |s| module.ckks_eval_lut(std::slice::from_mut(&mut stepped), &raised, &run, luts, keys, s),
        ),
        None => run_step::<BE>(
            "ckks_bootstrap_eval_mod",
            module.ckks_bootstrap_eval_mod_tmp_bytes(&stepped, &raised, &run, keys_layout),
            |s| module.ckks_bootstrap_eval_mod(&mut stepped, &raised, &run, keys, s),
        ),
    }
    steps.push(snapshot::<BE, _, Standard>(&stepped));
    steps
}

/// The masked CoeffsToSlots, the merged SlotsToCoeffs factor and the raise of a gap on fixture keys, each composed from
/// the steps: on the full ring and on the subring of gap 2, with and without a mask.
fn run_coeffs_mask<BE, F>(params: CKKSTestParams, module: &Module<BE>) -> Vec<Outcome>
where
    BE: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar + DftScalar + Sync,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + CKKSDFTOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + CKKSEncodingHostOps<BE, F>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSLinearTransformationOps<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GGLWEPreparedFactory<BE>
        + GLWEFill<BE>
        + crate::api::CKKSRightOperandPreparedFactory<BE>
        + CyclotomicOrder,
    Module<BE>: CKKSEncodingOps<BE, F, ConjugateInvariant>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE> + LWEInfos,
    FixtureKeys<BE>: BootstrappingKeys<BE> + Sync,
    <FixtureKeys<BE> as BootstrappingKeys<BE>>::TensorKey: Sync,
{
    let (n, b) = (module.n(), params.base2k);
    let log_n = n.ilog2() as usize;
    let log_delta = 12;
    let log_modulus_in = log_delta + LOG_MSG_RATIO;
    let mut results = Vec::new();
    for log_gap in [0, 1] {
        let plan = plan(BootstrappingPipeline::S2CFirst, log_n - 1 - log_gap, true, false, true)
            .with_log_gap(log_gap)
            .unwrap();
        let k_in = plan.input_k(log_modulus_in);
        let k_boot = plan.bootstrap_k(log_modulus_in + 2 * b, log_delta).next_multiple_of(b);
        let raise_layout = ct_layout(n, b, log_modulus_in, log_delta, SlotsKind::Real);
        let in_layout = ct_layout(n, b, k_in, log_delta, SlotsKind::Real);
        let out_layout = ct_layout(n, b, k_boot, log_delta, SlotsKind::Real);
        let atk = key_layout(n, b, k_boot, 2, 1, 1);
        let d2s = key_layout(n, b, k_in, 1, 1, 1);
        let first = CKKSTestParams {
            n,
            k: k_boot,
            rank: 1,
            ..params
        }
        .first_factor_key_layout(&plan, k_boot, log_modulus_in)
        .gglwe_layout();
        let elements = plan.key_widths(log_n, k_boot, k_in, log_delta).rotation_keys.into_keys();
        let layout = keys_layout(elements, &atk, &d2s, &d2s, Some(&first), true);
        let ctx = compile_context::<BE, F>(module, b.into(), &plan);
        let keys = fixture_keys(module, &plan, &atk, &d2s, &d2s, Some(&first));

        let keep: Vec<bool> = (0..n).map(|j| j < n / 2 && j.is_multiple_of(1 << (log_gap + 1))).collect();
        let mask_bytes = ctx.coeffs_mask_tmp_bytes(module).unwrap();
        let mask = with_scratch::<BE, _>(mask_bytes, |scratch| ctx.coeffs_mask(module, &keep, scratch)).unwrap();
        let run = ctx.run(raise_layout.meta);
        let raise_bytes = BE::ckks_mod_raise_tmp_bytes(module, &out_layout, &raise_layout, &run, &layout);
        let masked_bytes = module.ckks_dft_tmp_bytes(&out_layout, &out_layout, &ctx.masked(&mask).unwrap(), &atk);
        let eval_mod_bytes = BE::ckks_bootstrap_eval_mod_tmp_bytes(module, &out_layout, &out_layout, &run, &layout);
        for mask in [None, Some(&mask)] {
            let mut coeffs = fixture_ciphertext(module, &raise_layout, 107);
            let mut z = fixture_ciphertext(module, &out_layout, 109);
            let mut out = fixture_ciphertext(module, &out_layout, 113);
            let staged = with_scratch::<BE, _>(raise_bytes, |scratch| {
                BE::ckks_mod_raise(module, &mut z, &mut coeffs, &run, &keys, scratch)
            })
            .and_then(|()| match mask {
                Some(mask) => with_scratch::<BE, _>(masked_bytes, |scratch| {
                    module.ckks_coeffs_to_slots_assign(&mut z, &ctx.masked(mask)?, &keys, scratch)
                }),
                None => {
                    let bytes = module.ckks_dft_tmp_bytes(&out_layout, &out_layout, ctx.coeffs_to_slots(), &atk);
                    with_scratch::<BE, _>(bytes, |scratch| {
                        module.ckks_coeffs_to_slots_assign(&mut z, &run, &keys, scratch)
                    })
                }
            });
            // EvalMod clobbers `z`.
            let transformed = snapshot::<BE, _, Standard>(&z);
            let result = staged.and_then(|()| {
                with_scratch::<BE, _>(eval_mod_bytes, |scratch| {
                    BE::ckks_bootstrap_eval_mod(module, &mut out, &z, &run, &keys, scratch)
                })
            });
            assert!(result.is_ok(), "log_gap {log_gap}: {result:?}");
            results.push((
                result.map_err(|e| e.to_string()),
                vec![transformed, snapshot::<BE, _, Standard>(&out)],
            ));
        }

        // One kernel per coset, of dyadic coefficients of the subring.
        let cosets = ctx.merged_s2c_cosets();
        let kernels: Vec<Vec<F>> = (0..cosets)
            .map(|r| {
                let mut kernel = vec![F::zero(); n];
                kernel[(r + 1) << log_gap] = F::from_f64(0.25).unwrap();
                kernel
            })
            .collect();
        let factor_bytes = ctx.merged_s2c_factor_tmp_bytes(module, 14).unwrap();
        let merged = with_scratch::<BE, _>(factor_bytes, |scratch| ctx.merged_s2c_factor(module, &kernels, 14, scratch)).unwrap();
        let factor = merged.factor();
        let schedule = factor.index();
        let head = ctx.slots_to_coeffs().head();
        let head_bytes = module.ckks_dft_tmp_bytes(&in_layout, &in_layout, &head, &atk);
        let babies_bytes = module.glwe_linear_transformation_baby_steps_bytes_of(&schedule, &in_layout);
        let factor_eval_bytes = babies_bytes
            + module
                .ckks_linear_transformation_baby_steps_prepare_tmp_bytes(&in_layout, &schedule, &atk)
                .max(module.ckks_linear_transformation_eval_tmp_bytes(
                    &in_layout,
                    factor.first_diagonal_plaintext().unwrap(),
                    &schedule,
                    &atk,
                    &atk,
                ));
        let mut running = fixture_ciphertext(module, &in_layout, 127);
        let result = with_scratch::<BE, _>(head_bytes, |scratch| {
            module.ckks_slots_to_coeffs_assign(&mut running, &head, &keys.rotation_keys, scratch)
        })
        .and_then(|()| {
            with_scratch::<BE, _>(factor_eval_bytes, |scratch| {
                let (mut babies, mut scratch) = scratch
                    .borrow()
                    .take_glwe_linear_transformation_baby_steps_scratch(module, &schedule, &running);
                module.ckks_linear_transformation_baby_steps_prepare(&mut babies, &running, &keys.rotation_keys, &mut scratch)?;
                module.ckks_linear_transformation_eval_assign(&mut running, &babies, factor, &keys.rotation_keys, &mut scratch)
            })
        });
        assert!(result.is_ok(), "log_gap {log_gap}: {result:?}");
        results.push((result.map_err(|e| e.to_string()), vec![snapshot::<BE, _, Standard>(&running)]));
    }
    results
}

/// Compare selected bootstrapping implementations on the same fixture inputs
/// and key coefficients: C2S-first, S2C-first, real slots, EvalRound+,
/// encapsulation, functional bootstrapping, the raise alone, the steps composed by
/// hand, the masked CoeffsToSlots, the raise of a gap and the merged SlotsToCoeffs
/// factor, each within its own exact guarded scratch budget.
pub fn test_bootstrapping_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    BT: Backend<ZnxWord = i64> + CKKSBootstrappingImpl + HostStaged,
    F: CKKSEncodingScalar + DftScalar + Sync,
    Module<BR>: CKKSDFTMatrixPreparedFactory<BR>
        + CKKSDFTOps<BR>
        + CKKSDFTMatrixOps<BR, F>
        + CKKSEncodingOps<BR, F>
        + CKKSEncodingHostOps<BR, F>
        + CKKSAllOpsTmpBytes<BR>
        + CKKSLinearTransformationOps<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + GGLWEPreparedFactory<BR>
        + GLWEFill<BR>
        + crate::api::CKKSRightOperandPreparedFactory<BR>
        + CyclotomicOrder,
    Module<BT>: CKKSDFTMatrixPreparedFactory<BT>
        + CKKSDFTOps<BT>
        + CKKSDFTMatrixOps<BT, F>
        + CKKSEncodingOps<BT, F>
        + CKKSEncodingHostOps<BT, F>
        + CKKSAllOpsTmpBytes<BT>
        + CKKSLinearTransformationOps<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWETensorKeyPreparedFactory<BT>
        + GGLWEPreparedFactory<BT>
        + GLWEFill<BT>
        + crate::api::CKKSRightOperandPreparedFactory<BT>
        + CyclotomicOrder,
    Module<BR>: CKKSEncodingOps<BR, F, ConjugateInvariant>,
    Module<BT>: CKKSEncodingOps<BT, F, ConjugateInvariant>,
    CKKSPlaintextOwned<BR>: GLWEToBackendRef<BR> + CKKSCtBounds + GLWEDiagonalProd<BR> + LWEInfos,
    CKKSPlaintextOwned<BT>: GLWEToBackendRef<BT> + CKKSCtBounds + GLWEDiagonalProd<BT> + LWEInfos,
    FixtureKeys<BR>: BootstrappingKeys<BR> + Sync,
    <FixtureKeys<BR> as BootstrappingKeys<BR>>::TensorKey: Sync,
    FixtureKeys<BT>: BootstrappingKeys<BT> + Sync,
    <FixtureKeys<BT> as BootstrappingKeys<BT>>::TensorKey: Sync,
{
    assert_eq!(module_ref.n(), module_test.n());
    let (want, want_steps) = run_bootstrap_cases::<BR, F>(params, module_ref);
    assert_untagged(
        "bootstrapping",
        want.iter()
            .filter(|(result, _)| result.is_ok())
            .flat_map(|(_, outputs)| outputs),
    );
    assert_untagged("bootstrapping steps", want_steps.iter().flatten());
    let (got, got_steps) = run_bootstrap_cases::<BT, F>(params, module_test);
    assert_eq!(want, got, "bootstrapping differs");
    assert_eq!(want_steps, got_steps, "bootstrapping steps differ");
    let want = run_coeffs_mask::<BR, F>(params, module_ref);
    assert_untagged("coefficient mask", want.iter().flat_map(|(_, outputs)| outputs));
    assert_eq!(
        want,
        run_coeffs_mask::<BT, F>(params, module_test),
        "coefficient mask differs"
    );
}
