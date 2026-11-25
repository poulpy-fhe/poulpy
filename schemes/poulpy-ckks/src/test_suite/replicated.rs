//! Replicated slot layout: shaped encoding and encryption.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_replicated_encoding_is_shaped`] | shaped quantization keeps the data slots far above the scale's precision |
//! | [`test_replicated_unit_shaping_encodes_plainly`] | `U = 1` encodes as a full layout holding the scattered batch |
//! | [`test_replicated_encryption_is_shaped`] | a fresh encryption keeps the data slots far above CKKS at the same scale, with a noise estimate |
//! | [`test_replicated_encryption_refuses_before_drawing`] | a short arena, and on the conjugate-invariant ring a `U` without an invariant shift, refuse before any draw |
//! | [`test_replicated_round_then_add`] | the finer of two scales, rounded with shaping to the other, adds |
//! | [`test_replicated_encoding_rejects_batch_size`] | a batch of the wrong size |
//! | [`test_replicated_products_are_shaped`] | chains of every product, prepared product, dot product, multiply-accumulate and affine map keep the data slots far above CKKS, each call on exactly its query |
//! | [`test_replicated_product_key_width`] | a shaped product looks its key up at the operands' width on the one-pass path and at its tensor's on the tensor path, and refuses another before any write |
//! | [`test_replicated_layouts_do_not_mix`] | products and sums of different layouts, constant sums and `ckks_add_many`, and a product into a reused replicated destination |
//! | [`test_replicated_plain_rounding_is_rejected`] | operations that round plainly (dot products of prepared operands, affine maps, linear transformations, inner sums), and layouts or scales without a shaped encoding |
//! | [`test_replicated_eval_mod_is_shaped`] | EvalMod with masked constants keeps the message far above CKKS: an input above the plan scale rounded at entry with a fused last squaring, and one below it evaluated in place with an inverse and a shaped return; a packed plan refused off a replicated layout of two values, with an inverse, on real slots and past the module's slots |
//! | [`test_replicated_complex_eval_mod_is_shaped`] | the complex EvalMod with a fused last squaring keeps the message far above CKKS |
//! | [`test_replicated_polynomials_are_shaped`] | monomial and Chebyshev polynomials keep the data slots far above CKKS, on either ring, and constants of another layout are refused at entry |
//! | [`test_replicated_polynomial_fallback_is_shaped`] | a baby step whose masked constants differ in scale takes the products term by term, which keep the shaping |
//! | [`test_replicated_complex_polynomials_are_shaped`] | a complex polynomial keeps the data slots far above CKKS |
//! | [`test_replicated_automorphisms_are_shaped`] | rotations and conjugations, into and in place, keep the data slots far above CKKS |
//! | [`test_replicated_rotations_are_shaped`] | rotations keep the data slots far above CKKS, on either ring |
//! | [`test_replicated_rotations_are_admissible`] | rotations by a non-multiple of the period |
//! | [`test_replicated_masked_constants`] | a masked constant fills the data slots only and adds to a ciphertext, fills every slot of a full-layout plaintext, and a plaintext of half a period is refused |
//! | [`test_ckks_to_and_from_replicated`] | CKKS to the replicated layout and back keeps the input's precision, with dense and factored maps, and one value maps onto the layout with no key |
//! | [`test_ckks_bootstrap_replicated`] | the bootstrapping of a replicated layout per unit, period and maps, and every refusal of it and of its steps |
//! | [`test_ckks_dft_matrices_gen_replicated_matches_the_encoder`] | the dense maps of a replicated layout against the encoder's slots |
//! | [`test_ckks_bootstrap_replicated_ci`] | the bootstrapping of a conjugate-invariant replicated layout through the standard ring |
//! | [`test_ckks_to_and_from_replicated_apply_matrices`] | a complex matrix onto the data slots, and gathered then traced back to period `k` |

use std::collections::HashMap;

use anyhow::Result;
use poulpy_core::{
    GLWECIKeyEncryptSk, GLWEMulPreparedPlainSum, GLWETrace,
    api::GLWECIKeyPreparedFactory,
    layouts::{
        BackendGLWESecret, Base2K, GGLWEInfos, GGLWELayout, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout, GLWESecretAlloc,
        GLWEToBackendRef, LWEInfos, LinearTransformationBabySteps, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, ConjugateInvariant, HostBytesBackend, Module, PrepareHint, Ring, ScratchArena, ScratchOwned, Standard, ZnxView,
    },
    source::Source,
};

use super::helpers::{
    LoggedBootstrappingKeys, PT_PREC, QueryLog, TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar,
    alloc_ct, alloc_scratch, assert_ckks_error, assert_ckks_internal, assert_entry_then_widest, ckks_pt_cst, ckks_spec, dirty_ct,
    gen_atk, gen_ci_keys, gen_sk, gen_sk_with_raw, gen_tsk, next_test_seed, upload_pt,
};
use super::parity::helpers::{fixture_ciphertext, fixture_plaintext, snapshot, with_scratch};
use super::presets::{assert_refused, compile_context, paper_eval_mod};
use crate::{
    CKKSInfos, CKKSLayout, CoeffsMeta, GLWEPlaintextInfos, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos,
    SetGLWEPlaintextInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSCIRingMapOps,
        CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps, CKKSCopyOps, CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory,
        CKKSDFTOps, CKKSDecryptOps, CKKSDotProductOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSEncryptOps,
        CKKSEvalModOps, CKKSInnerSumOps, CKKSLeftOperandPreparedFactory, CKKSLinearTransformationOps, CKKSModuleInfos,
        CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSPolynomialEvaluationOps, CKKSReplicatedOps,
        CKKSRightOperandPreparedFactory, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeyWidths, BootstrappingKeysLayout, BootstrappingKeysPrepared,
        BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, CKKSCiphertextAlloc, CKKSCiphertextOwned,
        CKKSEncodingBuffer, CKKSPlaintextAlloc, CKKSPlaintextOwned, DFTPlan, DFTType, Decode, DftScalar, Encode, EvalModPlan,
        EvalModPoly, EvalModType, ReplicatedConversionPrepared, SparseSecretEncapsulation, ckks_eval_mod_compile,
        dft::DFTOutputFormat,
        dft_matrices::one_factor_matrix,
        replicated::{ReplicatedChain, ReplicatedConversionModule, gather_factor, period_factor, split_depth},
    },
    polynomial::{Basis, CKKSPolynomialEncode, ComplexPolynomial, Polynomial, SplitStrategy},
    presets::key_policy::BootstrappingKeyPolicy,
    shaping::{ShapingProfile, ShapingUnit, conjugate_invariant},
    test_suite::CKKSTestParams,
};

const LOG_DELTA: usize = 20;
const LOG_K: u8 = 2;

fn layout() -> ReplicatedLayout {
    ShapingUnit::U32_45.layout(LOG_K).unwrap()
}

/// Whether the ring `Rg` has real slots, where batches have no imaginary part.
fn real_slots<Rg: Ring>() -> bool {
    conjugate_invariant::<Rg>()
}

fn batch<Rg: Ring, F: TestScalar>(len: usize) -> (Vec<F>, Vec<F>) {
    let value = |x: f64| F::from_f64(x).unwrap();
    let im = if real_slots::<Rg>() { 0.0 } else { 0.7 };
    (
        (0..len).map(|i| value(0.9 * (i as f64 + 0.3).cos())).collect(),
        (0..len).map(|i| value(im * (2.0 * i as f64 + 0.1).sin())).collect(),
    )
}

fn plaintext<BE, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    slot_layout: SlotLayout,
    len: usize,
) -> CKKSPlaintextOwned<BE, Rg>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, Rg>,
{
    let mut pt = module.ckks_plaintext_alloc_from_slots::<Rg>(
        slots_of::<BE, Rg>(module, slot_layout, len),
        Base2K(params.base2k as u32),
        TorusPrecision((LOG_DELTA + 30) as u32),
    );
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(LOG_DELTA),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout,
    });
    pt
}

/// Slots holding a batch of `len` values in `slot_layout`; a full layout
/// repeats the batch up to the backend's smallest plaintext.
fn slots_of<BE: Backend, Rg: Ring>(module: &Module<BE>, slot_layout: SlotLayout, len: usize) -> usize {
    let slots = match slot_layout {
        SlotLayout::Full => len,
        SlotLayout::Replicated(layout) => len << layout.log_r(),
    };
    slots.max(module.ckks_slots_of_degree::<Rg>(BE::MIN_DEGREE))
}

/// Bits of precision: `-log2 max |got - want|` over both parts.
pub(crate) fn precision<F: TestScalar>(got: (&[F], &[F]), want: (&[F], &[F])) -> f64 {
    let err = |a: &[F], b: &[F]| {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).to_f64().unwrap().abs())
            .fold(0.0, f64::max)
    };
    -err(got.0, want.0).max(err(got.1, want.1)).log2()
}

fn decode<BE, F, Rg: Ring>(module: &Module<BE>, pt: &CKKSPlaintextOwned<BE, Rg>, len: usize) -> (Vec<F>, Vec<F>)
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let slots = module.ckks_slots_of_degree::<Rg>(pt.n().as_usize());
    let read = if pt.slot_layout() == SlotLayout::Full { slots } else { len };
    let (mut re, mut im) = (vec![F::zero(); read], vec![F::zero(); read]);
    with_scratch::<BE, _>(module.ckks_encoding_host_tmp_bytes(slots), |scratch| {
        module.ckks_decode_slots_host_into(pt, &mut re, &mut im, scratch)
    })
    .unwrap();
    re.truncate(len);
    im.truncate(len);
    (re, im)
}

fn encoded<BE, F, Rg: Ring>(module: &Module<BE>, pt: &mut CKKSPlaintextOwned<BE, Rg>, re: &[F], im: &[F]) -> Result<()>
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let slots = module.ckks_slots_of_degree::<Rg>(pt.n().as_usize());
    let tile = |values: &[F]| values.iter().copied().cycle().take(slots).collect::<Vec<_>>();
    let (re, im) = if pt.slot_layout() == SlotLayout::Full {
        (tile(re), tile(im))
    } else {
        (re.to_vec(), im.to_vec())
    };
    with_scratch::<BE, _>(module.ckks_encoding_host_tmp_bytes(slots), |scratch| {
        module.ckks_encode_slots_host_into(pt, &re, &im, scratch)
    })?;
    Ok(())
}

/// Asserts that `err` is `InvalidOperand` under `op`.
fn assert_invalid(err: &crate::CKKSError, op: &str) {
    assert!(
        matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op: got, .. }) if *got == op),
        "expected InvalidOperand from {op}, got {err}"
    );
}

/// [`assert_invalid`] for an encoding error.
fn assert_invalid_encoding(err: &anyhow::Error, op: &str) {
    let ckks = err
        .downcast_ref::<crate::CKKSError>()
        .unwrap_or_else(|| panic!("expected a CKKS error from {op}, got {err}"));
    assert_invalid(ckks, op);
}

/// `InvalidOperand` under `op` with a reason containing `reason`.
fn operand(op: &'static str, reason: impl Into<String>) -> impl Fn(&crate::CKKSCompositionError) -> bool {
    let reason = reason.into();
    move |e| matches!(e, crate::CKKSCompositionError::InvalidOperand { op: got, reason: why } if *got == op && why.contains(&reason))
}

/// Runs `$call`, which must refuse with `InvalidOperand` under `$op`, as `$matches` accepts, or with an error naming
/// `$message`, leaving `$operand`s as they were.
macro_rules! refused {
    ($rg:ty, $op:expr, $call:expr, [$($operand:expr),+ $(,)?]) => {{
        let before = ($(snapshot::<BE, _, $rg>(&$operand),)+);
        let err = $call.unwrap_err();
        assert_invalid(&err, $op);
        assert_eq!(($(snapshot::<BE, _, $rg>(&$operand),)+), before, "{} changed an operand", $op);
    }};
    ($rg:ty, $label:expr, $call:expr, [$($operand:expr),+ $(,)?], message = $message:expr) => {{
        let before = ($(snapshot::<BE, _, $rg>(&$operand),)+);
        let err = $call.unwrap_err();
        assert_ckks_internal($label, &err, &$message);
        assert_eq!(($(snapshot::<BE, _, $rg>(&$operand),)+), before, "{} changed an operand", $label);
    }};
    ($rg:ty, $label:expr, $call:expr, [$($operand:expr),+ $(,)?], $matches:expr) => {{
        let before = ($(snapshot::<BE, _, $rg>(&$operand),)+);
        let err = $call.unwrap_err();
        assert!(err.composition().is_some_and($matches), "{}: unexpected refusal {err}", $label);
        assert_eq!(($(snapshot::<BE, _, $rg>(&$operand),)+), before, "{} changed an operand", $label);
    }};
}

/// Decrypts `ct` and decodes its first `len` values.
pub(crate) fn decrypt_batch<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE, Rg>,
    ct: &CKKSCiphertextOwned<BE, Rg>,
    len: usize,
) -> (Vec<F>, Vec<F>)
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    super::helpers::assert_canonical_at_k::<BE, Rg>("decrypt", ct);
    // A few bits of budget hold the unit-scale message.
    let budget = ct.log_budget().min(10);
    let mut pt = module.ckks_plaintext_alloc_from_slots::<Rg>(
        module.ckks_slots_of_degree::<Rg>((ct.n().as_usize() >> ct.log_sparsity()).max(BE::MIN_DEGREE)),
        Base2K(params.base2k as u32),
        TorusPrecision((ct.log_delta() + budget) as u32),
    );
    pt.set_meta(ct.meta());
    pt.set_log_budget(budget);
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_decrypt_tmp_bytes(&pt, ct));
    module.ckks_decrypt(&mut pt, ct, sk, &mut scratch.borrow()).unwrap();
    decode::<BE, F, Rg>(module, &pt, len)
}

pub fn test_replicated_encoding_is_shaped<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let (re, im) = batch::<Rg, F>(1 << LOG_K);
    let mut full = plaintext(&params, module, SlotLayout::Full, re.len());
    encoded(module, &mut full, &re, &im)?;
    let mut shaped = plaintext(&params, module, SlotLayout::Replicated(layout()), re.len());
    encoded(module, &mut shaped, &re, &im)?;

    let (full_re, full_im) = decode::<BE, F, Rg>(module, &full, re.len());
    let (shaped_re, shaped_im) = decode::<BE, F, Rg>(module, &shaped, re.len());
    let full_bits = precision((&full_re, &full_im), (&re, &im));
    let shaped_bits = precision((&shaped_re, &shaped_im), (&re, &im));
    assert!(
        shaped_bits >= full_bits + 20.0,
        "replicated encoding at 2^{LOG_DELTA}: {shaped_bits:.2} bits against {full_bits:.2} for a full layout"
    );
    Ok(())
}

pub fn test_replicated_unit_shaping_encodes_plainly<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let unit = ReplicatedLayout::new(LOG_K + 2, 1, &[1, 0, 0, 0])?;
    let (re, im) = batch::<Rg, F>(1 << unit.log_k());
    let mut shaped = plaintext(&params, module, SlotLayout::Replicated(unit), re.len());
    encoded(module, &mut shaped, &re, &im)?;

    let scatter = |values: &[F]| {
        let mut slots = vec![F::zero(); values.len() << unit.log_r()];
        values.iter().enumerate().for_each(|(i, &v)| slots[i << unit.log_r()] = v);
        slots
    };
    let mut full = plaintext(&params, module, SlotLayout::Full, re.len() << unit.log_r());
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(re.len() << unit.log_r()));
    module.ckks_encode_all_slots_host_into(&mut full, &scatter(&re), &scatter(&im), &mut scratch.borrow())?;

    let digits = |pt: &CKKSPlaintextOwned<BE, Rg>| -> Vec<i64> {
        let glwe = GLWEToBackendRef::<BE, Rg>::to_backend_ref(pt);
        (0..glwe.size()).flat_map(|limb| glwe.data().at(0, limb).to_vec()).collect()
    };
    assert_eq!(digits(&shaped), digits(&full));
    Ok(())
}

pub fn test_replicated_encryption_is_shaped<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let (re, im) = batch::<Rg, F>(1 << LOG_K);
    let mut bits = Vec::new();
    for slot_layout in [SlotLayout::Full, SlotLayout::Replicated(layout())] {
        let mut pt = plaintext(&params, module, slot_layout, re.len());
        encoded(module, &mut pt, &re, &im)?;
        let mut ct = alloc_ct(&params, module, LOG_DELTA + 30);
        let (mut xe, mut xa) = (Source::new(next_test_seed(1)), Source::new(next_test_seed(2)));
        with_scratch::<BE, _>(module.ckks_encrypt_sk_tmp_bytes(&ct), |scratch| {
            module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, scratch)
        })?;
        assert_eq!(ct.slot_layout(), slot_layout);
        assert_eq!((ct.log_delta(), ct.k().as_usize()), (LOG_DELTA, LOG_DELTA + 30));
        // The shaped rounding leaves an estimate at the ciphertext's width, its masks rounded too.
        let noise = ct.noise().expect("a fresh encryption carries a noise estimate");
        assert_eq!(noise.precision(), ct.k());
        assert_eq!(
            noise.masks().iter().any(|mask| mask.variance() > 0.0),
            slot_layout != SlotLayout::Full
        );

        let (got_re, got_im) = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &ct, re.len());
        bits.push(precision((&got_re, &got_im), (&re, &im)));
    }
    assert!(
        bits[1] >= bits[0] + 20.0,
        "fresh replicated encryption at 2^{LOG_DELTA}: {:.2} bits against {:.2} for a full layout",
        bits[1],
        bits[0]
    );
    Ok(())
}

pub fn test_replicated_encoding_rejects_batch_size<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let (re, im) = batch::<Rg, F>(2 << LOG_K);
    let mut pt = plaintext(&params, module, SlotLayout::Replicated(layout()), 1 << LOG_K);
    let before = snapshot::<BE, _, Rg>(&pt);
    assert_invalid_encoding(
        &encoded(module, &mut pt, &re, &im).unwrap_err(),
        "ckks_encode_slots_host_into",
    );
    assert_eq!(snapshot::<BE, _, Rg>(&pt), before);
    Ok(())
}

/// A short arena, and on the conjugate-invariant ring a shaping polynomial
/// without an invariant shift, refuse before the encryption draws from its
/// sources or writes the ciphertext.
pub fn test_replicated_encryption_refuses_before_drawing<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let (re, im) = batch::<Rg, F>(1 << LOG_K);
    let mut pt = plaintext(&params, module, SlotLayout::Replicated(layout()), re.len());
    encoded(module, &mut pt, &re, &im)?;
    let mut ct = alloc_ct(&params, module, LOG_DELTA + 30);
    let bytes = module.ckks_encrypt_sk_tmp_bytes(&ct);
    let seeds = (next_test_seed(5), next_test_seed(6));
    let mut attempt = |pt: &CKKSPlaintextOwned<BE, Rg>, bytes: usize| -> crate::CKKSError {
        let (mut xe, mut xa) = (Source::new(seeds.0), Source::new(seeds.1));
        let before = snapshot::<BE, _, Rg>(&ct);
        let err = with_scratch::<BE, _>(bytes, |scratch| {
            module.ckks_encrypt_sk(&mut ct, pt, &sk, &mut xe, &mut xa, scratch)
        })
        .unwrap_err();
        assert_eq!(snapshot::<BE, _, Rg>(&ct), before);
        let fresh = (Source::new(seeds.0).next_i64(), Source::new(seeds.1).next_i64());
        assert_eq!(
            (xe.next_i64(), xa.next_i64()),
            fresh,
            "the refused encryption drew from its sources"
        );
        err
    };
    let err = attempt(&pt, bytes - 1);
    assert!(matches!(&err, crate::CKKSError::Internal(_)), "{err}");
    if real_slots::<Rg>() {
        // `U = 3 + 2Y^2` has no conjugate-invariant shift.
        pt.set_slot_layout(SlotLayout::Replicated(ReplicatedLayout::new(LOG_K, 1, &[3, 0, 2, 0])?));
        assert_invalid(&attempt(&pt, bytes), "ckks_encrypt_sk");
    }
    Ok(())
}

/// Two replicated operands at two scales do not add. Rounding the finer one
/// with shaping to the coarser scale makes them add, at the shaped precision.
pub fn test_replicated_round_then_add<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let len = 1 << LOG_K;
    let (a, b) = (unit_batch::<Rg, F>(len, 0), unit_batch::<Rg, F>(len, 1));
    let (k, finer) = (3 * LOG_DELTA, 4);
    let slot_layout = SlotLayout::Replicated(layout());
    let coarse = encrypt_batch(&params, module, &sk, slot_layout, (&a.0, &a.1), LOG_DELTA, k, 1)?;
    // `b` encrypted `finer` bits further under the scale.
    let mut pt = plaintext(&params, module, slot_layout, len);
    pt.set_scale(Scale::Log(LOG_DELTA + finer));
    encoded(module, &mut pt, &b.0, &b.1)?;
    let mut fine = alloc_ct(&params, module, k);
    let (mut xe, mut xa) = (Source::new(next_test_seed(3)), Source::new(next_test_seed(4)));
    with_scratch::<BE, _>(module.ckks_encrypt_sk_tmp_bytes(&fine), |scratch| {
        module.ckks_encrypt_sk(&mut fine, &pt, &sk, &mut xe, &mut xa, scratch)
    })?;
    let mut out = alloc_ct(&params, module, k);
    let mut scratch = alloc_scratch(&params, module);
    refused!(
        Rg,
        "ckks_add_into",
        module.ckks_add_into(&mut out, &coarse, &fine, &mut scratch.borrow()),
        [out, coarse, fine]
    );

    let mut rounded = alloc_ct(&params, module, k - finer);
    with_scratch::<BE, _>(module.ckks_round_tmp_bytes(&rounded, &fine), |s| {
        module.ckks_round(&mut rounded, &fine, s)
    })?;
    assert_eq!((rounded.slot_layout(), rounded.log_delta()), (slot_layout, LOG_DELTA));
    module.ckks_add_into(&mut out, &coarse, &rounded, &mut scratch.borrow())?;
    assert_eq!((out.slot_layout(), out.log_delta()), (slot_layout, LOG_DELTA));

    let want: (Vec<F>, Vec<F>) = (
        a.0.iter().zip(&b.0).map(|(&x, &y)| x + y).collect(),
        a.1.iter().zip(&b.1).map(|(&x, &y)| x + y).collect(),
    );
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &out, len);
    let bits = precision((&got.0, &got.1), (&want.0, &want.1));
    // A full layout at `2^20` keeps about 18 bits.
    assert!(bits >= 45.0, "shaped round then add at 2^{LOG_DELTA}: {bits:.2} bits");
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Product {
    Mul,
    MulAssign,
    MulPrepared,
    Square,
    SquareAssign,
    Pt,
    PtAssign,
    Const,
    ConstAssign,
    /// `acc·b + b·acc`, two pairs summed once.
    Dot,
    /// `acc·b + acc·b` with a plaintext `b`, two pairs summed and rounded once.
    DotPt,
    /// `c ± acc·b` into a fresh accumulator `c`.
    MulAdd,
    MulSub,
    MulAddPt,
    MulSubPt,
    MulAddConst,
    MulSubConst,
    /// `acc·b + c` with plaintexts `b` and `c`.
    AffinePt,
    AffinePtAssign,
}

impl Product {
    const ALL: [Self; 19] = [
        Self::Mul,
        Self::MulAssign,
        Self::MulPrepared,
        Self::Square,
        Self::SquareAssign,
        Self::Pt,
        Self::PtAssign,
        Self::Const,
        Self::ConstAssign,
        Self::Dot,
        Self::DotPt,
        Self::MulAdd,
        Self::MulSub,
        Self::MulAddPt,
        Self::MulSubPt,
        Self::MulAddConst,
        Self::MulSubConst,
        Self::AffinePt,
        Self::AffinePtAssign,
    ];
}

const STEPS: usize = 3;
/// Exactly representable at the constant plaintext's 8 bits, so that only the
/// rescale rounds.
const CONSTANT: f64 = 0.75;

#[expect(clippy::too_many_arguments, reason = "the batch's layout, values, scale, width and seed")]
pub(crate) fn encrypt_batch<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE, Rg>,
    slot_layout: SlotLayout,
    values: (&[F], &[F]),
    log_delta: usize,
    k: usize,
    seed: u8,
) -> Result<CKKSCiphertextOwned<BE, Rg>>
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let mut pt = plaintext(params, module, slot_layout, values.0.len());
    pt.set_scale(Scale::Log(log_delta));
    encoded(module, &mut pt, values.0, values.1)?;
    let mut ct = alloc_ct(params, module, k);
    let (mut xe, mut xa) = (
        Source::new(next_test_seed(seed)),
        Source::new(next_test_seed(seed.wrapping_add(128))),
    );
    with_scratch::<BE, _>(module.ckks_encrypt_sk_tmp_bytes(&ct), |scratch| {
        module.ckks_encrypt_sk(&mut ct, &pt, sk, &mut xe, &mut xa, scratch)
    })?;
    Ok(ct)
}

/// Unit-modulus operands, so that the chained products stay bounded. Real
/// slots take the real parts, of modulus at most one.
fn unit_batch<Rg: Ring, F: TestScalar>(len: usize, step: usize) -> (Vec<F>, Vec<F>) {
    let angle = |i: usize| 0.37 * (i + 1) as f64 + 1.1 * step as f64;
    let value = |x: f64| F::from_f64(x).unwrap();
    let real = real_slots::<Rg>();
    (
        (0..len).map(|i| value(angle(i).cos())).collect(),
        (0..len).map(|i| value(if real { 0.0 } else { angle(i).sin() })).collect(),
    )
}

fn complex_mul<F: TestScalar>(a: (&[F], &[F]), b: (&[F], &[F])) -> (Vec<F>, Vec<F>) {
    (
        (0..a.0.len()).map(|i| a.0[i] * b.0[i] - a.1[i] * b.1[i]).collect(),
        (0..a.0.len()).map(|i| a.0[i] * b.1[i] + a.1[i] * b.0[i]).collect(),
    )
}

/// Precision of `STEPS` chained products of one kind.
fn product_chain<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
    product: Product,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let len = 1 << LOG_K;
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let tsk = gen_tsk(params, module, &sk_raw, &mut scratch.borrow());
    let k = LOG_DELTA * (STEPS + 2);

    let mut want = unit_batch::<Rg, F>(len, 0);
    let mut acc = encrypt_batch(params, module, &sk, slot_layout, (&want.0, &want.1), LOG_DELTA, k, 1)?;
    let cst = ckks_pt_cst::<BE, F, Rg>(host_module, module, params.base2k.into(), PT_PREC, Some(CONSTANT), None);
    let c = F::from_f64(CONSTANT).unwrap();
    let constant = (vec![c; len], vec![F::zero(); len]);
    let add = |x: (Vec<F>, Vec<F>), y: &(Vec<F>, Vec<F>), sign: F| -> (Vec<F>, Vec<F>) {
        (
            x.0.iter().zip(&y.0).map(|(&u, &v)| u + sign * v).collect(),
            x.1.iter().zip(&y.1).map(|(&u, &v)| u + sign * v).collect(),
        )
    };
    let (one, minus) = (F::one(), -F::one());
    for step in 1..=STEPS {
        let operand = unit_batch::<Rg, F>(len, step);
        let addend = unit_batch::<Rg, F>(len, step + STEPS);
        let fresh = encrypt_batch(
            params,
            module,
            &sk,
            slot_layout,
            (&operand.0, &operand.1),
            LOG_DELTA,
            k,
            step as u8 + 1,
        )?;
        let mut accumulator = encrypt_batch(
            params,
            module,
            &sk,
            slot_layout,
            (&addend.0, &addend.1),
            LOG_DELTA,
            k,
            step as u8 + 64,
        )?;
        let mut pt = plaintext(params, module, slot_layout, len);
        encoded(module, &mut pt, &operand.0, &operand.1)?;
        // The offset fits the budget the affine product leaves at the last step.
        let mut offset = module.ckks_plaintext_alloc_from_slots::<Rg>(
            slots_of::<BE, Rg>(module, slot_layout, len),
            Base2K(params.base2k as u32),
            TorusPrecision((LOG_DELTA + 10) as u32),
        );
        offset.set_meta(pt.meta());
        encoded(module, &mut offset, &addend.0, &addend.1)?;
        let mut out = alloc_ct(params, module, k);
        let times_operand = complex_mul((&want.0, &want.1), (&operand.0, &operand.1));
        let times_constant = complex_mul((&want.0, &want.1), (&constant.0, &constant.1));
        // Each call runs on exactly its own query's bytes.
        want = match product {
            Product::Mul => {
                with_scratch::<BE, _>(module.ckks_mul_tmp_bytes(&out, &acc, &fresh, &tsk), |s| {
                    module.ckks_mul_into(&mut out, &acc, &fresh, &tsk, s)
                })?;
                acc = out;
                times_operand
            }
            Product::MulAssign => {
                with_scratch::<BE, _>(module.ckks_mul_tmp_bytes(&acc, &acc, &fresh, &tsk), |s| {
                    module.ckks_mul_assign(&mut acc, &fresh, &tsk, s)
                })?;
                times_operand
            }
            Product::MulPrepared => {
                let prepared = with_scratch::<BE, _>(module.ckks_right_operand_prepare_tmp_bytes(&fresh), |s| {
                    module.ckks_right_operand_prepare_alloc(&fresh, PrepareHint::OneShot, s)
                })?;
                with_scratch::<BE, _>(module.ckks_mul_tmp_bytes(&acc, &acc, &prepared, &tsk), |s| {
                    module.ckks_mul_prepared_assign(&mut acc, &prepared, &tsk, s)
                })?;
                times_operand
            }
            Product::Square => {
                with_scratch::<BE, _>(module.ckks_square_tmp_bytes(&out, &acc, &tsk), |s| {
                    module.ckks_square_into(&mut out, &acc, &tsk, s)
                })?;
                acc = out;
                complex_mul((&want.0, &want.1), (&want.0, &want.1))
            }
            Product::SquareAssign => {
                with_scratch::<BE, _>(module.ckks_square_tmp_bytes(&acc, &acc, &tsk), |s| {
                    module.ckks_square_assign(&mut acc, &tsk, s)
                })?;
                complex_mul((&want.0, &want.1), (&want.0, &want.1))
            }
            Product::Pt => {
                with_scratch::<BE, _>(module.ckks_mul_pt_tmp_bytes(&out, &acc, &pt), |s| {
                    module.ckks_mul_pt_into(&mut out, &acc, &pt, s)
                })?;
                acc = out;
                times_operand
            }
            Product::PtAssign => {
                with_scratch::<BE, _>(module.ckks_mul_pt_tmp_bytes(&acc, &acc, &pt), |s| {
                    module.ckks_mul_pt_assign(&mut acc, &pt, s)
                })?;
                times_operand
            }
            Product::Const => {
                with_scratch::<BE, _>(module.ckks_mul_pt_tmp_bytes(&out, &acc, &cst), |s| {
                    module.ckks_mul_pt_into(&mut out, &acc, &cst, s)
                })?;
                acc = out;
                times_constant
            }
            Product::ConstAssign => {
                with_scratch::<BE, _>(module.ckks_mul_pt_tmp_bytes(&acc, &acc, &cst), |s| {
                    module.ckks_mul_pt_assign(&mut acc, &cst, s)
                })?;
                times_constant
            }
            Product::Dot => {
                with_scratch::<BE, _>(module.ckks_dot_product_tmp_bytes(2, &out, &acc, &fresh, &tsk), |s| {
                    module.ckks_dot_product(&mut out, &[&acc, &fresh], &[&fresh, &acc], &tsk, s)
                })?;
                acc = out;
                add(times_operand.clone(), &times_operand, one)
            }
            Product::DotPt => {
                let bytes = module.ckks_dot_product_pt_tmp_bytes(&out, &fresh, &pt);
                let before = snapshot::<BE, _, Rg>(&out);
                let short = with_scratch::<BE, _>(bytes - 1, |s| {
                    module.ckks_dot_product_pt(&mut out, &[&acc, &fresh], &[&pt, &cst], s)
                })
                .unwrap_err();
                assert_ckks_internal(
                    "dot product one byte short",
                    &short,
                    "ckks_dot_product_pt needs a scratch arena",
                );
                assert_eq!(snapshot::<BE, _, Rg>(&out), before, "a short arena wrote dst");
                with_scratch::<BE, _>(bytes, |s| {
                    module.ckks_dot_product_pt(&mut out, &[&acc, &fresh], &[&pt, &cst], s)
                })?;
                acc = out;
                let operand_times_constant = complex_mul((&operand.0, &operand.1), (&constant.0, &constant.1));
                add(times_operand, &operand_times_constant, one)
            }
            Product::MulAdd | Product::MulSub => {
                let bytes = module.ckks_mul_add_tmp_bytes(&accumulator, &acc, &fresh, &tsk);
                with_scratch::<BE, _>(bytes, |s| {
                    if matches!(product, Product::MulAdd) {
                        module.ckks_mul_add_into(&mut accumulator, &acc, &fresh, &tsk, s)
                    } else {
                        module.ckks_mul_sub_into(&mut accumulator, &acc, &fresh, &tsk, s)
                    }
                })?;
                acc = accumulator;
                add(
                    addend,
                    &times_operand,
                    if matches!(product, Product::MulAdd) { one } else { minus },
                )
            }
            Product::MulAddPt | Product::MulSubPt => {
                with_scratch::<BE, _>(module.ckks_mul_add_pt_tmp_bytes(&accumulator, &acc, &pt), |s| {
                    if matches!(product, Product::MulAddPt) {
                        module.ckks_mul_add_pt_into(&mut accumulator, &acc, &pt, s)
                    } else {
                        module.ckks_mul_sub_pt_into(&mut accumulator, &acc, &pt, s)
                    }
                })?;
                acc = accumulator;
                add(
                    addend,
                    &times_operand,
                    if matches!(product, Product::MulAddPt) { one } else { minus },
                )
            }
            Product::MulAddConst | Product::MulSubConst => {
                with_scratch::<BE, _>(module.ckks_mul_add_pt_tmp_bytes(&accumulator, &acc, &cst), |s| {
                    if matches!(product, Product::MulAddConst) {
                        module.ckks_mul_add_pt_into(&mut accumulator, &acc, &cst, s)
                    } else {
                        module.ckks_mul_sub_pt_into(&mut accumulator, &acc, &cst, s)
                    }
                })?;
                acc = accumulator;
                add(
                    addend,
                    &times_constant,
                    if matches!(product, Product::MulAddConst) { one } else { minus },
                )
            }
            Product::AffinePt => {
                with_scratch::<BE, _>(module.ckks_affine_pt_tmp_bytes(&out, &acc, &pt), |s| {
                    module.ckks_affine_pt_into(&mut out, &acc, &pt, &offset, s)
                })?;
                acc = out;
                add(times_operand, &addend, one)
            }
            Product::AffinePtAssign => {
                with_scratch::<BE, _>(module.ckks_affine_pt_tmp_bytes(&acc, &acc, &pt), |s| {
                    module.ckks_affine_pt_assign(&mut acc, &pt, &offset, s)
                })?;
                add(times_operand, &addend, one)
            }
        };
        assert_eq!(acc.slot_layout(), slot_layout, "{product:?}");
    }
    let got = decrypt_batch::<BE, F, Rg>(params, module, &sk, &acc, len);
    Ok(precision((&got.0, &got.1), (&want.0, &want.1)))
}

pub fn test_replicated_products_are_shaped<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    for product in Product::ALL {
        let full = product_chain::<BE, F, Rg>(&params, module, host_module, SlotLayout::Full, product)?;
        let shaped = product_chain::<BE, F, Rg>(&params, module, host_module, SlotLayout::Replicated(layout()), product)?;
        assert!(
            shaped >= full + 15.0,
            "{product:?} chain at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
        );
    }
    Ok(())
}

/// Every unit of [`ShapingUnit::ALL`] shapes a product chain: the data slots
/// gain the unit's target suppression over a full layout, within 4 bits, up to
/// the 35 bits the chain resolves.
pub fn test_replicated_units_shape_products<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let full = product_chain::<BE, F, Rg>(&params, module, host_module, SlotLayout::Full, Product::Mul)?;
    for unit in ShapingUnit::ALL {
        let layout = SlotLayout::Replicated(unit.layout(LOG_K)?);
        let shaped = product_chain::<BE, F, Rg>(&params, module, host_module, layout, Product::Mul)?;
        let want = (unit.target_bits as f64).min(35.0) - 4.0;
        assert!(
            shaped >= full + want,
            "R = {}, target {}: {shaped:.2} bits against {full:.2} for a full layout",
            1 << unit.log_r,
            unit.target_bits
        );
    }
    Ok(())
}

pub fn test_replicated_layouts_do_not_mix<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSMulAddOps<BE, Rg> + CKKSMulSubOps<BE, Rg>,
    F: TestScalar,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let values = unit_batch::<Rg, F>(1 << LOG_K, 0);
    let k = 3 * LOG_DELTA;
    let shaped = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        k,
        1,
    )?;
    let full = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Full,
        (&values.0, &values.1),
        LOG_DELTA,
        k,
        2,
    )?;
    let mut pt = plaintext(&params, module, SlotLayout::Full, values.0.len());
    encoded(module, &mut pt, &values.0, &values.1)?;
    let mut out = alloc_ct(&params, module, k);
    let s = &mut scratch.borrow();
    refused!(
        Rg,
        "ckks_mul_into",
        module.ckks_mul_into(&mut out, &shaped, &full, &tsk, s),
        [out, shaped, full]
    );
    refused!(
        Rg,
        "ckks_add_into",
        module.ckks_add_into(&mut out, &shaped, &full, s),
        [out, shaped, full]
    );
    refused!(
        Rg,
        "ckks_mul_pt_into",
        module.ckks_mul_pt_into(&mut out, &shaped, &pt, s),
        [out, shaped, pt]
    );
    refused!(
        Rg,
        "ckks_add_into",
        module.ckks_add_into(&mut out, &shaped, &pt, s),
        [out, shaped, pt]
    );
    let cst = ckks_pt_cst::<BE, F, Rg>(host_module, module, params.base2k.into(), PT_PREC, Some(CONSTANT), None);
    refused!(
        Rg,
        "ckks_add_into",
        module.ckks_add_into(&mut out, &shaped, &cst, s),
        [out, shaped]
    );
    module.ckks_mul_pt_into(&mut out, &shaped, &cst, s)?;

    // The sum of many terms checks every layout before the first addition writes `out`.
    let shaped_too = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        k,
        3,
    )?;
    refused!(
        Rg,
        "ckks_add_many",
        module.ckks_add_many(1, &mut out, &[&shaped, &shaped_too, &full], s),
        [out, shaped, shaped_too, full]
    );

    // A product leaves a reused replicated destination in the full layout.
    let mut reused = shaped.clone();
    module.ckks_mul_into(&mut reused, &full, &full, &tsk, s)?;
    assert_eq!(reused.slot_layout(), SlotLayout::Full);
    // A fused multiply-accumulate checks its accumulator's layout before the product, under its own name.
    let mut acc = shaped_too.clone();
    refused!(
        Rg,
        "ckks_mul_add_into",
        module.ckks_mul_add_into(&mut acc, &full, &full, &tsk, s),
        [acc, full]
    );
    refused!(
        Rg,
        "ckks_mul_sub_pt_into",
        module.ckks_mul_sub_pt_into(&mut acc, &full, &pt, s),
        [acc, full, pt]
    );
    refused!(
        Rg,
        "ckks_mul_add_pt_into",
        module.ckks_mul_add_pt_into(&mut acc, &full, &cst, s),
        [acc, full]
    );
    Ok(())
}

pub fn test_replicated_plain_rounding_is_rejected<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg>
        + CKKSEncodingOps<BE, F, Rg>
        + CKKSAffineOps<BE, Rg>
        + CKKSDotProductOps<BE, Rg>
        + CKKSLinearTransformationOps<BE, Rg>
        + CKKSInnerSumOps<BE, Rg>
        + GLWEMulPreparedPlainSum<BE, Rg>,
    F: TestScalar,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let values = unit_batch::<Rg, F>(1 << LOG_K, 0);
    let k = 3 * LOG_DELTA;
    let shaped = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        k,
        1,
    )?;
    let mut out = alloc_ct(&params, module, k);
    let cst = ckks_pt_cst::<BE, F, Rg>(host_module, module, params.base2k.into(), PT_PREC, Some(CONSTANT), None);
    refused!(
        Rg,
        "ckks_affine_pt_into",
        module.ckks_affine_pt_into(&mut out, &shaped, &cst, &cst, &mut scratch.borrow()),
        [out, shaped]
    );

    let mut pt = plaintext(&params, module, SlotLayout::Full, values.0.len());
    encoded(module, &mut pt, &values.0, &values.1)?;
    let left = module.ckks_left_operand_prepare_alloc(&shaped, PrepareHint::OneShot, &mut scratch.borrow())?;
    let right = module.ckks_right_operand_prepare_alloc(&pt, PrepareHint::OneShot, &mut scratch.borrow())?;
    refused!(
        Rg,
        "ckks_dot_product_prepared_pt",
        module.ckks_dot_product_prepared_pt(&mut out, &[&left], &[&right], &[1], &mut scratch.borrow()),
        [out]
    );

    let gal = module.ckks_galois_element::<Rg>(1);
    let atks = HashMap::from([(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()))]);
    let mut babies = LinearTransformationBabySteps::alloc(module, &[poulpy_core::layouts::galois_of_rotation(1)], &shaped);
    refused!(
        Rg,
        "ckks_linear_transformation_baby_steps_prepare",
        module.ckks_linear_transformation_baby_steps_prepare(&mut babies, &shaped, &atks, &mut scratch.borrow()),
        [shaped]
    );

    // The inner sum's traces key-switch with plain rounding.
    refused!(
        Rg,
        "ckks_inner_sum",
        module.ckks_inner_sum(0, 1, false, &mut out, &shaped, &atks, &mut scratch.borrow()),
        [out, shaped]
    );

    // No shaping pass fits `i128` above 126 torus bits.
    let mut wide = module.ckks_plaintext_alloc_from_slots::<Rg>(
        slots_of::<BE, Rg>(module, SlotLayout::Replicated(layout()), values.0.len()),
        Base2K(params.base2k as u32),
        TorusPrecision(130),
    );
    wide.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(100),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout: SlotLayout::Replicated(layout()),
    });
    let wide_before = snapshot::<BE, _, Rg>(&wide);
    assert_invalid_encoding(
        &encoded(module, &mut wide, &values.0, &values.1).unwrap_err(),
        "ckks_quantize_coeffs_into",
    );
    assert_eq!(snapshot::<BE, _, Rg>(&wide), wide_before);

    // A sum at two scales would drop the finer operand's bits plainly.
    let mut coarse = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        k,
        2,
    )?;
    coarse.set_log_delta(LOG_DELTA - 4);
    refused!(
        Rg,
        "ckks_add_into",
        module.ckks_add_into(&mut out, &shaped, &coarse, &mut scratch.borrow()),
        [out, shaped, coarse]
    );
    let mut fine = plaintext(&params, module, SlotLayout::Replicated(layout()), values.0.len());
    encoded(module, &mut fine, &values.0, &values.1)?;
    let mut acc = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        k,
        3,
    )?;
    acc.set_log_delta(LOG_DELTA - 4);
    refused!(
        Rg,
        "ckks_add_assign",
        module.ckks_add_assign(&mut acc, &fine, &mut scratch.borrow()),
        [acc, fine]
    );

    {
        let before = snapshot::<BE, _, _>(&out);
        let source_before = snapshot::<BE, _, _>(&acc);
        let pt_before = snapshot::<BE, _, _>(&fine);
        for add in [true, false] {
            let bytes = if add {
                module.ckks_add_tmp_bytes(out.size())
            } else {
                module.ckks_sub_tmp_bytes(out.size())
            };
            let err = with_scratch::<BE, _>(bytes, |scratch| {
                if add {
                    module.ckks_add_into(&mut out, &acc, &fine, scratch)
                } else {
                    module.ckks_sub_into(&mut out, &acc, &fine, scratch)
                }
            })
            .unwrap_err();
            let expected_op = if add { "ckks_add_into" } else { "ckks_sub_into" };
            assert!(
                matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op, .. }) if *op == expected_op)
            );
            assert_eq!(snapshot::<BE, _, _>(&out), before);
            assert_eq!(snapshot::<BE, _, _>(&acc), source_before);
            assert_eq!(snapshot::<BE, _, _>(&fine), pt_before);
        }
    }

    // `N/2` values of period `R` exceed the slots of the ring.
    let log_n = params.n.ilog2() as u8;
    let oversized = ReplicatedLayout::new(log_n - 1, layout().log_r() as u8, layout().shaping())?;
    let mut big = plaintext(&params, module, SlotLayout::Full, params.n / 2);
    big.set_meta(GLWEPlaintextMeta {
        slot_layout: SlotLayout::Replicated(oversized),
        ..big.meta()
    });
    let many = unit_batch::<Rg, F>(params.n / 2, 0);
    let big_before = snapshot::<BE, _, Rg>(&big);
    assert_invalid_encoding(
        &encoded(module, &mut big, &many.0, &many.1).unwrap_err(),
        "ckks_encode_slots_host_into",
    );
    assert_eq!(snapshot::<BE, _, Rg>(&big), big_before);
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum PolynomialCase {
    Monomial,
    Chebyshev,
}

fn coefficients<F: TestScalar>(len: usize, shift: usize) -> Vec<F> {
    (0..len)
        .map(|i| F::from_f64(0.37 * (1.7 * (i + shift) as f64 + 0.2).sin()).unwrap())
        .collect()
}

/// An encrypted batch of real inputs in `[-1, 1]`, and the setup of a
/// polynomial evaluation on it with constants at the ciphertext scale.
struct PolynomialSetup<BE: Backend, F, Rg: Ring = Standard> {
    sk: GLWESecretPrepared<BE::OwnedBuf, BE, Rg>,
    tsk: poulpy_core::layouts::GLWETensorKeyPrepared<BE::OwnedBuf, BE, Rg>,
    xs: Vec<F>,
    x: CKKSCiphertextOwned<BE, Rg>,
    out: CKKSCiphertextOwned<BE, Rg>,
    coeff_meta: CoeffsMeta,
    scratch: ScratchOwned<BE>,
    encoding: ScratchOwned<BE>,
}

fn polynomial_setup<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
) -> Result<PolynomialSetup<BE, F, Rg>>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let len = 1 << LOG_K;
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let tsk = gen_tsk(params, module, &sk_raw, &mut scratch.borrow());
    let k = 7 * LOG_DELTA;
    let xs: Vec<F> = (0..len)
        .map(|i| F::from_f64(0.95 * (1.3 * i as f64 + 0.4).cos()).unwrap())
        .collect();
    let zero = vec![F::zero(); len];
    let x = encrypt_batch(params, module, &sk, slot_layout, (&xs, &zero), LOG_DELTA, k, 1)?;
    let out = alloc_ct(params, module, k);
    let coeff_prec = ckks_spec(params.n, params.base2k, LOG_DELTA, 10);
    let scratch = ScratchOwned::<BE>::alloc(module.ckks_all_ops_tmp_bytes(&x, &coeff_prec, &params.tsk_layout()));
    let encoding = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(params.n));
    let mut coeff_meta = CoeffsMeta::from(coeff_prec);
    coeff_meta.meta.slot_layout = slot_layout;
    Ok(PolynomialSetup {
        sk,
        tsk,
        xs,
        x,
        out,
        coeff_meta,
        scratch,
        encoding,
    })
}

/// Precision of a real polynomial evaluated on real inputs in `[-1, 1]`, with
/// constants at the ciphertext scale, in the layout of the input.
fn polynomial_precision<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
    case: PolynomialCase,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSPolynomialEvaluationOps<BE, Rg>,
{
    let len = 1 << LOG_K;
    let mut setup = polynomial_setup::<BE, F, Rg>(params, module, host_module, slot_layout)?;
    let base2k = params.base2k.into();
    let (basis, degree) = match case {
        PolynomialCase::Monomial => (Basis::Monomial, 7),
        PolynomialCase::Chebyshev => (Basis::Chebyshev, 13),
    };
    let poly = Polynomial::new(basis, coefficients::<F>(degree + 1, 0));
    let bsgs = poly
        .encode_bsgs(host_module, base2k, setup.coeff_meta)?
        .map_baby_steps_ref(|pt| upload_pt(module, pt));
    module.ckks_eval_poly_real_const_coeffs(&mut setup.out, &setup.x, &bsgs, &setup.tsk, &mut setup.scratch.borrow())?;
    assert_eq!(setup.out.slot_layout(), slot_layout, "{case:?}");
    let got = decrypt_batch::<BE, F, Rg>(params, module, &setup.sk, &setup.out, len);
    let want: Vec<F> = setup.xs.iter().map(|&x| poly.evaluate(x)).collect();
    Ok(precision((&got.0, &got.1), (&want, &vec![F::zero(); len])))
}

/// [`polynomial_precision`] of a complex polynomial, on the standard ring.
fn complex_polynomial_precision<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSComplexPolynomialEvaluationOps<BE>,
{
    let len = 1 << LOG_K;
    let mut setup = polynomial_setup::<BE, F, Standard>(params, module, host_module, slot_layout)?;
    let base2k = params.base2k.into();
    let poly = ComplexPolynomial::new(Basis::Monomial, coefficients::<F>(6, 0), coefficients::<F>(6, 2));
    let bsgs = poly
        .encode_bsgs(host_module, base2k, setup.coeff_meta)?
        .map_baby_steps_ref(|pt| upload_pt(module, pt));
    module.ckks_eval_poly_complex_const_coeffs(&mut setup.out, &setup.x, &bsgs, &setup.tsk, &mut setup.scratch.borrow())?;
    assert_eq!(setup.out.slot_layout(), slot_layout);
    let got = decrypt_batch::<BE, F, Standard>(params, module, &setup.sk, &setup.out, len);
    let want: (Vec<F>, Vec<F>) = setup.xs.iter().map(|&x| poly.evaluate(x)).unzip();
    Ok(precision((&got.0, &got.1), (&want.0, &want.1)))
}

pub fn test_replicated_complex_polynomials_are_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSComplexPolynomialEvaluationOps<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let full = complex_polynomial_precision::<BE, F>(&params, module, host_module, SlotLayout::Full)?;
    let shaped = complex_polynomial_precision::<BE, F>(&params, module, host_module, SlotLayout::Replicated(layout()))?;
    assert!(
        shaped >= full + 15.0,
        "complex polynomial at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
    );
    Ok(())
}

pub fn test_replicated_polynomials_are_shaped<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSPolynomialEvaluationOps<BE, Rg>,
    F: TestScalar,
{
    for case in [PolynomialCase::Monomial, PolynomialCase::Chebyshev] {
        let full = polynomial_precision::<BE, F, Rg>(&params, module, host_module, SlotLayout::Full, case)?;
        let shaped = polynomial_precision::<BE, F, Rg>(&params, module, host_module, SlotLayout::Replicated(layout()), case)?;
        assert!(
            shaped >= full + 15.0,
            "{case:?} polynomial at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
        );
    }

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let k = 7 * LOG_DELTA;
    let values = unit_batch::<Rg, F>(1 << LOG_K, 0);
    let x = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        k,
        1,
    )?;
    let narrower = ShapingUnit::U16_20.layout(LOG_K)?;
    let identity = Polynomial::new(Basis::Monomial, vec![F::zero(), F::one()]);
    let mut narrower_meta = CoeffsMeta::from_delta_budget(LOG_DELTA, 10);
    narrower_meta.meta.slot_layout = SlotLayout::Replicated(narrower);
    let bsgs = identity
        .encode_bsgs(host_module, params.base2k.into(), narrower_meta)?
        .map_baby_steps_ref(|pt| upload_pt(module, pt));
    let mut out = alloc_ct(&params, module, k);
    // Refused at entry, before any work or scratch check.
    refused!(
        Rg,
        "ckks_eval_poly_real_const_coeffs",
        with_scratch::<BE, _>(0, |s| module.ckks_eval_poly_real_const_coeffs(&mut out, &x, &bsgs, &tsk, s)),
        [out, x]
    );
    // Constants of the full layout.
    let plain = identity
        .encode_bsgs(
            host_module,
            params.base2k.into(),
            CoeffsMeta::from_delta_budget(LOG_DELTA, 10),
        )?
        .map_baby_steps_ref(|pt| upload_pt(module, pt));
    refused!(
        Rg,
        "ckks_eval_poly_real_const_coeffs",
        with_scratch::<BE, _>(0, |s| module.ckks_eval_poly_real_const_coeffs(&mut out, &x, &plain, &tsk, s)),
        [out, x]
    );
    Ok(())
}

/// A baby step whose masked constants differ in scale leaves the shaped linear
/// combination for the products term by term, which keep the shaping.
pub fn test_replicated_polynomial_fallback_is_shaped<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSPolynomialEvaluationOps<BE, Rg>,
    F: TestScalar,
{
    let len = 1 << LOG_K;
    let full = polynomial_precision::<BE, F, Rg>(&params, module, host_module, SlotLayout::Full, PolynomialCase::Monomial)?;
    let mut setup = polynomial_setup::<BE, F, Rg>(&params, module, host_module, SlotLayout::Replicated(layout()))?;
    let base2k: Base2K = params.base2k.into();
    let poly = Polynomial::new(Basis::Monomial, coefficients::<F>(8, 0));
    let mut coeff_meta = setup.coeff_meta;
    coeff_meta.meta.slots = SlotsKind::Real;
    // The constant of degree one of every baby step, one bit finer than the others.
    // `polynomial_setup` encodes its constants at `2^LOG_DELTA` with 10 bits of budget.
    let mut finer = CoeffsMeta::from_delta_budget(LOG_DELTA + 1, 10);
    finer.meta.slots = SlotsKind::Real;
    let degree = crate::encoding::linear_transformation::period_degree::<_, Rg>(module, 1 << layout().log_r());
    let encoding = &mut setup.encoding;
    let bsgs = crate::layouts::eval_mod::encode_bsgs_with(&poly, SplitStrategy::MinDepth, false, &mut |values: &[F]| {
        values
            .iter()
            .enumerate()
            .map(|(j, &value)| {
                let meta = if j == 1 { finer } else { coeff_meta };
                let mut pt = module.ckks_plaintext_alloc::<Rg>(degree, base2k, meta.k);
                pt.set_meta(GLWEPlaintextMeta {
                    slots: SlotsKind::Real,
                    slot_layout: SlotLayout::Replicated(layout()),
                    ..meta.meta
                });
                module.ckks_encode_masked_constant_host_into(&mut pt, value, F::zero(), &mut encoding.borrow())?;
                Ok(pt)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    // Polynomial evaluation has no query of its own: it runs on exactly the shared bound.
    let coeff_prec = ckks_spec(params.n, params.base2k, LOG_DELTA, 10);
    let bytes = module.ckks_all_ops_tmp_bytes(&setup.x, &coeff_prec, &params.tsk_layout());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_eval_poly_real_const_coeffs(&mut setup.out, &setup.x, &bsgs, &setup.tsk, s)
    })?;
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &setup.sk, &setup.out, len);
    let want: Vec<F> = setup.xs.iter().map(|&x| poly.evaluate(x)).collect();
    let shaped = precision((&got.0, &got.1), (&want, &vec![F::zero(); len]));
    assert!(
        shaped >= full + 15.0,
        "polynomial term by term at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
    );
    Ok(())
}

/// A shaped product looks its tensor key up at the width its relinearization
/// decomposes: the operands' on the one-pass path, and on the tensor path, which
/// a key missing at the operands' width forces, its tensor intermediate,
/// `λ` bits under the result.
pub fn test_replicated_product_key_width<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    use super::dot_product::TensorKeyAt;
    let len = 1 << LOG_K;
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let k = LOG_DELTA * (STEPS + 2);
    let slot_layout = SlotLayout::Replicated(layout());
    let (a_values, b_values) = (unit_batch::<Rg, F>(len, 0), unit_batch::<Rg, F>(len, 1));
    let a = encrypt_batch(&params, module, &sk, slot_layout, (&a_values.0, &a_values.1), LOG_DELTA, k, 1)?;
    let b = encrypt_batch(&params, module, &sk, slot_layout, (&b_values.0, &b_values.1), LOG_DELTA, k, 2)?;
    // Equal operands into a destination as wide: the result keeps `a.log_budget()` bits.
    let lambda = crate::shaping::ShapingProfile::<Rg>::try_new(&layout())?.shaping_depth(a.log_delta());
    let tensor_k = TorusPrecision(a.k().as_usize().max(a.log_budget() + lambda) as u32);
    assert!(tensor_k > a.k(), "the shaped tensor is wider than the operands");
    let want = complex_mul((&a_values.0, &a_values.1), (&b_values.0, &b_values.1));

    // Rank 1 takes the one-pass product with a key at the operands' width; a wider rank always takes the tensor path.
    let accepted = if params.rank == 1 {
        vec![a.k(), tensor_k]
    } else {
        vec![tensor_k]
    };
    for key_k in accepted {
        let key = TensorKeyAt(&tsk, key_k);
        let mut out = alloc_ct(&params, module, k);
        with_scratch::<BE, _>(module.ckks_mul_tmp_bytes(&out, &a, &b, &tsk), |s| {
            module.ckks_mul_into(&mut out, &a, &b, &key, s)
        })?;
        let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &out, len);
        let bits = precision((&got.0, &got.1), (&want.0, &want.1));
        assert!(bits >= 35.0, "product with a key at {}: {bits:.2} bits", key_k.as_u32());
    }

    // A key at neither width is refused at the tensor's, before any write.
    let mut out = alloc_ct(&params, module, k);
    let before = snapshot::<BE, _, Rg>(&out);
    let wrong = TensorKeyAt(&tsk, TorusPrecision(tensor_k.as_u32() + 1));
    let err = with_scratch::<BE, _>(0, |s| module.ckks_mul_into(&mut out, &a, &b, &wrong, s)).unwrap_err();
    assert!(
        matches!(
            err.composition(),
            Some(crate::CKKSCompositionError::MissingRelinearizationKey { op: "ckks_mul_into", k, .. }) if *k == tensor_k.as_u32()
        ),
        "expected MissingRelinearizationKey at {} from ckks_mul_into, got {err}",
        tensor_k.as_u32()
    );
    assert_eq!(snapshot::<BE, _, Rg>(&out), before, "a refused product wrote its destination");
    Ok(())
}

/// Precision after a rotation into a new ciphertext, a rotation in place, a
/// conjugation into a new ciphertext and two in place, against `conj(v[i + 2])`.
fn automorphism_chain<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
{
    let len = 1 << LOG_K;
    let step = match slot_layout {
        SlotLayout::Replicated(layout) => 1i64 << layout.log_r(),
        SlotLayout::Full => 1,
    };
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let mut atks = HashMap::new();
    let gal = module.ckks_galois_element::<Standard>(step);
    atks.insert(gal, gen_atk(params, module, gal, &sk_raw, &mut scratch.borrow()));
    let conj = gen_atk(params, module, -1, &sk_raw, &mut scratch.borrow());

    let values = unit_batch::<Standard, F>(len, 0);
    let k = 3 * LOG_DELTA;
    let ct = encrypt_batch(params, module, &sk, slot_layout, (&values.0, &values.1), LOG_DELTA, k, 1)?;
    let mut rotated = alloc_ct(params, module, k);
    // Each call runs on exactly its own query's bytes.
    // The queries take the source, whose layout the output gets.
    let rotate_bytes = module.ckks_rotate_tmp_bytes(&ct, &atks[&gal]);
    with_scratch::<BE, _>(rotate_bytes, |s| module.ckks_rotate_into(step, &mut rotated, &ct, &atks, s))?;
    with_scratch::<BE, _>(rotate_bytes, |s| module.ckks_rotate_assign(step, &mut rotated, &atks, s))?;
    let mut out = alloc_ct(params, module, k);
    let conjugate_bytes = module.ckks_conjugate_tmp_bytes(&rotated, &conj);
    with_scratch::<BE, _>(conjugate_bytes, |s| module.ckks_conjugate_into(&mut out, &rotated, &conj, s))?;
    with_scratch::<BE, _>(conjugate_bytes, |s| module.ckks_conjugate_assign(&mut out, &conj, s))?;
    with_scratch::<BE, _>(conjugate_bytes, |s| module.ckks_conjugate_assign(&mut out, &conj, s))?;
    assert_eq!(out.slot_layout(), slot_layout);

    let want_re: Vec<F> = (0..len).map(|i| values.0[(i + 2) % len]).collect();
    let want_im: Vec<F> = (0..len).map(|i| -values.1[(i + 2) % len]).collect();
    let got = decrypt_batch::<BE, F, Standard>(params, module, &sk, &out, len);
    Ok(precision((&got.0, &got.1), (&want_re, &want_im)))
}

pub fn test_replicated_automorphisms_are_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let full = automorphism_chain::<BE, F>(&params, module, host_module, SlotLayout::Full)?;
    let shaped = automorphism_chain::<BE, F>(&params, module, host_module, SlotLayout::Replicated(layout()))?;
    assert!(
        shaped >= full + 15.0,
        "rotations and conjugation at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
    );
    Ok(())
}

/// Precision after a rotation into a new ciphertext and a rotation in place,
/// against `v[i + 2]`.
fn rotation_chain<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let len = 1 << LOG_K;
    let step = match slot_layout {
        SlotLayout::Replicated(layout) => 1i64 << layout.log_r(),
        SlotLayout::Full => 1,
    };
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let gal = module.ckks_galois_element::<Rg>(step);
    let atks = HashMap::from([(gal, gen_atk(params, module, gal, &sk_raw, &mut scratch.borrow()))]);

    let values = unit_batch::<Rg, F>(len, 0);
    let k = 3 * LOG_DELTA;
    let ct = encrypt_batch(params, module, &sk, slot_layout, (&values.0, &values.1), LOG_DELTA, k, 1)?;
    let mut out = alloc_ct(params, module, k);
    // Each call runs on exactly its own query's bytes.
    // The query takes the source, whose layout the output gets.
    let bytes = module.ckks_rotate_tmp_bytes(&ct, &atks[&gal]);
    with_scratch::<BE, _>(bytes, |s| module.ckks_rotate_into(step, &mut out, &ct, &atks, s))?;
    with_scratch::<BE, _>(bytes, |s| module.ckks_rotate_assign(step, &mut out, &atks, s))?;
    assert_eq!(out.slot_layout(), slot_layout);

    let want_re: Vec<F> = (0..len).map(|i| values.0[(i + 2) % len]).collect();
    let want_im: Vec<F> = (0..len).map(|i| values.1[(i + 2) % len]).collect();
    let got = decrypt_batch::<BE, F, Rg>(params, module, &sk, &out, len);
    Ok(precision((&got.0, &got.1), (&want_re, &want_im)))
}

/// The rotations of [`test_replicated_automorphisms_are_shaped`], on any ring.
pub fn test_replicated_rotations_are_shaped<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let full = rotation_chain::<BE, F, Rg>(&params, module, host_module, SlotLayout::Full)?;
    let shaped = rotation_chain::<BE, F, Rg>(&params, module, host_module, SlotLayout::Replicated(layout()))?;
    assert!(
        shaped >= full + 15.0,
        "rotations at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
    );
    Ok(())
}

pub fn test_replicated_rotations_are_admissible<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let mut atks = HashMap::new();
    for r in [1, 1i64 << layout().log_r()] {
        let gal = module.ckks_galois_element::<Rg>(r);
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }
    let values = unit_batch::<Rg, F>(1 << LOG_K, 0);
    let mut ct = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        3 * LOG_DELTA,
        1,
    )?;
    let mut out = alloc_ct(&params, module, 3 * LOG_DELTA);
    // The refusal comes before the scratch check.
    refused!(
        Rg,
        "ckks_rotate_into",
        with_scratch::<BE, _>(0, |s| module.ckks_rotate_into(1, &mut out, &ct, &atks, s)),
        [out, ct]
    );
    refused!(
        Rg,
        "ckks_rotate_assign",
        with_scratch::<BE, _>(0, |s| module.ckks_rotate_assign(-1, &mut ct, &atks, s)),
        [ct]
    );
    let step = 1i64 << layout().log_r();
    let bytes = module.ckks_rotate_tmp_bytes(&ct, &atks[&module.ckks_galois_element::<Rg>(step)]);
    with_scratch::<BE, _>(bytes, |s| module.ckks_rotate_into(step, &mut out, &ct, &atks, s))?;
    Ok(())
}

pub fn test_replicated_masked_constants<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
    F: TestScalar,
{
    let len = 1usize << LOG_K;
    let slots = len << layout().log_r();
    let c_im = if real_slots::<Rg>() { 0.0 } else { -0.2 };
    let (c_re, c_im) = (F::from_f64(0.3).unwrap(), F::from_f64(c_im).unwrap());
    let mut cst = plaintext(&params, module, SlotLayout::Replicated(layout()), len);
    // Each call runs on exactly its own query's bytes.
    let encode_bytes = CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(cst.n().as_usize());
    with_scratch::<BE, _>(encode_bytes, |s| {
        module.ckks_encode_masked_constant_host_into(&mut cst, c_re, c_im, s)
    })?;

    let (mut re, mut im) = (vec![F::zero(); slots], vec![F::zero(); slots]);
    with_scratch::<BE, _>(module.ckks_encoding_host_tmp_bytes(slots), |s| {
        module.ckks_decode_all_slots_host_into(&cst, &mut re, &mut im, s)
    })?;
    // Data slots hold the constant; guards hold only the shaped quantization
    // error, amplified by the guard classes of `g`.
    for t in 0..slots {
        let data = t % (1 << layout().log_r()) == 0;
        let want = if data { (c_re, c_im) } else { (F::zero(), F::zero()) };
        let err = (re[t] - want.0).abs().max((im[t] - want.1).abs()).to_f64().unwrap();
        let bound = if data { -40 } else { 8 - LOG_DELTA as i32 };
        assert!(err < 2f64.powi(bound), "slot {t}: error 2^{:.2}", err.log2());
    }

    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let values = unit_batch::<Rg, F>(len, 0);
    let mut ct = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        LOG_DELTA,
        3 * LOG_DELTA,
        1,
    )?;
    with_scratch::<BE, _>(module.ckks_add_tmp_bytes(ct.max_size()), |s| {
        module.ckks_add_assign(&mut ct, &cst, s)
    })?;
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &ct, len);
    let want_re: Vec<F> = values.0.iter().map(|&x| x + c_re).collect();
    let want_im: Vec<F> = values.1.iter().map(|&x| x + c_im).collect();
    let bits = precision((&got.0, &got.1), (&want_re, &want_im));
    assert!(bits >= 40.0, "masked constant sum: {bits:.2} bits");

    // The masked one, added twice and subtracted once.
    let (add_one, sub_one) = (
        module.ckks_add_one_tmp_bytes(ct.max_size()),
        module.ckks_sub_one_tmp_bytes(ct.max_size()),
    );
    with_scratch::<BE, _>(add_one, |s| module.ckks_add_one_assign(&mut ct, s))?;
    with_scratch::<BE, _>(add_one, |s| module.ckks_add_one_assign(&mut ct, s))?;
    with_scratch::<BE, _>(sub_one, |s| module.ckks_sub_one_assign(&mut ct, s))?;
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &ct, len);
    let want_re: Vec<F> = want_re.iter().map(|&x| x + F::one()).collect();
    let bits = precision((&got.0, &got.1), (&want_re, &want_im));
    assert!(bits >= 40.0, "masked one: {bits:.2} bits");

    // The masked one of a period whose `2R` coefficients lie below the
    // backend's smallest plaintext.
    let small = ReplicatedLayout::new(LOG_K, 1, &[1, -1, 0, 1])?;
    let mut ct = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(small),
        (&values.0, &values.1),
        LOG_DELTA,
        3 * LOG_DELTA,
        5,
    )?;
    with_scratch::<BE, _>(module.ckks_add_one_tmp_bytes(ct.max_size()), |s| {
        module.ckks_add_one_assign(&mut ct, s)
    })?;
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &ct, len);
    let want_re: Vec<F> = values.0.iter().map(|&x| x + F::one()).collect();
    let bits = precision((&got.0, &got.1), (&want_re, &values.1));
    // `1 - Y + Y^3` barely suppresses, so the precision stays near the
    // scale's, about half a bit lower on the conjugate-invariant ring, whose
    // products have twice the variance.
    assert!(bits >= LOG_DELTA as f64 - 5.0, "masked one of period 2: {bits:.2} bits");

    // A replicated plaintext of half a period is refused before any write.
    const ENCODE: &str = "ckks_encode_masked_constant_host_into";
    let (base2k, k) = (Base2K(params.base2k as u32), TorusPrecision((LOG_DELTA + 30) as u32));
    let r = 1usize << layout().log_r();
    let period = if real_slots::<Rg>() { r } else { 2 * r };
    let half_period = r / 2;
    let short_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: half_period.into(),
            base2k,
            k,
            rank: 0usize.into(),
        },
        meta: cst.meta(),
    };
    let mut short = fixture_plaintext::<BE, Rg>(module, &short_layout, 11);
    refused!(
        Rg,
        ENCODE,
        with_scratch::<BE, _>(0, |s| {
            module.ckks_encode_masked_constant_host_into(&mut short, c_re, c_im, s)
        }),
        [short],
        operand(
            ENCODE,
            format!("a masked constant of period {r} needs a degree multiple of {period}, got {half_period}")
        )
    );

    // On the full layout, the constant fills every slot of a plaintext of degree 2 and above.
    for d in [2usize, 4] {
        let mut full: CKKSPlaintextOwned<BE, Rg> = module.ckks_plaintext_alloc(d.into(), base2k, k);
        full.set_meta(GLWEPlaintextMeta {
            slot_layout: SlotLayout::Full,
            ..cst.meta()
        });
        with_scratch::<BE, _>(CKKSEncodingBuffer::<BE::OwnedBuf, F>::bytes_of(d), |s| {
            module.ckks_encode_masked_constant_host_into(&mut full, c_re, c_im, s)
        })?;
        let slots = module.ckks_slots_of_degree::<Rg>(d);
        let (mut re, mut im) = (vec![F::zero(); slots], vec![F::zero(); slots]);
        with_scratch::<BE, _>(module.ckks_encoding_host_tmp_bytes(slots), |s| {
            module.ckks_decode_all_slots_host_into(&full, &mut re, &mut im, s)
        })?;
        for t in 0..slots {
            let err = (re[t] - c_re).abs().max((im[t] - c_im).abs()).to_f64().unwrap();
            // Quantized plainly, at the scale.
            assert!(
                err < 2f64.powi(1 - LOG_DELTA as i32),
                "full layout, degree {d}, slot {t}: error 2^{:.2}",
                err.log2()
            );
        }
    }
    Ok(())
}

/// CKKS to the replicated layout and back keeps the input's precision, with the maps dense and factored into
/// placements of two-bit groups, at the module's degree and at half of it; one value maps onto the layout with no key,
/// on exactly its query and refused one byte below.
pub fn test_ckks_to_and_from_replicated<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, Rg> + ReplicatedConversionModule<BE, F, Rg> + CKKSReplicatedOps<BE, Rg>,
    F: TestScalar,
{
    let half = CKKSTestParams {
        n: params.n / 2,
        hw: params.hw.min(params.n / 4),
        ..params
    };
    for placement_factors in [None, Some(2)] {
        conversions_round_trip_at::<BE, F, Rg>(params, module, host_module, placement_factors)?;
        conversions_round_trip_at::<BE, F, Rg>(half, module, host_module, placement_factors)?;
    }
    // One value: its map onto the layout switches no key, and the arena it checks is sized from none.
    let base2k = Base2K(params.base2k as u32);
    let single = ShapingUnit::U16_40.layout(0)?;
    let (conv, (log_delta, .., extra_bits), k) =
        round_trip_conversion::<BE, F, Rg>(module, (params.n, single, base2k), (40, 20, 30), None)?;
    assert!(conv.galois_elements().is_empty(), "the map of one value rotates nothing");
    let (_, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let values = unit_batch::<Rg, F>(1, 0);
    let ckks = encrypt_batch(&params, module, &sk, SlotLayout::Full, (&values.0, &values.1), 40, k, 9)?;
    let keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE, Rg>> = HashMap::new();
    let mut res = alloc_ct(&params, module, k);
    let want = converted_meta(&conv, true, &ckks, k, (log_delta, extra_bits));
    let bytes = module.ckks_replicated_conversion_tmp_bytes(&res, &ckks, &conv, &params.atk_layout());
    refused!(
        Rg,
        "ckks_to_replicated",
        with_scratch::<BE, _>(bytes - 1, |s| module.ckks_to_replicated(&mut res, &ckks, &conv, &keys, s)),
        [res],
        message = "ckks_to_replicated needs a scratch arena"
    );
    with_scratch::<BE, _>(bytes, |s| module.ckks_to_replicated(&mut res, &ckks, &conv, &keys, s))?;
    assert_eq!(res.slot_layout(), SlotLayout::Replicated(single));
    assert_eq!(stamped(&res), want);
    let fresh = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &ckks, 1);
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &res, 1);
    let fresh_bits = precision((&fresh.0, &fresh.1), (&values.0, &values.1));
    let bits = precision((&got.0, &got.1), (&values.0, &values.1));
    // The map of one value is its diagonal alone: at most 0.44 bits under the fresh encryption measured on the portable
    // backends, both rings, both scalars and rank 2: 0.56 bits of margin.
    assert!(
        bits >= fresh_bits - 1.0,
        "one value onto the layout: {bits:.2} bits from {fresh_bits:.2}"
    );
    Ok(())
}

/// The replicated scale, the scale of the diagonals and of the inner ones, and the extra bits of a conversion.
type ConversionScales = (usize, usize, usize, usize);

/// The conversions of `layout` for ciphertexts of degree `n`, at the scales `(log_delta, log_delta_diag,
/// inner_log_delta_diag, extra_bits)`: dense, or factored into `placement_factors` placement factors.
pub(crate) fn conversion<BE, F, Rg: Ring>(
    module: &Module<BE>,
    (n, layout, base2k): (usize, ReplicatedLayout, Base2K),
    (log_delta, log_delta_diag, inner_log_delta_diag, extra_bits): ConversionScales,
    placement_factors: Option<usize>,
    scratch: &mut ScratchArena<'_, BE>,
) -> anyhow::Result<ReplicatedConversionPrepared<BE, Rg>>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: ReplicatedConversionModule<BE, F, Rg>,
{
    match placement_factors {
        None => ReplicatedConversionPrepared::new::<F>(
            module,
            n.into(),
            layout,
            base2k,
            log_delta,
            log_delta_diag,
            extra_bits,
            scratch,
        ),
        Some(factors) => ReplicatedConversionPrepared::new_factored::<F>(
            module,
            n.into(),
            layout,
            base2k,
            log_delta,
            log_delta_diag,
            inner_log_delta_diag,
            extra_bits,
            factors,
            true,
            scratch,
        ),
    }
}

/// The conversion a round trip of `layout` takes on ciphertexts of degree `n`: the CKKS side at `2^log_delta_ckks`, the
/// replicated one at `2^log_delta`, the diagonals at `2^log_delta_diag` and the inner factors of `to` quantized plainly,
/// a few bits above the working precision, built on exactly the arena of its query. With its scales, the last its
/// extra bits, and the width that holds both conversions with a margin of one diagonal scale.
pub(crate) fn round_trip_conversion<BE, F, Rg: Ring>(
    module: &Module<BE>,
    (n, layout, base2k): (usize, ReplicatedLayout, Base2K),
    (log_delta_ckks, log_delta, log_delta_diag): (usize, usize, usize),
    placement_factors: Option<usize>,
) -> anyhow::Result<(ReplicatedConversionPrepared<BE, Rg>, ConversionScales, usize)>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: ReplicatedConversionModule<BE, F, Rg> + CKKSModuleInfos,
{
    let extra_bits =
        log_delta_ckks - log_delta - (module.ckks_slots_of_degree::<Rg>(n) >> layout.log_k()).trailing_zeros() as usize;
    let inner_diag = log_delta + extra_bits + 4;
    let bytes =
        ReplicatedConversionPrepared::<BE, Rg>::new_tmp_bytes::<F>(module, &layout, base2k, log_delta_diag.max(inner_diag))?;
    let scales = (log_delta, log_delta_diag, inner_diag, extra_bits);
    let conv = with_scratch::<BE, _>(bytes, |s| {
        conversion::<BE, F, Rg>(module, (n, layout, base2k), scales, placement_factors, s)
    })?;
    let from_bits = conv.from.as_ref().map_or(0, |from| from.consumed);
    let k = log_delta_ckks + conv.to.consumed + from_bits + log_delta_diag;
    Ok((conv, scales, k))
}

/// The `(k, log_delta)` and `(slots, log_sparsity)` a conversion through `conv`, at the replicated scale `log_delta` and
/// `extra_bits`, stamps from `a` on a destination of `res_k` bits. Onto the layout when `to`: the map consumes its
/// diagonals' bits and lands at the replicated scale, keeping `a`'s slots on the sparsity of the period `kR`. Back
/// otherwise: the gather consumes its diagonals' bits, and the trace moves its `L` levels under the scale, `extra_bits`
/// above `a`'s, closing the slot cycle at them.
fn converted_meta<BE: Backend, Rg: Ring>(
    conv: &ReplicatedConversionPrepared<BE, Rg>,
    to: bool,
    a: &impl CKKSInfos,
    res_k: usize,
    (log_delta, extra_bits): (usize, usize),
) -> ((usize, usize), (SlotsKind, usize)) {
    let trace = conv.trace_levels;
    if to {
        let k = (a.log_budget() - conv.to.consumed + log_delta).min(res_k);
        return ((k, log_delta), (a.slots(), trace - conv.layout.log_r()));
    }
    let consumed = conv.from.as_ref().map_or(0, |from| from.consumed);
    let log_delta = a.log_delta() + extra_bits + trace;
    let k = (a.log_budget() - consumed - trace + log_delta).min(res_k);
    ((k, log_delta), (a.slots(), trace))
}

/// The `(k, log_delta)` and `(slots, log_sparsity)` of `ct`.
fn stamped(ct: &impl CKKSInfos) -> ((usize, usize), (SlotsKind, usize)) {
    ((ct.k().as_usize(), ct.log_delta()), (ct.slots(), ct.log_sparsity()))
}

/// The conversions round trip on ciphertexts of degree `params.n`, and at the module's degree their refusals.
fn conversions_round_trip_at<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    placement_factors: Option<usize>,
) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, Rg> + ReplicatedConversionModule<BE, F, Rg> + CKKSReplicatedOps<BE, Rg>,
    F: TestScalar,
{
    // `R = 8` divides `k = 8`. The return lands on `20 + extra + log2(S/k)`
    // for the `S` slots of the ring.
    let (log_k, log_delta_ckks, log_delta, log_delta_diag) = (3u8, 40, 20, 30);
    let layout = ShapingUnit::U16_40.layout(log_k)?;
    let len = 1usize << log_k;
    let base2k = Base2K(params.base2k as u32);
    let shape = (params.n, layout, base2k);
    let (conv, scales, k) =
        round_trip_conversion::<BE, F, Rg>(module, shape, (log_delta_ckks, log_delta, log_delta_diag), placement_factors)?;
    let (extra_bits, from_bits_consumed) = (scales.3, conv.from.as_ref().map_or(0, |from| from.consumed));
    // Keys covering both conversions.
    let params = CKKSTestParams {
        k: params.k.max(k + extra_bits + conv.trace_levels),
        ..params
    };

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch::<BE, Rg>(&params, module);
    let mut atks = HashMap::new();
    for gal in conv.galois_elements() {
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }

    let values = unit_batch::<Rg, F>(len, 0);
    let ckks = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Full,
        (&values.0, &values.1),
        log_delta_ckks,
        k,
        9,
    )?;
    let fresh = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &ckks, len);
    let fresh_bits = precision((&fresh.0, &fresh.1), (&values.0, &values.1));

    // `ckks_to_replicated` when `$to`, else `ckks_from_replicated`, through `conv` on `$bytes` of scratch.
    macro_rules! convert {
        ($to:expr, $res:expr, $a:expr, $keys:expr, $bytes:expr) => {
            with_scratch::<BE, _>($bytes, |s| {
                if $to {
                    module.ckks_to_replicated($res, $a, &conv, $keys, s)
                } else {
                    module.ckks_from_replicated($res, $a, &conv, $keys, s)
                }
            })
        };
    }
    let atk_infos = params.atk_layout();
    let mut replicated = alloc_ct(&params, module, k);
    let conv_bytes = module.ckks_replicated_conversion_tmp_bytes(&replicated, &ckks, &conv, &atk_infos);
    let conversion_scales = (log_delta, extra_bits);
    let to_meta = converted_meta(&conv, true, &ckks, replicated.k().as_usize(), conversion_scales);
    let logged = QueryLog::new(&atks);
    // The entry check looks every key of `chain` up at its widest use on a working copy of `start` bits.
    let lookups = |label, chain: &ReplicatedChain<BE, Rg>, start: usize| {
        let entry: Vec<_> = chain.keys.iter().map(|&(p, below)| (p, start - below)).collect();
        let seen: Vec<_> = logged.take().into_iter().map(|(p, k)| (p, k.as_usize())).collect();
        assert_entry_then_widest(label, &seen, &entry);
    };
    convert!(true, &mut replicated, &ckks, &logged, conv_bytes)?;
    lookups("to", &conv.to, ckks.k().as_usize() + extra_bits);
    assert_eq!(replicated.slot_layout(), SlotLayout::Replicated(layout));
    assert_eq!(stamped(&replicated), to_meta);
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &replicated, len);
    let to_bits = precision((&got.0, &got.1), (&values.0, &values.1));
    // The map runs `extra_bits` under the replicated scale, which bounds the
    // result as the fresh encryption does; half the module's degree, one more
    // working bit, measures 2.4 bits under that bound.
    let working_bits = (log_delta + extra_bits) as f64;
    assert!(
        to_bits >= fresh_bits.min(working_bits) - 3.0,
        "to replicated: {to_bits:.2} bits from {fresh_bits:.2}, working at {working_bits}"
    );

    let mut back = alloc_ct(&params, module, k);
    let from_meta = converted_meta(&conv, false, &replicated, back.k().as_usize(), conversion_scales);
    let ((from_k, from_log_delta), _) = from_meta;
    let back_bytes = module.ckks_replicated_conversion_tmp_bytes(&back, &replicated, &conv, &atk_infos);
    convert!(false, &mut back, &replicated, &logged, back_bytes)?;
    let from_start = replicated.k().as_usize() + extra_bits;
    lookups("from", conv.from.as_ref().unwrap(), from_start);
    assert_eq!(back.slot_layout(), SlotLayout::Full);
    assert_eq!(stamped(&back), from_meta);
    assert_eq!(back.log_delta(), log_delta_ckks);
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &back, len);
    let from_bits = precision((&got.0, &got.1), (&values.0, &values.1));
    // Plain gather diagonals bound the return near `log_delta_diag` bits: 26.63 measured at least (CI ring, n = 256, factored).
    assert!(
        from_bits >= fresh_bits.min(log_delta_diag as f64) - 4.0,
        "from replicated: {from_bits:.2} bits from {fresh_bits:.2}"
    );

    // Destinations narrower than the results, off the limb grid, take them through a modulus drop at the same scale
    // and precision.
    let (to, from) = ("ckks_to_replicated", "ckks_from_replicated");
    let op = |to_dir: bool| if to_dir { to } else { from };
    let source = |to_dir: bool| if to_dir { &ckks } else { &replicated };
    let narrow_from_k = from_log_delta + (from_k - from_log_delta) / 2;
    for (to_dir, res_k, bits) in [(true, to_meta.0.0 - 7, to_bits), (false, narrow_from_k, from_bits)] {
        let mut narrow = alloc_ct(&params, module, res_k);
        let want = converted_meta(&conv, to_dir, source(to_dir), res_k, conversion_scales);
        assert_eq!(want.0.0, res_k, "a narrower destination bounds the result");
        let bytes = module.ckks_replicated_conversion_tmp_bytes(&narrow, source(to_dir), &conv, &atk_infos);
        convert!(to_dir, &mut narrow, source(to_dir), &atks, bytes)?;
        assert_eq!(stamped(&narrow), want);
        let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &narrow, len);
        let narrow_bits = precision((&got.0, &got.1), (&values.0, &values.1));
        // A gap of 0.00 bits measured on the portable and oracle backends, both rings, both scalars: 0.5 bits of margin.
        assert!(
            (narrow_bits - bits).abs() < 0.5,
            "{}, narrow: {narrow_bits:.2} bits against {bits:.2}",
            op(to_dir)
        );
    }

    if params.n != module.n() {
        return Ok(());
    }
    use crate::CKKSCompositionError as E;
    let dirty = |n: usize, rank: usize, seed: u8| {
        let glwe_layout = GLWELayout {
            n: n.into(),
            rank: rank.into(),
            ..ckks.glwe_layout()
        };
        fixture_ciphertext::<BE, Rg>(
            module,
            &CKKSLayout {
                glwe_layout,
                meta: ckks.meta(),
            },
            seed,
        )
    };
    let rank = ckks.rank().as_usize();
    for (n, rank, reason) in [
        (params.n, rank + 1, format!("of one rank, got {rank} and {}", rank + 1)),
        (params.n / 2, rank, "of one degree, which the module serves".to_string()),
    ] {
        let mut bad = dirty(n, rank, 31);
        for to_dir in [true, false] {
            refused!(
                Rg,
                op(to_dir),
                convert!(to_dir, &mut bad, source(to_dir), &atks, 0),
                [bad, ckks, replicated],
                operand(op(to_dir), reason.as_str())
            );
        }
    }
    // Operands below the module's degree, which the compiled slots do not fit.
    let (half_src, mut half_dst) = (dirty(params.n / 2, rank, 32), dirty(params.n / 2, rank, 33));
    for to_dir in [true, false] {
        refused!(
            Rg,
            op(to_dir),
            convert!(to_dir, &mut half_dst, &half_src, &atks, 0),
            [half_dst],
            operand(op(to_dir), format!("the degree of its context, got {}", params.n / 2))
        );
    }
    // Conversions at a degree off a power of two or above the module's, of a layout past the slots of its degree, and
    // factored into a placement count off `1..=log2(k)`.
    let slots = module.ckks_slots_of_degree::<Rg>(params.n);
    let past = ShapingUnit::U16_40.layout(((2 * slots) >> layout.log_r()).ilog2() as u8)?;
    let off_degree = |d: usize| format!("the degree {d} is not a power of two at most the module's {}", params.n);
    let factor_count = format!("the placement takes between one and log2(k) = {log_k} factors");
    for (n, layout, factors, reason) in [
        (3 * params.n / 4, layout, placement_factors, off_degree(3 * params.n / 4)),
        (2 * params.n, layout, placement_factors, off_degree(2 * params.n)),
        (
            params.n,
            past,
            placement_factors,
            format!(
                "{} values of period {} need more than the {slots} slots of degree {}",
                1 << past.log_k(),
                1 << past.log_r(),
                params.n
            ),
        ),
        (params.n, layout, Some(0), factor_count.clone()),
        (params.n, layout, Some(log_k as usize + 1), factor_count),
    ] {
        let err = with_scratch::<BE, _>(0, |s| {
            conversion::<BE, F, Rg>(module, (n, layout, base2k), scales, factors, s).err()
        });
        assert_eq!(err.expect("a refused conversion").to_string(), reason);
    }
    // A source of the other layout, at a scale more than the working bits under the replicated scale, or without the
    // budget of the map or of the gather and the trace; a destination narrower than the result's scale, where no
    // modulus drop lands it; and a key set lacking the keys the run looks up.
    let capacity = |op: &'static str| move |e: &E| matches!(e, E::InsufficientHomomorphicCapacity { op: got, .. } if *got == op);
    let missing = |op: &'static str| move |e: &E| matches!(e, E::MissingAutomorphismKey { op: got, .. } if *got == op);
    let mut coarse = ckks.clone();
    coarse.set_log_delta(log_delta - extra_bits - 1);
    let (mut spent_to, mut spent_from) = (ckks.clone(), replicated.clone());
    spent_to.set_log_budget(conv.to.consumed - 1);
    spent_from.set_log_budget(from_bits_consumed + conv.trace_levels - 1);
    let no_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE, Rg>> = HashMap::new();
    for (to_dir, res_k, a, keys, matches) in [
        (true, k, &spent_to, &atks, &capacity(to) as &dyn Fn(&E) -> bool),
        (false, k, &spent_from, &atks, &capacity(from)),
        (true, log_delta - 1, &ckks, &atks, &capacity(to)),
        (false, from_log_delta - 1, &replicated, &atks, &capacity(from)),
        (true, k, &replicated, &atks, &operand(to, "support a replicated slot layout")),
        (false, k, &ckks, &atks, &operand(from, "the conversion's replicated layout")),
        (true, k, &coarse, &atks, &operand(to, "exceeds the working scale")),
        (true, k, &ckks, &no_keys, &missing(to)),
        (false, k, &replicated, &no_keys, &missing(from)),
    ] {
        let mut res = dirty_ct(module, &ckks, res_k);
        refused!(Rg, op(to_dir), convert!(to_dir, &mut res, a, keys, 0), [res], matches);
    }
    // An arena a byte below the query of each direction.
    let mut out = dirty_ct(module, &ckks, k);
    for to_dir in [true, false] {
        let bytes = module.ckks_replicated_conversion_tmp_bytes(&out, source(to_dir), &conv, &atk_infos);
        let short = format!("{} needs a scratch arena", op(to_dir));
        refused!(
            Rg,
            op(to_dir),
            convert!(to_dir, &mut out, source(to_dir), &atks, bytes - 1),
            [out],
            message = short
        );
    }
    // A period above the batch: the conversion has no gather.
    let wide = ShapingUnit::U16_40.layout(2)?;
    let scales_of_wide = (log_delta_ckks, log_delta, log_delta_diag);
    let (ungathered, ..) = round_trip_conversion::<BE, F, Rg>(module, (params.n, wide, base2k), scales_of_wide, None)?;
    assert!(ungathered.from.is_none());
    let mut other = replicated.clone();
    other.set_slot_layout(SlotLayout::Replicated(wide));
    refused!(
        Rg,
        "from, no gather",
        with_scratch::<BE, _>(0, |s| module.ckks_from_replicated(&mut out, &other, &ungathered, &atks, s)),
        [out],
        operand(from, "the gather needs the period")
    );
    Ok(())
}

/// `(Mv)` for `M(c, j) = (cos(c·j + 0.1)/2, sin(c + 2j)/3)`, real on real slots.
fn matrix<Rg: Ring, F: TestScalar>(c: usize, j: usize) -> (F, F) {
    let value = |x: f64| F::from_f64(x).unwrap();
    let im = if real_slots::<Rg>() { 0.0 } else { 1.0 / 3.0 };
    (
        value(((c * j) as f64 + 0.1).cos() / 2.0),
        value(((c + 2 * j) as f64).sin() * im),
    )
}

fn apply_matrix<Rg: Ring, F: TestScalar>(v: (&[F], &[F])) -> (Vec<F>, Vec<F>) {
    let len = v.0.len();
    let (mut re, mut im) = (vec![F::zero(); len], vec![F::zero(); len]);
    for c in 0..len {
        for j in 0..len {
            let (a, b) = matrix::<Rg, F>(c, j);
            re[c] = re[c] + a * v.0[j] - b * v.1[j];
            im[c] = im[c] + a * v.1[j] + b * v.0[j];
        }
    }
    (re, im)
}

pub fn test_ckks_to_and_from_replicated_apply_matrices<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, Rg> + ReplicatedConversionModule<BE, F, Rg> + CKKSReplicatedOps<BE, Rg>,
    F: TestScalar,
{
    let log_k = 3u8;
    let (len, base2k) = (1usize << log_k, Base2K(params.base2k as u32));
    let layout = ShapingUnit::U16_40.layout(log_k)?;
    let (log_delta_diag, extra_bits) = (36, 16);
    let n = params.n.into();
    let bytes = ReplicatedConversionPrepared::<BE, Rg>::new_tmp_bytes::<F>(module, &layout, base2k, log_delta_diag)?;
    let mut setup = ScratchOwned::<BE>::alloc(bytes);
    let conversion = |to, from, scratch: &mut ScratchArena<'_, BE>| {
        ReplicatedConversionPrepared::<BE, Rg>::from_factors(module, n, layout, base2k, LOG_DELTA, extra_bits, to, from, scratch)
    };
    if real_slots::<Rg>() {
        let imaginary = |_: usize, _: usize| (F::zero(), F::one());
        let to = vec![period_factor(layout, len, 14, imaginary)?];
        let err = conversion(to, None, &mut setup.borrow())
            .err()
            .expect("complex diagonals on real slots");
        assert_ckks_error(
            "complex diagonals on real slots",
            &err.into(),
            crate::CKKSCompositionError::InvalidOperand {
                op: "ckks_linear_transformation_encode_from_diagonals",
                reason: "complex linear-transformation diagonals cannot be encoded into real slots".into(),
            },
        );
    }
    // Shaped diagonals far coarser than the result: each BSGS giant step needs
    // its own rotated shaping polynomial to keep their error off the data slots.
    let to = period_factor(layout, len, 14, |s, j| {
        if s % (1 << layout.log_r()) == 0 {
            matrix::<Rg, F>((s >> layout.log_r()) % len, j)
        } else {
            (F::zero(), F::zero())
        }
    })?;
    let gather = gather_factor(layout, log_delta_diag, matrix::<Rg, F>)?;
    let conv = conversion(vec![to], Some(vec![gather]), &mut setup.borrow())?;

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch::<BE, Rg>(&params, module);
    let mut atks = HashMap::new();
    for gal in conv.galois_elements() {
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }
    let values = unit_batch::<Rg, F>(len, 0);
    let want = apply_matrix::<Rg, F>((&values.0, &values.1));
    let k = 40 + 2 * log_delta_diag;
    let ckks = encrypt_batch(&params, module, &sk, SlotLayout::Full, (&values.0, &values.1), 40, k, 9)?;
    let atk_infos = params.atk_layout();
    let mut onto = alloc_ct(&params, module, k);
    let bytes = module.ckks_replicated_conversion_tmp_bytes(&onto, &ckks, &conv, &atk_infos);
    with_scratch::<BE, _>(bytes, |s| module.ckks_to_replicated(&mut onto, &ckks, &conv, &atks, s))?;
    assert_eq!(onto.slot_layout(), SlotLayout::Replicated(layout));
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &onto, len);
    let onto_bits = precision((&got.0, &got.1), (&want.0, &want.1));
    // 33.96 bits measured at least on the portable, oracle and rayon backends, both rings: 3.96 bits of margin.
    assert!(onto_bits >= 30.0, "period-k map onto the data slots: {onto_bits:.2} bits");

    let slot_layout = SlotLayout::Replicated(layout);
    let replicated = encrypt_batch(&params, module, &sk, slot_layout, (&values.0, &values.1), LOG_DELTA, k, 3)?;
    let mut gathered = alloc_ct(&params, module, k);
    let bytes = module.ckks_replicated_conversion_tmp_bytes(&gathered, &replicated, &conv, &atk_infos);
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_from_replicated(&mut gathered, &replicated, &conv, &atks, s)
    })?;
    assert_eq!(gathered.slot_layout(), SlotLayout::Full);
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &gathered, len);
    let gather_bits = precision((&got.0, &got.1), (&want.0, &want.1));
    // Plainly quantized diagonals weighted `R·M(c, j)` over `k` terms: 28.53 bits measured at least on the portable,
    // oracle and rayon backends, both rings: 2.53 bits of margin.
    assert!(gather_bits >= 26.0, "gather and trace: {gather_bits:.2} bits");
    Ok(())
}

fn eval_mod_plan() -> EvalModPlan {
    EvalModPlan {
        eval_mod_type: EvalModType::CosHK,
        log_msg_ratio: 13,
        f_mod_degree: 30,
        f_mod_interval: 16,
        f_mod_log_interval_reduction: 3,
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: CoeffsMeta::from_delta_budget(26, 20),
        f_mod_log_delta: 28,
        packed: false,
    }
}

/// An EvalMod run: its plan and the scale of its input.
#[derive(Clone, Copy)]
struct EvalModCase {
    plan: EvalModPlan,
    log_delta_in: usize,
}

impl EvalModCase {
    /// The input above the plan scale, so that the entry rounds it with shaping into a carved copy, and the last
    /// squaring returns to the input scale.
    fn rounded() -> Self {
        Self {
            plan: eval_mod_plan(),
            log_delta_in: 36,
        }
    }

    /// The input below the plan scale, in an allocation holding the working width, so that EvalMod runs on it in
    /// place, and an inverse after the range extension, so that no squaring is fused and the output returns to the
    /// input scale with a shaped rounding of its own.
    fn inverse() -> Self {
        Self {
            plan: EvalModPlan {
                f_mod_inv_degree: Some(3),
                ..eval_mod_plan()
            },
            log_delta_in: 24,
        }
    }

    /// The complex exponential, whose last complex squaring is fused.
    fn complex() -> Self {
        Self {
            plan: EvalModPlan {
                eval_mod_type: EvalModType::ExpCmplx,
                ..eval_mod_plan()
            },
            log_delta_in: 36,
        }
    }
}

/// The EvalMod of [`eval_mod_precision`].
#[derive(Clone, Copy)]
enum EvalModMode {
    Full,
    Replicated,
    /// Two batches side by side, the second returned times `i`.
    Packed,
}

/// EvalMod of `x = I + m/2^13` at `t = x/K`, against the circuit's own
/// polynomials, in message units.
fn eval_mod_precision<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    case: EvalModCase,
    mode: EvalModMode,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSEvalModOps<BE, Rg>,
{
    let EvalModCase { plan, log_delta_in } = case;
    let ratio = (1u64 << plan.log_msg_ratio) as f64;
    let interval = plan.f_mod_interval as f64;
    let dsize = 2;
    // The tensor key's gadget covers the working width at the plan scale, which the allocation of the input holds.
    let test_params = CKKSTestParams {
        k: (plan.consumed_bits() + log_delta_in + 2 * params.base2k).next_multiple_of(dsize * params.base2k) + log_delta_in
            - plan.f_mod_log_delta,
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta_in),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        },
        prec_log_budget: 10,
        dsize,
        ..*params
    };
    let k = test_params.k;
    let (slot_layout, len) = match mode {
        EvalModMode::Full => (SlotLayout::Full, 1 << LOG_K),
        EvalModMode::Replicated => (SlotLayout::Replicated(layout()), 1 << LOG_K),
        EvalModMode::Packed => (SlotLayout::Replicated(ShapingUnit::U32_45.layout(LOG_K + 1)?), 2 << LOG_K),
    };
    let xs: Vec<f64> = [
        (15.0, 0.9),
        (-15.0, -0.95),
        (3.0, 0.3),
        (-7.0, -0.6),
        (12.0, -0.4),
        (-1.0, 0.8),
        (0.0, 0.05),
        (-11.0, 0.7),
    ]
    .into_iter()
    .take(len)
    .map(|(integer, message)| (integer + message / ratio) / interval)
    .collect();
    let re: Vec<F> = xs.iter().map(|&x| F::from_f64(x).unwrap()).collect();
    let zero = vec![F::zero(); len];

    let (sk_raw, sk) = gen_sk_with_raw(&test_params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&test_params, module);
    let tsk = gen_tsk(&test_params, module, &sk_raw, &mut scratch.borrow());
    let mut pt = module.ckks_plaintext_alloc_from_slots(
        slots_of::<BE, Rg>(module, slot_layout, len),
        Base2K(params.base2k as u32),
        TorusPrecision((log_delta_in + 10) as u32),
    );
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(log_delta_in),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout,
    });
    encoded(module, &mut pt, &re, &zero)?;
    let mut ct = alloc_ct(&test_params, module, k);
    let (mut xe, mut xa) = (Source::new([1; 32]), Source::new([2; 32]));
    with_scratch::<BE, _>(module.ckks_encrypt_sk_tmp_bytes(&ct), |s| {
        module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, s)
    })?;

    let mut compile = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(params.n));
    let base2k = Base2K(params.base2k as u32);
    let mut res = alloc_ct(&test_params, module, k);
    let tsk_layout = test_params.tsk_layout();
    // The evaluation runs on exactly its query's bytes.
    let packed = matches!(mode, EvalModMode::Packed);
    let eval_mod = ckks_eval_mod_compile::<BE, F, Rg>(
        base2k,
        plan.with_slot_layout(slot_layout).with_packed(packed),
        module,
        &mut compile.borrow(),
    )?;
    with_scratch::<BE, _>(module.ckks_eval_mod_tmp_bytes(&res, &ct, &eval_mod, &tsk_layout), |s| {
        module.ckks_eval_mod(&mut res, &mut ct, &eval_mod, &tsk, s)
    })?;
    let scale = eval_mod.range_extension_scale();
    let (f_mod_poly, f_mod_inv_poly) = (eval_mod.f_mod_poly, eval_mod.f_mod_inv_poly);
    assert_eq!(res.slot_layout(), slot_layout);
    assert_eq!(res.log_delta(), log_delta_in);

    let steps = plan.f_mod_log_interval_reduction;
    let two = F::one() + F::one();
    let to_message = F::from_f64(ratio).unwrap();
    let want: Vec<(F, F)> = match &f_mod_poly {
        EvalModPoly::Real(poly) => re
            .iter()
            .map(|&t| {
                let mut p = poly.evaluate(t);
                for i in 0..steps {
                    p = two * p * p - F::from_f64(scale.powi(1 << (i + 1))).unwrap();
                }
                if let Some(inverse) = &f_mod_inv_poly {
                    p = inverse.evaluate(p);
                }
                (p * to_message, F::zero())
            })
            .collect(),
        EvalModPoly::Complex(poly) => re
            .iter()
            .map(|&t| {
                let (mut a, mut b) = poly.evaluate(t);
                for _ in 0..steps {
                    (a, b) = (a * a - b * b, two * a * b);
                }
                (a * to_message, b * to_message)
            })
            .collect(),
    };
    // A packed second half comes out on the imaginary parts.
    let half = if let EvalModMode::Packed = mode { len / 2 } else { len };
    let want_re: Vec<F> = (0..len).map(|j| if j < half { want[j].0 } else { F::zero() }).collect();
    let want_im: Vec<F> = (0..len).map(|j| if j < half { want[j].1 } else { want[j].0 }).collect();
    let got = decrypt_batch::<BE, F, Rg>(&test_params, module, &sk, &res, len);
    let got_re: Vec<F> = got.0.iter().map(|&v| v * to_message).collect();
    let got_im: Vec<F> = got.1.iter().map(|&v| v * to_message).collect();
    Ok(precision((&got_re, &got_im), (&want_re, &want_im)))
}

/// Precisions of `case` on a full layout and on a replicated one.
fn eval_mod_gain<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    case: EvalModCase,
) -> Result<(f64, f64)>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSEvalModOps<BE, Rg>,
{
    Ok((
        eval_mod_precision::<BE, F, Rg>(params, module, host_module, case, EvalModMode::Full)?,
        eval_mod_precision::<BE, F, Rg>(params, module, host_module, case, EvalModMode::Replicated)?,
    ))
}

pub fn test_replicated_eval_mod_is_shaped<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSEvalModOps<BE, Rg>,
    F: TestScalar,
{
    let (full, shaped) = eval_mod_gain::<BE, F, Rg>(&params, module, host_module, EvalModCase::rounded())?;
    assert!(
        shaped >= 30.0 && shaped >= full + 25.0,
        "EvalMod at 2^28: {shaped:.2} bits against {full:.2} for a full layout"
    );
    let (full, shaped) = eval_mod_gain::<BE, F, Rg>(&params, module, host_module, EvalModCase::inverse())?;
    // Measured on the portable and oracle backends, both rings: 35.4 to 37.1 bits against 3.6 to 4.5.
    assert!(
        shaped >= 30.0 && shaped >= full + 25.0,
        "EvalMod with an inverse from 2^24: {shaped:.2} bits against {full:.2} for a full layout"
    );
    // A packed plan off a replicated layout of two values or more, or other than a cosine without inverse.
    let mut compile = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(params.n));
    let mut compiled = |plan: EvalModPlan| {
        ckks_eval_mod_compile::<BE, F, Rg>(Base2K(params.base2k as u32), plan, module, &mut compile.borrow()).err()
    };
    let packed = eval_mod_plan().with_packed(true);
    let one_value = SlotLayout::Replicated(ShapingUnit::U32_45.layout(0)?);
    let inverse = EvalModPlan {
        f_mod_inv_degree: Some(3),
        ..packed.with_slot_layout(SlotLayout::Replicated(layout()))
    };
    for (plan, reason) in [
        (packed, "a packed EvalMod takes a replicated layout"),
        (packed.with_slot_layout(one_value), "a packed EvalMod needs two batches"),
        (inverse, "a packed EvalMod needs a cosine EvalMod without inverse"),
    ] {
        let err = compiled(plan).expect(reason);
        let want = crate::CKKSCompositionError::InvalidPlan {
            plan: "EvalModPlan",
            reason: reason.into(),
        };
        assert_eq!(err.downcast_ref::<crate::CKKSCompositionError>(), Some(&want), "{err}");
    }
    // Real slots have no imaginary parts to pack a second batch on.
    if real_slots::<Rg>() {
        let two_batches = SlotLayout::Replicated(ShapingUnit::U32_45.layout(LOG_K + 1)?);
        let err = compiled(packed.with_slot_layout(two_batches)).expect("a packed EvalMod on real slots");
        let ckks = err.downcast_ref::<crate::CKKSError>().and_then(crate::CKKSError::composition);
        assert!(
            ckks.is_some_and(operand(
                "ckks_eval_mod_compile",
                "a packed EvalMod needs the complex slots of the standard ring"
            )),
            "{err}"
        );
        return Ok(());
    }
    // A packed layout past the module's slots.
    let unit = ShapingUnit::U32_45;
    let log_k = (module.ckks_max_slots::<Rg>().ilog2() as usize + 1 - unit.layout(0)?.log_r()) as u8;
    let err =
        compiled(packed.with_slot_layout(SlotLayout::Replicated(unit.layout(log_k)?))).expect("a packed EvalMod past the slots");
    let ckks = err.downcast_ref::<crate::CKKSError>().and_then(crate::CKKSError::composition);
    assert!(
        ckks.is_some_and(operand("ckks_eval_mod_compile", "exceeds the module's")),
        "{err}"
    );
    let full = eval_mod_precision::<BE, F, Rg>(&params, module, host_module, EvalModCase::rounded(), EvalModMode::Full)?;
    let packed = eval_mod_precision::<BE, F, Rg>(&params, module, host_module, EvalModCase::rounded(), EvalModMode::Packed)?;
    assert!(
        packed >= 30.0 && packed >= full + 25.0,
        "packed EvalMod at 2^28: {packed:.2} bits against {full:.2} for a full layout"
    );
    Ok(())
}

/// The complex EvalMod, whose last complex squaring is fused, keeps the message far above CKKS.
pub fn test_replicated_complex_eval_mod_is_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSEvalModOps<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let (full, shaped) = eval_mod_gain::<BE, F, Standard>(&params, module, host_module, EvalModCase::complex())?;
    // Measured on the portable backends: 35.4 to 37.3 bits against 5.6 to 6.2.
    assert!(
        shaped >= 30.0 && shaped >= full + 25.0,
        "complex EvalMod at 2^28: {shaped:.2} bits against {full:.2} for a full layout"
    );
    Ok(())
}

/// One bootstrapping of `k = 2^log_k` values per row with the parameters of the paper, from the leveled scale, with
/// full-width and row-offset return keys: dense maps of `U16_40` at the module's degree, with every refusal of the
/// bootstrapping and of its steps, at half of it, unpacked, and unpacked on real messages, which take one EvalMod on the
/// real part; the periods `R = 4` and `R = 2`, whose precision the units' suppression bounds, `U4_15` also on `k = 2`
/// values, a subring of degree 4, below the smallest degree of every backend; and `SlotsToCoeffs` and `CoeffsToSlots`
/// factored into two DFT factors each, the placements in groups of two bits.
pub fn test_ckks_bootstrap_replicated<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: ReplicatedBootstrapModule<BE, F>,
    F: TestScalar + DftScalar,
{
    let half = CKKSTestParams {
        n: params.n / 2,
        hw: params.hw.min(params.n / 4),
        ..params
    };
    let (dense, factored) = (false, true);
    let (packed, unpacked, real_messages) = ((true, false), (false, false), (false, true));
    // Least precision on the portable and oracle backends over both key sets, against each row's floor: `U16_40` 23.26
    // bits (4.26 bits of margin), 23.27 at half the degree (4.27), 23.26 unpacked (4.26), 23.33 unpacked on real
    // messages (4.33) and 23.33 factored (4.33), and `U8_40` 23.11 (4.11), the 40-bit units bounded alike by the
    // bootstrapping; `U4_15`, whose 15.26 bits of suppression bound it, 18.21 on 8 values (3.71) and 19.39 on 2 (4.89).
    for (params, host_module, unit, log_k, transforms, (pack, real), refusals) in [
        (params, host_module, ShapingUnit::U16_40, 3, dense, packed, true),
        (half, host_module, ShapingUnit::U16_40, 3, dense, packed, false),
        (params, host_module, ShapingUnit::U16_40, 3, dense, unpacked, false),
        (params, host_module, ShapingUnit::U16_40, 3, dense, real_messages, false),
        (params, host_module, ShapingUnit::U8_40, 3, dense, packed, false),
        (params, host_module, ShapingUnit::U4_15, 3, dense, packed, false),
        (params, host_module, ShapingUnit::U4_15, 1, dense, packed, false),
        (params, host_module, ShapingUnit::U16_40, 3, factored, packed, false),
    ] {
        let (min_bits, min_agreement) = match unit.target_bits {
            15 => (14.5, 16.0),
            _ => (19.0, 28.0),
        };
        let layout = unit.layout(log_k)?;
        let plan = replicated_plan(layout, params.n.ilog2() as usize, transforms, pack, Some(16))?;
        let ctx = compile_context::<BE, F>(module, Base2K(params.base2k as u32), &plan);
        let (k, r) = (1usize << log_k, 1usize << layout.log_r());
        assert_eq!(ctx.eval_mod_plan().packed, pack && 4 * k * r <= params.n);
        let k_in = replicated_input_k(&plan);
        let k_out = k_in + plan.s2c_guard_bits() + LOG_DELTA;
        let (key_layouts, widths) = replicated_test_keys(&params, &plan, k_out)?;
        // The output keeps the input's slots, on the sparsity of the period `kR`.
        let sparsity = (params.n / (2 * k * r)).ilog2() as usize;
        let (sk_raw, sk) = gen_sk_with_raw::<_, Standard>(&params, module, host_module, [0u8; 32]);
        let values = match real {
            true => unit_batch::<ConjugateInvariant, F>(k, 0),
            false => unit_batch::<Standard, F>(k, 0),
        };
        let slot_layout = SlotLayout::Replicated(layout);
        let mut input = encrypt_batch(&params, module, &sk, slot_layout, (&values.0, &values.1), LOG_DELTA, k_in, 1)?;
        if real {
            input.set_slots(SlotsKind::Real);
        }
        let mut full_layouts = key_layouts.clone();
        let full = full_layouts.encapsulation.as_mut().unwrap().sparse_to_dense.as_mut().unwrap();
        full.k_offset = 0usize.into();
        full.dnum = GGLWELayout::dnum_for_input(plan.bootstrap_k(k_out, 0).into(), full.k_digit);
        let mut decoded = Vec::new();
        for (offset_keys, layouts) in [(false, &full_layouts), (true, &key_layouts)] {
            let (mut xs, mut xe, mut xa) = (Source::new([3; 32]), Source::new([4; 32]), Source::new([5; 32]));
            let mut keygen = ScratchOwned::<BE>::alloc(layouts.generate_tmp_bytes(module).max(layouts.prepare_tmp_bytes(module)));
            let keys = ctx
                .generate_keys(module, &sk_raw, layouts, &mut xs, &mut xe, &mut xa, &mut keygen.borrow())?
                .prepare(module, &mut keygen.borrow())?;
            let mut out = alloc_ct::<_, Standard>(&params, module, plan.bootstrap_k(k_out, 0));
            let boot_bytes = module.ckks_bootstrap_tmp_bytes(&out, &input, &ctx, layouts);
            let logged = LoggedBootstrappingKeys::new::<BE>(&keys);
            with_scratch::<BE, _>(boot_bytes, |s| module.ckks_bootstrap(&mut out, &input, &ctx, &logged, s))?;
            logged.assert_lookups_at(&widths, &format!("{unit:?} at degree {}", params.n));
            assert_eq!(out.slot_layout(), slot_layout);
            assert_eq!((out.slots(), out.log_sparsity()), (input.slots(), sparsity));
            assert_eq!((out.log_delta(), out.k().as_usize()), (LOG_DELTA, k_out));
            let got = decrypt_batch::<BE, F, Standard>(&params, module, &sk, &out, k);
            let bits = precision((&got.0, &got.1), (&values.0, &values.1));
            assert!(bits >= min_bits, "2^{log_k} values of {unit:?}: {bits:.2} bits");
            decoded.push(got);
            if refusals && offset_keys {
                replicated_bootstrap_refusals(&params, module, &ctx, &sk_raw, &keys, layouts, &input, k_out)?;
            }
        }
        let bits = precision((&decoded[0].0, &decoded[0].1), (&decoded[1].0, &decoded[1].1));
        // Least agreement on the same backends: `U16_40` 31.23 bits, 32.38 at half the degree, 31.64 unpacked, 32.25
        // unpacked on real messages and 30.42 factored, `U8_40` 31.97, and `U4_15` 17.79 on 8 values and 19.35 on 2,
        // floored at 28 and 16 bits.
        assert!(bits >= min_agreement, "{unit:?}: the key sets differ by {bits:.2} bits");
    }
    Ok(())
}

/// The dense maps of a replicated layout against the slots the encoder decodes from each monomial `Y'^j` of the
/// subring of degree `2k`, `V[c][j]`: `SlotsToCoeffs` is `V·BR`, the generator's one-factor `Decode`, and
/// `CoeffsToSlots` is `BR·V^H/(2Kk)`, its `Encode` scaled by `1/(2K)`, at every degree `2k` from 4 to the module's.
pub fn test_ckks_dft_matrices_gen_replicated_matches_the_encoder<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let max_slots = module.ckks_max_slots::<Standard>();
    let off = |(re, im): (F, F), (x, y): (f64, f64)| (re.to_f64().unwrap() - x).abs().max((im.to_f64().unwrap() - y).abs());
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(max_slots));
    // From two values, the first with a DFT layer.
    let mut k = 2;
    while k <= max_slots {
        let log_k = k.trailing_zeros() as usize;
        let eval_mod = paper_eval_mod(&ShapingUnit::U4_15.layout(log_k as u8)?);
        let meta = CoeffsMeta::from_delta_budget(26, 0);
        let dft = |kind| DFTPlan::new(kind, vec![(log_k, 1)], DFTOutputFormat::SplitRealAndImag, meta);
        let decode = one_factor_matrix::<F>(&dft(DFTType::Decode)?);
        let encode = one_factor_matrix::<F>(&dft(DFTType::Encode)?.with_scaling(eval_mod.input_scaling()?)?);
        let scale = 1.0 / (2 * k * eval_mod.f_mod_interval) as f64;
        let bit_reversed = |j: usize| j.reverse_bits().checked_shr(usize::BITS - log_k as u32).unwrap_or(0);
        for j in 0..k {
            // At the degree `2k` of the map, below the backend's floor too.
            let mut pt = module.ckks_plaintext_alloc_from_slots::<Standard>(
                k,
                Base2K(params.base2k as u32),
                TorusPrecision((LOG_DELTA + 30) as u32),
            );
            pt.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(40),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
                slot_layout: SlotLayout::Full,
            });
            let mut coeffs = vec![F::zero(); 2 * k];
            coeffs[bit_reversed(j)] = F::one();
            module.ckks_encode_coeffs_host_into(&mut pt, &coeffs, &mut scratch.borrow())?;
            let (mut re, mut im) = (vec![F::zero(); k], vec![F::zero(); k]);
            module.ckks_decode_slots_host_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
            for c in 0..k {
                let (v_re, v_im) = (re[c].to_f64().unwrap(), im[c].to_f64().unwrap());
                // The encoding rounds the monomial at `2^40`.
                let err = off(decode[c][j], (v_re, v_im));
                assert!(err < 1e-9, "k = {k}: SlotsToCoeffs at ({c}, {j}) is off by {err:e}");
                let err = off(encode[j][c], (v_re * scale, -v_im * scale)) / scale;
                assert!(err < 1e-9, "k = {k}: CoeffsToSlots at ({j}, {c}) is off by {err:e}");
            }
        }
        k *= 2;
    }
    Ok(())
}

/// What compiling, keying and running a replicated bootstrapping at the scalar `F` needs of a module.
pub trait ReplicatedBootstrapModule<BE: Backend, F: TestScalar>:
    TestContextModule<BE>
    + TestContextSharedModule<BE, ConjugateInvariant>
    + CKKSEncodingOps<BE, F>
    + CKKSEncodingOps<BE, F, ConjugateInvariant>
    + CKKSBootstrappingOps<BE>
    + CKKSDFTOps<BE>
    + CKKSDFTMatrixOps<BE, F>
    + CKKSDFTMatrixPreparedFactory<BE>
    + CKKSLinearTransformationOps<BE>
    + GLWETrace<BE>
    + GLWECIKeyEncryptSk<BE>
    + GLWECIKeyPreparedFactory<BE>
{
}

impl<BE: Backend, F: TestScalar, M> ReplicatedBootstrapModule<BE, F> for M where
    M: TestContextModule<BE>
        + TestContextSharedModule<BE, ConjugateInvariant>
        + CKKSEncodingOps<BE, F>
        + CKKSEncodingOps<BE, F, ConjugateInvariant>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSDFTMatrixOps<BE, F>
        + CKKSDFTMatrixPreparedFactory<BE>
        + CKKSLinearTransformationOps<BE>
        + GLWETrace<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
{
}

/// The `k` of an input of `plan` at the leveled scale `2^LOG_DELTA`.
fn replicated_input_k(plan: &BootstrappingPlan) -> usize {
    plan.input_k(replicated_raise_input_k(plan, LOG_DELTA))
}

/// The `k` the raise of `plan` reads for a message at scale `2^log_delta`: the guard and working bits of SlotsToCoeffs
/// and the message ratio above the scale.
pub(crate) fn replicated_raise_input_k(plan: &BootstrappingPlan, log_delta: usize) -> usize {
    log_delta + plan.s2c_guard_bits() + plan.working_bits() + plan.eval_mod().log_msg_ratio
}

/// The key layouts of `plan` for an output of `k_out` bits and an input at the leveled scale, each key covering its
/// width ([`BootstrappingPlan::key_widths`]), shaped by the shared policy under bounds that every test width fits, and
/// the widths.
fn replicated_test_keys(
    params: &CKKSTestParams,
    plan: &BootstrappingPlan,
    k_out: usize,
) -> Result<(BootstrappingKeysLayout, BootstrappingKeyWidths)> {
    let (raised, k_in) = (plan.bootstrap_k(k_out, 0), replicated_input_k(plan));
    let policy = BootstrappingKeyPolicy {
        log_n: params.n.ilog2() as usize,
        base2k: params.base2k,
        rank: 1,
        dense_log_modulus: 1 << 12,
        sparse_log_modulus: 1 << 12,
        max_dsize: None,
        dense_to_sparse_dsize: 1,
    };
    let layout = policy.keys_layout(plan, raised, k_in, LOG_DELTA)?;
    Ok((layout, plan.key_widths(policy.log_n, raised, k_in, LOG_DELTA)))
}

/// The replicated plan of ePrint 2026/2513 for `layout` at degree `2^log_n`, the maps in one factor each, or `factored`
/// into two DFT factors and two placement factors with the inner CoeffsToSlots factors at `2^44`, `packed` when the two
/// batches fit, and its sparse secret of weight `hamming_weight`.
pub(crate) fn replicated_plan(
    layout: ReplicatedLayout,
    log_n: usize,
    factored: bool,
    packed: bool,
    hamming_weight: Option<usize>,
) -> Result<BootstrappingPlan> {
    let factors = 1 + usize::from(factored);
    let dft = |kind, log_delta| -> Result<DFTPlan> {
        let schedule: Vec<_> = split_depth(layout.log_k(), factors)
            .into_iter()
            .map(|depth| (depth, 1))
            .collect();
        let meta = CoeffsMeta::from_delta_budget(log_delta, 0);
        let plan = DFTPlan::new(kind, schedule, DFTOutputFormat::SplitRealAndImag, meta)?;
        Ok(match factored {
            true => plan.with_placement_factors(factors).with_optimal_bsgs(log_n),
            false => plan,
        })
    };
    let coeffs_to_slots = match factored {
        true => dft(DFTType::Encode, 44)?.with_last_factor_log_delta(34),
        false => dft(DFTType::Encode, 34)?,
    };
    let packed = packed && 4 << (layout.log_k() + layout.log_r()) <= 1usize << log_n;
    let techniques = BootstrappingTechniques {
        sparse_secret_encapsulation: hamming_weight.map(|hamming_weight| SparseSecretEncapsulation { hamming_weight }),
        eval_round_plus: None,
    };
    let eval_mod = paper_eval_mod(&layout).with_packed(packed);
    BootstrappingPlan::new(
        BootstrappingPipeline::S2CFirst,
        techniques,
        coeffs_to_slots,
        eval_mod,
        dft(DFTType::Decode, 26)?,
    )?
    .with_slot_layout(SlotLayout::Replicated(layout), log_n)?
    .with_c2s_guard_bits(14)?
    .with_working_bits(8)?
    .with_s2c_guard_bits(10)
}

/// Every refusal of a replicated bootstrapping and of its steps fires before any write, each with its variant, `op`
/// and reason, `res` left as it was: an input off the plan's layout or without the budget of SlotsToCoeffs; an output
/// of another rank, degree or radix, or narrower than the stages; keys whose encapsulation the plan does not have, a
/// rotation key or a tensor key short of its `k`, a tensor key decomposing less than EvalMod's product; a layout above
/// the module's degree; an arena a byte below the query; the operands of the replicated
/// SlotsToCoeffs, CoeffsToSlots and EvalMod steps, the keys of both transforms, a plain EvalMod and the tensor key of
/// EvalMod; and, before any draw, key generation from a secret of another degree than the plan's.
#[allow(clippy::too_many_arguments)]
fn replicated_bootstrap_refusals<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    ctx: &BootstrappingContextPrepared<BE, F>,
    sk_raw: &BackendGLWESecret<BE>,
    keys: &BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
    key_layouts: &BootstrappingKeysLayout,
    input: &CKKSCiphertextOwned<BE>,
    k_out: usize,
) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: ReplicatedBootstrapModule<BE, F>,
    F: TestScalar + DftScalar,
{
    use crate::CKKSCompositionError as E;
    const OP: &str = "ckks_bootstrap";
    let (n, b, level) = (params.n, params.base2k, LOG_DELTA);
    let shape = |k: usize, rank: usize, n: usize, base2k: usize| GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: rank.into(),
    };
    let zero = |k: usize, n: usize| module.ckks_ciphertext_alloc_from_glwe_infos::<_, Standard>(&shape(k, 1, n, b));
    // A refused destination holds a fixture, at the input's metadata where its width holds the scale.
    let alloc = |k: usize, rank: usize, n: usize, base2k: usize| {
        let meta = if k >= input.log_delta() {
            input.meta()
        } else {
            GLWEPlaintextMeta::default()
        };
        let glwe_layout = shape(k, rank, n, base2k);
        fixture_ciphertext::<BE, Standard>(module, &CKKSLayout { glwe_layout, meta }, 7)
    };
    let raised_k = ctx.plan.bootstrap_k(k_out, 0);
    let out = || alloc(raised_k, 1, n, b);
    let refuses = |label: &str,
                   mut res: CKKSCiphertextOwned<BE>,
                   a: &CKKSCiphertextOwned<BE>,
                   ctx: &BootstrappingContextPrepared<BE, F>,
                   keys: &BootstrappingKeysPrepared<BE::OwnedBuf, BE>,
                   matches: &dyn Fn(&E) -> bool| {
        let before = snapshot::<BE, _, Standard>(&res);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut res, a, ctx, keys, s)).unwrap_err();
        assert!(err.composition().is_some_and(matches), "{label}: unexpected refusal {err}");
        assert_eq!(snapshot::<BE, _, Standard>(&res), before, "{label}: the refusal wrote res");
    };

    // An input off the plan's layout and one without the budget of its chain, which the SlotsToCoeffs step refuses too.
    let run = ctx.run(input.meta());
    let (chain, eval_mod) = ("ckks_dft_matrix_eval_assign", "ckks_bootstrap_eval_mod");
    let (mut full, mut spent_input) = (input.clone(), input.clone());
    full.set_slot_layout(SlotLayout::Full);
    spent_input.set_log_budget(ctx.slots_to_coeffs().consumed_bits() - 1);
    let off_layout = "the replicated layout of SlotsToCoeffs";
    let spent = |op: &'static str| move |e: &E| matches!(e, E::InsufficientHomomorphicCapacity { op: o, .. } if *o == op);
    for (mut a, at_entry, in_step) in [
        (
            full,
            &operand(OP, off_layout) as &dyn Fn(&E) -> bool,
            &operand(chain, off_layout) as &dyn Fn(&E) -> bool,
        ),
        (spent_input, &spent(OP), &spent(chain)),
    ] {
        let mut res = out();
        // Refused before the arena check, on an empty arena.
        refused!(
            Standard,
            "SlotsToCoeffs input",
            with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut res, &a, ctx, keys, s)),
            [res],
            at_entry
        );
        refused!(
            Standard,
            "SlotsToCoeffs input",
            with_scratch::<BE, _>(0, |s| module.ckks_slots_to_coeffs_assign(&mut a, &run, keys, s)),
            [a],
            in_step
        );
    }
    // The narrowest output holds the scale the raise labels, what SlotsToCoeffs leaves of the lifted input plus the
    // raise's C2S guard bits, and the bits CoeffsToSlots and EvalMod consume.
    let narrower = "narrower than the bits its stages consume";
    let lifted = input.k().as_usize() + ctx.plan.s2c_guard_bits() + ctx.plan.working_bits();
    let raised_scale = lifted - ctx.slots_to_coeffs().consumed_bits() + ctx.plan.mod_raise().c2s_guard_bits();
    let narrowest = raised_scale + ctx.plan.post_mod_up_consumed_bits() - ctx.plan.c2s_guard_bits();
    for (res, reason) in [
        (alloc(raised_k, 2, n, b), "rank-1"),
        (alloc(raised_k, 1, n / 2, b), "one degree"),
        (alloc(raised_k, 1, n, b - 1), "one base2k"),
        (alloc(level - 1, 1, n, b), narrower),
        (alloc(narrowest - 1, 1, n, b), narrower),
    ] {
        refuses(reason, res, input, ctx, keys, &operand(OP, reason));
    }
    // At the narrowest output the entry moves past the stages, to a later refusal.
    let mut res = alloc(narrowest, 1, n, b);
    let before = snapshot::<BE, _, Standard>(&res);
    let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut res, input, ctx, keys, s)).unwrap_err();
    assert!(
        !err.composition().is_some_and(operand(OP, narrower)),
        "the narrowest output: {err}"
    );
    assert_eq!(
        snapshot::<BE, _, Standard>(&res),
        before,
        "the narrowest output: the refusal wrote res"
    );
    let log_n = n.ilog2() as usize;
    let SlotLayout::Replicated(layout) = input.slot_layout() else {
        unreachable!("a replicated input")
    };
    let plain = replicated_plan(layout, log_n, false, true, None)?;
    // Above the module, the plan names the degree its layout states.
    let above = plain.check_log_gap_fits(log_n - 1).unwrap_err();
    assert_eq!(
        above.downcast_ref::<E>(),
        Some(&E::InvalidPlan {
            plan: "BootstrappingPlan",
            reason: format!(
                "a replicated layout takes the degree it states, 2^{log_n}, above the module's 2^{}",
                log_n - 1
            ),
        })
    );
    let plain_ctx = compile_context::<BE, F>(module, Base2K(b as u32), &plain);
    refuses("encapsulation", out(), input, &plain_ctx, keys, &operand(OP, "encapsulation"));
    // The coefficient mask and the merged SlotsToCoeffs factor take a context on the full layout.
    let full_layout = "on the full layout";
    let mask = with_scratch::<BE, _>(0, |s| ctx.coeffs_mask(module, &vec![false; n], s));
    assert_refused("a replicated mask", mask, "coeffs_mask", full_layout);
    let merged = with_scratch::<BE, _>(0, |s| ctx.merged_s2c_factor(module, &[vec![F::zero(); n]], LOG_DELTA, s));
    assert_refused("a replicated merged factor", merged, "merged_s2c_factor", full_layout);
    // An output wide enough that a rotation key falls short of its width.
    let (input_k, log_delta) = (input.k().as_usize(), input.log_delta());
    let widths_at = |k: usize| ctx.plan.key_widths(log_n, ctx.plan.bootstrap_k(k, 0), input_k, log_delta);
    let covered = |k: usize| {
        widths_at(k)
            .rotation_keys
            .iter()
            .all(|(p, &w)| w <= key_layouts.rotation_keys[p].gglwe_layout().max_input_k().as_usize())
    };
    let wide_k = (k_out..).find(|&k| !covered(k)).unwrap();
    let narrow = |e: &E| matches!(e, E::AutomorphismKeyTooNarrow { op, .. } if *op == OP);
    let wide = alloc(ctx.plan.bootstrap_k(wide_k, 0), 1, n, b);
    refuses("rotation key short", wide, input, ctx, keys, &narrow);
    // Keys whose tensor key digits cover less than EvalMod's widest product, and keys whose encapsulation leaves the
    // raise short: the ephemeral key narrower than the raise's input, and the return key's rows ending below the raised
    // width, which the entry refuses before SlotsToCoeffs runs.
    let shortened = |shorten: &dyn Fn(&mut BootstrappingKeysLayout)| -> Result<_> {
        let mut layouts = key_layouts.clone();
        shorten(&mut layouts);
        let (mut xs, mut xe, mut xa) = (Source::new([9; 32]), Source::new([10; 32]), Source::new([11; 32]));
        let mut keygen = ScratchOwned::<BE>::alloc(layouts.generate_tmp_bytes(module).max(layouts.prepare_tmp_bytes(module)));
        ctx.generate_keys(module, sk_raw, &layouts, &mut xs, &mut xe, &mut xa, &mut keygen.borrow())?
            .prepare(module, &mut keygen.borrow())
    };
    let tensor_k = widths_at(k_out).tensor_key;
    let short_tensor = shortened(&|layouts| {
        let tensor = &mut layouts.tensor_key;
        tensor.k_digit = ((tensor_k - 1) / tensor.dnum.as_usize()).into();
    })?;
    let relinearization =
        |want: usize| move |e: &E| matches!(e, E::RelinearizationKeyTooNarrow { op, k, .. } if *op == OP && *k as usize == want);
    refuses(
        "tensor key short",
        out(),
        input,
        ctx,
        &short_tensor,
        &relinearization(tensor_k),
    );
    // A tensor key the lookup at `tensor_k` takes, whose digits stop short of EvalMod's product.
    let product_k = ctx.plan.tensor_key_max_input_k(raised_k, log_delta);
    let max_input_k_at = |k_digit: usize| {
        let mut tensor = key_layouts.tensor_key;
        tensor.k_digit = k_digit.into();
        tensor.gglwe_layout().max_input_k().as_usize()
    };
    let k_digit = (1..=key_layouts.tensor_key.k_digit.as_usize())
        .rev()
        .find(|&k_digit| max_input_k_at(k_digit) < product_k)
        .unwrap();
    assert!(
        max_input_k_at(k_digit) >= tensor_k,
        "no digit lands between the lookup and the product"
    );
    let product_short = shortened(&|layouts| layouts.tensor_key.k_digit = k_digit.into())?;
    refuses(
        "tensor key short of the product",
        out(),
        input,
        ctx,
        &product_short,
        &relinearization(product_k),
    );
    let lift = ctx.plan.s2c_guard_bits() + ctx.plan.working_bits();
    let raise_input_k = replicated_raise_input_k(&ctx.plan, level);
    let ephemeral = shortened(&|layouts| {
        let key = &mut layouts.encapsulation.as_mut().unwrap().dense_to_sparse;
        key.k_digit = ((raise_input_k - 1) / key.dnum.as_usize()).into();
    })?;
    let ephemeral_short = operand(OP, "the input is wider than an encapsulation key's max_input_k");
    refuses("ephemeral key short", out(), input, ctx, &ephemeral, &ephemeral_short);
    let short_return = shortened(&|layouts| {
        let key = layouts.encapsulation.as_mut().unwrap().sparse_to_dense.as_mut().unwrap();
        key.k_digit = (key.k_digit.as_usize() - 1).into();
    })?;
    refuses(
        "return key short",
        out(),
        input,
        ctx,
        &short_return,
        &operand(OP, "the sparse-to-dense key takes"),
    );
    // Key generation from a secret of another degree than the plan's.
    let sk = module.glwe_secret_alloc_from_infos(&shape(1, 1, n / 2, b));
    let fresh = || [12, 13, 14].map(|b| Source::new([b; 32]));
    let [xs, xe, xa] = &mut fresh();
    let result = with_scratch::<BE, _>(0, |s| ctx.generate_keys(module, &sk, key_layouts, xs, xe, xa, s));
    let off_degree = E::InvalidPlan {
        plan: "BootstrappingPlan",
        reason: "a replicated layout takes the degree it states".into(),
    };
    assert_eq!(result.err().unwrap().downcast_ref::<E>(), Some(&off_degree));
    assert!([xs, xe, xa].map(|s| s.new_seed()) == fresh().map(|mut s| s.new_seed()));
    // An arena a byte below the query, then EvalMod on an input off the layout CoeffsToSlots writes.
    let bytes = module.ckks_bootstrap_tmp_bytes(&out(), input, ctx, key_layouts);
    let mut res = out();
    refused!(
        Standard,
        "short arena",
        with_scratch::<BE, _>(bytes - 1, |s| module.ckks_bootstrap(&mut res, input, ctx, keys, s)),
        [res],
        message = "ckks_bootstrap needs a scratch arena"
    );
    refused!(
        Standard,
        "EvalMod, replicated layout",
        with_scratch::<BE, _>(0, |s| module.ckks_bootstrap_eval_mod(&mut res, input, &run, keys, s)),
        [res],
        operand(eval_mod, "the output of the replicated CoeffsToSlots")
    );

    // The steps on the operands of a bootstrapping, before any write: the chain of each transform refuses its degree,
    // budget and keys, then its arena, a byte below the query over the keys it uses. SlotsToCoeffs runs on a copy of the
    // input, which it lifts by the S2C guard and working bits, as the bootstrap composes it.
    let mut coeffs = zero(input.k().as_usize() + lift, n);
    with_scratch::<BE, _>(bytes, |s| module.ckks_copy(&mut coeffs, input, s))?;
    let offsets = ctx.rotation_key_offsets();
    let step_bytes = |decode: bool, query: &dyn Fn(&GLWEAutomorphismKeyLayout) -> usize| {
        let used = offsets
            .iter()
            .filter(|key| (if decode { key.slots_to_coeffs } else { key.raised }).is_some());
        used.map(|key| query(&key_layouts.rotation_keys[&key.p])).max().unwrap()
    };
    let s2c_bytes = step_bytes(true, &|key| {
        module.ckks_dft_tmp_bytes::<Decode, _, _, _, _, _>(&coeffs, &coeffs, &run, key)
    });
    let short_arena = |op: &str| format!("{op} needs a scratch arena");
    let mut a = coeffs.clone();
    refused!(
        Standard,
        "SlotsToCoeffs, short arena",
        with_scratch::<BE, _>(s2c_bytes - 1, |s| module.ckks_slots_to_coeffs_assign(&mut a, &run, keys, s)),
        [a],
        message = short_arena(chain)
    );
    // An input whose allocation ends at its `k`, on the limb boundary below the input's, the lift above it.
    let mut tight = zero(input.k().as_usize() / b * b, n);
    with_scratch::<BE, _>(bytes, |s| module.ckks_copy(&mut tight, input, s))?;
    assert!(
        tight.k().as_usize() + lift > tight.max_k().as_usize() && tight.log_budget() >= ctx.slots_to_coeffs().consumed_bits(),
        "the fixture lacks the room of the lift alone"
    );
    refused!(
        Standard,
        "SlotsToCoeffs, no room for the lift",
        with_scratch::<BE, _>(0, |s| module.ckks_slots_to_coeffs_assign(&mut tight, &run, keys, s)),
        [tight],
        operand(chain, "the input does not hold the lift of its plan")
    );
    let mut decoded = coeffs.clone();
    with_scratch::<BE, _>(s2c_bytes, |s| module.ckks_slots_to_coeffs_assign(&mut decoded, &run, keys, s))?;
    let mut raised = zero(ctx.plan.bootstrap_k(k_out, 0), n);
    with_scratch::<BE, _>(bytes, |s| module.ckks_mod_raise(&mut raised, &mut decoded, &run, keys, s))?;
    let half = CKKSLayout {
        glwe_layout: shape(raised.k().as_usize(), 1, n / 2, b),
        meta: raised.meta(),
    };
    let (half, mut spent) = (fixture_ciphertext::<BE, Standard>(module, &half, 8), raised.clone());
    spent.set_log_budget(ctx.coeffs_to_slots().consumed_bits() - 1);
    let layout = operand(chain, "replicated slot layout");
    let degree = operand(chain, "the slots of the input's degree must hold the layout of the plan");
    let budget = |e: &E| matches!(e, E::InsufficientHomomorphicCapacity { op, .. } if *op == chain);
    for (label, a, matches) in [
        ("CoeffsToSlots, replicated layout", input, &layout as &dyn Fn(&E) -> bool),
        ("CoeffsToSlots, half degree", &half, &degree),
        ("CoeffsToSlots, spent input", &spent, &budget),
    ] {
        let mut a = a.clone();
        refused!(
            Standard,
            label,
            with_scratch::<BE, _>(0, |s| module.ckks_coeffs_to_slots_assign(&mut a, &run, keys, s)),
            [a],
            matches
        );
    }
    // A key of each step's trace, and one of its transform that the trace does not use, hidden one at a time.
    let mut hiding = shortened(&|_| {})?;
    for (decode, start) in [(true, &coeffs), (false, &raised)] {
        let dft = if decode {
            ctx.plan.slots_to_coeffs()
        } else {
            ctx.plan.coeffs_to_slots()
        };
        let trace = module.glwe_trace_galois_elements(n.into(), dft.trace_levels(log_n).unwrap(), false);
        let transform = offsets
            .iter()
            .find(|key| (if decode { key.slots_to_coeffs } else { key.raised }).is_some() && !trace.contains(&key.p));
        let transform = transform.expect("each transform rotates by a key its trace does not use").p;
        for p in [trace[0], transform] {
            let key = hiding.rotation_keys.remove(&p).unwrap();
            let mut a = start.clone();
            refused!(
                Standard,
                "hidden key",
                with_scratch::<BE, _>(0, |s| match decode {
                    true => module.ckks_slots_to_coeffs_assign(&mut a, &run, &hiding, s),
                    false => module.ckks_coeffs_to_slots_assign(&mut a, &run, &hiding, s),
                }),
                [a],
                |e: &E| matches!(e, E::MissingAutomorphismKey { op, rotation, .. } if *op == chain && *rotation == p)
            );
            hiding.rotation_keys.insert(p, key);
        }
    }
    let c2s_bytes = step_bytes(false, &|key| {
        module.ckks_dft_tmp_bytes::<Encode, _, _, _, _, _>(&raised, &raised, &run, key)
    });
    let mut a = raised.clone();
    refused!(
        Standard,
        "CoeffsToSlots, short arena",
        with_scratch::<BE, _>(c2s_bytes - 1, |s| module.ckks_coeffs_to_slots_assign(&mut a, &run, keys, s)),
        [a],
        message = short_arena(chain)
    );
    with_scratch::<BE, _>(c2s_bytes, |s| module.ckks_coeffs_to_slots_assign(&mut raised, &run, keys, s))?;
    // EvalMod on an output of another rank, degree or width, on an input short of its working bits, and on one without
    // the budget EvalMod consumes.
    let mut cooled = raised.clone();
    cooled.set_log_delta(level + ctx.plan.s2c_guard_bits() + ctx.plan.c2s_guard_bits() - 1);
    let mut starved = raised.clone();
    starved.set_log_budget(ctx.plan.eval_mod().consumed_bits() - 1);
    let (rank, degree, narrow) = (
        operand(eval_mod, "rank-1"),
        operand(eval_mod, "one degree"),
        operand(eval_mod, "lands the result in an output of"),
    );
    let working = operand(eval_mod, "bits under the raised scale");
    let capacity = |e: &E| matches!(e, E::InsufficientHomomorphicCapacity { op, .. } if *op == eval_mod);
    for (label, mut res, a, matches) in [
        (
            "EvalMod, rank 2",
            alloc(k_out, 2, n, b),
            &raised,
            &rank as &dyn Fn(&E) -> bool,
        ),
        ("EvalMod, half degree", alloc(k_out, 1, n / 2, b), &raised, &degree),
        ("EvalMod, narrow", alloc((k_out - 1) / b * b, 1, n, b), &raised, &narrow),
        ("EvalMod, off its working bits", out(), &cooled, &working),
        ("EvalMod, spent input", out(), &starved, &capacity),
    ] {
        refused!(
            Standard,
            label,
            with_scratch::<BE, _>(0, |s| module.ckks_bootstrap_eval_mod(&mut res, a, &run, keys, s)),
            [res],
            matches
        );
    }
    // A plain EvalMod, which has no message to land the output at.
    let mut res = out();
    refused!(
        Standard,
        "EvalMod, plain",
        with_scratch::<BE, _>(0, |s| module.ckks_bootstrap_eval_mod(
            &mut res,
            &raised,
            ctx.eval_mod().unwrap(),
            keys,
            s
        )),
        [res],
        operand(eval_mod, "a plain EvalMod takes the full layout")
    );
    // A tensor key of another radix covering EvalMod's widest product, but not the tensor intermediate its products
    // then carve under the shaping depth.
    let widest = ctx.plan.eval_mod().relinearization_k(raised.log_budget());
    let other_radix = shortened(&|layouts| {
        let tensor = &mut layouts.tensor_key;
        tensor.base2k = (b - 1).into();
        tensor.k_digit = widest.div_ceil(tensor.dnum.as_usize()).into();
    })?;
    let mut res = out();
    refused!(
        Standard,
        "EvalMod, tensor key short of the intermediate",
        with_scratch::<BE, _>(0, |s| module.ckks_bootstrap_eval_mod(
            &mut res,
            &raised,
            &run,
            &other_radix,
            s
        )),
        [res],
        |e: &E| matches!(e, E::RelinearizationKeyTooNarrow { op, .. } if *op == eval_mod)
    );
    // A byte below the step's query, which an override's own workspace is part of, then exactly the query.
    let mut res = out();
    let eval_mod_bytes = module.ckks_bootstrap_eval_mod_tmp_bytes(&res, &raised, &run, key_layouts);
    refused!(
        Standard,
        "EvalMod, short arena",
        with_scratch::<BE, _>(eval_mod_bytes - 1, |s| module
            .ckks_bootstrap_eval_mod(&mut res, &raised, &run, keys, s)),
        [res],
        message = short_arena(eval_mod)
    );
    with_scratch::<BE, _>(eval_mod_bytes, |s| {
        module.ckks_bootstrap_eval_mod(&mut res, &raised, &run, keys, s)
    })?;
    Ok(())
}

/// [`test_ckks_bootstrap_replicated`] on the conjugate-invariant ring of degree `N` on one `module` of degree `2N`: the input
/// embedded into the standard ring by `ckks_ci_embed`, bootstrapped one bit wider than the output by `ckks_bootstrap`,
/// and traced back by `ckks_ci_trace`, each ring map at the shaping depth.
pub fn test_ckks_bootstrap_replicated_ci<BE>(params: CKKSTestParams, module: &Module<BE>) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: ReplicatedBootstrapModule<BE, f64> + CKKSCIRingMapOps<BE>,
{
    assert_eq!(module.n(), 2 * params.n, "the module serves both rings at degrees N and 2N");
    let standard_params = CKKSTestParams {
        n: 2 * params.n,
        ..params
    };
    let host = Module::<HostBytesBackend>::new(standard_params.n as u64);
    let layout = ShapingUnit::U16_40.layout(3)?;
    let plan = replicated_plan(layout, standard_params.n.ilog2() as usize, false, true, Some(16))?;
    let base2k = Base2K(params.base2k as u32);
    let ctx = compile_context::<BE, f64>(module, base2k, &plan);

    // The standard bootstrapping runs one bit wider than the output, for the trace to spend.
    let k_in = replicated_input_k(&plan);
    let k_out = k_in + plan.s2c_guard_bits() + LOG_DELTA;
    let (key_layouts, widths) = replicated_test_keys(&standard_params, &plan, k_out + 1)?;
    let (ci_sk_raw, ci_sk) = gen_sk_with_raw::<_, ConjugateInvariant>(&params, module, &host, [0u8; 32]);
    let (sk_raw, _) = gen_sk_with_raw::<_, Standard>(&standard_params, module, &host, [1u8; 32]);

    let len = 1 << layout.log_k();
    let values = unit_batch::<ConjugateInvariant, f64>(len, 0);
    let slot_layout = SlotLayout::Replicated(layout);
    let input = encrypt_batch(
        &params,
        module,
        &ci_sk,
        slot_layout,
        (&values.0, &values.1),
        LOG_DELTA,
        k_in,
        1,
    )?;

    // The conversion keys cover the ring maps at the shaping depth.
    let profile = ShapingProfile::new(&layout);
    let width = (k_in + profile.shaping_depth(input.log_delta())).max(k_out + 1 + profile.shaping_depth(LOG_DELTA));
    let conversion_layout = standard_params.ksk_layout(width);
    let mut keygen = ScratchOwned::<BE>::alloc(
        module
            .glwe_ci_key_encrypt_sk_tmp_bytes(&conversion_layout)
            .max(module.glwe_ci_key_prepare_tmp_bytes(&conversion_layout))
            .max(key_layouts.generate_tmp_bytes(module))
            .max(key_layouts.prepare_tmp_bytes(module)),
    );
    let (embed_key, trace_key) = gen_ci_keys(&standard_params, module, &ci_sk_raw, &sk_raw, width, &mut keygen.borrow());
    let (mut xs, mut xe, mut xa) = (Source::new([3; 32]), Source::new([4; 32]), Source::new([5; 32]));
    let keys = ctx
        .generate_keys(module, &sk_raw, &key_layouts, &mut xs, &mut xe, &mut xa, &mut keygen.borrow())?
        .prepare(module, &mut keygen.borrow())?;

    let mut embedded = alloc_ct::<_, Standard>(&standard_params, module, k_in);
    let mut refreshed = alloc_ct::<_, Standard>(&standard_params, module, plan.bootstrap_k(k_out + 1, 0));
    let mut out = alloc_ct::<_, ConjugateInvariant>(&params, module, k_out);
    let embedded_meta = (
        GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            ..input.meta()
        },
        input.k(),
    );
    let bytes = module.ckks_ci_embed_tmp_bytes(&embedded, &input, &conversion_layout);
    with_scratch::<BE, _>(bytes, |s| module.ckks_ci_embed(&mut embedded, &input, &embed_key, s))?;
    assert_eq!((embedded.meta(), embedded.k()), embedded_meta);
    let bytes = module.ckks_bootstrap_tmp_bytes(&refreshed, &embedded, &ctx, &key_layouts);
    let logged = LoggedBootstrappingKeys::new::<BE>(&keys);
    with_scratch::<BE, _>(bytes, |s| module.ckks_bootstrap(&mut refreshed, &embedded, &ctx, &logged, s))?;
    logged.assert_lookups_at(&widths, "conjugate-invariant replicated bootstrapping");
    let bytes = module.ckks_ci_trace_tmp_bytes(&out, &refreshed, &conversion_layout);
    with_scratch::<BE, _>(bytes, |s| module.ckks_ci_trace(&mut out, &refreshed, &trace_key, s))?;
    assert_eq!(out.slot_layout(), slot_layout);
    assert_eq!(out.slots(), SlotsKind::Real);
    assert_eq!((out.log_delta(), out.k().as_usize()), (LOG_DELTA, k_out));
    let got = decrypt_batch::<BE, f64, ConjugateInvariant>(&params, module, &ci_sk, &out, len);
    let bits = precision((&got.0, &got.1), (&values.0, &values.1));
    // 23.33 bits measured at least on the portable, oracle and rayon backends: 4.33 bits of margin.
    assert!(bits >= 19.0, "conjugate-invariant replicated bootstrapping: {bits:.2} bits");
    Ok(())
}
