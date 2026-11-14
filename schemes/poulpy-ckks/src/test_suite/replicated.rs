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
//! | [`test_replicated_eval_mod_is_shaped`] | EvalMod with masked constants keeps the message far above CKKS: an input above the plan scale rounded at entry with a fused last squaring, and one below it evaluated in place with an inverse and a shaped return; a packed EvalMod refused on real slots |
//! | [`test_replicated_complex_eval_mod_is_shaped`] | the complex EvalMod with a fused last squaring keeps the message far above CKKS |
//! | [`test_replicated_polynomials_are_shaped`] | monomial and Chebyshev polynomials keep the data slots far above CKKS, on either ring, and constants without masked plaintexts or of another layout are refused at entry |
//! | [`test_replicated_polynomial_fallback_is_shaped`] | a baby step whose masked constants differ in scale takes the products term by term, which keep the shaping |
//! | [`test_replicated_complex_polynomials_are_shaped`] | a complex polynomial keeps the data slots far above CKKS |
//! | [`test_replicated_automorphisms_are_shaped`] | rotations and conjugations, into and in place, keep the data slots far above CKKS |
//! | [`test_replicated_rotations_are_shaped`] | rotations keep the data slots far above CKKS, on either ring |
//! | [`test_replicated_rotations_are_admissible`] | rotations by a non-multiple of the period |
//! | [`test_replicated_masked_constants`] | a masked constant fills the data slots only, and adds to a ciphertext |
//! | [`test_replicated_conversions_round_trip`] | CKKS to the replicated layout and back keeps the input's precision |
//! | [`test_replicated_ci_bootstrap`] | the bootstrapping of a conjugate-invariant replicated layout through the standard ring |
//! | [`test_replicated_transforms_apply_matrices`] | a complex matrix onto the data slots, and gathered then traced back to period `k` |

use std::collections::HashMap;

use anyhow::Result;
use poulpy_core::{
    GLWECIKeyEncryptSk, GLWEMulPreparedPlainSum, GLWETrace,
    api::GLWECIKeyPreparedFactory,
    layouts::{
        Base2K, GLWEInfos, GLWELayout, GLWEToBackendRef, LWEInfos, LinearTransformationBabySteps, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ConjugateInvariant, HostBytesBackend, Module, PrepareHint, Ring, ScratchOwned, Standard, ZnxView},
    source::Source,
};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, alloc_ct, alloc_scratch,
    ckks_pt_cst_full, ckks_spec, gen_atk, gen_ci_keys, gen_sk, gen_sk_with_raw, gen_tsk, next_test_seed, upload_pt,
};
use super::parity::helpers::{snapshot, with_scratch};
use crate::{
    CKKSInfos, CKKSLayout, CoeffsMeta, GLWEPlaintextInfos, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos,
    SetGLWEPlaintextInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps,
        CKKSCopyOps, CKKSDecryptOps, CKKSDotProductOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSEvalModOps,
        CKKSInnerSumOps, CKKSLeftOperandPreparedFactory, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSMulAddOps, CKKSMulOps,
        CKKSMulSubOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSReplicatedBootstrapOps, CKKSReplicatedCIBootstrapOps,
        CKKSReplicatedOps, CKKSRightOperandPreparedFactory, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::{
        CKKSCiphertextAlloc, CKKSCiphertextOwned, CKKSEncodingBuffer, CKKSPlaintextAlloc, CKKSPlaintextOwned, EvalModPlan,
        EvalModPoly, EvalModType, ReplicatedBootstrapPlan, ReplicatedBootstrapping, ReplicatedConversion, ReplicatedTransform,
        ReplicatedTransforms, ckks_eval_mod_compile, ckks_eval_mod_replicated_compile, ckks_eval_mod_replicated_packed_compile,
        encode_bsgs_replicated, encode_complex_bsgs_replicated,
    },
    oep::CKKSEncapsulatedModUpImpl,
    oep::derived::replicated::{wide_gather_trace, wide_onto_replicated},
    polynomial::{Basis, CKKSPolynomialEncode, ComplexPolynomial, Polynomial, SplitStrategy},
    shaping::{ShapingUnit, conjugate_invariant},
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
fn precision<F: TestScalar>(got: (&[F], &[F]), want: (&[F], &[F])) -> f64 {
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

/// Runs `$call`, which must refuse with `InvalidOperand` under `$op`, leaving `$operand`s as they were.
macro_rules! refused {
    ($rg:ty, $op:expr, $call:expr, [$($operand:expr),+ $(,)?]) => {{
        let before = ($(snapshot::<BE, _, $rg>(&$operand),)+);
        let err = $call.unwrap_err();
        assert_invalid(&err, $op);
        assert_eq!(($(snapshot::<BE, _, $rg>(&$operand),)+), before, "{} changed an operand", $op);
    }};
}

/// Decrypts `ct` and decodes its first `len` values.
fn decrypt_batch<BE, F, Rg: Ring>(
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
    let coarse = encrypt_batch(&params, module, &sk, slot_layout, (&a.0, &a.1), k, 1)?;
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
    PtVec,
    PtVecAssign,
    PtConst,
    PtConstAssign,
    /// `acc·b + b·acc`, two pairs summed once.
    Dot,
    /// `c ± acc·b` into a fresh accumulator `c`.
    MulAddCt,
    MulSubCt,
    MulAddPtVec,
    MulSubPtVec,
    MulAddPtConst,
    MulSubPtConst,
    /// `acc·b + c` with plaintexts `b` and `c`.
    AffinePtVec,
    AffinePtVecAssign,
}

impl Product {
    const ALL: [Self; 18] = [
        Self::Mul,
        Self::MulAssign,
        Self::MulPrepared,
        Self::Square,
        Self::SquareAssign,
        Self::PtVec,
        Self::PtVecAssign,
        Self::PtConst,
        Self::PtConstAssign,
        Self::Dot,
        Self::MulAddCt,
        Self::MulSubCt,
        Self::MulAddPtVec,
        Self::MulSubPtVec,
        Self::MulAddPtConst,
        Self::MulSubPtConst,
        Self::AffinePtVec,
        Self::AffinePtVecAssign,
    ];
}

const STEPS: usize = 3;
/// Exactly representable at the constant plaintext's 8 bits, so that only the
/// rescale rounds.
const CONSTANT: f64 = 0.75;

fn encrypt_batch<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE, Rg>,
    slot_layout: SlotLayout,
    values: (&[F], &[F]),
    k: usize,
    seed: u8,
) -> Result<CKKSCiphertextOwned<BE, Rg>>
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let mut pt = plaintext(params, module, slot_layout, values.0.len());
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
    let mut acc = encrypt_batch(params, module, &sk, slot_layout, (&want.0, &want.1), k, 1)?;
    let cst = ckks_pt_cst_full::<BE, F, Rg>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        params.n / 2,
        Some(CONSTANT),
        None,
    );
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
        let fresh = encrypt_batch(params, module, &sk, slot_layout, (&operand.0, &operand.1), k, step as u8 + 1)?;
        let mut accumulator = encrypt_batch(params, module, &sk, slot_layout, (&addend.0, &addend.1), k, step as u8 + 64)?;
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
            Product::PtVec => {
                with_scratch::<BE, _>(module.ckks_mul_pt_vec_tmp_bytes(&out, &acc, &pt), |s| {
                    module.ckks_mul_pt_vec_into(&mut out, &acc, &pt, s)
                })?;
                acc = out;
                times_operand
            }
            Product::PtVecAssign => {
                with_scratch::<BE, _>(module.ckks_mul_pt_vec_tmp_bytes(&acc, &acc, &pt), |s| {
                    module.ckks_mul_pt_vec_assign(&mut acc, &pt, s)
                })?;
                times_operand
            }
            Product::PtConst => {
                with_scratch::<BE, _>(module.ckks_mul_pt_const_tmp_bytes(&out, &acc, &cst), |s| {
                    module.ckks_mul_pt_const_into(&mut out, &acc, &cst, 0, s)
                })?;
                acc = out;
                times_constant
            }
            Product::PtConstAssign => {
                with_scratch::<BE, _>(module.ckks_mul_pt_const_tmp_bytes(&acc, &acc, &cst), |s| {
                    module.ckks_mul_pt_const_assign(&mut acc, &cst, 0, s)
                })?;
                times_constant
            }
            Product::Dot => {
                with_scratch::<BE, _>(module.ckks_dot_product_ct_tmp_bytes(2, &out, &acc, &fresh, &tsk), |s| {
                    module.ckks_dot_product_ct(&mut out, &[&acc, &fresh], &[&fresh, &acc], &tsk, s)
                })?;
                acc = out;
                add(times_operand.clone(), &times_operand, one)
            }
            Product::MulAddCt | Product::MulSubCt => {
                let bytes = module.ckks_mul_add_ct_tmp_bytes(&accumulator, &acc, &fresh, &tsk);
                with_scratch::<BE, _>(bytes, |s| {
                    if matches!(product, Product::MulAddCt) {
                        module.ckks_mul_add_ct_into(&mut accumulator, &acc, &fresh, &tsk, s)
                    } else {
                        module.ckks_mul_sub_ct_into(&mut accumulator, &acc, &fresh, &tsk, s)
                    }
                })?;
                acc = accumulator;
                add(
                    addend,
                    &times_operand,
                    if matches!(product, Product::MulAddCt) { one } else { minus },
                )
            }
            Product::MulAddPtVec | Product::MulSubPtVec => {
                with_scratch::<BE, _>(module.ckks_mul_add_pt_vec_tmp_bytes(&accumulator, &acc, &pt), |s| {
                    if matches!(product, Product::MulAddPtVec) {
                        module.ckks_mul_add_pt_vec_into(&mut accumulator, &acc, &pt, s)
                    } else {
                        module.ckks_mul_sub_pt_vec_into(&mut accumulator, &acc, &pt, s)
                    }
                })?;
                acc = accumulator;
                add(
                    addend,
                    &times_operand,
                    if matches!(product, Product::MulAddPtVec) { one } else { minus },
                )
            }
            Product::MulAddPtConst | Product::MulSubPtConst => {
                with_scratch::<BE, _>(module.ckks_mul_add_pt_const_tmp_bytes(&accumulator, &acc, &cst), |s| {
                    if matches!(product, Product::MulAddPtConst) {
                        module.ckks_mul_add_pt_const_into(&mut accumulator, &acc, &cst, 0, s)
                    } else {
                        module.ckks_mul_sub_pt_const_into(&mut accumulator, &acc, &cst, 0, s)
                    }
                })?;
                acc = accumulator;
                add(
                    addend,
                    &times_constant,
                    if matches!(product, Product::MulAddPtConst) {
                        one
                    } else {
                        minus
                    },
                )
            }
            Product::AffinePtVec => {
                with_scratch::<BE, _>(module.ckks_affine_pt_vec_tmp_bytes(&out, &acc, &pt), |s| {
                    module.ckks_affine_pt_vec_into(&mut out, &acc, &pt, &offset, s)
                })?;
                acc = out;
                add(times_operand, &addend, one)
            }
            Product::AffinePtVecAssign => {
                with_scratch::<BE, _>(module.ckks_affine_pt_vec_tmp_bytes(&acc, &acc, &pt), |s| {
                    module.ckks_affine_pt_vec_assign(&mut acc, &pt, &offset, s)
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
        k,
        1,
    )?;
    let full = encrypt_batch(&params, module, &sk, SlotLayout::Full, (&values.0, &values.1), k, 2)?;
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
        "ckks_mul_pt_vec_into",
        module.ckks_mul_pt_vec_into(&mut out, &shaped, &pt, s),
        [out, shaped, pt]
    );
    refused!(
        Rg,
        "ckks_add_pt_vec_into",
        module.ckks_add_pt_vec_into(&mut out, &shaped, &pt, s),
        [out, shaped, pt]
    );
    let cst = ckks_pt_cst_full::<BE, F, Rg>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        params.n / 2,
        Some(CONSTANT),
        None,
    );
    refused!(
        Rg,
        "ckks_add_pt_const_into",
        module.ckks_add_pt_const_into(&mut out, 0, &shaped, &cst, 0, s),
        [out, shaped]
    );
    module.ckks_mul_pt_const_into(&mut out, &shaped, &cst, 0, s)?;

    // The sum of many terms checks every layout before the first addition writes `out`.
    let shaped_too = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
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
        "ckks_mul_add_ct_into",
        module.ckks_mul_add_ct_into(&mut acc, &full, &full, &tsk, s),
        [acc, full]
    );
    refused!(
        Rg,
        "ckks_mul_sub_pt_vec_into",
        module.ckks_mul_sub_pt_vec_into(&mut acc, &full, &pt, s),
        [acc, full, pt]
    );
    refused!(
        Rg,
        "ckks_mul_add_pt_const_into",
        module.ckks_mul_add_pt_const_into(&mut acc, &full, &cst, 0, s),
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
        k,
        1,
    )?;
    let mut out = alloc_ct(&params, module, k);
    let cst = ckks_pt_cst_full::<BE, F, Rg>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        params.n / 2,
        Some(CONSTANT),
        None,
    );
    refused!(
        Rg,
        "ckks_affine_pt_const_into",
        module.ckks_affine_pt_const_into(&mut out, &shaped, &cst, 0, 0, &mut scratch.borrow()),
        [out, shaped]
    );

    let mut pt = plaintext(&params, module, SlotLayout::Full, values.0.len());
    encoded(module, &mut pt, &values.0, &values.1)?;
    let left = module.ckks_left_operand_prepare_alloc(&shaped, PrepareHint::OneShot, &mut scratch.borrow())?;
    let right = module.ckks_right_operand_prepare_alloc(&pt, PrepareHint::OneShot, &mut scratch.borrow())?;
    refused!(
        Rg,
        "ckks_dot_product_prepared_pt_vec",
        module.ckks_dot_product_prepared_pt_vec(&mut out, &[&left], &[&right], &[1], &mut scratch.borrow()),
        [out]
    );

    let gal = module.ckks_galois_element::<Rg>(1);
    let atks = HashMap::from([(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()))]);
    let mut babies = LinearTransformationBabySteps::alloc(module, &[1], &shaped);
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
        k,
        3,
    )?;
    acc.set_log_delta(LOG_DELTA - 4);
    refused!(
        Rg,
        "ckks_add_pt_vec_assign",
        module.ckks_add_pt_vec_assign(&mut acc, &fine, &mut scratch.borrow()),
        [acc, fine]
    );

    {
        let before = snapshot::<BE, _, _>(&out);
        let source_before = snapshot::<BE, _, _>(&acc);
        let pt_before = snapshot::<BE, _, _>(&fine);
        for add in [true, false] {
            let bytes = if add {
                module.ckks_add_pt_vec_tmp_bytes(out.size())
            } else {
                module.ckks_sub_pt_vec_tmp_bytes(out.size())
            };
            let err = with_scratch::<BE, _>(bytes, |scratch| {
                if add {
                    module.ckks_add_pt_vec_into(&mut out, &acc, &fine, scratch)
                } else {
                    module.ckks_sub_pt_vec_into(&mut out, &acc, &fine, scratch)
                }
            })
            .unwrap_err();
            let expected_op = if add { "ckks_add_pt_vec_into" } else { "ckks_sub_pt_vec_into" };
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
    let x = encrypt_batch(params, module, &sk, slot_layout, (&xs, &zero), k, 1)?;
    let out = alloc_ct(params, module, k);
    let coeff_prec = ckks_spec(params.n, params.base2k, LOG_DELTA, 10);
    let scratch = ScratchOwned::<BE>::alloc(module.ckks_all_ops_tmp_bytes(&x, &coeff_prec, &params.tsk_layout()));
    let encoding = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(params.n));
    Ok(PolynomialSetup {
        sk,
        tsk,
        xs,
        x,
        out,
        coeff_meta: CoeffsMeta::from(coeff_prec),
        scratch,
        encoding,
    })
}

/// Precision of a real polynomial evaluated on real inputs in `[-1, 1]`, with
/// constants at the ciphertext scale: scalar constants on a full layout, and
/// masked constants on a replicated layout.
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
    let s = &mut setup.scratch.borrow();
    match slot_layout {
        SlotLayout::Replicated(layout) => {
            let bsgs = encode_bsgs_replicated(
                &poly,
                module,
                base2k,
                setup.coeff_meta,
                SplitStrategy::MinDepth,
                layout,
                &mut setup.encoding.borrow(),
            )?;
            module.ckks_eval_poly_real_const_coeffs(&mut setup.out, &setup.x, &bsgs, &setup.tsk, s)?;
        }
        SlotLayout::Full => {
            let bsgs = poly
                .encode_bsgs(host_module, base2k, setup.coeff_meta)?
                .map_baby_steps_ref(|pt| upload_pt(module, pt));
            module.ckks_eval_poly_real_const_coeffs(&mut setup.out, &setup.x, &bsgs, &setup.tsk, s)?;
        }
    }
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
    let s = &mut setup.scratch.borrow();
    match slot_layout {
        SlotLayout::Replicated(layout) => {
            let bsgs = encode_complex_bsgs_replicated(
                &poly,
                module,
                base2k,
                setup.coeff_meta,
                SplitStrategy::MinDepth,
                layout,
                &mut setup.encoding.borrow(),
            )?;
            module.ckks_eval_poly_complex_const_coeffs(&mut setup.out, &setup.x, &bsgs, &setup.tsk, s)?;
        }
        SlotLayout::Full => {
            let bsgs = poly
                .encode_bsgs(host_module, base2k, setup.coeff_meta)?
                .map_baby_steps_ref(|pt| upload_pt(module, pt));
            module.ckks_eval_poly_complex_const_coeffs(&mut setup.out, &setup.x, &bsgs, &setup.tsk, s)?;
        }
    }
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
        k,
        1,
    )?;
    let narrower = ShapingUnit::U16_20.layout(LOG_K)?;
    let identity = Polynomial::new(Basis::Monomial, vec![F::zero(), F::one()]);
    let mut encoding = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(params.n));
    let bsgs = encode_bsgs_replicated(
        &identity,
        module,
        params.base2k.into(),
        CoeffsMeta::from_delta_budget(LOG_DELTA, 10),
        SplitStrategy::MinDepth,
        narrower,
        &mut encoding.borrow(),
    )?;
    let mut out = alloc_ct(&params, module, k);
    // Refused at entry, before any work or scratch check.
    refused!(
        Rg,
        "ckks_eval_poly_real_const_coeffs",
        with_scratch::<BE, _>(0, |s| module.ckks_eval_poly_real_const_coeffs(&mut out, &x, &bsgs, &tsk, s)),
        [out, x]
    );
    // Plain constants, without their masked plaintexts.
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
    let degree = crate::encoding::linear_transformation::period_degree::<BE, _, Rg>(module, 1 << layout().log_r());
    let encoding = &mut setup.encoding;
    let bsgs = crate::layouts::eval_mod::encode_bsgs_with(&poly, SplitStrategy::MinDepth, false, &mut |values: &[F]| {
        let scalars = crate::layouts::eval_mod::encode_scalar_constants::<BE, F, Rg>(
            module,
            base2k,
            coeff_meta,
            values,
            &mut encoding.borrow(),
        )?;
        let masked = values
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
            .collect::<Result<Vec<_>>>()?;
        Ok(crate::layouts::MaskedCoeffs::new(scalars, masked))
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
    let a = encrypt_batch(&params, module, &sk, slot_layout, (&a_values.0, &a_values.1), k, 1)?;
    let b = encrypt_batch(&params, module, &sk, slot_layout, (&b_values.0, &b_values.1), k, 2)?;
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
    let ct = encrypt_batch(params, module, &sk, slot_layout, (&values.0, &values.1), k, 1)?;
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
    let ct = encrypt_batch(params, module, &sk, slot_layout, (&values.0, &values.1), k, 1)?;
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
        3 * LOG_DELTA,
        1,
    )?;
    with_scratch::<BE, _>(module.ckks_add_pt_vec_tmp_bytes(ct.max_size()), |s| {
        module.ckks_add_pt_vec_assign(&mut ct, &cst, s)
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
    Ok(())
}

/// Full-layout encryption of `values` with period `values.len()`.
fn encrypt_full<BE, F, Rg: Ring>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE, Rg>,
    values: (&[F], &[F]),
    log_delta: usize,
    k: usize,
) -> Result<CKKSCiphertextOwned<BE, Rg>>
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg>,
{
    let mut pt = module.ckks_plaintext_alloc_from_slots::<Rg>(
        slots_of::<BE, Rg>(module, SlotLayout::Full, values.0.len()),
        Base2K(params.base2k as u32),
        TorusPrecision((log_delta + 4) as u32),
    );
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(log_delta),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout: SlotLayout::Full,
    });
    encoded(module, &mut pt, values.0, values.1)?;
    let mut ct = alloc_ct(params, module, k);
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_encrypt_sk_tmp_bytes(&ct));
    let (mut xe, mut xa) = (Source::new([9; 32]), Source::new([10; 32]));
    module.ckks_encrypt_sk(&mut ct, &pt, sk, &mut xe, &mut xa, &mut scratch.borrow())?;
    Ok(ct)
}

pub fn test_replicated_conversions_round_trip<BE, F, Rg: Ring>(
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
        + CKKSReplicatedOps<BE, Rg>
        + CKKSLinearTransformationOps<BE, Rg>
        + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    conversions_round_trip_with::<BE, F, Rg>(params, module, host_module, None)
}

/// [`test_replicated_conversions_round_trip`] with the conversions factored
/// into placements of two-bit groups.
pub fn test_replicated_conversions_round_trip_factored<BE, F, Rg: Ring>(
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
        + CKKSReplicatedOps<BE, Rg>
        + CKKSLinearTransformationOps<BE, Rg>
        + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    conversions_round_trip_with::<BE, F, Rg>(params, module, host_module, Some(2))
}

fn conversions_round_trip_with<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    placement_factors: Option<usize>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, Rg>
        + CKKSEncodingOps<BE, F, Rg>
        + CKKSReplicatedOps<BE, Rg>
        + CKKSLinearTransformationOps<BE, Rg>
        + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    // `R = 8` divides `k = 8`. The return lands on `20 + extra + log2(S/k)`
    // for the `S` slots of the ring.
    let (log_k, log_delta_ckks, log_delta, log_delta_diag) = (3u8, 40, 20, 30);
    let extra_bits = log_delta_ckks - log_delta - (module.ckks_max_slots::<Rg>() >> log_k).trailing_zeros() as usize;
    let layout = ShapingUnit::U16_40.layout(log_k)?;
    let len = 1usize << log_k;
    let base2k = Base2K(params.base2k as u32);
    // The inner factors of `to` are quantized plainly, a few bits above the working precision.
    let inner_log_delta_diag = log_delta + extra_bits + 4;
    let mut setup = ScratchOwned::<BE>::alloc(ReplicatedConversion::<BE, Rg>::new_tmp_bytes::<F>(
        module,
        &layout,
        base2k,
        log_delta_diag.max(inner_log_delta_diag),
    )?);
    let conv = match placement_factors {
        None => ReplicatedConversion::<BE, Rg>::new::<F>(
            module,
            layout,
            base2k,
            log_delta,
            log_delta_diag,
            extra_bits,
            &mut setup.borrow(),
        )?,
        Some(placement_factors) => ReplicatedConversion::<BE, Rg>::new_factored::<F>(
            module,
            layout,
            base2k,
            log_delta,
            log_delta_diag,
            inner_log_delta_diag,
            extra_bits,
            placement_factors,
            &mut setup.borrow(),
        )?,
    };
    let from_bits_consumed = conv.from.as_ref().map_or(0, ReplicatedTransform::consumed_bits);
    // Room for both conversions and a margin of one diagonal scale, and keys covering it.
    let k = log_delta_ckks + conv.to.consumed_bits() + from_bits_consumed + log_delta_diag;
    let params = CKKSTestParams {
        k: params.k.max(k + extra_bits + conv.trace.len()),
        ..params
    };

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch::<BE, Rg>(&params, module);
    let mut atks = HashMap::new();
    for r in conv.rotations() {
        let gal = module.ckks_galois_element::<Rg>(r);
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }

    let values = unit_batch::<Rg, F>(len, 0);
    let ckks = encrypt_full(&params, module, &sk, (&values.0, &values.1), log_delta_ckks, k)?;
    let fresh = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &ckks, len);
    let fresh_bits = precision((&fresh.0, &fresh.1), (&values.0, &values.1));

    let atk_infos = params.atk_layout();
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_replicated_conversion_tmp_bytes(&ckks, &atk_infos, &conv));
    let mut replicated = alloc_ct(&params, module, k);
    let mut to_workspace = conv.alloc_to_workspace(module, &ckks);
    module.ckks_to_replicated(&mut replicated, &ckks, &conv, &mut to_workspace, &atks, &mut scratch.borrow())?;
    assert_eq!(replicated.slot_layout(), SlotLayout::Replicated(layout));
    assert_eq!(replicated.log_delta(), log_delta);
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &replicated, len);
    let to_bits = precision((&got.0, &got.1), (&values.0, &values.1));
    // The map runs `extra_bits` under the replicated scale, which bounds the
    // result as the fresh encryption does.
    let working_bits = (log_delta + extra_bits) as f64;
    assert!(
        to_bits >= fresh_bits.min(working_bits) - 2.0,
        "to replicated: {to_bits:.2} bits from {fresh_bits:.2}, working at {working_bits}"
    );

    let mut back = alloc_ct(&params, module, k);
    let mut from_workspace = conv.alloc_from_workspace(module, &replicated)?;
    module.ckks_from_replicated(
        &mut back,
        &replicated,
        &conv,
        &mut from_workspace,
        &atks,
        &mut scratch.borrow(),
    )?;
    assert_eq!(back.slot_layout(), SlotLayout::Full);
    assert_eq!(back.log_delta(), log_delta_ckks);
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &back, len);
    let from_bits = precision((&got.0, &got.1), (&values.0, &values.1));
    // The gather's diagonals are quantized plainly, which bounds the return
    // to about `log_delta_diag` bits.
    assert!(
        from_bits >= fresh_bits.min(log_delta_diag as f64) - 3.0,
        "from replicated: {from_bits:.2} bits from {fresh_bits:.2}"
    );

    let mut out = alloc_ct(&params, module, k);
    {
        use crate::test_suite::parity::helpers::snapshot;
        let before = snapshot::<BE, _, Rg>(&out);
        let input_before = snapshot::<BE, _, Rg>(&ckks);
        let replicated_before = snapshot::<BE, _, Rg>(&replicated);
        for (n, rank, reason) in [
            (params.n, ckks.rank().as_usize() + 1, "source and destination ranks differ"),
            (params.n / 2, ckks.rank().as_usize(), "source and destination degrees differ"),
        ] {
            let mut bad = module.ckks_ciphertext_alloc_from_glwe_infos::<_, Rg>(&GLWELayout {
                n: n.into(),
                rank: rank.into(),
                ..ckks.glwe_layout()
            });
            bad.set_meta(ckks.meta());
            let bad_before = snapshot::<BE, _, Rg>(&bad);
            for to in [true, false] {
                let err = if to {
                    module.ckks_to_replicated(&mut bad, &ckks, &conv, &mut to_workspace, &atks, &mut scratch.borrow())
                } else {
                    module.ckks_from_replicated(
                        &mut bad,
                        &replicated,
                        &conv,
                        &mut from_workspace,
                        &atks,
                        &mut scratch.borrow(),
                    )
                }
                .unwrap_err();
                let expected_op = if to { "ckks_to_replicated" } else { "ckks_from_replicated" };
                assert!(
                    matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op, reason: actual }) if *op == expected_op && actual == reason)
                );
                assert_eq!(snapshot::<BE, _, Rg>(&bad), bad_before);
                assert_eq!(snapshot::<BE, _, Rg>(&ckks), input_before);
                assert_eq!(snapshot::<BE, _, Rg>(&replicated), replicated_before);
            }
        }
        // Operands below the module's degree, which the compiled slots do not fit.
        let half = GLWELayout {
            n: (params.n / 2).into(),
            ..ckks.glwe_layout()
        };
        let mut half_src = module.ckks_ciphertext_alloc_from_glwe_infos::<_, Rg>(&half);
        half_src.set_meta(ckks.meta());
        let mut half_dst = module.ckks_ciphertext_alloc_from_glwe_infos::<_, Rg>(&half);
        half_dst.set_meta(ckks.meta());
        let half_before = snapshot::<BE, _, Rg>(&half_dst);
        let err = module
            .ckks_to_replicated(
                &mut half_dst,
                &half_src,
                &conv,
                &mut to_workspace,
                &atks,
                &mut scratch.borrow(),
            )
            .unwrap_err();
        assert!(
            matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op: "ckks_to_replicated", reason }) if reason == "takes ciphertexts of the module's degree")
        );
        assert_eq!(snapshot::<BE, _, Rg>(&half_dst), half_before);
        let mut wrong_width = conv.alloc_to_workspace(
            module,
            &GLWELayout {
                k: (ckks.k().as_usize() - 1).into(),
                ..ckks.glwe_layout()
            },
        );
        let mut wrong_rotations = conv.alloc_to_workspace(module, &ckks);
        wrong_rotations.babies[0] = LinearTransformationBabySteps::alloc(module, &[1], &wrong_rotations.input);
        for workspace in [&mut wrong_width, &mut wrong_rotations] {
            let err = module
                .ckks_to_replicated(&mut out, &ckks, &conv, workspace, &atks, &mut scratch.borrow())
                .unwrap_err();
            assert!(
                matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op: "ckks_to_replicated", reason }) if reason == "workspace does not match the transform input layout or rotations")
            );
            assert_eq!(snapshot::<BE, _, Rg>(&out), before);
            assert_eq!(snapshot::<BE, _, Rg>(&ckks), input_before);
        }
    }
    assert!(
        module
            .ckks_to_replicated(&mut out, &replicated, &conv, &mut to_workspace, &atks, &mut scratch.borrow())
            .is_err()
    );
    assert!(
        module
            .ckks_from_replicated(&mut out, &ckks, &conv, &mut from_workspace, &atks, &mut scratch.borrow())
            .is_err()
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

pub fn test_replicated_transforms_apply_matrices<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>:
        TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSLinearTransformationOps<BE, Rg> + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    let log_k = 3u8;
    let (len, base2k) = (1usize << log_k, Base2K(params.base2k as u32));
    let layout = ShapingUnit::U16_40.layout(log_k)?;
    let (log_delta_diag, extra_bits) = (36, 16);
    let mut setup = ScratchOwned::<BE>::alloc(
        ReplicatedTransform::<BE, Rg>::encode_tmp_bytes::<F>(module, &layout, base2k, 14)?.max(
            ReplicatedTransform::<BE, Rg>::encode_tmp_bytes::<F>(module, &layout, base2k, log_delta_diag)?,
        ),
    );
    if real_slots::<Rg>() {
        let imaginary = |_: usize, _: usize| (F::zero(), F::one());
        assert!(
            ReplicatedTransform::<BE, Rg>::from_period(module, layout, base2k, 14, true, imaginary, &mut setup.borrow()).is_err()
        );
    }
    // Shaped diagonals far coarser than the result: each BSGS giant step needs
    // its own rotated shaping polynomial to keep their error off the data slots.
    let to = ReplicatedTransform::<BE, Rg>::from_period(
        module,
        layout,
        base2k,
        14,
        true,
        |s, j| {
            if s % (1 << layout.log_r()) == 0 {
                matrix::<Rg, F>((s >> layout.log_r()) % len, j)
            } else {
                (F::zero(), F::zero())
            }
        },
        &mut setup.borrow(),
    )?;
    let gather = ReplicatedTransform::<BE, Rg>::gather(
        module,
        layout,
        base2k,
        log_delta_diag,
        false,
        matrix::<Rg, F>,
        &mut setup.borrow(),
    )?;
    let trace: Vec<i64> = (0..(module.ckks_max_slots::<Rg>() / len).trailing_zeros())
        .map(|l| (len << l) as i64)
        .collect();

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch::<BE, Rg>(&params, module);
    let mut atks = HashMap::new();
    for r in to
        .rotations()
        .into_iter()
        .chain(gather.rotations())
        .chain(trace.iter().copied())
    {
        let gal = module.ckks_galois_element::<Rg>(r);
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }
    let values = unit_batch::<Rg, F>(len, 0);
    let want = apply_matrix::<Rg, F>((&values.0, &values.1));
    let k = 40 + 2 * log_delta_diag;
    let ckks = encrypt_full(&params, module, &sk, (&values.0, &values.1), 40, k)?;
    let wide = CKKSLayout {
        glwe_layout: GLWELayout {
            k: (k + extra_bits + trace.len()).into(),
            ..ckks.glwe_layout()
        },
        meta: ckks.meta(),
    };
    let buffers = 3 * BE::scratch_aligned(BE::vec_znx_bytes_of(params.n, wide.rank().as_usize() + 1, wide.size()));
    let atk_infos = params.atk_layout();
    let ops = crate::oep::derived::replicated::transform_tmp_bytes(module, &wide, &atk_infos, &to)
        .max(crate::oep::derived::replicated::transform_tmp_bytes(
            module, &wide, &atk_infos, &gather,
        ))
        .max(module.ckks_rotate_tmp_bytes(&wide, &atk_infos))
        .max(module.ckks_add_tmp_bytes(wide.size()))
        .max(module.ckks_copy_tmp_bytes(&wide, &ckks))
        .max(module.ckks_round_tmp_bytes(&ckks, &wide))
        .max(module.ckks_div_pow2_tmp_bytes(wide.size()));
    let mut scratch = ScratchOwned::<BE>::alloc(buffers + ops);
    let mut onto = alloc_ct(&params, module, k);
    let mut to_workspace = to.alloc_workspace(
        module,
        &GLWELayout {
            k: (ckks.k().as_usize() + extra_bits).into(),
            ..ckks.glwe_layout()
        },
    );
    wide_onto_replicated(
        module,
        &mut onto,
        &ckks,
        &to,
        &mut to_workspace,
        layout,
        LOG_DELTA,
        extra_bits,
        &atks,
        &mut scratch.borrow(),
    )?;
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &onto, len);
    let onto_bits = precision((&got.0, &got.1), (&want.0, &want.1));
    assert!(onto_bits >= 30.0, "period-k map onto the data slots: {onto_bits:.2} bits");

    let replicated = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout),
        (&values.0, &values.1),
        k,
        3,
    )?;
    let mut gathered = alloc_ct(&params, module, k);
    let mut gather_workspace = gather.alloc_workspace(
        module,
        &GLWELayout {
            k: (replicated.k().as_usize() + extra_bits).into(),
            ..replicated.glwe_layout()
        },
    );
    wide_gather_trace(
        module,
        &mut gathered,
        &replicated,
        &gather,
        &mut gather_workspace,
        &trace,
        extra_bits,
        &atks,
        &mut scratch.borrow(),
    )?;
    let got = decrypt_batch::<BE, F, Rg>(&params, module, &sk, &gathered, len);
    let gather_bits = precision((&got.0, &got.1), (&want.0, &want.1));
    // Plainly quantized diagonals weighted `R·M(c, j)` over `k` terms.
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
    // The evaluation runs on exactly its query's bytes; the host polynomials and range-extension scale come back.
    macro_rules! run {
        ($eval_mod:expr) => {{
            let eval_mod = $eval_mod;
            with_scratch::<BE, _>(module.ckks_eval_mod_tmp_bytes(&res, &ct, &eval_mod, &tsk_layout), |s| {
                module.ckks_eval_mod(&mut res, &mut ct, &eval_mod, &tsk, s)
            })?;
            let scale = eval_mod.range_extension_scale();
            (eval_mod.f_mod_poly, eval_mod.f_mod_inv_poly, scale)
        }};
    }
    let (f_mod_poly, f_mod_inv_poly, scale) =
        match slot_layout {
            SlotLayout::Replicated(layout) if matches!(mode, EvalModMode::Packed) => run!(
                ckks_eval_mod_replicated_packed_compile::<BE, F, Rg>(base2k, plan, layout, module, &mut compile.borrow())?
            ),
            SlotLayout::Replicated(layout) => run!(ckks_eval_mod_replicated_compile::<BE, F, Rg>(
                base2k,
                plan,
                layout,
                module,
                &mut compile.borrow()
            )?),
            SlotLayout::Full => run!(ckks_eval_mod_compile::<BE, F, Rg>(
                base2k,
                plan,
                module,
                &mut compile.borrow()
            )?),
        };
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
    // Real slots have no imaginary parts to pack a second batch on.
    if real_slots::<Rg>() {
        let mut compile = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(params.n));
        let err = ckks_eval_mod_replicated_packed_compile::<BE, F, Rg>(
            Base2K(params.base2k as u32),
            eval_mod_plan(),
            ShapingUnit::U32_45.layout(LOG_K + 1)?,
            module,
            &mut compile.borrow(),
        )
        .err()
        .expect("a packed EvalMod on real slots");
        assert_invalid_encoding(&err, "ckks_eval_mod_replicated_packed_compile");
        return Ok(());
    }
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

/// One bootstrapping of `k = 8` values with the parameters of the paper, from
/// the leveled scale, with full-width and row-offset return keys.
pub fn test_replicated_bootstrap<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend + CKKSEncapsulatedModUpImpl,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWETrace<BE>
        + GLWECIKeyPreparedFactory<BE>
        + poulpy_core::api::GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>,
    F: TestScalar,
    E: ReimNew<F>,
{
    replicated_bootstrap_with::<BE, F>(
        params,
        module,
        host_module,
        ShapingUnit::U16_40,
        ReplicatedTransforms::Dense,
        19.0,
    )
}

/// [`test_replicated_bootstrap`] at the periods `R = 4` and `R = 2`, whose
/// precision the units' suppression bounds: 41 bits for `U8_40`, 15 for `U4_15`.
pub fn test_replicated_bootstrap_small_periods<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend + CKKSEncapsulatedModUpImpl,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWETrace<BE>
        + GLWECIKeyPreparedFactory<BE>
        + poulpy_core::api::GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>,
    F: TestScalar,
    E: ReimNew<F>,
{
    // `U4_15` suppresses 15.26 bits, which caps its precision: the noise draws move it by a few tenths.
    for (unit, min_bits) in [(ShapingUnit::U8_40, 21.0), (ShapingUnit::U4_15, 14.5)] {
        replicated_bootstrap_with::<BE, F>(params, module, host_module, unit, ReplicatedTransforms::Dense, min_bits)?;
    }
    Ok(())
}

/// [`test_replicated_bootstrap`] with `SlotsToCoeffs` and `CoeffsToSlots`
/// factored: two DFT factors each, the placements in groups of two bits.
pub fn test_replicated_bootstrap_factored<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend + CKKSEncapsulatedModUpImpl,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWETrace<BE>
        + GLWECIKeyPreparedFactory<BE>
        + poulpy_core::api::GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let transforms = ReplicatedTransforms::Factored {
        s2c_factors: 2,
        c2s_factors: 2,
        placement_factors: 2,
        c2s_inner_log_delta_diag: 44,
    };
    replicated_bootstrap_with::<BE, F>(params, module, host_module, ShapingUnit::U16_40, transforms, 19.0)
}

/// One bootstrapping of `k = 8` values of `unit`, with full-width and
/// row-offset return keys, each at least `min_bits` precise.
fn replicated_bootstrap_with<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    unit: ShapingUnit,
    transforms: ReplicatedTransforms,
    min_bits: f64,
) -> Result<()>
where
    BE: TestContextBackend + CKKSEncapsulatedModUpImpl,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWETrace<BE>
        + GLWECIKeyPreparedFactory<BE>
        + poulpy_core::api::GLWETensorKeyPreparedFactory<BE, ConjugateInvariant>,
    F: TestScalar,
{
    let layout = unit.layout(3)?;
    let plan = ReplicatedBootstrapPlan {
        transforms,
        ..ReplicatedBootstrapPlan::paper(layout)
    };
    let base2k = Base2K(params.base2k as u32);
    let mut compile = ScratchOwned::<BE>::alloc(ReplicatedBootstrapping::<BE, F>::new_tmp_bytes(module, &plan, base2k)?);
    let ctx = ReplicatedBootstrapping::<BE, F>::new(module, plan, base2k, &mut compile.borrow())?;
    assert!(ctx.is_packed());
    let k_out = plan.k_in() + plan.level_log_delta;
    let key_layouts = plan.key_layouts(
        &ctx.galois_elements(module),
        params.n.into(),
        base2k,
        k_out,
        params.base2k,
        params.base2k + 1,
        true,
    );

    let (sk_raw, sk) = gen_sk_with_raw::<_, Standard>(&params, module, host_module, [0u8; 32]);

    let len = 1 << layout.log_k();
    let values = unit_batch::<Standard, F>(len, 0);
    let input = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout),
        (&values.0, &values.1),
        plan.k_in() - (plan.log_delta - plan.level_log_delta),
        1,
    )?;
    let mut full_layouts = key_layouts.clone();
    let full = full_layouts.encapsulation.as_mut().unwrap().sparse_to_dense.as_mut().unwrap();
    full.k_offset = 0usize.into();
    full.dnum = poulpy_core::layouts::GGLWELayout::dnum_for_input(plan.k_boot(k_out).into(), full.k_digit);
    let mut decoded = Vec::new();
    for key_layouts in [&full_layouts, &key_layouts] {
        let (mut xs, mut xe, mut xa) = (Source::new([3; 32]), Source::new([4; 32]), Source::new([5; 32]));
        let mut keygen = ScratchOwned::<BE>::alloc(
            ctx.generate_keys_tmp_bytes(module, key_layouts)
                .max(key_layouts.prepare_tmp_bytes(module)),
        );
        let keys = ctx
            .generate_keys(
                module,
                &sk_raw,
                key_layouts,
                16,
                &mut xs,
                &mut xe,
                &mut xa,
                &mut keygen.borrow(),
            )
            .prepare(module, &mut keygen.borrow())?;
        let encapsulation = key_layouts.encapsulation.as_ref().unwrap();
        let mut out = alloc_ct::<_, Standard>(&params, module, k_out);
        let mut boot_scratch = ScratchOwned::<BE>::alloc(module.ckks_replicated_bootstrap_tmp_bytes(
            &out,
            &input,
            &ctx,
            key_layouts.rotation_keys.values().next().expect("the bootstrapping rotates"),
            &key_layouts.tensor_key,
            Some((
                &encapsulation.dense_to_sparse,
                encapsulation.sparse_to_dense.as_ref().unwrap(),
            )),
        ));
        let mut workspace = ctx.alloc_workspace(module, &out);
        module.ckks_replicated_bootstrap(&mut out, &input, &ctx, &mut workspace, &keys, &mut boot_scratch.borrow())?;
        assert_eq!(out.slot_layout(), SlotLayout::Replicated(layout));
        assert_eq!((out.log_delta(), out.k().as_usize()), (plan.level_log_delta, k_out));
        let got = decrypt_batch::<BE, F, Standard>(&params, module, &sk, &out, len);
        let bits = precision((&got.0, &got.1), (&values.0, &values.1));
        assert!(bits >= min_bits, "replicated bootstrapping of {unit:?}: {bits:.2} bits");
        decoded.push(got);
    }
    let bits = precision((&decoded[0].0, &decoded[0].1), (&decoded[1].0, &decoded[1].1));
    assert!(
        bits >= min_bits - 1.0,
        "offset and full-width replicated bootstrapping differ by {bits:.2} bits"
    );
    Ok(())
}

/// [`test_replicated_bootstrap`] on the conjugate-invariant ring of degree `N`
/// on one `module` of degree `2N`, which encrypts, bootstraps and decrypts the
/// conjugate-invariant operands and holds the standard ones of the bootstrapping.
pub fn test_replicated_ci_bootstrap<BE>(params: CKKSTestParams, module: &Module<BE>) -> Result<()>
where
    BE: TestContextBackend + CKKSEncapsulatedModUpImpl,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant>
        + CKKSEncodingOps<BE, f64, ConjugateInvariant>
        + TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSReplicatedCIBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + GLWETrace<BE>,
{
    assert_eq!(module.n(), 2 * params.n, "the module serves both rings at degrees N and 2N");
    let standard_params = CKKSTestParams {
        n: 2 * params.n,
        ..params
    };
    let host = Module::<HostBytesBackend>::new(standard_params.n as u64);
    let layout = ShapingUnit::U16_40.layout(3)?;
    let plan = ReplicatedBootstrapPlan::paper(layout);
    let base2k = Base2K(params.base2k as u32);
    let mut compile = ScratchOwned::<BE>::alloc(ReplicatedBootstrapping::<BE, f64>::new_tmp_bytes(module, &plan, base2k)?);
    let ctx = ReplicatedBootstrapping::<BE, f64>::new(module, plan, base2k, &mut compile.borrow())?;

    // The standard bootstrapping runs one bit wider than the output.
    let k_out = plan.k_in() + plan.level_log_delta;
    let key_layouts = plan.key_layouts(
        &ctx.galois_elements(module),
        standard_params.n.into(),
        base2k,
        k_out + 1,
        params.base2k,
        params.base2k + 1,
        true,
    );
    let (ci_sk_raw, ci_sk) = gen_sk_with_raw::<_, ConjugateInvariant>(&params, module, &host, [0u8; 32]);
    let (sk_raw, _) = gen_sk_with_raw::<_, Standard>(&standard_params, module, &host, [1u8; 32]);

    let len = 1 << layout.log_k();
    let values = unit_batch::<ConjugateInvariant, f64>(len, 0);
    let k_in = plan.k_in() - (plan.log_delta - plan.level_log_delta);
    let input = encrypt_batch(
        &params,
        module,
        &ci_sk,
        SlotLayout::Replicated(layout),
        (&values.0, &values.1),
        k_in,
        1,
    )?;

    // The conversion keys cover the ring maps at the shaping depth.
    let profile = crate::shaping::ShapingProfile::new(&layout);
    let width = (k_in + profile.shaping_depth(input.log_delta())).max(k_out + 1 + profile.shaping_depth(plan.level_log_delta));
    let conversion_layout = standard_params.ksk_layout(width);
    let mut keygen = ScratchOwned::<BE>::alloc(
        module
            .glwe_ci_key_encrypt_sk_tmp_bytes(&conversion_layout)
            .max(module.glwe_ci_key_prepare_tmp_bytes(&conversion_layout))
            .max(ctx.generate_keys_tmp_bytes(module, &key_layouts))
            .max(key_layouts.prepare_tmp_bytes(module)),
    );
    let (embed_key, trace_key) = gen_ci_keys(&standard_params, module, &ci_sk_raw, &sk_raw, width, &mut keygen.borrow());
    let (mut xs, mut xe, mut xa) = (Source::new([3; 32]), Source::new([4; 32]), Source::new([5; 32]));
    let keys = ctx
        .generate_keys(
            module,
            &sk_raw,
            &key_layouts,
            16,
            &mut xs,
            &mut xe,
            &mut xa,
            &mut keygen.borrow(),
        )
        .prepare(module, &mut keygen.borrow())?;

    let encapsulation = key_layouts.encapsulation.as_ref().unwrap();
    let mut out = alloc_ct::<_, ConjugateInvariant>(&params, module, k_out);
    let bytes = module.ckks_replicated_ci_bootstrap_tmp_bytes(
        &out,
        &input,
        &ctx,
        key_layouts.rotation_keys.values().next().expect("the bootstrapping rotates"),
        &key_layouts.tensor_key,
        Some((
            &encapsulation.dense_to_sparse,
            encapsulation.sparse_to_dense.as_ref().unwrap(),
        )),
        &standard_params.ksk_layout(width),
    );
    let mut boot_scratch = ScratchOwned::<BE>::alloc(bytes);
    let mut workspace = ctx.alloc_workspace(
        module,
        &GLWELayout {
            n: standard_params.n.into(),
            k: (k_out + 1).into(),
            ..out.glwe_layout()
        },
    );
    module.ckks_replicated_ci_bootstrap(
        &mut out,
        &input,
        &ctx,
        &mut workspace,
        &keys,
        (&embed_key, &trace_key),
        &mut boot_scratch.borrow(),
    )?;
    assert_eq!(out.slot_layout(), SlotLayout::Replicated(layout));
    assert_eq!(out.slots(), SlotsKind::Real);
    assert_eq!((out.log_delta(), out.k().as_usize()), (plan.level_log_delta, k_out));
    let got = decrypt_batch::<BE, f64, ConjugateInvariant>(&params, module, &ci_sk, &out, len);
    let bits = precision((&got.0, &got.1), (&values.0, &values.1));
    assert!(bits >= 19.0, "conjugate-invariant replicated bootstrapping: {bits:.2} bits");
    Ok(())
}
