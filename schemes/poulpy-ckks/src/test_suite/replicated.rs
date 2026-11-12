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
//! | [`test_replicated_layouts_do_not_mix`] | sums of different layouts, constant sums and `ckks_add_many`, and a product into a reused replicated destination |
//! | [`test_replicated_conjugation_is_rejected`] | conjugation on the standard ring |
//! | [`test_replicated_plain_rounding_is_rejected`] | operations that round plainly (products, automorphisms, dot products, affine maps, linear transformations, inner sums), and layouts or scales without a shaped encoding |

use std::collections::HashMap;

use anyhow::Result;
use poulpy_core::{
    GLWEMulPreparedPlainSum,
    layouts::{Base2K, GLWEToBackendRef, LWEInfos, LinearTransformationBabySteps, TorusPrecision, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, PrepareHint, Ring, ScratchOwned, Standard, ZnxView},
    source::Source,
};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextSharedModule, TestScalar, alloc_ct, alloc_scratch, ckks_pt_cst_full, gen_atk, gen_sk,
    gen_sk_with_raw, gen_tsk, next_test_seed,
};
use super::parity::helpers::{snapshot, with_scratch};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout,
    SlotsKind,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSConjugateOps, CKKSDecryptOps, CKKSDotProductOps, CKKSEncodingHostOps,
        CKKSEncodingOps, CKKSEncryptOps, CKKSInnerSumOps, CKKSLeftOperandPreparedFactory, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSMulOps, CKKSRightOperandPreparedFactory, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::{CKKSCiphertextOwned, CKKSPlaintextAlloc, CKKSPlaintextOwned},
    shaping::{ShapingRing, ShapingUnit},
    test_suite::CKKSTestParams,
};

const LOG_DELTA: usize = 20;
const LOG_K: u8 = 2;

fn layout() -> ReplicatedLayout {
    ShapingUnit::U32_45.layout(LOG_K).unwrap()
}

/// Whether the ring `Rg` has real slots, where batches have no imaginary part.
fn real_slots<Rg: Ring>() -> bool {
    ShapingRing::of::<Rg>() == ShapingRing::ConjugateInvariant
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
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(slots));
    let read = if pt.slot_layout() == SlotLayout::Full { slots } else { len };
    let (mut re, mut im) = (vec![F::zero(); read], vec![F::zero(); read]);
    module
        .ckks_decode_slots_host_into(pt, &mut re, &mut im, &mut scratch.borrow())
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
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(slots));
    if pt.slot_layout() == SlotLayout::Full {
        let tile = |values: &[F]| values.iter().copied().cycle().take(slots).collect::<Vec<_>>();
        module.ckks_encode_slots_host_into(pt, &tile(re), &tile(im), &mut scratch.borrow())?;
    } else {
        module.ckks_encode_slots_host_into(pt, re, im, &mut scratch.borrow())?;
    }
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

pub fn test_replicated_layouts_do_not_mix<BE, F, Rg: Ring>(
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

    // Products, automorphisms and the sums built on them round plainly, so
    // they refuse a replicated layout, whatever keys they are given.
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let mut replicated_pt = plaintext(&params, module, SlotLayout::Replicated(layout()), values.0.len());
    encoded(module, &mut replicated_pt, &values.0, &values.1)?;
    let mut acc = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        k,
        4,
    )?;
    let period = 1i64 << layout().log_r();
    let s = &mut scratch.borrow();
    refused!(
        Rg,
        "ckks_mul_into",
        module.ckks_mul_into(&mut out, &shaped, &acc, &tsk, s),
        [out, shaped, acc]
    );
    refused!(
        Rg,
        "ckks_mul_assign",
        module.ckks_mul_assign(&mut acc, &shaped, &tsk, s),
        [acc, shaped]
    );
    refused!(
        Rg,
        "ckks_square_into",
        module.ckks_square_into(&mut out, &shaped, &tsk, s),
        [out, shaped]
    );
    refused!(
        Rg,
        "ckks_mul_pt_vec_into",
        module.ckks_mul_pt_vec_into(&mut out, &shaped, &replicated_pt, s),
        [out, shaped, replicated_pt]
    );
    refused!(
        Rg,
        "ckks_mul_pt_const_into",
        module.ckks_mul_pt_const_into(&mut out, &shaped, &cst, 0, s),
        [out, shaped]
    );
    refused!(
        Rg,
        "ckks_dot_product_ct",
        module.ckks_dot_product_ct(&mut out, &[&shaped], &[&acc], &tsk, s),
        [out, shaped, acc]
    );
    refused!(Rg, "ckks_add_pt_const_assign", module.ckks_add_one_assign(&mut acc, s), [acc]);
    refused!(
        Rg,
        "ckks_rotate_into",
        module.ckks_rotate_into(period, &mut out, &shaped, &atks, s),
        [out, shaped]
    );
    refused!(
        Rg,
        "ckks_rotate_assign",
        module.ckks_rotate_assign(period, &mut acc, &atks, s),
        [acc]
    );
    refused!(
        Rg,
        "ckks_inner_sum",
        module.ckks_inner_sum(0, 1, false, &mut out, &shaped, &atks, s),
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

/// Conjugation key-switches with plain rounding, so it refuses a replicated
/// layout on the standard ring, whatever key it is given.
pub fn test_replicated_conjugation_is_rejected<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F> + CKKSConjugateOps<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let values = unit_batch::<Standard, F>(1 << LOG_K, 0);
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
    let mut acc = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        k,
        2,
    )?;
    // The Galois element of the conjugation.
    let gal = -1;
    let atks = HashMap::from([(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()))]);
    let mut out = alloc_ct(&params, module, k);
    let refused = |result: crate::CKKSResult<()>| {
        matches!(
            result.unwrap_err().composition(),
            Some(crate::CKKSCompositionError::InvalidOperand { .. })
        )
    };
    let s = &mut scratch.borrow();
    assert!(refused(module.ckks_conjugate_into(&mut out, &shaped, &atks, s)));
    assert!(refused(module.ckks_conjugate_assign(&mut acc, &atks, s)));
    Ok(())
}
