//! Replicated slot layout: shaped encoding and encryption.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_replicated_encoding_is_shaped`] | shaped quantization keeps the data slots far above the scale's precision |
//! | [`test_replicated_unit_shaping_encodes_plainly`] | `U = 1` encodes as a full layout holding the scattered batch |
//! | [`test_replicated_encryption_is_shaped`] | a fresh encryption keeps the data slots far above CKKS at the same scale |
//! | [`test_replicated_encoding_rejects_batch_size`] | a batch of the wrong size |
//! | [`test_replicated_products_are_shaped`] | chains of every ct × ct and ct × pt product keep the data slots far above CKKS |
//! | [`test_replicated_layouts_do_not_mix`] | products and sums of different layouts, and constant sums |
//! | [`test_replicated_plain_rounding_is_rejected`] | operations that round plainly, and layouts or scales without a shaped encoding |
//! | [`test_replicated_eval_mod_is_shaped`] | EvalMod with masked constants and a shaped last squaring keeps the message far above CKKS |
//! | [`test_replicated_polynomials_are_shaped`] | monomial, Chebyshev and complex polynomials keep the data slots far above CKKS |
//! | [`test_replicated_automorphisms_are_shaped`] | rotations and conjugation keep the data slots far above CKKS |
//! | [`test_replicated_rotations_are_shaped`] | rotations keep the data slots far above CKKS, on either ring |
//! | [`test_replicated_rotations_are_admissible`] | rotations by a non-multiple of the period |
//! | [`test_replicated_masked_constants`] | a masked constant fills the data slots only, and adds to a ciphertext |
//! | [`test_replicated_conversions_round_trip`] | CKKS to the replicated layout and back keeps the input's precision |
//! | [`test_replicated_transforms_apply_matrices`] | a complex matrix onto the data slots, and gathered then traced back to period `k` |

use std::collections::HashMap;

use anyhow::Result;
use poulpy_core::{
    GLWEMulPreparedPlainSum, GLWETrace,
    layouts::{
        Base2K, GLWEInfos, GLWELayout, GLWEToBackendRef, LWEInfos, LinearTransformationBabySteps, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, PrepareHint, ScratchOwned, Standard, ZnxView},
    source::Source,
};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, alloc_ct, alloc_scratch,
    ckks_pt_cst_full, ckks_spec, gen_atk, gen_sk, gen_sk_with_raw, gen_tsk, upload_pt,
};
use crate::{
    CKKSInfos, CoeffsMeta, GLWEPlaintextInfos, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps, CKKSDecryptOps,
        CKKSDotProductOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSEvalModOps, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSReplicatedBootstrapOps, CKKSReplicatedOps, CKKSRotateOps,
        CKKSSubOps,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, EvalModPlan, EvalModPoly, EvalModType, ReplicatedBootstrapPlan,
        ReplicatedBootstrapping, ReplicatedConversion, ReplicatedTransform, compile_eval_mod, compile_eval_mod_replicated,
        compile_eval_mod_replicated_packed, encode_bsgs_replicated, encode_complex_bsgs_replicated,
    },
    oep::CKKSEncapsulatedModUpImpl,
    polynomial::{Basis, ComplexPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
    reference::replicated::{wide_gather_trace, wide_onto_replicated},
    shaping::{ShapingRing, ShapingUnit},
    test_suite::CKKSTestParams,
};

const LOG_DELTA: usize = 20;
const LOG_K: u8 = 2;

fn layout() -> ReplicatedLayout {
    ShapingUnit::U32_45.layout(LOG_K, 24).unwrap()
}

/// Whether the ring of `BE` has real slots, where batches have no imaginary part.
fn real_slots<BE: Backend>() -> bool {
    ShapingRing::of::<BE::Ring>() == ShapingRing::ConjugateInvariant
}

fn batch<BE: Backend, F: TestScalar>(len: usize) -> (Vec<F>, Vec<F>) {
    let value = |x: f64| F::from_f64(x).unwrap();
    let im = if real_slots::<BE>() { 0.0 } else { 0.7 };
    (
        (0..len).map(|i| value(0.9 * (i as f64 + 0.3).cos())).collect(),
        (0..len).map(|i| value(im * (2.0 * i as f64 + 0.1).sin())).collect(),
    )
}

fn plaintext<BE>(params: &CKKSTestParams, module: &Module<BE>, slot_layout: SlotLayout, len: usize) -> CKKSPlaintextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
{
    let mut pt = module.ckks_pt_vec_alloc(
        slots_of(module, slot_layout, len),
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
fn slots_of<BE: Backend>(module: &Module<BE>, slot_layout: SlotLayout, len: usize) -> usize {
    match slot_layout {
        SlotLayout::Full => len.max(module.ckks_slots_of_degree(BE::MIN_DEGREE)),
        SlotLayout::Replicated(layout) => len << layout.log_r(),
    }
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

fn decode<BE, F>(module: &Module<BE>, pt: &CKKSPlaintextOwned<BE>, len: usize) -> (Vec<F>, Vec<F>)
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
{
    let slots = module.ckks_slots_of_degree(pt.n().as_usize());
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(slots));
    let read = if pt.slot_layout() == SlotLayout::Full { slots } else { len };
    let (mut re, mut im) = (vec![F::zero(); read], vec![F::zero(); read]);
    module
        .ckks_decode_reim_into(pt, &mut re, &mut im, &mut scratch.borrow())
        .unwrap();
    re.truncate(len);
    im.truncate(len);
    (re, im)
}

fn encoded<BE, F>(module: &Module<BE>, pt: &mut CKKSPlaintextOwned<BE>, re: &[F], im: &[F]) -> Result<()>
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
{
    let slots = module.ckks_slots_of_degree(pt.n().as_usize());
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(slots));
    if pt.slot_layout() == SlotLayout::Full {
        let tile = |values: &[F]| values.iter().copied().cycle().take(slots).collect::<Vec<_>>();
        module.ckks_encode_reim_into(pt, &tile(re), &tile(im), &mut scratch.borrow())?;
    } else {
        module.ckks_encode_reim_into(pt, re, im, &mut scratch.borrow())?;
    }
    Ok(())
}

pub fn test_replicated_encoding_is_shaped<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let (re, im) = batch::<BE, F>(1 << LOG_K);
    let mut full = plaintext(&params, module, SlotLayout::Full, re.len());
    encoded(module, &mut full, &re, &im)?;
    let mut shaped = plaintext(&params, module, SlotLayout::Replicated(layout()), re.len());
    encoded(module, &mut shaped, &re, &im)?;

    let (full_re, full_im) = decode::<BE, F>(module, &full, re.len());
    let (shaped_re, shaped_im) = decode::<BE, F>(module, &shaped, re.len());
    let full_bits = precision((&full_re, &full_im), (&re, &im));
    let shaped_bits = precision((&shaped_re, &shaped_im), (&re, &im));
    assert!(
        shaped_bits >= full_bits + 20.0,
        "replicated encoding at 2^{LOG_DELTA}: {shaped_bits:.2} bits against {full_bits:.2} for a full layout"
    );
    Ok(())
}

pub fn test_replicated_unit_shaping_encodes_plainly<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let unit = ReplicatedLayout::new(LOG_K + 2, 1, 0, &[1, 0, 0, 0])?;
    let (re, im) = batch::<BE, F>(1 << unit.log_k());
    let mut shaped = plaintext(&params, module, SlotLayout::Replicated(unit), re.len());
    encoded(module, &mut shaped, &re, &im)?;

    let scatter = |values: &[F]| {
        let mut slots = vec![F::zero(); values.len() << unit.log_r()];
        values.iter().enumerate().for_each(|(i, &v)| slots[i << unit.log_r()] = v);
        slots
    };
    let mut full = plaintext(&params, module, SlotLayout::Full, re.len() << unit.log_r());
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(re.len() << unit.log_r()));
    module.ckks_encode_reim_slots_into(&mut full, &scatter(&re), &scatter(&im), &mut scratch.borrow())?;

    let digits = |pt: &CKKSPlaintextOwned<BE>| -> Vec<i64> {
        let glwe = GLWEToBackendRef::<BE>::to_backend_ref(pt);
        (0..glwe.size()).flat_map(|limb| glwe.data().at(0, limb).to_vec()).collect()
    };
    assert_eq!(digits(&shaped), digits(&full));
    Ok(())
}

pub fn test_replicated_encryption_is_shaped<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let (re, im) = batch::<BE, F>(1 << LOG_K);
    let mut bits = Vec::new();
    for slot_layout in [SlotLayout::Full, SlotLayout::Replicated(layout())] {
        let mut pt = plaintext(&params, module, slot_layout, re.len());
        encoded(module, &mut pt, &re, &im)?;
        let mut ct = alloc_ct(&params, module, LOG_DELTA + 30);
        let mut scratch = ScratchOwned::<BE>::alloc(
            module
                .ckks_encrypt_sk_tmp_bytes(&ct)
                .max(module.ckks_decrypt_tmp_bytes(&pt, &ct)),
        );
        let (mut xe, mut xa) = (Source::new([1; 32]), Source::new([2; 32]));
        module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, &mut scratch.borrow())?;
        assert_eq!(ct.slot_layout(), slot_layout);
        assert_eq!((ct.log_delta(), ct.k().as_usize()), (LOG_DELTA, LOG_DELTA + 30));

        let mut decrypted = plaintext(&params, module, slot_layout, re.len());
        decrypted.set_meta(ct.meta());
        module.ckks_decrypt(&mut decrypted, &ct, &sk, &mut scratch.borrow())?;
        let (got_re, got_im) = decode::<BE, F>(module, &decrypted, re.len());
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

pub fn test_replicated_encoding_rejects_batch_size<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let (re, im) = batch::<BE, F>(2 << LOG_K);
    let mut pt = plaintext(&params, module, SlotLayout::Replicated(layout()), 1 << LOG_K);
    assert!(encoded(module, &mut pt, &re, &im).is_err());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Product {
    Mul,
    MulAssign,
    Square,
    SquareAssign,
    PtVec,
    PtVecAssign,
    PtConst,
    PtConstAssign,
}

const STEPS: usize = 3;
/// Exactly representable at the constant plaintext's 8 bits, so that only the
/// rescale rounds.
const CONSTANT: f64 = 0.75;

fn encrypt_batch<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    slot_layout: SlotLayout,
    values: (&[F], &[F]),
    k: usize,
    seed: u8,
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
{
    let mut pt = plaintext(params, module, slot_layout, values.0.len());
    encoded(module, &mut pt, values.0, values.1)?;
    let mut ct = alloc_ct(params, module, k);
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_encrypt_sk_tmp_bytes(&ct));
    let (mut xe, mut xa) = (Source::new([seed; 32]), Source::new([seed.wrapping_add(128); 32]));
    module.ckks_encrypt_sk(&mut ct, &pt, sk, &mut xe, &mut xa, &mut scratch.borrow())?;
    Ok(ct)
}

fn decrypt_batch<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    ct: &CKKSCiphertextOwned<BE>,
    len: usize,
) -> (Vec<F>, Vec<F>)
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
{
    // A few bits of budget hold the unit-scale message.
    let budget = ct.log_budget().min(10);
    let mut pt = module.ckks_pt_vec_alloc(
        module.ckks_slots_of_degree(ct.n().as_usize() >> ct.log_sparsity()),
        Base2K(params.base2k as u32),
        TorusPrecision((ct.log_delta() + budget) as u32),
    );
    pt.set_meta(ct.meta());
    pt.set_log_budget(budget);
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_decrypt_tmp_bytes(&pt, ct));
    module.ckks_decrypt(&mut pt, ct, sk, &mut scratch.borrow()).unwrap();
    decode::<BE, F>(module, &pt, len)
}

/// Unit-modulus operands, so that the chained products stay bounded. Real
/// slots take the real parts, of modulus at most one.
fn unit_batch<BE: Backend, F: TestScalar>(len: usize, step: usize) -> (Vec<F>, Vec<F>) {
    let angle = |i: usize| 0.37 * (i + 1) as f64 + 1.1 * step as f64;
    let value = |x: f64| F::from_f64(x).unwrap();
    let real = real_slots::<BE>();
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
fn product_chain<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
    slot_layout: SlotLayout,
    product: Product,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
{
    let len = 1 << LOG_K;
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let tsk = gen_tsk(params, module, &sk_raw, &mut scratch.borrow());
    let k = LOG_DELTA * (STEPS + 2);

    let mut want = unit_batch::<BE, F>(len, 0);
    let mut acc = encrypt_batch(params, module, &sk, slot_layout, (&want.0, &want.1), k, 1)?;
    let cst = ckks_pt_cst_full::<BE, F>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        params.n / 2,
        Some(CONSTANT),
        None,
    );
    let zero = F::zero();
    for step in 1..=STEPS {
        let operand = unit_batch::<BE, F>(len, step);
        let fresh = encrypt_batch(params, module, &sk, slot_layout, (&operand.0, &operand.1), k, step as u8 + 1)?;
        let mut pt = plaintext(params, module, slot_layout, len);
        encoded(module, &mut pt, &operand.0, &operand.1)?;
        let mut out = alloc_ct(params, module, k);
        let s = &mut scratch.borrow();
        want = match product {
            Product::Mul => {
                module.ckks_mul_into(&mut out, &acc, &fresh, &tsk, s)?;
                acc = out;
                complex_mul((&want.0, &want.1), (&operand.0, &operand.1))
            }
            Product::MulAssign => {
                module.ckks_mul_assign(&mut acc, &fresh, &tsk, s)?;
                complex_mul((&want.0, &want.1), (&operand.0, &operand.1))
            }
            Product::Square => {
                module.ckks_square_into(&mut out, &acc, &tsk, s)?;
                acc = out;
                complex_mul((&want.0, &want.1), (&want.0, &want.1))
            }
            Product::SquareAssign => {
                module.ckks_square_assign(&mut acc, &tsk, s)?;
                complex_mul((&want.0, &want.1), (&want.0, &want.1))
            }
            Product::PtVec => {
                module.ckks_mul_pt_vec_into(&mut out, &acc, &pt, s)?;
                acc = out;
                complex_mul((&want.0, &want.1), (&operand.0, &operand.1))
            }
            Product::PtVecAssign => {
                module.ckks_mul_pt_vec_assign(&mut acc, &pt, s)?;
                complex_mul((&want.0, &want.1), (&operand.0, &operand.1))
            }
            Product::PtConst | Product::PtConstAssign => {
                if matches!(product, Product::PtConst) {
                    module.ckks_mul_pt_const_into(&mut out, &acc, &cst, 0, s)?;
                    acc = out;
                } else {
                    module.ckks_mul_pt_const_assign(&mut acc, &cst, 0, s)?;
                }
                let c = F::from_f64(CONSTANT).unwrap();
                complex_mul((&want.0, &want.1), (&vec![c; len], &vec![zero; len]))
            }
        };
        assert_eq!(acc.slot_layout(), slot_layout, "{product:?}");
    }
    let got = decrypt_batch::<BE, F>(params, module, &sk, &acc, len);
    Ok(precision((&got.0, &got.1), (&want.0, &want.1)))
}

pub fn test_replicated_products_are_shaped<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    use Product::*;
    for product in [
        Mul,
        MulAssign,
        Square,
        SquareAssign,
        PtVec,
        PtVecAssign,
        PtConst,
        PtConstAssign,
    ] {
        let full = product_chain::<BE, F>(&params, module, host_module, SlotLayout::Full, product)?;
        let shaped = product_chain::<BE, F>(&params, module, host_module, SlotLayout::Replicated(layout()), product)?;
        assert!(
            shaped >= full + 15.0,
            "{product:?} chain at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
        );
    }
    Ok(())
}

pub fn test_replicated_layouts_do_not_mix<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let values = unit_batch::<BE, F>(1 << LOG_K, 0);
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
    assert!(module.ckks_mul_into(&mut out, &shaped, &full, &tsk, s).is_err());
    assert!(module.ckks_add_into(&mut out, &shaped, &full, s).is_err());
    assert!(module.ckks_mul_pt_vec_into(&mut out, &shaped, &pt, s).is_err());
    assert!(module.ckks_add_pt_vec_into(&mut out, &shaped, &pt, s).is_err());
    let cst = ckks_pt_cst_full::<BE, F>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        params.n / 2,
        Some(CONSTANT),
        None,
    );
    assert!(module.ckks_add_pt_const_into(&mut out, &shaped, 0, &cst, 0, s).is_err());
    module.ckks_mul_pt_const_into(&mut out, &shaped, &cst, 0, s)?;
    Ok(())
}

pub fn test_replicated_plain_rounding_is_rejected<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSAffineOps<BE>
        + CKKSDotProductOps<BE>
        + CKKSLinearTransformationOps<BE>
        + GLWEMulPreparedPlainSum<BE>,
    F: TestScalar,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let values = unit_batch::<BE, F>(1 << LOG_K, 0);
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
    let cst = ckks_pt_cst_full::<BE, F>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        params.n / 2,
        Some(CONSTANT),
        None,
    );
    assert!(
        module
            .ckks_affine_pt_const_into(&mut out, &shaped, &cst, 0, 0, &mut scratch.borrow())
            .is_err()
    );

    let mut pt = plaintext(&params, module, SlotLayout::Full, values.0.len());
    encoded(module, &mut pt, &values.0, &values.1)?;
    let left = module.ckks_prepare_left(&shaped, PrepareHint::OneShot, &mut scratch.borrow())?;
    let right = module.ckks_prepare_right(&pt, PrepareHint::OneShot, &mut scratch.borrow())?;
    assert!(
        module
            .ckks_dot_product_prepared_pt(&mut out, &[&left], &[&right], &[1], &mut scratch.borrow())
            .is_err()
    );

    let gal = module.ckks_galois_element(1);
    let atks = HashMap::from([(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()))]);
    let mut babies = LinearTransformationBabySteps::alloc(module, &[1], &shaped);
    assert!(
        module
            .ckks_prepare_linear_transformation_baby_steps(&mut babies, &shaped, &atks, &mut scratch.borrow())
            .is_err()
    );

    // No shaping pass fits `i128` above 126 torus bits.
    let mut wide = module.ckks_pt_vec_alloc(
        slots_of(module, SlotLayout::Replicated(layout()), values.0.len()),
        Base2K(params.base2k as u32),
        TorusPrecision(130),
    );
    wide.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(100),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout: SlotLayout::Replicated(layout()),
    });
    assert!(encoded(module, &mut wide, &values.0, &values.1).is_err());

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
    assert!(
        module
            .ckks_add_into(&mut out, &shaped, &coarse, &mut scratch.borrow())
            .is_err()
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
    assert!(module.ckks_add_pt_vec_assign(&mut acc, &fine, &mut scratch.borrow()).is_err());

    {
        use crate::test_suite::parity::helpers::{snapshot, with_scratch};
        let before = snapshot::<BE, _>(&out);
        let source_before = snapshot::<BE, _>(&acc);
        let pt_before = snapshot::<BE, _>(&fine);
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
            assert_eq!(snapshot::<BE, _>(&out), before);
            assert_eq!(snapshot::<BE, _>(&acc), source_before);
            assert_eq!(snapshot::<BE, _>(&fine), pt_before);
        }
    }

    // `N/2` values of period `R` exceed the slots of the ring.
    let log_n = params.n.ilog2() as u8;
    let oversized = ReplicatedLayout::new(log_n - 1, layout().log_r() as u8, layout().w() as u8, layout().shaping())?;
    let mut big = plaintext(&params, module, SlotLayout::Full, params.n / 2);
    big.set_meta(GLWEPlaintextMeta {
        slot_layout: SlotLayout::Replicated(oversized),
        ..big.meta()
    });
    let many = unit_batch::<BE, F>(params.n / 2, 0);
    assert!(encoded(module, &mut big, &many.0, &many.1).is_err());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum PolynomialCase {
    Monomial,
    Chebyshev,
    Complex,
}

fn coefficients<F: TestScalar>(len: usize, shift: usize) -> Vec<F> {
    (0..len)
        .map(|i| F::from_f64(0.37 * (1.7 * (i + shift) as f64 + 0.2).sin()).unwrap())
        .collect()
}

/// Precision of a polynomial evaluated on real inputs in `[-1, 1]`, with
/// constants at the ciphertext scale: scalar constants on a full layout, and
/// masked constants on a replicated layout.
fn polynomial_precision<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
    case: PolynomialCase,
) -> Result<f64>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>:
        TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSPolynomialEvaluationOps<BE> + CKKSComplexPolynomialEvaluationOps<BE>,
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
    let mut out = alloc_ct(params, module, k);
    let base2k = params.base2k.into();
    let coeff_prec = ckks_spec(params.n, params.base2k, LOG_DELTA, 10);
    let coeff_meta = CoeffsMeta::from(coeff_prec);
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_all_ops_tmp_bytes(&x, &params.tsk_layout(), &coeff_prec));
    let s = &mut scratch.borrow();
    let mut encoding = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(params.n / 2));
    let want: Vec<(F, F)> = match case {
        PolynomialCase::Monomial | PolynomialCase::Chebyshev => {
            let (basis, degree) = match case {
                PolynomialCase::Monomial => (Basis::Monomial, 7),
                _ => (Basis::Chebyshev, 13),
            };
            let poly = Polynomial::new(basis, coefficients::<F>(degree + 1, 0));
            match slot_layout {
                SlotLayout::Replicated(layout) => {
                    let bsgs = encode_bsgs_replicated(
                        &poly,
                        module,
                        base2k,
                        coeff_meta,
                        SplitStrategy::MinDepth,
                        layout,
                        &mut encoding.borrow(),
                    )?;
                    module.ckks_eval_poly_real_const_coeffs(&mut out, &x, &bsgs, &tsk, s)?;
                }
                SlotLayout::Full => {
                    let bsgs = poly
                        .encode_bsgs(host_module, base2k, coeff_meta)?
                        .map_baby_steps_ref(|pt| upload_pt(module, pt));
                    module.ckks_eval_poly_real_const_coeffs(&mut out, &x, &bsgs, &tsk, s)?;
                }
            }
            xs.iter().map(|&x| (poly.evaluate(x), F::zero())).collect()
        }
        PolynomialCase::Complex => {
            let poly = ComplexPolynomial::new(Basis::Monomial, coefficients::<F>(6, 0), coefficients::<F>(6, 2));
            match slot_layout {
                SlotLayout::Replicated(layout) => {
                    let bsgs = encode_complex_bsgs_replicated(
                        &poly,
                        module,
                        base2k,
                        coeff_meta,
                        SplitStrategy::MinDepth,
                        layout,
                        &mut encoding.borrow(),
                    )?;
                    module.ckks_eval_poly_complex_const_coeffs(&mut out, &x, &bsgs, &tsk, s)?;
                }
                SlotLayout::Full => {
                    let bsgs = poly
                        .encode_bsgs(host_module, base2k, coeff_meta)?
                        .map_baby_steps_ref(|pt| upload_pt(module, pt));
                    module.ckks_eval_poly_complex_const_coeffs(&mut out, &x, &bsgs, &tsk, s)?;
                }
            }
            xs.iter().map(|&x| poly.evaluate(x)).collect()
        }
    };
    assert_eq!(out.slot_layout(), slot_layout, "{case:?}");
    let got = decrypt_batch::<BE, F>(params, module, &sk, &out, len);
    let want: (Vec<F>, Vec<F>) = want.into_iter().unzip();
    Ok(precision((&got.0, &got.1), (&want.0, &want.1)))
}

pub fn test_replicated_polynomials_are_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>:
        TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSPolynomialEvaluationOps<BE> + CKKSComplexPolynomialEvaluationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    for case in [PolynomialCase::Monomial, PolynomialCase::Chebyshev, PolynomialCase::Complex] {
        let full = polynomial_precision::<BE, F>(&params, module, host_module, SlotLayout::Full, case)?;
        let shaped = polynomial_precision::<BE, F>(&params, module, host_module, SlotLayout::Replicated(layout()), case)?;
        assert!(
            shaped >= full + 15.0,
            "{case:?} polynomial at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
        );
    }

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let k = 7 * LOG_DELTA;
    let values = unit_batch::<BE, F>(1 << LOG_K, 0);
    let x = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        k,
        1,
    )?;
    let narrower = ShapingUnit::U16_20.layout(LOG_K, 24)?;
    let identity = Polynomial::new(Basis::Monomial, vec![F::zero(), F::one()]);
    let mut encoding = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(params.n / 2));
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
    assert!(
        module
            .ckks_eval_poly_real_const_coeffs(&mut out, &x, &bsgs, &tsk, &mut scratch.borrow())
            .is_err()
    );
    Ok(())
}

/// Precision after a rotation into a new ciphertext, a rotation in place and a
/// conjugation, against `conj(v[i + 2])`.
fn automorphism_chain<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
) -> Result<f64>
where
    BE: TestContextBackend<Ring = Standard>,
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
    let gal = module.ckks_galois_element(step);
    atks.insert(gal, gen_atk(params, module, gal, &sk_raw, &mut scratch.borrow()));
    let conj = gen_atk(params, module, -1, &sk_raw, &mut scratch.borrow());

    let values = unit_batch::<BE, F>(len, 0);
    let k = 3 * LOG_DELTA;
    let ct = encrypt_batch(params, module, &sk, slot_layout, (&values.0, &values.1), k, 1)?;
    let mut rotated = alloc_ct(params, module, k);
    let s = &mut scratch.borrow();
    module.ckks_rotate_into(&mut rotated, &ct, step, &atks, s)?;
    module.ckks_rotate_assign(&mut rotated, step, &atks, s)?;
    let mut out = alloc_ct(params, module, k);
    module.ckks_conjugate_into(&mut out, &rotated, &conj, s)?;
    assert_eq!(out.slot_layout(), slot_layout);

    let want_re: Vec<F> = (0..len).map(|i| values.0[(i + 2) % len]).collect();
    let want_im: Vec<F> = (0..len).map(|i| -values.1[(i + 2) % len]).collect();
    let got = decrypt_batch::<BE, F>(params, module, &sk, &out, len);
    Ok(precision((&got.0, &got.1), (&want_re, &want_im)))
}

pub fn test_replicated_automorphisms_are_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
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
fn rotation_chain<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
    slot_layout: SlotLayout,
) -> Result<f64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
{
    let len = 1 << LOG_K;
    let step = match slot_layout {
        SlotLayout::Replicated(layout) => 1i64 << layout.log_r(),
        SlotLayout::Full => 1,
    };
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let gal = module.ckks_galois_element(step);
    let atks = HashMap::from([(gal, gen_atk(params, module, gal, &sk_raw, &mut scratch.borrow()))]);

    let values = unit_batch::<BE, F>(len, 0);
    let k = 3 * LOG_DELTA;
    let ct = encrypt_batch(params, module, &sk, slot_layout, (&values.0, &values.1), k, 1)?;
    let mut out = alloc_ct(params, module, k);
    let s = &mut scratch.borrow();
    module.ckks_rotate_into(&mut out, &ct, step, &atks, s)?;
    module.ckks_rotate_assign(&mut out, step, &atks, s)?;
    assert_eq!(out.slot_layout(), slot_layout);

    let want_re: Vec<F> = (0..len).map(|i| values.0[(i + 2) % len]).collect();
    let want_im: Vec<F> = (0..len).map(|i| values.1[(i + 2) % len]).collect();
    let got = decrypt_batch::<BE, F>(params, module, &sk, &out, len);
    Ok(precision((&got.0, &got.1), (&want_re, &want_im)))
}

/// The rotations of [`test_replicated_automorphisms_are_shaped`], on any ring.
pub fn test_replicated_rotations_are_shaped<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let full = rotation_chain::<BE, F>(&params, module, host_module, SlotLayout::Full)?;
    let shaped = rotation_chain::<BE, F>(&params, module, host_module, SlotLayout::Replicated(layout()))?;
    assert!(
        shaped >= full + 15.0,
        "rotations at 2^{LOG_DELTA}: {shaped:.2} bits against {full:.2} for a full layout"
    );
    Ok(())
}

pub fn test_replicated_rotations_are_admissible<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let mut atks = HashMap::new();
    for r in [1, 1i64 << layout().log_r()] {
        let gal = module.ckks_galois_element(r);
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }
    let values = unit_batch::<BE, F>(1 << LOG_K, 0);
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
    let s = &mut scratch.borrow();
    assert!(module.ckks_rotate_into(&mut out, &ct, 1, &atks, s).is_err());
    assert!(module.ckks_rotate_assign(&mut ct, -1, &atks, s).is_err());
    module.ckks_rotate_into(&mut out, &ct, 1 << layout().log_r(), &atks, s)?;
    Ok(())
}

pub fn test_replicated_masked_constants<BE, F>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextSharedModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    let len = 1usize << LOG_K;
    let slots = len << layout().log_r();
    let c_im = if real_slots::<BE>() { 0.0 } else { -0.2 };
    let (c_re, c_im) = (F::from_f64(0.3).unwrap(), F::from_f64(c_im).unwrap());
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(slots));
    let mut cst = plaintext(&params, module, SlotLayout::Replicated(layout()), len);
    module.ckks_encode_masked_constant_into(&mut cst, c_re, c_im, &mut scratch.borrow())?;

    let (mut re, mut im) = (vec![F::zero(); slots], vec![F::zero(); slots]);
    module.ckks_decode_reim_slots_into(&cst, &mut re, &mut im, &mut scratch.borrow())?;
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
    let values = unit_batch::<BE, F>(len, 0);
    let mut ct = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(layout()),
        (&values.0, &values.1),
        3 * LOG_DELTA,
        1,
    )?;
    let mut add_scratch = alloc_scratch(&params, module);
    module.ckks_add_pt_vec_assign(&mut ct, &cst, &mut add_scratch.borrow())?;
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &ct, len);
    let want_re: Vec<F> = values.0.iter().map(|&x| x + c_re).collect();
    let want_im: Vec<F> = values.1.iter().map(|&x| x + c_im).collect();
    let bits = precision((&got.0, &got.1), (&want_re, &want_im));
    assert!(bits >= 40.0, "masked constant sum: {bits:.2} bits");

    // The masked one, added twice and subtracted once.
    let s = &mut add_scratch.borrow();
    module.ckks_add_one_assign(&mut ct, s)?;
    module.ckks_add_one_assign(&mut ct, s)?;
    module.ckks_sub_one_assign(&mut ct, s)?;
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &ct, len);
    let want_re: Vec<F> = want_re.iter().map(|&x| x + F::one()).collect();
    let bits = precision((&got.0, &got.1), (&want_re, &want_im));
    assert!(bits >= 40.0, "masked one: {bits:.2} bits");

    // The masked one of a period whose `2R` coefficients lie below the
    // backend's smallest plaintext.
    let small = ReplicatedLayout::new(LOG_K, 1, 24, &[1, -1, 0, 1])?;
    let mut ct = encrypt_batch(
        &params,
        module,
        &sk,
        SlotLayout::Replicated(small),
        (&values.0, &values.1),
        3 * LOG_DELTA,
        5,
    )?;
    module.ckks_add_one_assign(&mut ct, &mut add_scratch.borrow())?;
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &ct, len);
    let want_re: Vec<F> = values.0.iter().map(|&x| x + F::one()).collect();
    let bits = precision((&got.0, &got.1), (&want_re, &values.1));
    // `1 - Y + Y^3` barely suppresses, so the precision stays near the
    // scale's, about half a bit lower on the conjugate-invariant ring, whose
    // products have twice the variance.
    assert!(bits >= LOG_DELTA as f64 - 5.0, "masked one of period 2: {bits:.2} bits");
    Ok(())
}

/// Full-layout encryption of `values` with period `values.len()`.
fn encrypt_full<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    values: (&[F], &[F]),
    log_delta: usize,
    k: usize,
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: TestContextBackend,
    F: TestScalar,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
{
    let mut pt = module.ckks_pt_vec_alloc(
        slots_of(module, SlotLayout::Full, values.0.len()),
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

pub fn test_replicated_conversions_round_trip<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSReplicatedOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    // `R = 8` divides `k = 8`; the return lands on `20 + 16 + log2(N/(2k))`.
    let (log_k, log_delta_ckks, log_delta, log_delta_diag) = (3u8, 40, 20, 30);
    let extra_bits = log_delta_ckks - log_delta - (params.n / (2 << log_k)).trailing_zeros() as usize;
    let layout = ShapingUnit::U16_40.layout(log_k, 24)?;
    let len = 1usize << log_k;
    let conv = ReplicatedConversion::new::<F>(
        module,
        layout,
        Base2K(params.base2k as u32),
        log_delta,
        log_delta_diag,
        extra_bits,
    )?;

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let mut atks = HashMap::new();
    for r in conv.rotations() {
        let gal = module.ckks_galois_element(r);
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }

    let values = unit_batch::<BE, F>(len, 0);
    let k = log_delta_ckks + 3 * log_delta_diag;
    let ckks = encrypt_full(&params, module, &sk, (&values.0, &values.1), log_delta_ckks, k)?;
    let fresh = decrypt_batch::<BE, F>(&params, module, &sk, &ckks, len);
    let fresh_bits = precision((&fresh.0, &fresh.1), (&values.0, &values.1));

    let atk_infos = params.atk_layout();
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_replicated_conversion_tmp_bytes(&ckks, &atk_infos, &conv));
    let mut replicated = alloc_ct(&params, module, k);
    let mut to_workspace = conv.alloc_to_workspace(module, &ckks);
    module.ckks_to_replicated_into(&mut replicated, &ckks, &conv, &mut to_workspace, &atks, &mut scratch.borrow())?;
    assert_eq!(replicated.slot_layout(), SlotLayout::Replicated(layout));
    assert_eq!(replicated.log_delta(), log_delta);
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &replicated, len);
    let to_bits = precision((&got.0, &got.1), (&values.0, &values.1));
    assert!(
        to_bits >= fresh_bits - 2.0,
        "to replicated: {to_bits:.2} bits from {fresh_bits:.2}"
    );

    let mut back = alloc_ct(&params, module, k);
    let mut from_workspace = conv.alloc_from_workspace(module, &replicated)?;
    module.ckks_from_replicated_into(
        &mut back,
        &replicated,
        &conv,
        &mut from_workspace,
        &atks,
        &mut scratch.borrow(),
    )?;
    assert_eq!(back.slot_layout(), SlotLayout::Full);
    assert_eq!(back.log_delta(), log_delta_ckks);
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &back, len);
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
        let before = snapshot::<BE, _>(&out);
        let input_before = snapshot::<BE, _>(&ckks);
        let replicated_before = snapshot::<BE, _>(&replicated);
        for (n, rank, reason) in [
            (params.n, ckks.rank().as_usize() + 1, "source and destination ranks differ"),
            (params.n / 2, ckks.rank().as_usize(), "source and destination degrees differ"),
        ] {
            let mut bad = module.ckks_ciphertext_alloc_from_glwe_infos(&GLWELayout {
                n: n.into(),
                rank: rank.into(),
                ..ckks.glwe_layout()
            });
            bad.set_meta(ckks.meta());
            let bad_before = snapshot::<BE, _>(&bad);
            for to in [true, false] {
                let err = if to {
                    module.ckks_to_replicated_into(&mut bad, &ckks, &conv, &mut to_workspace, &atks, &mut scratch.borrow())
                } else {
                    module.ckks_from_replicated_into(
                        &mut bad,
                        &replicated,
                        &conv,
                        &mut from_workspace,
                        &atks,
                        &mut scratch.borrow(),
                    )
                }
                .unwrap_err();
                let expected_op = if to {
                    "ckks_to_replicated_into"
                } else {
                    "ckks_from_replicated_into"
                };
                assert!(
                    matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op, reason: actual }) if *op == expected_op && actual == reason)
                );
                assert_eq!(snapshot::<BE, _>(&bad), bad_before);
                assert_eq!(snapshot::<BE, _>(&ckks), input_before);
                assert_eq!(snapshot::<BE, _>(&replicated), replicated_before);
            }
        }
        let mut wrong_width = conv.alloc_to_workspace(
            module,
            &GLWELayout {
                k: (ckks.k().as_usize() - 1).into(),
                ..ckks.glwe_layout()
            },
        );
        let mut wrong_rotations = conv.alloc_to_workspace(module, &ckks);
        wrong_rotations.babies = LinearTransformationBabySteps::alloc(module, &[1], &wrong_rotations.input);
        for workspace in [&mut wrong_width, &mut wrong_rotations] {
            let err = module
                .ckks_to_replicated_into(&mut out, &ckks, &conv, workspace, &atks, &mut scratch.borrow())
                .unwrap_err();
            assert!(
                matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op: "ckks_to_replicated_into", reason }) if reason == "workspace does not match the transform input layout or rotations")
            );
            assert_eq!(snapshot::<BE, _>(&out), before);
            assert_eq!(snapshot::<BE, _>(&ckks), input_before);
        }
    }
    assert!(
        module
            .ckks_to_replicated_into(&mut out, &replicated, &conv, &mut to_workspace, &atks, &mut scratch.borrow())
            .is_err()
    );
    assert!(
        module
            .ckks_from_replicated_into(&mut out, &ckks, &conv, &mut from_workspace, &atks, &mut scratch.borrow())
            .is_err()
    );
    Ok(())
}

/// `(Mv)` for `M(c, j) = (cos(c·j + 0.1)/2, sin(c + 2j)/3)`.
fn matrix<F: TestScalar>(c: usize, j: usize) -> (F, F) {
    let value = |x: f64| F::from_f64(x).unwrap();
    (
        value(((c * j) as f64 + 0.1).cos() / 2.0),
        value(((c + 2 * j) as f64).sin() / 3.0),
    )
}

fn apply_matrix<F: TestScalar>(v: (&[F], &[F])) -> (Vec<F>, Vec<F>) {
    let len = v.0.len();
    let (mut re, mut im) = (vec![F::zero(); len], vec![F::zero(); len]);
    for c in 0..len {
        for j in 0..len {
            let (a, b) = matrix::<F>(c, j);
            re[c] = re[c] + a * v.0[j] - b * v.1[j];
            im[c] = im[c] + a * v.1[j] + b * v.0[j];
        }
    }
    (re, im)
}

pub fn test_replicated_transforms_apply_matrices<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_k = 3u8;
    let (len, base2k) = (1usize << log_k, Base2K(params.base2k as u32));
    let layout = ShapingUnit::U16_40.layout(log_k, 24)?;
    let (log_delta_diag, extra_bits) = (36, 16);
    // Shaped diagonals far coarser than the result: each BSGS giant step needs
    // its own rotated shaping polynomial to keep their error off the data slots.
    let to = ReplicatedTransform::from_period(module, layout, base2k, 14, true, |s, j| {
        if s % (1 << layout.log_r()) == 0 {
            matrix::<F>((s >> layout.log_r()) % len, j)
        } else {
            (F::zero(), F::zero())
        }
    })?;
    let gather = ReplicatedTransform::gather(module, layout, base2k, log_delta_diag, false, matrix::<F>)?;
    let trace: Vec<i64> = (0..(params.n / (2 * len)).trailing_zeros())
        .map(|l| (len << l) as i64)
        .collect();

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let mut atks = HashMap::new();
    for r in to
        .rotations()
        .into_iter()
        .chain(gather.rotations())
        .chain(trace.iter().copied())
    {
        let gal = module.ckks_galois_element(r);
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }
    let values = unit_batch::<BE, F>(len, 0);
    let want = apply_matrix((&values.0, &values.1));
    let k = 40 + 2 * log_delta_diag;
    let mut scratch = ScratchOwned::<BE>::alloc(1 << 24);

    let ckks = encrypt_full(&params, module, &sk, (&values.0, &values.1), 40, k)?;
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
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &onto, len);
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
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &gathered, len);
    let gather_bits = precision((&got.0, &got.1), (&want.0, &want.1));
    // Plainly quantized diagonals weighted `R·M(c, j)` over `k` terms.
    assert!(gather_bits >= 26.0, "gather and trace: {gather_bits:.2} bits");
    Ok(())
}

/// Input scale of the EvalMod test, above its plan scale so that the entry
/// rounds and the output keeps the message precision under the ratio.
const EVAL_MOD_LOG_DELTA_IN: usize = 36;

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
fn eval_mod_precision<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    mode: EvalModMode,
) -> Result<f64>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSEvalModOps<BE>,
{
    let plan = eval_mod_plan();
    let ratio = (1u64 << plan.log_msg_ratio) as f64;
    let interval = plan.f_mod_interval as f64;
    let dsize = 2;
    // The tensor key's gadget covers the working width at the plan scale.
    let test_params = CKKSTestParams {
        k: (plan.consumed_bits() + EVAL_MOD_LOG_DELTA_IN + 2 * params.base2k).next_multiple_of(dsize * params.base2k)
            + EVAL_MOD_LOG_DELTA_IN
            - plan.f_mod_log_delta,
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(EVAL_MOD_LOG_DELTA_IN),
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
        EvalModMode::Packed => (SlotLayout::Replicated(ShapingUnit::U32_45.layout(LOG_K + 1, 24)?), 2 << LOG_K),
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
    let mut pt = module.ckks_pt_vec_alloc(
        slots_of(module, slot_layout, len),
        Base2K(params.base2k as u32),
        TorusPrecision((EVAL_MOD_LOG_DELTA_IN + 10) as u32),
    );
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(EVAL_MOD_LOG_DELTA_IN),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout,
    });
    encoded(module, &mut pt, &re, &zero)?;
    let mut ct = alloc_ct(&test_params, module, k);
    let (mut xe, mut xa) = (Source::new([1; 32]), Source::new([2; 32]));
    module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, &mut scratch.borrow())?;

    let mut compile = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(params.n / 2));
    let base2k = Base2K(params.base2k as u32);
    let mut res = alloc_ct(&test_params, module, k);
    let tsk_layout = test_params.tsk_layout();
    let (f_mod_poly, steps, scale) = match slot_layout {
        SlotLayout::Replicated(layout) => {
            let eval_mod = if let EvalModMode::Packed = mode {
                compile_eval_mod_replicated_packed::<BE, F>(base2k, plan, layout, module, &mut compile.borrow())?
            } else {
                compile_eval_mod_replicated::<BE, F>(base2k, plan, layout, module, &mut compile.borrow())?
            };
            let mut eval = ScratchOwned::<BE>::alloc(module.ckks_eval_mod_tmp_bytes(&res, &ct, &eval_mod, &tsk_layout));
            module.ckks_eval_mod(&mut res, &ct, &eval_mod, &tsk, &mut eval.borrow())?;
            let scale = eval_mod.range_extension_scale();
            (eval_mod.f_mod_poly, plan.f_mod_log_interval_reduction, scale)
        }
        SlotLayout::Full => {
            let eval_mod = compile_eval_mod::<BE, F>(base2k, plan, module, &mut compile.borrow())?;
            let mut eval = ScratchOwned::<BE>::alloc(module.ckks_eval_mod_tmp_bytes(&res, &ct, &eval_mod, &tsk_layout));
            module.ckks_eval_mod(&mut res, &ct, &eval_mod, &tsk, &mut eval.borrow())?;
            let scale = eval_mod.range_extension_scale();
            (eval_mod.f_mod_poly, plan.f_mod_log_interval_reduction, scale)
        }
    };
    assert_eq!(res.slot_layout(), slot_layout);
    assert_eq!(res.log_delta(), EVAL_MOD_LOG_DELTA_IN);

    let EvalModPoly::Real(poly) = f_mod_poly else {
        unreachable!("CosHK is a real polynomial")
    };
    let two = F::one() + F::one();
    let to_message = F::from_f64(ratio).unwrap();
    let want: Vec<F> = re
        .iter()
        .map(|&t| {
            let mut p = poly.evaluate(t);
            for i in 0..steps {
                p = two * p * p - F::from_f64(scale.powi(1 << (i + 1))).unwrap();
            }
            p * to_message
        })
        .collect();
    // A packed second half comes out on the imaginary parts.
    let half = if let EvalModMode::Packed = mode { len / 2 } else { len };
    let want_re: Vec<F> = (0..len).map(|j| if j < half { want[j] } else { F::zero() }).collect();
    let want_im: Vec<F> = (0..len).map(|j| if j < half { F::zero() } else { want[j] }).collect();
    let got = decrypt_batch::<BE, F>(&test_params, module, &sk, &res, len);
    let got_re: Vec<F> = got.0.iter().map(|&v| v * to_message).collect();
    let got_im: Vec<F> = got.1.iter().map(|&v| v * to_message).collect();
    Ok(precision((&got_re, &got_im), (&want_re, &want_im)))
}

pub fn test_replicated_eval_mod_is_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSEvalModOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let full = eval_mod_precision::<BE, F>(&params, module, host_module, EvalModMode::Full)?;
    let shaped = eval_mod_precision::<BE, F>(&params, module, host_module, EvalModMode::Replicated)?;
    let packed = eval_mod_precision::<BE, F>(&params, module, host_module, EvalModMode::Packed)?;
    assert!(
        shaped >= 30.0 && shaped >= full + 25.0,
        "EvalMod at 2^28: {shaped:.2} bits against {full:.2} for a full layout"
    );
    assert!(
        packed >= 30.0 && packed >= full + 25.0,
        "packed EvalMod at 2^28: {packed:.2} bits against {full:.2} for a full layout"
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
    BE: TestContextBackend<Ring = Standard> + CKKSEncapsulatedModUpImpl,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSReplicatedBootstrapOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + GLWETrace<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let layout = ShapingUnit::U16_40.layout(3, 24)?;
    let plan = ReplicatedBootstrapPlan::paper(layout);
    let base2k = Base2K(params.base2k as u32);
    let mut compile = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(params.n / 2));
    let ctx = ReplicatedBootstrapping::<BE, F>::new(module, plan, base2k, &mut compile.borrow())?;
    assert!(ctx.is_packed());
    let k_out = plan.k_in() + plan.level_log_delta;
    let key_layouts = plan.key_layouts(params.n.into(), base2k, k_out, params.base2k, params.base2k + 1, true);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);

    let len = 1 << layout.log_k();
    let values = unit_batch::<BE, F>(len, 0);
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
        let mut keygen = ScratchOwned::<BE>::alloc(1 << 26);
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
            .prepare(module, &mut keygen.borrow());
        let encapsulation = key_layouts.encapsulation.as_ref().unwrap();
        let mut out = alloc_ct(&params, module, k_out);
        let mut boot_scratch = ScratchOwned::<BE>::alloc(module.ckks_replicated_bootstrap_tmp_bytes(
            &out,
            &input,
            &ctx,
            &key_layouts.automorphism_key,
            &key_layouts.tensor_key,
            Some((
                &encapsulation.dense_to_sparse,
                encapsulation.sparse_to_dense.as_ref().unwrap(),
            )),
        ));
        let mut workspace = ctx.alloc_workspace(module, &out);
        module.ckks_replicated_bootstrap_into(&mut out, &input, &ctx, &mut workspace, &keys, &mut boot_scratch.borrow())?;
        assert_eq!(out.slot_layout(), SlotLayout::Replicated(layout));
        assert_eq!((out.log_delta(), out.k().as_usize()), (plan.level_log_delta, k_out));
        let got = decrypt_batch::<BE, F>(&params, module, &sk, &out, len);
        let bits = precision((&got.0, &got.1), (&values.0, &values.1));
        assert!(bits >= 19.0, "replicated bootstrapping: {bits:.2} bits");
        decoded.push(got);
    }
    let bits = precision((&decoded[0].0, &decoded[0].1), (&decoded[1].0, &decoded[1].1));
    assert!(
        bits >= 18.0,
        "offset and full-width replicated bootstrapping differ by {bits:.2} bits"
    );
    Ok(())
}
