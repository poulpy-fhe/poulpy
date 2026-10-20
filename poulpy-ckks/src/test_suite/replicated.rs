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
//! | [`test_replicated_layouts_do_not_mix`] | products and sums of different layouts, constant sums, and polynomial evaluation |
//! | [`test_replicated_automorphisms_are_shaped`] | rotations and conjugation keep the data slots far above CKKS |
//! | [`test_replicated_rotations_are_admissible`] | rotations by a non-multiple of the period |
//! | [`test_replicated_masked_constants`] | a masked constant fills the data slots only, and adds to a ciphertext |
//! | [`test_replicated_conversions_round_trip`] | CKKS to the replicated layout and back keeps the input's precision |

use std::collections::HashMap;

use anyhow::Result;
use poulpy_core::layouts::{Base2K, GLWEToBackendRef, LWEInfos, TorusPrecision, prepared::GLWESecretPrepared};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchOwned, Standard, ZnxView},
    source::Source,
};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, ckks_pt_cst_full, gen_atk, gen_sk,
    gen_sk_with_raw, gen_tsk, upload_pt,
};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSModuleInfos,
        CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSReplicatedOps, CKKSRotateOps, CKKSSubOps,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, ReplicatedConversion},
    polynomial::{Basis, EncodeBSGS, Polynomial},
    shaping::ShapingUnit,
    test_suite::CKKSTestParams,
};

const LOG_DELTA: usize = 20;
const LOG_K: u8 = 2;

fn layout() -> ReplicatedLayout {
    ShapingUnit::U32_45.layout(LOG_K, 24).unwrap()
}

fn batch<F: TestScalar>(len: usize) -> (Vec<F>, Vec<F>) {
    let value = |x: f64| F::from_f64(x).unwrap();
    (
        (0..len).map(|i| value(0.9 * (i as f64 + 0.3).cos())).collect(),
        (0..len).map(|i| value(0.7 * (2.0 * i as f64 + 0.1).sin())).collect(),
    )
}

fn plaintext<BE>(params: &CKKSTestParams, module: &Module<BE>, slot_layout: SlotLayout, len: usize) -> CKKSPlaintextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
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

pub fn test_replicated_encoding_is_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let (re, im) = batch::<F>(1 << LOG_K);
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

pub fn test_replicated_unit_shaping_encodes_plainly<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let unit = ReplicatedLayout::new(LOG_K + 2, 1, 0, &[1, 0, 0, 0])?;
    let (re, im) = batch::<F>(1 << unit.log_k());
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

pub fn test_replicated_encryption_is_shaped<BE, F, E>(
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let (re, im) = batch::<F>(1 << LOG_K);
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

pub fn test_replicated_encoding_rejects_batch_size<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let (re, im) = batch::<F>(2 << LOG_K);
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
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

/// Unit-modulus operands, so that the chained products stay bounded.
fn unit_batch<F: TestScalar>(len: usize, step: usize) -> (Vec<F>, Vec<F>) {
    let angle = |i: usize| 0.37 * (i + 1) as f64 + 1.1 * step as f64;
    let value = |x: f64| F::from_f64(x).unwrap();
    (
        (0..len).map(|i| value(angle(i).cos())).collect(),
        (0..len).map(|i| value(angle(i).sin())).collect(),
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
    host_module: &Module<HostBytesBackend>,
    slot_layout: SlotLayout,
    product: Product,
) -> Result<f64>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
{
    let len = 1 << LOG_K;
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let tsk = gen_tsk(params, module, &sk_raw, &mut scratch.borrow());
    let k = LOG_DELTA * (STEPS + 2);

    let mut want = unit_batch::<F>(len, 0);
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
        let operand = unit_batch::<F>(len, step);
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

pub fn test_replicated_products_are_shaped<BE, F, E>(
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

pub fn test_replicated_layouts_do_not_mix<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSPolynomialEvaluationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let values = unit_batch::<F>(1 << LOG_K, 0);
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
    let mut shifted = shaped.clone();
    assert!(module.ckks_sub_one_assign(&mut shifted, s).is_err());
    module.ckks_mul_pt_const_into(&mut out, &shaped, &cst, 0, s)?;

    // Polynomial evaluation adds scalar constants and fuses plain roundings.
    let poly = Polynomial::new(Basis::Monomial, vec![0.125f64, -0.25])
        .encode_bsgs(host_module, params.base2k.into(), PT_PREC.into())?
        .map_baby_steps_ref(|pt| upload_pt(module, pt));
    assert!(
        module
            .ckks_eval_poly_real_const_coeffs(&mut out, &shaped, &poly, &tsk, s)
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

    let values = unit_batch::<F>(len, 0);
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

pub fn test_replicated_rotations_are_admissible<BE, F, E>(
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
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let mut atks = HashMap::new();
    for r in [1, 1i64 << layout().log_r()] {
        let gal = module.ckks_galois_element(r);
        atks.insert(gal, gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
    }
    let values = unit_batch::<F>(1 << LOG_K, 0);
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

pub fn test_replicated_masked_constants<BE, F, E>(
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
    let len = 1usize << LOG_K;
    let slots = len << layout().log_r();
    let (c_re, c_im) = (F::from_f64(0.3).unwrap(), F::from_f64(-0.2).unwrap());
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
    let values = unit_batch::<F>(len, 0);
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSReplicatedOps<BE>,
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

    let values = unit_batch::<F>(len, 0);
    let k = log_delta_ckks + 3 * log_delta_diag;
    let ckks = encrypt_full(&params, module, &sk, (&values.0, &values.1), log_delta_ckks, k)?;
    let fresh = decrypt_batch::<BE, F>(&params, module, &sk, &ckks, len);
    let fresh_bits = precision((&fresh.0, &fresh.1), (&values.0, &values.1));

    let atk_infos = params.atk_layout();
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_replicated_conversion_tmp_bytes(&ckks, &atk_infos, &conv));
    let mut replicated = alloc_ct(&params, module, k);
    module.ckks_to_replicated_into(&mut replicated, &ckks, &conv, &atks, &mut scratch.borrow())?;
    assert_eq!(replicated.slot_layout(), SlotLayout::Replicated(layout));
    assert_eq!(replicated.log_delta(), log_delta);
    let got = decrypt_batch::<BE, F>(&params, module, &sk, &replicated, len);
    let to_bits = precision((&got.0, &got.1), (&values.0, &values.1));
    assert!(
        to_bits >= fresh_bits - 2.0,
        "to replicated: {to_bits:.2} bits from {fresh_bits:.2}"
    );

    let mut back = alloc_ct(&params, module, k);
    module.ckks_from_replicated_into(&mut back, &replicated, &conv, &atks, &mut scratch.borrow())?;
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
    assert!(
        module
            .ckks_to_replicated_into(&mut out, &replicated, &conv, &atks, &mut scratch.borrow())
            .is_err()
    );
    assert!(
        module
            .ckks_from_replicated_into(&mut out, &ckks, &conv, &atks, &mut scratch.borrow())
            .is_err()
    );
    Ok(())
}
