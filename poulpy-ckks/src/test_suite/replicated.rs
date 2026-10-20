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

use anyhow::Result;
use poulpy_core::layouts::{Base2K, GLWEToBackendRef, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchOwned, Standard, ZnxView},
    source::Source,
};

use super::helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_ct, gen_sk};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSModuleInfos},
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned},
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
