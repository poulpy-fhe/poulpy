//! Slot and coefficient encoding round trips of the portable backend types in
//! `f64` and `Quad`.

#![cfg(feature = "enable-ckks")]

use poulpy_ckks::{
    GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{CKKSEncodingBuffer, CKKSPlaintextAlloc, ScratchArenaTakeCKKSPlaintext},
};
use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, TorusPrecision};
use poulpy_cpu_portable::{FFT64Portable, NTT4x30Portable};
use poulpy_hal::{
    AlignedBuf,
    api::ScratchOwnedAlloc,
    layouts::{Backend, Module, ScratchOwned, Standard},
};

fn assert_close<F: CKKSEncodingScalar>(got_re: &[F], got_im: &[F], re: &[F], im: &[F]) {
    for (got, want) in got_re.iter().chain(got_im).zip(re.iter().chain(im)) {
        assert!((got.to_f64().unwrap() - want.to_f64().unwrap()).abs() < 1e-8);
    }
}

fn roundtrip_all_dimensions<BE, F>(module: &Module<BE>)
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingOps<BE, F>,
{
    for slots in [1usize, 2, 4, 8, 16] {
        let re: Vec<F> = (0..slots).map(|i| F::from_f64((i as f64 + 1.0) / 17.0).unwrap()).collect();
        let im: Vec<F> = (0..slots).map(|i| F::from_f64(-(i as f64 + 1.0) / 29.0).unwrap()).collect();
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_encoding_host_tmp_bytes(slots));
        let mut scratch = scratch.arena();

        let degree = 2 * slots;
        let mut pt = module.ckks_plaintext_alloc(Degree(degree as u32), Base2K(20), TorusPrecision(80));
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(40),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        });
        module.ckks_encode_slots_host_into(&mut pt, &re, &im, &mut scratch).unwrap();
        let mut got_re = vec![F::zero(); slots];
        let mut got_im = vec![F::zero(); slots];
        module
            .ckks_decode_slots_host_into(&pt, &mut got_re, &mut got_im, &mut scratch)
            .unwrap();
        assert_close(&got_re, &got_im, &re, &im);
    }
}

#[test]
fn fft64_precision_families_coexist_in_one_module() {
    let module = Module::<FFT64Portable>::new(32);
    roundtrip_all_dimensions::<FFT64Portable, f64>(&module);
    roundtrip_all_dimensions::<FFT64Portable, poulpy_ckks::Quad>(&module);
}

#[test]
fn ntt_encoding_precisions_coexist_independently_of_ring_transform() {
    let module = Module::<NTT4x30Portable>::new(32);
    roundtrip_all_dimensions::<NTT4x30Portable, f64>(&module);
    roundtrip_all_dimensions::<NTT4x30Portable, poulpy_ckks::Quad>(&module);
}

#[test]
fn coefficient_only_encoding_accepts_scratch_plaintext_views() {
    let module = Module::<FFT64Portable>::new(32);
    let mut layout = module.ckks_plaintext_alloc::<Standard>(Degree(16), Base2K(20), TorusPrecision(80));
    layout.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(40),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout: SlotLayout::Full,
    });
    let bytes = GLWEPlaintext::<AlignedBuf, i64>::bytes_of_from_infos(&layout);
    let mut pt_scratch = ScratchOwned::<FFT64Portable>::alloc(bytes);
    let (mut pt, _) = pt_scratch.arena().take_ckks_plaintext_like_scratch::<_, Standard>(&layout);

    let slots = 8;
    let re: Vec<f64> = (0..slots).map(|i| (i as f64 + 1.0) / 17.0).collect();
    let im: Vec<f64> = (0..slots).map(|i| -(i as f64 + 1.0) / 29.0).collect();
    let values = re.iter().chain(&im).copied().collect::<Vec<_>>();
    let mut values = CKKSEncodingBuffer::<AlignedBuf, f64>::from_host::<FFT64Portable>(&values);
    module.ckks_encode_slots_into(&mut pt, &mut values).unwrap();
    let coeffs = values.to_host::<FFT64Portable>();

    let mut decoded = CKKSEncodingBuffer::<AlignedBuf, f64>::from_host::<FFT64Portable>(&vec![0.0; 2 * slots]);
    module.ckks_decode_slots_into(&pt, &mut decoded).unwrap();
    let values = decoded.to_host::<FFT64Portable>();
    let (got_re, got_im) = values.split_at(slots);
    assert_close(got_re, got_im, &re, &im);

    let mut coeff_scratch = ScratchOwned::<FFT64Portable>::alloc(
        CKKSEncodingHostOps::<FFT64Portable, f64>::ckks_encoding_host_tmp_bytes(&module, slots),
    );
    let mut coeff_scratch = coeff_scratch.arena();
    module
        .ckks_encode_coeffs_host_into(&mut pt, &coeffs, &mut coeff_scratch)
        .unwrap();
    let mut got_coeffs = vec![0.0; coeffs.len()];
    module
        .ckks_decode_coeffs_host_into(&pt, &mut got_coeffs, &mut coeff_scratch)
        .unwrap();
    for (got, want) in got_coeffs.iter().zip(&coeffs) {
        assert!((got - want).abs() < 1e-8);
    }

    let mut reim_scratch = ScratchOwned::<FFT64Portable>::alloc(
        CKKSEncodingHostOps::<FFT64Portable, f64>::ckks_encoding_host_tmp_bytes(&module, slots),
    );
    let mut reim_scratch = reim_scratch.arena();
    module
        .ckks_encode_slots_host_into(&mut pt, &re, &im, &mut reim_scratch)
        .unwrap();
    let mut got_re = vec![0.0; slots];
    let mut got_im = vec![0.0; slots];
    module
        .ckks_decode_slots_host_into(&pt, &mut got_re, &mut got_im, &mut reim_scratch)
        .unwrap();
    assert_close(&got_re, &got_im, &re, &im);
}

#[test]
fn encoding_rejects_invalid_slot_shapes() {
    let module = Module::<FFT64Portable>::new(32);
    let mut pt = module.ckks_plaintext_alloc::<Standard>(Degree(16), Base2K(20), TorusPrecision(80));
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(40),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
        slot_layout: SlotLayout::Full,
    });

    let three = [0.0; 3];
    let mut scratch = ScratchOwned::<FFT64Portable>::alloc(
        CKKSEncodingHostOps::<FFT64Portable, f64>::ckks_encoding_host_tmp_bytes(&module, 8),
    );
    let mut scratch = scratch.arena();
    assert!(
        module
            .ckks_encode_slots_host_into(&mut pt, &three, &three, &mut scratch)
            .is_err()
    );

    let re = [0.0; 8];
    let im = [0.0; 7];
    assert!(module.ckks_encode_slots_host_into(&mut pt, &re, &im, &mut scratch).is_err());

    let four = [0.0; 4];
    assert!(
        module
            .ckks_encode_slots_host_into(&mut pt, &four, &four, &mut scratch)
            .is_err()
    );

    let im = [0.0; 8];
    assert!(module.ckks_encode_slots_host_into(&mut pt, &re, &im, &mut scratch).is_ok());

    let required = CKKSEncodingHostOps::<FFT64Portable, f64>::ckks_encoding_host_tmp_bytes(&module, 8);
    let mut undersized = ScratchOwned::<FFT64Portable>::alloc(required - <FFT64Portable>::SCRATCH_ALIGN);
    assert!(
        module
            .ckks_encode_slots_host_into(&mut pt, &re, &im, &mut undersized.arena())
            .is_err()
    );
}
