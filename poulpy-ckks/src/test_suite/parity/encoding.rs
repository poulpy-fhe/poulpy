//! Production encoding parity with caller-selected backends and scalar precision.
//!
//! Scalar buffers cross the boundary through explicit transfers. Integer
//! plaintexts are compared after downloading and decoding their canonical
//! coefficients, independently of either backend's floating-point decoder.

use poulpy_core::layouts::{GLWELayout, IntPolyInfos, LWEInfos};
use poulpy_hal::layouts::{Backend, Module, ZnxView};

use crate::{
    CKKSInfos, CKKSLayout, CKKSMeta, SlotsKind,
    api::{CKKSEncodingOps, CKKSEncodingScalar, PaCoScalar, ShipScalar},
    layouts::{
        CKKSEncodingBuffer, CKKSEncodingBufferToBackendMut, CKKSPlaintextOwned, PaCoDFTPlan, PaCoPlan, PaCoSlotOrder, ShipPlan,
    },
    oep::{CKKSEncodingImpl, CKKSPaCoCoeffEncodingImpl, CKKSShipCoeffEncodingImpl},
    test_suite::CKKSTestParams,
};

use super::helpers::{Snapshot, fixture_ciphertext, fixture_plaintext, snapshot, with_scratch};

fn scalar_close<F: CKKSEncodingScalar>(expected: &[F], actual: &[F], tolerance: F) {
    assert_eq!(expected.len(), actual.len(), "encoding scalar count differs");
    assert!(
        expected.iter().zip(actual).all(|(&a, &b)| (a - b).abs() <= tolerance),
        "encoding numerical parity failed"
    );
}

fn transform_tolerance<F: CKKSEncodingScalar>(n: usize) -> F {
    // A conservative allowance for the permutation/FFT accumulation; preserve
    // the selected scalar precision instead of reducing comparison to f64.
    F::epsilon() * F::from_usize(64 * n).unwrap()
}

fn coefficients<B, F>(pt: &CKKSPlaintextOwned<B>) -> Vec<F>
where
    B: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
{
    let host = pt.to_host_owned::<B>();
    let mut raw = vec![F::zero(); host.n().as_usize()];
    let base2k = host.base2k().as_usize();
    let radix = F::from_usize(base2k).unwrap().exp2();
    // Plaintexts are integer polynomials across their whole allocation. Fold
    // every limb so high-width corruption cannot disappear through truncation
    // to a fixed-width host integer.
    assert_eq!(host.encoded_k().as_usize(), host.data().size() * base2k);
    for limb in 0..host.data().size() {
        for (out, &digit) in raw.iter_mut().zip(host.data().at(0, limb)) {
            *out = *out * radix + F::from_i64(digit).unwrap();
        }
    }
    let inverse_scale = (-F::from_usize(pt.log_delta()).unwrap()).exp2();
    raw.into_iter().map(|x| x * inverse_scale).collect()
}

fn compare_plaintexts<BR, BT, F>(r: &CKKSPlaintextOwned<BR>, t: &CKKSPlaintextOwned<BT>)
where
    BR: Backend<ZnxWord = i64>,
    BT: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
{
    assert!(
        snapshot::<BR, _>(r).layout == snapshot::<BT, _>(t).layout,
        "encoding metadata differs"
    );
    let quantization = F::from_f64(2.0).unwrap() * (-F::from_usize(r.log_delta()).unwrap()).exp2();
    scalar_close(
        &coefficients::<BR, F>(r),
        &coefficients::<BT, F>(t),
        transform_tolerance::<F>(r.n().as_usize()) + quantization,
    );
}

fn encoding_transforms<B, F>(module: &Module<B>, n: usize) -> Vec<Vec<F>>
where
    B: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
{
    let plans = B::ckks_encoding_plans_create_impl(module).unwrap();
    // Repeat the geometric family in reverse order to reuse the same cached
    // plans after their first production-dispatch construction.
    let dimensions: Vec<_> = (1..=n.ilog2()).map(|log| 1usize << log).collect();
    let mut observed = Vec::new();
    for len in dimensions.iter().chain(dimensions.iter().rev()).copied() {
        for real in [true, false] {
            let input: Vec<F> = (0..len)
                .map(|i| {
                    if real && i >= len / 2 {
                        F::zero()
                    } else {
                        F::from_i64((i % 19) as i64 - 9).unwrap() / F::from_f64(32.0).unwrap()
                    }
                })
                .collect();
            let mut cached = CKKSEncodingBuffer::<B::OwnedBuf, F>::from_host::<B>(&input);
            let mut fresh = CKKSEncodingBuffer::<B::OwnedBuf, F>::from_host::<B>(&input);
            module.ckks_slots_to_coeffs_assign(&mut cached).unwrap();
            B::ckks_slots_to_coeffs_assign_impl(
                module,
                &plans,
                &mut CKKSEncodingBufferToBackendMut::<B, F>::to_backend_mut(&mut fresh),
            )
            .unwrap();
            let got = cached.to_host::<B>();
            scalar_close(&got, &fresh.to_host::<B>(), transform_tolerance::<F>(len));
            observed.push(got);
            module.ckks_coeffs_to_slots_assign(&mut cached).unwrap();
            B::ckks_coeffs_to_slots_assign_impl(
                module,
                &plans,
                &mut CKKSEncodingBufferToBackendMut::<B, F>::to_backend_mut(&mut fresh),
            )
            .unwrap();
            scalar_close(&input, &cached.to_host::<B>(), transform_tolerance::<F>(len));
            scalar_close(&input, &fresh.to_host::<B>(), transform_tolerance::<F>(len));
            observed.push(cached.to_host::<B>());
        }
    }
    for len in [0, 1, 3, 2 * n] {
        let input = vec![F::from_f64(0.375).unwrap(); len];
        let mut values = CKKSEncodingBuffer::<B::OwnedBuf, F>::from_host::<B>(&input);
        assert!(module.ckks_slots_to_coeffs_assign(&mut values).is_err());
        assert!(values.to_host::<B>() == input, "invalid transform changed its input");
        assert!(module.ckks_coeffs_to_slots_assign(&mut values).is_err());
        assert!(values.to_host::<B>() == input, "invalid transform changed its input");
    }
    observed
}

fn coefficient_codec<B, F>(module: &Module<B>, params: CKKSTestParams) -> Vec<Snapshot>
where
    B: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
{
    let mut observations = Vec::new();
    for (k, log_delta) in [(31usize, 20usize), (65, 40), (91, 70)] {
        for log_sparsity in [0, 2] {
            for slots in [SlotsKind::Real, SlotsKind::Complex] {
                let layout = CKKSLayout {
                    ring_kind: crate::CKKSRingKind::Standard,
                    glwe_layout: GLWELayout {
                        n: params.n.into(),
                        base2k: params.base2k.into(),
                        k: k.into(),
                        rank: 0usize.into(),
                    },
                    meta: CKKSMeta {
                        log_delta,
                        log_sparsity,
                        slots,
                    },
                };
                let count = params.n >> log_sparsity;
                let scale = F::from_usize(log_delta).unwrap().exp2();
                // Dyadic inputs include positive and negative ties. Both scalar
                // integer widths run, as do non-limb-aligned plaintext widths.
                let input: Vec<F> = (0..count)
                    .map(|i| {
                        F::from_i64((i % 17) as i64 - 8).unwrap() / F::from_f64(32.0).unwrap()
                            + F::from_f64(if i % 2 == 0 { 0.5 } else { -0.5 }).unwrap() / scale
                    })
                    .collect();
                let buffer = CKKSEncodingBuffer::<B::OwnedBuf, F>::from_host::<B>(&input);
                let mut pt = fixture_plaintext(module, &layout, 47);
                let original_layout = snapshot::<B, _>(&pt).layout;
                module.ckks_encode_coeffs_into(&mut pt, &buffer).unwrap();
                assert!(buffer.to_host::<B>() == input, "coefficient encoding changed its input");
                let encoded = snapshot::<B, _>(&pt);
                assert!(encoded.layout == original_layout, "coefficient encoding changed metadata");
                let want: Vec<F> = input.iter().map(|&x| (x * scale).round() / scale).collect();
                let canonical = coefficients::<B, F>(&pt);
                let gap = 1usize << log_sparsity;
                for (i, &value) in canonical.iter().enumerate() {
                    let expected = if i % gap == 0 { want[i / gap] } else { F::zero() };
                    assert!(value == expected, "coefficient quantization or sparse zeroing differs");
                }
                let mut decoded = CKKSEncodingBuffer::<B::OwnedBuf, F>::from_host::<B>(&vec![F::nan(); count]);
                module.ckks_decode_coeffs_into(&pt, &mut decoded).unwrap();
                assert!(decoded.to_host::<B>() == want, "coefficient decoding differs");
                assert!(snapshot::<B, _>(&pt) == encoded, "coefficient decoding changed plaintext");
                observations.push(encoded);
                for invalid_count in [0, 3, params.n + 1] {
                    let bad_input = vec![F::one(); invalid_count];
                    let mut bad = CKKSEncodingBuffer::<B::OwnedBuf, F>::from_host::<B>(&bad_input);
                    let before = snapshot::<B, _>(&pt);
                    assert!(module.ckks_encode_coeffs_into(&mut pt, &bad).is_err());
                    assert!(
                        snapshot::<B, _>(&pt) == before,
                        "invalid coefficient encoding changed plaintext"
                    );
                    assert!(module.ckks_decode_coeffs_into(&pt, &mut bad).is_err());
                    assert!(
                        bad.to_host::<B>() == bad_input,
                        "invalid coefficient decoding changed destination"
                    );
                }
                let integer_limit = F::from_usize(if k <= 63 { 63 } else { 127 }).unwrap().exp2();
                for invalid_value in [
                    F::nan(),
                    F::infinity(),
                    F::neg_infinity(),
                    integer_limit / scale,
                    F::max_value(),
                    -F::max_value(),
                ] {
                    let mut invalid = input.clone();
                    invalid[count - 1] = invalid_value;
                    let bad = CKKSEncodingBuffer::<B::OwnedBuf, F>::from_host::<B>(&invalid);
                    let before = snapshot::<B, _>(&pt);
                    assert!(module.ckks_encode_coeffs_into(&mut pt, &bad).is_err());
                    assert!(snapshot::<B, _>(&pt) == before, "failed quantization changed plaintext");
                }
            }
        }
    }
    observations
}

/// Pairs every encoding OEP, including plan creation/cache reuse and both
/// coefficient codecs. The independently decoded integer representation,
/// quantization ties, sparse tails, metadata, and failure preservation are
/// checked before comparing the two supplied implementations.
pub fn test_encoding_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
{
    assert!(params.n >= 8 && params.n.is_power_of_two());
    let reference = encoding_transforms::<BR, F>(r, params.n);
    let tested = encoding_transforms::<BT, F>(t, params.n);
    assert_eq!(reference.len(), tested.len());
    for (reference, tested) in reference.iter().zip(&tested) {
        scalar_close(reference, tested, transform_tolerance::<F>(params.n));
    }
    assert!(
        coefficient_codec::<BR, F>(r, params) == coefficient_codec::<BT, F>(t, params),
        "coefficient codec parity failed"
    );
}

/// Pairs PaCo coefficient embedding for both slot-order conventions. Each
/// backend receives equivalent exhausted ciphertexts and its own scratch bound;
/// prepared plaintexts are compared by decoded canonical coefficients.
pub fn test_paco_encoding_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSPaCoCoeffEncodingImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSPaCoCoeffEncodingImpl,
    F: PaCoScalar,
{
    for order in [PaCoSlotOrder::Natural, PaCoSlotOrder::BitRevLow] {
        let plan = PaCoPlan::new(params.log_n(), 2, 2, 17)
            .unwrap()
            .with_evaluation(
                40,
                5,
                PaCoDFTPlan::uniform(3, 1, 1, 0, 0).unwrap(),
                PaCoDFTPlan::uniform(1, 1, 1, 0, 0).unwrap(),
            )
            .unwrap()
            .with_slot_order(order);
        let layout = CKKSLayout {
            ring_kind: crate::CKKSRingKind::Standard,
            glwe_layout: GLWELayout {
                n: params.n.into(),
                base2k: params.base2k.into(),
                k: 17usize.into(),
                rank: 1usize.into(),
            },
            meta: CKKSMeta {
                log_delta: 10,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            },
        };
        let cr = fixture_ciphertext(r, &layout, 72);
        let ct = fixture_ciphertext(t, &layout, 72);
        let sr = snapshot::<BR, _>(&cr);
        let st = snapshot::<BT, _>(&ct);
        let bytes_r = BR::ckks_paco_coeff_encodings_tmp_bytes_impl::<F>(r, &plan).unwrap();
        let bytes_t = BT::ckks_paco_coeff_encodings_tmp_bytes_impl::<F>(t, &plan).unwrap();
        let pr = with_scratch::<BR, _>(bytes_r, |s| {
            BR::ckks_paco_coeff_encodings_impl::<F, _>(r, &cr, &plan, params.base2k.into(), s)
        })
        .unwrap();
        let pt = with_scratch::<BT, _>(bytes_t, |s| {
            BT::ckks_paco_coeff_encodings_impl::<F, _>(t, &ct, &plan, params.base2k.into(), s)
        })
        .unwrap();
        for (a, b) in pr.iter().zip(&pt) {
            compare_plaintexts::<BR, BT, F>(a, b);
            assert!(a.n().as_usize() == plan.n(), "PaCo degree differs");
            assert!(a.log_delta() == plan.log_delta_bsk(), "PaCo scale differs");
            assert!(a.log_budget() == plan.log_beta_budget(), "PaCo budget differs");
            assert!(a.log_sparsity() == 0, "PaCo sparsity differs");
            assert!(a.slots() == SlotsKind::Complex);
        }
        assert!(
            snapshot::<BR, _>(&cr) == sr && snapshot::<BT, _>(&ct) == st,
            "PaCo encoding changed its input"
        );
        let mut invalid = layout;
        invalid.glwe_layout.rank = 2usize.into();
        let ir = fixture_ciphertext(r, &invalid, 73);
        let it = fixture_ciphertext(t, &invalid, 73);
        let before_r = snapshot::<BR, _>(&ir);
        let before_t = snapshot::<BT, _>(&it);
        assert!(
            with_scratch::<BR, _>(bytes_r, |s| BR::ckks_paco_coeff_encodings_impl::<F, _>(
                r,
                &ir,
                &plan,
                params.base2k.into(),
                s
            ))
            .is_err()
        );
        assert!(
            with_scratch::<BT, _>(bytes_t, |s| BT::ckks_paco_coeff_encodings_impl::<F, _>(
                t,
                &it,
                &plan,
                params.base2k.into(),
                s
            ))
            .is_err()
        );
        assert!(
            snapshot::<BR, _>(&ir) == before_r && snapshot::<BT, _>(&it) == before_t,
            "failed PaCo encoding changed its input"
        );
    }
}

/// Pairs SHIP's real and complex coefficient embeddings, including both pt0
/// halves and every rotated pi vector. Shape failures preserve the bottom
/// ciphertext, and each backend executes with its own advertised scratch.
pub fn test_ship_encoding_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSShipCoeffEncodingImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSShipCoeffEncodingImpl,
    F: ShipScalar,
{
    let plan = ShipPlan::new(params.log_n(), 4, 40, 3, 2, 2, 2, 2).unwrap();
    let base2k = params.base2k;
    let layout = CKKSLayout {
        ring_kind: crate::CKKSRingKind::Standard,
        glwe_layout: GLWELayout {
            n: params.n.into(),
            base2k: base2k.into(),
            k: base2k.into(),
            rank: 1usize.into(),
        },
        meta: CKKSMeta {
            log_delta: base2k - 4,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        },
    };
    let cr = fixture_ciphertext(r, &layout, 83);
    let ct = fixture_ciphertext(t, &layout, 83);
    let sr = snapshot::<BR, _>(&cr);
    let st = snapshot::<BT, _>(&ct);
    for complex in [false, true] {
        let bytes_r = BR::ckks_ship_coeff_encodings_tmp_bytes_impl::<F>(r, &plan, base2k.into(), complex).unwrap();
        let bytes_t = BT::ckks_ship_coeff_encodings_tmp_bytes_impl::<F>(t, &plan, base2k.into(), complex).unwrap();
        let pr = with_scratch::<BR, _>(bytes_r, |s| {
            BR::ckks_ship_coeff_encodings_impl::<F, _>(r, &cr, &plan, base2k.into(), complex, s)
        })
        .unwrap();
        let pt = with_scratch::<BT, _>(bytes_t, |s| {
            BT::ckks_ship_coeff_encodings_impl::<F, _>(t, &ct, &plan, base2k.into(), complex, s)
        })
        .unwrap();
        compare_plaintexts::<BR, BT, F>(&pr.pt0, &pt.pt0);
        assert!(pr.pt0.k().as_usize() == plan.raised_k(base2k), "SHIP width differs");
        assert!(pr.pt0.log_delta() == plan.log_delta_work(), "SHIP scale differs");
        assert!(pr.pt0_2.is_some() == complex && pt.pt0_2.is_some() == complex);
        if let (Some(a), Some(b)) = (&pr.pt0_2, &pt.pt0_2) {
            compare_plaintexts::<BR, BT, F>(a, b);
        }
        assert_eq!(pr.pi.len(), plan.sparse_hamming_weight());
        assert_eq!(pt.pi.len(), pr.pi.len());
        for (a, b) in pr.pi.iter().zip(&pt.pi) {
            assert_eq!(a.len(), 4 * plan.theta());
            assert_eq!(b.len(), a.len());
            for (a, b) in a.iter().zip(b) {
                compare_plaintexts::<BR, BT, F>(a, b);
                assert!(a.k().as_usize() == plan.log_delta_work() + base2k, "SHIP pi width differs");
                assert!(a.log_delta() == plan.log_delta_work(), "SHIP pi scale differs");
            }
        }
        assert!(
            snapshot::<BR, _>(&cr) == sr && snapshot::<BT, _>(&ct) == st,
            "SHIP encoding changed its input"
        );
        let mut invalid = layout;
        invalid.glwe_layout.k = (base2k + 1).into();
        let ir = fixture_ciphertext(r, &invalid, 84);
        let it = fixture_ciphertext(t, &invalid, 84);
        let before_r = snapshot::<BR, _>(&ir);
        let before_t = snapshot::<BT, _>(&it);
        assert!(
            with_scratch::<BR, _>(bytes_r, |s| BR::ckks_ship_coeff_encodings_impl::<F, _>(
                r,
                &ir,
                &plan,
                base2k.into(),
                complex,
                s
            ))
            .is_err()
        );
        assert!(
            with_scratch::<BT, _>(bytes_t, |s| BT::ckks_ship_coeff_encodings_impl::<F, _>(
                t,
                &it,
                &plan,
                base2k.into(),
                complex,
                s
            ))
            .is_err()
        );
        assert!(
            snapshot::<BR, _>(&ir) == before_r && snapshot::<BT, _>(&it) == before_t,
            "failed SHIP encoding changed its input"
        );
    }
}
