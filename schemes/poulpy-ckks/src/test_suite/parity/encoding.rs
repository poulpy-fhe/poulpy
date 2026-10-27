//! Production encoding parity with caller-selected backends and scalar precision.
//!
//! Scalar buffers cross the boundary through explicit transfers. Integer
//! plaintexts are compared after downloading and decoding their canonical
//! coefficients, independently of either backend's floating-point decoder.

use crate::Scale;
use poulpy_core::layouts::GLWEPlaintextInfos;
use poulpy_core::{
    GLWEMaskFill,
    layouts::{Diagonals, GLWELayout, IntPolyInfos, LWEInfos, LinearTransformationStrategy},
};
use poulpy_hal::layouts::{Backend, Module, ZnxView};

use crate::{
    CKKSInfos, CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, SlotsKind,
    api::{CKKSEncodingOps, CKKSEncodingScalar, CKKSModuleInfos, PaCoScalar, ShipScalar},
    encoding::{ckks_encode_linear_transformation_from_diagonals, ckks_encode_linear_transformation_tmp_bytes},
    layouts::{
        CKKSEncodingBuffer, CKKSEncodingBufferToBackendMut, CKKSPlaintextOwned, ComplexDiagonals, PaCoContext, PaCoDFTPlan,
        PaCoPlan, PaCoSlotOrder, ShipPlan, slot_coeff_count,
    },
    oep::{CKKSEncodingImpl, CKKSPaCoCoeffEncodingImpl, CKKSShipCoeffEncodingImpl},
    reference::encoding::CKKSSlotEmbedding,
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

fn coefficients<BE, F>(pt: &CKKSPlaintextOwned<BE>) -> Vec<F>
where
    BE: Backend<ZnxWord = i64>,
    F: CKKSEncodingScalar,
{
    let host = pt.to_host_owned::<BE>();
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

fn encoding_transforms<BE, F>(module: &Module<BE>) -> Vec<Vec<F>>
where
    BE: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
{
    let plans = BE::ckks_encoding_plans_create(module).unwrap();
    // Repeat the geometric family in reverse order to reuse the same cached
    // plans after their first production-dispatch construction.
    let cap = 2 * module.ckks_max_slots();
    let dimensions: Vec<_> = (1..=cap.ilog2()).map(|log| 1usize << log).collect();
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
            let mut cached = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&input);
            let mut fresh = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&input);
            module.ckks_slots_to_coeffs_assign(&mut cached).unwrap();
            BE::ckks_slots_to_coeffs_assign_with_plan(
                module,
                &plans,
                &mut CKKSEncodingBufferToBackendMut::<BE, F>::to_backend_mut(&mut fresh),
            )
            .unwrap();
            let got = cached.to_host::<BE>();
            scalar_close(&got, &fresh.to_host::<BE>(), transform_tolerance::<F>(len));
            observed.push(got);
            module.ckks_coeffs_to_slots_assign(&mut cached).unwrap();
            BE::ckks_coeffs_to_slots_assign_with_plan(
                module,
                &plans,
                &mut CKKSEncodingBufferToBackendMut::<BE, F>::to_backend_mut(&mut fresh),
            )
            .unwrap();
            scalar_close(&cached.to_host::<BE>(), &fresh.to_host::<BE>(), transform_tolerance::<F>(len));
            // Real slots round-trip on every ring; the invariant ring drops imaginary parts.
            if real {
                scalar_close(&input, &cached.to_host::<BE>(), transform_tolerance::<F>(len));
            }
            observed.push(cached.to_host::<BE>());
        }
    }
    for len in [0, 1, 3, 2 * cap] {
        let input = vec![F::from_f64(0.375).unwrap(); len];
        let mut values = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&input);
        assert!(module.ckks_slots_to_coeffs_assign(&mut values).is_err());
        assert!(values.to_host::<BE>() == input, "invalid transform changed its input");
        assert!(module.ckks_coeffs_to_slots_assign(&mut values).is_err());
        assert!(values.to_host::<BE>() == input, "invalid transform changed its input");
    }
    observed
}

fn coefficient_codec<BE, F>(module: &Module<BE>, params: CKKSTestParams) -> Vec<Snapshot>
where
    BE: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    BE::Ring: CKKSSlotEmbedding,
    F: CKKSEncodingScalar,
    Module<BE>: GLWEMaskFill<BE>,
{
    let mut observations = Vec::new();
    for (k, log_delta) in [(31usize, 20usize), (65, 40), (91, 70)] {
        for slots in [SlotsKind::Real, SlotsKind::Complex] {
            let layout = CKKSLayout {
                glwe_layout: GLWELayout {
                    n: params.n.into(),
                    base2k: params.base2k.into(),
                    k: k.into(),
                    rank: 0usize.into(),
                },
                meta: GLWEPlaintextMeta {
                    scale: Scale::Log(log_delta),
                    slots,
                    log_sparsity: 0,
                },
            };
            let count = params.n;
            let scale = F::from_usize(log_delta).unwrap().exp2();
            // Dyadic inputs include positive and negative ties. Both scalar
            // integer widths run, as do non-limb-aligned plaintext widths.
            let input: Vec<F> = (0..count)
                .map(|i| {
                    F::from_i64((i % 17) as i64 - 8).unwrap() / F::from_f64(32.0).unwrap()
                        + F::from_f64(if i % 2 == 0 { 0.5 } else { -0.5 }).unwrap() / scale
                })
                .collect();
            let buffer = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&input);
            let mut pt = fixture_plaintext(module, &layout, 47);
            let mut expected_layout = snapshot::<BE, _>(&pt).layout;
            expected_layout.meta.slots = slots.meet(BE::Ring::SLOTS);
            module.ckks_encode_coeffs_into(&mut pt, &buffer).unwrap();
            assert!(buffer.to_host::<BE>() == input, "coefficient encoding changed its input");
            let encoded = snapshot::<BE, _>(&pt);
            assert!(
                encoded.layout == expected_layout,
                "coefficient encoding changed metadata other than the ring's slot kind"
            );
            let want: Vec<F> = input.iter().map(|&x| (x * scale).round() / scale).collect();
            let canonical = coefficients::<BE, F>(&pt);
            assert!(canonical == want, "coefficient quantization differs");
            let mut decoded = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&vec![F::nan(); count]);
            module.ckks_decode_coeffs_into(&pt, &mut decoded).unwrap();
            assert!(decoded.to_host::<BE>() == want, "coefficient decoding differs");
            assert!(snapshot::<BE, _>(&pt) == encoded, "coefficient decoding changed plaintext");
            observations.push(encoded);
            for invalid_count in [0, 3, params.n / 2, params.n + 1] {
                let bad_input = vec![F::one(); invalid_count];
                let mut bad = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&bad_input);
                let before = snapshot::<BE, _>(&pt);
                assert!(module.ckks_encode_coeffs_into(&mut pt, &bad).is_err());
                assert!(
                    snapshot::<BE, _>(&pt) == before,
                    "invalid coefficient encoding changed plaintext"
                );
                assert!(module.ckks_decode_coeffs_into(&pt, &mut bad).is_err());
                assert!(
                    bad.to_host::<BE>() == bad_input,
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
                let bad = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&invalid);
                let before = snapshot::<BE, _>(&pt);
                assert!(module.ckks_encode_coeffs_into(&mut pt, &bad).is_err());
                assert!(snapshot::<BE, _>(&pt) == before, "failed quantization changed plaintext");
            }
        }
    }
    observations
}

/// Encodes and decodes planar slots through the two slot-composition OEP
/// methods, for every slot count up to the module's capacity whose degree
/// reaches `floor`.
fn slot_codec<BE, F>(module: &Module<BE>, params: CKKSTestParams, floor: usize) -> Vec<(CKKSPlaintextOwned<BE>, Vec<F>)>
where
    BE: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BE>: GLWEMaskFill<BE>,
{
    let layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: params.n.into(),
            base2k: params.base2k.into(),
            k: 65usize.into(),
            rank: 0usize.into(),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(40),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
    };
    let max_slots = module.ckks_max_slots();
    (0..=max_slots.ilog2())
        .map(|log| 2 << log)
        .filter(|&len| slot_coeff_count(module, len) >= floor)
        .map(|len| {
            let layout = CKKSLayout {
                glwe_layout: GLWELayout {
                    n: slot_coeff_count(module, len).into(),
                    ..layout.glwe_layout
                },
                ..layout
            };
            let input: Vec<F> = (0..len)
                .map(|i| F::from_i64((i % 13) as i64 - 6).unwrap() / F::from_f64(32.0).unwrap())
                .collect();
            let mut slots = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&input);
            let mut pt = fixture_plaintext(module, &layout, 53);
            BE::ckks_encode_slots_assign_into(module, &mut pt, &mut slots).unwrap();
            let mut decoded = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&vec![F::nan(); len]);
            let before = snapshot::<BE, _>(&pt);
            BE::ckks_decode_slots_into(module, &pt, &mut decoded).unwrap();
            assert!(snapshot::<BE, _>(&pt) == before, "slot decoding changed plaintext");
            (pt, decoded.to_host::<BE>())
        })
        .collect()
}

/// Pairs every encoding OEP, including plan creation/cache reuse and both
/// coefficient codecs. The independently decoded integer representation,
/// quantization ties, metadata, and failure preservation are
/// checked before comparing the two supplied implementations.
pub fn test_encoding_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    BR::Ring: CKKSSlotEmbedding,
    BT::Ring: CKKSSlotEmbedding,
    F: CKKSEncodingScalar,
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    assert!(params.n >= 8 && params.n.is_power_of_two());
    let reference = encoding_transforms::<BR, F>(module_ref);
    let tested = encoding_transforms::<BT, F>(module_test);
    assert_eq!(reference.len(), tested.len());
    let tolerance = transform_tolerance::<F>(2 * module_ref.ckks_max_slots());
    for (reference, tested) in reference.iter().zip(&tested) {
        scalar_close(reference, tested, tolerance);
    }
    assert!(
        coefficient_codec::<BR, F>(module_ref, params) == coefficient_codec::<BT, F>(module_test, params),
        "coefficient codec parity failed"
    );
    diagonal_within_query::<BR, F>(module_ref, params);
    diagonal_within_query::<BT, F>(module_test, params);
}

/// A map of period 1 encodes within [`ckks_encode_linear_transformation_tmp_bytes`] exactly, on a plaintext at the
/// backend's minimum degree whose slots repeat the one value.
fn diagonal_within_query<BE, F>(module: &Module<BE>, params: CKKSTestParams)
where
    BE: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
{
    let slots = module.ckks_max_slots();
    let mut re = Diagonals::new(slots);
    re.set(0, vec![F::from_f64(0.5).unwrap()]);
    let map = ComplexDiagonals::new(re, Diagonals::new(slots));
    let bytes = ckks_encode_linear_transformation_tmp_bytes::<BE, _, F>(module, map.period());
    let lt = with_scratch::<BE, _>(bytes, |scratch| {
        ckks_encode_linear_transformation_from_diagonals::<BE, F>(
            module,
            params.base2k.into(),
            CoeffsMeta::from_delta_budget(8, 12),
            &map,
            LinearTransformationStrategy::Direct,
            false,
            scratch,
        )
    })
    .unwrap();
    assert_eq!(lt.first_diagonal_plaintext().unwrap().n().as_usize(), BE::MIN_DEGREE);
}

/// Pairs the two slot-composition OEP methods exactly, for every slot count up
/// to the module's capacity: the encoded plaintexts and the decoded slots must
/// be bit-identical.
pub fn test_slot_encoding_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    // Both sides encode the slot counts whose degree both backends allocate.
    let floor = BR::MIN_DEGREE.max(BT::MIN_DEGREE);
    let reference = slot_codec::<BR, F>(module_ref, params, floor);
    let tested = slot_codec::<BT, F>(module_test, params, floor);
    assert_eq!(reference.len(), tested.len());
    for ((pt_r, slots_r), (pt_t, slots_t)) in reference.iter().zip(&tested) {
        assert!(snapshot::<BR, _>(pt_r) == snapshot::<BT, _>(pt_t), "slot encoding differs");
        assert!(
            bytemuck::cast_slice::<F, u8>(slots_r) == bytemuck::cast_slice::<F, u8>(slots_t),
            "slot decoding differs"
        );
    }
}

fn paco_context<BE, F>(module: &Module<BE>, params: CKKSTestParams, plan: &PaCoPlan) -> PaCoContext<BE, F>
where
    BE: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSPaCoCoeffEncodingImpl,
    F: PaCoScalar,
{
    let bytes = ckks_encode_linear_transformation_tmp_bytes::<BE, _, F>(module, module.ckks_max_slots());
    with_scratch::<BE, _>(bytes, |s| PaCoContext::compile(module, params.base2k.into(), plan.clone(), s)).unwrap()
}

/// Pairs PaCo coefficient embedding for both slot-order conventions. Each
/// backend receives equivalent exhausted ciphertexts and its own scratch bound;
/// prepared plaintexts are compared by decoded canonical coefficients.
pub fn test_paco_encoding_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSPaCoCoeffEncodingImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSPaCoCoeffEncodingImpl,
    F: PaCoScalar,
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    for order in [PaCoSlotOrder::Natural, PaCoSlotOrder::BitRevLow] {
        let plan = PaCoPlan::new(params.log_n(), 2, 2, 17)
            .unwrap()
            .with_evaluation(
                40,
                5,
                PaCoDFTPlan::uniform(3, 1, 1, 30, 8).unwrap(),
                PaCoDFTPlan::uniform(1, 1, 1, 30, 8).unwrap(),
            )
            .unwrap()
            .with_slot_order(order);
        let layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: params.n.into(),
                base2k: params.base2k.into(),
                k: 17usize.into(),
                rank: 1usize.into(),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(10),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
            },
        };
        let cr = fixture_ciphertext(module_ref, &layout, 72);
        let ct = fixture_ciphertext(module_test, &layout, 72);
        let sr = snapshot::<BR, _>(&cr);
        let st = snapshot::<BT, _>(&ct);
        let ctx_r = paco_context::<BR, F>(module_ref, params, &plan);
        let ctx_t = paco_context::<BT, F>(module_test, params, &plan);
        let bytes_r = BR::ckks_paco_coeff_encodings_tmp_bytes::<F>(module_ref, &ctx_r).unwrap();
        let bytes_t = BT::ckks_paco_coeff_encodings_tmp_bytes::<F>(module_test, &ctx_t).unwrap();
        let pr = with_scratch::<BR, _>(bytes_r, |s| BR::ckks_paco_coeff_encodings::<F, _>(module_ref, &cr, &ctx_r, s)).unwrap();
        let pt = with_scratch::<BT, _>(bytes_t, |s| {
            BT::ckks_paco_coeff_encodings::<F, _>(module_test, &ct, &ctx_t, s)
        })
        .unwrap();
        for (a, b) in pr.iter().zip(&pt) {
            compare_plaintexts::<BR, BT, F>(a, b);
            assert!(a.n().as_usize() == plan.n(), "PaCo degree differs");
            assert!(a.log_delta() == plan.log_delta_bsk(), "PaCo scale differs");
            assert!(a.log_budget() == plan.log_beta_budget(), "PaCo budget differs");
            assert!(a.slots() == SlotsKind::Complex);
        }
        assert!(
            snapshot::<BR, _>(&cr) == sr && snapshot::<BT, _>(&ct) == st,
            "PaCo encoding changed its input"
        );
        let mut invalid = layout;
        invalid.glwe_layout.rank = 2usize.into();
        let ir = fixture_ciphertext(module_ref, &invalid, 73);
        let it = fixture_ciphertext(module_test, &invalid, 73);
        let before_r = snapshot::<BR, _>(&ir);
        let before_t = snapshot::<BT, _>(&it);
        assert!(with_scratch::<BR, _>(bytes_r, |s| BR::ckks_paco_coeff_encodings::<F, _>(module_ref, &ir, &ctx_r, s)).is_err());
        assert!(
            with_scratch::<BT, _>(bytes_t, |s| BT::ckks_paco_coeff_encodings::<F, _>(
                module_test,
                &it,
                &ctx_t,
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
pub fn test_ship_encoding_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSShipCoeffEncodingImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncodingImpl<F> + CKKSShipCoeffEncodingImpl,
    F: ShipScalar,
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    let plan = ShipPlan::new(params.log_n(), 4, 40, 3, 2, 2, 2, 2).unwrap();
    let base2k = params.base2k;
    let layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: params.n.into(),
            base2k: base2k.into(),
            k: base2k.into(),
            rank: 1usize.into(),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(base2k - 4),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
    };
    let cr = fixture_ciphertext(module_ref, &layout, 83);
    let ct = fixture_ciphertext(module_test, &layout, 83);
    let sr = snapshot::<BR, _>(&cr);
    let st = snapshot::<BT, _>(&ct);
    for complex in [false, true] {
        let bytes_r = BR::ckks_ship_coeff_encodings_tmp_bytes::<F>(module_ref, base2k.into(), complex, &plan).unwrap();
        let bytes_t = BT::ckks_ship_coeff_encodings_tmp_bytes::<F>(module_test, base2k.into(), complex, &plan).unwrap();
        let pr = with_scratch::<BR, _>(bytes_r, |s| {
            BR::ckks_ship_coeff_encodings::<F, _>(module_ref, base2k.into(), complex, &cr, &plan, s)
        })
        .unwrap();
        let pt = with_scratch::<BT, _>(bytes_t, |s| {
            BT::ckks_ship_coeff_encodings::<F, _>(module_test, base2k.into(), complex, &ct, &plan, s)
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
        let ir = fixture_ciphertext(module_ref, &invalid, 84);
        let it = fixture_ciphertext(module_test, &invalid, 84);
        let before_r = snapshot::<BR, _>(&ir);
        let before_t = snapshot::<BT, _>(&it);
        assert!(
            with_scratch::<BR, _>(bytes_r, |s| BR::ckks_ship_coeff_encodings::<F, _>(
                module_ref,
                base2k.into(),
                complex,
                &ir,
                &plan,
                s
            ))
            .is_err()
        );
        assert!(
            with_scratch::<BT, _>(bytes_t, |s| BT::ckks_ship_coeff_encodings::<F, _>(
                module_test,
                base2k.into(),
                complex,
                &it,
                &plan,
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
