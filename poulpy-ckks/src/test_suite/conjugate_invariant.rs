use std::collections::HashMap;

use crate::{
    CKKSInfos, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSDecryptOps, CKKSEncodingHostOps, CKKSImagOps, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSMulOps, CKKSNegOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
    layouts::CKKSModuleAlloc,
};
use poulpy_core::layouts::{GGLWEInfos, LWEInfos};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{HostBytesBackend, HostDataMut, HostDataRef, Module},
};

use super::{
    CKKSTestParams,
    helpers::{
        TestContextBackend, TestContextHostModule, TestContextModule, alloc_ct, alloc_scratch, ckks_encrypt_pt, gen_atk,
        gen_sk_with_raw, gen_tsk,
    },
};

pub fn test_conjugate_invariant_leveled<BE>(
    mut params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::ConjugateInvariant>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingHostOps<BE, f64>
        + CKKSLinearTransformationOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + poulpy_hal::api::CnvPVecAlloc<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
{
    params.prec_meta.slots = SlotsKind::Real;
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let slots = params.n;
    let re1 = (0..slots).map(|i| (i as f64 + 1.0) / 521.0).collect::<Vec<_>>();
    let re2 = (0..slots).map(|i| (slots - i) as f64 / 389.0).collect::<Vec<_>>();
    let im = vec![0.0; slots];

    let mut pt1 = module.ckks_pt_vec_alloc(params.base2k.into(), params.prec().k());
    pt1.set_meta(params.prec().meta());
    module
        .ckks_encode_reim_into(&mut pt1, &re1, &im, &mut scratch.borrow())
        .unwrap();
    let ct1 = ckks_encrypt_pt(
        &params,
        module,
        &sk,
        params.k,
        &pt1.to_host_owned::<BE>(),
        &mut scratch.borrow(),
    );

    let mut pt2 = module.ckks_pt_vec_alloc(params.base2k.into(), params.prec().k());
    pt2.set_meta(params.prec().meta());
    module
        .ckks_encode_reim_into(&mut pt2, &re2, &im, &mut scratch.borrow())
        .unwrap();
    let ct2 = ckks_encrypt_pt(
        &params,
        module,
        &sk,
        params.k,
        &pt2.to_host_owned::<BE>(),
        &mut scratch.borrow(),
    );

    let mut decrypted = module.ckks_pt_vec_alloc(params.base2k.into(), params.prec().k());
    let mut got_re = vec![0.0; slots];
    let mut got_im = vec![0.0; slots];
    macro_rules! assert_output {
        ($ct:expr, $want:expr) => {{
            decrypted.set_meta(params.prec().meta());
            module.ckks_decrypt(&mut decrypted, $ct, &sk, &mut scratch.borrow()).unwrap();
            module
                .ckks_decode_reim_into(&decrypted, &mut got_re, &mut got_im, &mut scratch.borrow())
                .unwrap();
            assert_reim_close(&got_re, &got_im, $want);
            assert_eq!($ct.slots(), SlotsKind::Real);
        }};
    }
    assert_output!(&ct1, &re1);

    let mut sum = alloc_ct(&params, module, params.k);
    module.ckks_add_into(&mut sum, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    let want_sum = re1.iter().zip(&re2).map(|(a, b)| a + b).collect::<Vec<_>>();
    assert_output!(&sum, &want_sum);

    let mut difference = alloc_ct(&params, module, params.k);
    module
        .ckks_sub_into(&mut difference, &ct1, &ct2, &mut scratch.borrow())
        .unwrap();
    let want_difference = re1.iter().zip(&re2).map(|(a, b)| a - b).collect::<Vec<_>>();
    assert_output!(&difference, &want_difference);

    let mut negated = alloc_ct(&params, module, params.k);
    module.ckks_neg_into(&mut negated, &ct1, &mut scratch.borrow()).unwrap();
    let want_negated = re1.iter().map(|value| -value).collect::<Vec<_>>();
    assert_output!(&negated, &want_negated);

    let mut copied = alloc_ct(&params, module, params.k);
    module.ckks_copy(&mut copied, &ct1, &mut scratch.borrow()).unwrap();
    assert_output!(&copied, &re1);

    let mut add_plaintext = alloc_ct(&params, module, params.k);
    module
        .ckks_add_pt_vec_into(&mut add_plaintext, &ct1, &pt2, &mut scratch.borrow())
        .unwrap();
    assert_output!(&add_plaintext, &want_sum);

    let mut multiply_plaintext = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut multiply_plaintext, &ct1, &pt2, &mut scratch.borrow())
        .unwrap();
    let want_product = re1.iter().zip(&re2).map(|(a, b)| a * b).collect::<Vec<_>>();
    assert_output!(&multiply_plaintext, &want_product);

    let mut product = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_into(&mut product, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_output!(&product, &want_product);

    let mut square = alloc_ct(&params, module, params.k);
    module
        .ckks_square_into(&mut square, &ct1, &tsk, &mut scratch.borrow())
        .unwrap();
    let want_square = re1.iter().map(|value| value * value).collect::<Vec<_>>();
    assert_output!(&square, &want_square);

    let mut scaled = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pow2_into(&mut scaled, &ct1, 2, &mut scratch.borrow())
        .unwrap();
    let want_scaled = re1.iter().map(|value| 4.0 * value).collect::<Vec<_>>();
    assert_output!(&scaled, &want_scaled);

    let rotations = [
        3,
        -2,
        (slots / 2) as i64,
        slots as i64 + 3,
        -(slots as i64) - 2,
        i64::MIN + 1,
        i64::MAX,
    ];
    let mut keys = HashMap::new();
    for rotation in rotations {
        let p = module.ckks_galois_element(rotation);
        keys.insert(p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    for rotation in rotations {
        let mut rotated = alloc_ct(&params, module, params.k);
        module
            .ckks_rotate_into(&mut rotated, &ct1, rotation, &keys, &mut scratch.borrow())
            .unwrap();
        let want = (0..slots)
            .map(|i| re1[(i + rotation.rem_euclid(slots as i64) as usize) % slots])
            .collect::<Vec<_>>();
        assert_output!(&rotated, &want);
        let mut assigned = ct1.clone();
        module
            .ckks_rotate_assign(&mut assigned, rotation, &keys, &mut scratch.borrow())
            .unwrap();
        assert_output!(&assigned, &want);
    }

    let empty = HashMap::<i64, poulpy_core::layouts::GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>::new();
    for rotation in [0, slots as i64, -(slots as i64), i64::MIN] {
        let mut rotated = ct1.clone();
        module
            .ckks_rotate_into(&mut rotated, &ct1, rotation, &empty, &mut scratch.arena())
            .unwrap();
        module
            .ckks_rotate_assign(&mut rotated, rotation, &empty, &mut scratch.arena())
            .unwrap();
        assert_output!(&rotated, &re1);
    }
    let mut rotation_scratch =
        poulpy_hal::layouts::ScratchOwned::<BE>::alloc(module.ckks_rotate_tmp_bytes(&ct1, &params.atk_layout()));
    let mut narrowed = alloc_ct(&params, module, params.k - params.base2k);
    module
        .ckks_rotate_into(&mut narrowed, &ct1, slots as i64, &empty, &mut rotation_scratch.arena())
        .unwrap();
    assert_eq!(narrowed.log_budget(), ct1.log_budget() - params.base2k);
    assert_eq!(narrowed.meta(), ct1.meta());
    assert_output!(&narrowed, &re1);
    let mut exhausted = alloc_ct(&params, module, ct1.log_delta() - 1);
    let exhausted_before = exhausted.clone();
    assert!(
        module
            .ckks_rotate_into(&mut exhausted, &ct1, 0, &empty, &mut rotation_scratch.arena())
            .is_err()
    );
    assert_eq!(exhausted.meta(), exhausted_before.meta());
    assert_eq!(exhausted.k(), exhausted_before.k());
    assert_eq!(exhausted.data().data().as_ref(), exhausted_before.data().data().as_ref());

    let sparse_slots = slots / 4;
    let mut compact = module.ckks_pt_vec_alloc_compact(sparse_slots, params.base2k.into(), params.prec().k());
    compact.set_meta(params.prec().meta());
    compact.set_log_sparsity(2);
    module
        .ckks_encode_reim_into(&mut compact, &re2[..sparse_slots], &im[..sparse_slots], &mut scratch.arena())
        .unwrap();
    let sparse = ckks_encrypt_pt(
        &params,
        module,
        &sk,
        params.k,
        &compact.to_host_owned::<BE>(),
        &mut scratch.arena(),
    );
    assert_eq!(sparse.log_sparsity(), 2);
    let sparse_want = (0..slots).map(|i| re2[i % sparse_slots]).collect::<Vec<_>>();
    assert_output!(&sparse, &sparse_want);
    let mut mixed = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut mixed, &ct1, &compact, &mut scratch.arena())
        .unwrap();
    let mixed_want = (0..slots).map(|i| re1[i] * sparse_want[i]).collect::<Vec<_>>();
    assert_output!(&mixed, &mixed_want);

    use crate::polynomial::{Basis, EncodeBSGS, Polynomial};
    for basis in [Basis::Monomial, Basis::Chebyshev] {
        let poly = Polynomial::new(basis, vec![0.125f64, 0.25, 0.125, -0.0625]);
        let host_poly = poly
            .encode_bsgs::<BE::Ring>(host_module, params.base2k.into(), params.prec().into())
            .unwrap();
        let encoded = host_poly.map_baby_steps_ref(|pt| super::helpers::upload_pt(module, pt));
        let mut result = alloc_ct(&params, module, params.k);
        module
            .ckks_eval_poly_real_const_coeffs(&mut result, &ct1, &encoded, &tsk, &mut scratch.arena())
            .unwrap();
        let want = re1
            .iter()
            .map(|&x| match basis {
                Basis::Monomial => 0.125 + 0.25 * x + 0.125 * x * x - 0.0625 * x * x * x,
                Basis::Chebyshev => 0.125 + 0.25 * x + 0.125 * (2.0 * x * x - 1.0) - 0.0625 * (4.0 * x * x * x - 3.0 * x),
            })
            .collect::<Vec<_>>();
        assert_output!(&result, &want);
    }

    use crate::layouts::ComplexDiagonals;
    use poulpy_core::layouts::{Diagonals, Evaluate, LinearTransformationStrategy};
    use poulpy_hal::{
        api::ScratchOwnedAlloc,
        layouts::{CyclotomicOrder, ScratchOwned},
    };
    let mut lt_scratch =
        ScratchOwned::<BE>::alloc(module.ckks_eval_linear_transformation_streamed_tmp_bytes(&ct1, &params.atk_layout()));
    for diagonal_slots in [slots, sparse_slots] {
        let mut re = Diagonals::new(diagonal_slots);
        for d in [0, 1, 3, 5] {
            re.set(d, (0..diagonal_slots).map(|i| ((i + d as usize) % 7) as f64 / 16.0).collect());
        }
        let diagonals = ComplexDiagonals::new(re, Diagonals::new(diagonal_slots));
        let mut complex = diagonals.clone();
        complex.im.set(0, vec![0.25; diagonal_slots]);
        assert!(
            crate::reference::ckks_encode_linear_transformation_from_diagonals(
                module,
                params.base2k.into(),
                params.prec().into(),
                &complex,
                LinearTransformationStrategy::Direct,
                false,
                &mut scratch.arena(),
            )
            .is_err()
        );
        for strategy in [
            LinearTransformationStrategy::Direct,
            LinearTransformationStrategy::Bsgs { giant_step: 2 },
        ] {
            let lt = crate::reference::ckks_encode_linear_transformation_from_diagonals(
                module,
                params.base2k.into(),
                params.prec().into(),
                &diagonals,
                strategy,
                false,
                &mut scratch.arena(),
            )
            .unwrap();
            let mut keys = HashMap::new();
            for p in lt.galois_elements(module.cyclotomic_order()) {
                keys.entry(p)
                    .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.arena()));
            }
            let input = if diagonal_slots == slots { &ct1 } else { &sparse };
            let values = if diagonal_slots == slots {
                &re1[..]
            } else {
                &re2[..sparse_slots]
            };
            let (want, _) = diagonals.evaluate((values, &im[..diagonal_slots]), strategy);
            let want = (0..slots).map(|i| want[i % diagonal_slots]).collect::<Vec<_>>();
            let mut result = alloc_ct(&params, module, params.k);
            module
                .ckks_eval_linear_transformation_self_into(&mut result, input, &lt, &keys, &mut lt_scratch.arena())
                .unwrap();
            assert_output!(&result, &want);
            let first = lt.first_diagonal_plaintext().unwrap();
            let mut prepared =
                crate::api::LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
            module
                .ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut lt_scratch.arena())
                .unwrap();
            module
                .ckks_eval_linear_transformation_self_into(&mut result, input, &prepared, &keys, &mut lt_scratch.arena())
                .unwrap();
            assert_output!(&result, &want);
        }
    }

    let mut rejected = ct1.clone();
    let original = rejected.to_host_owned::<BE>();
    assert!(
        module
            .ckks_conjugate_into(&mut rejected, &ct1, &keys, &mut scratch.borrow())
            .is_err()
    );
    assert!(
        module
            .ckks_conjugate_assign(&mut rejected, &keys, &mut scratch.borrow())
            .is_err()
    );
    assert!(module.ckks_mul_i_into(&mut rejected, &ct1, &mut scratch.borrow()).is_err());
    assert!(module.ckks_div_i_assign(&mut rejected, &mut scratch.borrow()).is_err());
    assert_eq!(
        rejected.to_host_owned::<BE>().data().data().as_ref(),
        original.data().data().as_ref()
    );
    assert_eq!(rejected.meta(), original.meta());
}

fn assert_reim_close(re: &[f64], im: &[f64], want: &[f64]) {
    for (actual, expected) in re.iter().zip(want) {
        assert!((actual - expected).abs() < 1e-5, "actual={actual}, expected={expected}");
    }
    assert!(im.iter().all(|&value| value == 0.0));
}

pub fn test_conjugate_invariant_encoding<BE, F>(module: &Module<BE>, log_delta: usize, tolerance: f64)
where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::ConjugateInvariant>,
    F: crate::api::CKKSEncodingScalar,
    Module<BE>: CKKSModuleAlloc<BE> + crate::api::CKKSEncodingOps<BE, F>,
{
    use crate::{CKKSMeta, api::CKKSEncodingOps, layouts::CKKSEncodingBuffer};
    use poulpy_hal::{AlignedBuf, api::ScratchOwnedAlloc, layouts::ScratchOwned};

    assert_eq!(module.ckks_max_slots(), module.n());
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(module.n()));
    for log_slots in 0..=module.n().ilog2() {
        let slots = 1usize << log_slots;
        let re = (0..slots)
            .map(|i| F::from_f64(((17 * i + 3) % 31) as f64 / 37.0).unwrap())
            .collect::<Vec<_>>();
        let im = vec![F::from_f64(0.25).unwrap(); slots];
        for compact in [false, true] {
            let mut pt = if compact {
                module.ckks_pt_vec_alloc_compact(slots, 19usize.into(), (log_delta + 36).into())
            } else {
                module.ckks_pt_vec_alloc(19usize.into(), (log_delta + 36).into())
            };
            assert_eq!(pt.slots(), SlotsKind::Real);
            if compact {
                assert_eq!(pt.n().as_usize(), slots.max(BE::MIN_DEGREE).min(module.n()));
            }
            pt.set_meta(CKKSMeta {
                log_delta,
                log_sparsity: (module.n() / slots).ilog2() as usize,
                slots: SlotsKind::Real,
            });
            module.ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch.arena()).unwrap();
            let mut got_re = vec![F::zero(); slots];
            let mut got_im = vec![F::one(); slots];
            module
                .ckks_decode_reim_into(&pt, &mut got_re, &mut got_im, &mut scratch.arena())
                .unwrap();
            for (got, expected) in got_re.iter().zip(&re) {
                assert!((*got - *expected).abs().to_f64().unwrap() < tolerance);
            }
            assert!(got_im.iter().all(|&value| value == F::zero()));

            let mut coefficients = vec![F::zero(); slots];
            module
                .ckks_decode_coeffs_host_into(&pt, &mut coefficients, &mut scratch.arena())
                .unwrap();
            let mut power = 1usize;
            for value in &re {
                let expected = coefficients[0].to_f64().unwrap()
                    + coefficients
                        .iter()
                        .enumerate()
                        .skip(1)
                        .map(|(j, c)| {
                            2.0 * c.to_f64().unwrap() * (std::f64::consts::TAU * (power * j) as f64 / (4 * slots) as f64).cos()
                        })
                        .sum::<f64>();
                assert!((value.to_f64().unwrap() - expected).abs() < 1e-8);
                power = (power * 5) % (4 * slots);
            }

            let mut raw = module.ckks_plaintext_alloc(pt.n(), pt.base2k(), pt.k());
            raw.set_meta(pt.meta());
            module
                .ckks_encode_coeffs_host_into(&mut raw, &coefficients, &mut scratch.arena())
                .unwrap();
            module
                .ckks_decode_reim_into(&raw, &mut got_re, &mut got_im, &mut scratch.arena())
                .unwrap();
            for (got, expected) in got_re.iter().zip(&re) {
                assert!((*got - *expected).abs().to_f64().unwrap() < tolerance);
            }
        }
        let mut values = CKKSEncodingBuffer::<AlignedBuf, F>::from_host::<BE>(&re.iter().chain(&im).copied().collect::<Vec<_>>());
        module.ckks_slots_to_coeffs_assign(&mut values).unwrap();
        module.ckks_coeffs_to_slots_assign(&mut values).unwrap();
        let values = values.to_host::<BE>();
        for (got, expected) in values[..slots].iter().zip(&re) {
            assert!((*got - *expected).abs().to_f64().unwrap() < tolerance);
        }
        assert!(values[slots..].iter().all(|&value| value == F::zero()));
    }
    let mut oversized = CKKSEncodingBuffer::<AlignedBuf, F>::from_host::<BE>(&vec![F::zero(); 4 * module.n()]);
    assert!(module.ckks_slots_to_coeffs_assign(&mut oversized).is_err());
    let mut pt = module.ckks_pt_vec_alloc(19usize.into(), (log_delta + 36).into());
    assert!(module.ckks_encode_coeffs_into(&mut pt, &oversized).is_err());
}

/// Instantiates conjugate invariant encoding and leveled-operation tests.
#[macro_export]
macro_rules! conjugate_invariant_ckks_test_suite {
    ($name:ident, $backend:ty, $standard:ty, $params:expr) => {
        mod $name {
            use poulpy_hal::{
                api::ModuleNew,
                layouts::{HostBytesBackend, Module},
            };
            #[test]
            fn ckks_ci_encoding_f64() {
                let module = Module::<$backend>::new(256);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_encoding::<$backend, f64>(&module, 40, 1e-8);
            }
            #[test]
            fn ckks_ci_encoding_quad() {
                let module = Module::<$backend>::new(256);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_encoding::<$backend, $crate::Quad>(
                    &module, 80, 1e-20,
                );
            }
            #[test]
            fn ckks_ci_bootstrap_s2c() {
                let params = $params;
                let ci = Module::<$backend>::new(params.n as u64);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_bootstrapping(
                    params,
                    ci,
                    Module::<$standard>::new((2 * params.n) as u64),
                    true,
                    true,
                    0,
                    false,
                    6,
                );
            }
            #[test]
            fn ckks_ci_bootstrap_s2c_without_guards() {
                let params = $params;
                let ci = Module::<$backend>::new(params.n as u64);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_bootstrapping(
                    params,
                    ci,
                    Module::<$standard>::new((2 * params.n) as u64),
                    true,
                    true,
                    0,
                    false,
                    0,
                );
            }
            #[test]
            fn ckks_ci_bootstrap_s2c_sparse() {
                let params = $params;
                let ci = Module::<$backend>::new(params.n as u64);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_bootstrapping(
                    params,
                    ci,
                    Module::<$standard>::new((2 * params.n) as u64),
                    true,
                    true,
                    2,
                    false,
                    6,
                );
            }
            #[test]
            fn ckks_ci_bootstrap_s2c_without_encapsulation() {
                let params = $params;
                let ci = Module::<$backend>::new(params.n as u64);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_bootstrapping(
                    params,
                    ci,
                    Module::<$standard>::new((2 * params.n) as u64),
                    true,
                    false,
                    0,
                    false,
                    6,
                );
            }
            #[test]
            fn ckks_ci_bootstrap_c2s() {
                let params = $params;
                let ci = Module::<$backend>::new(params.n as u64);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_bootstrapping(
                    params,
                    ci,
                    Module::<$standard>::new((2 * params.n) as u64),
                    false,
                    true,
                    0,
                    false,
                    0,
                );
            }
            #[test]
            fn ckks_ci_bootstrap_eval_round() {
                let params = $params;
                let ci = Module::<$backend>::new(params.n as u64);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_bootstrapping(
                    params,
                    ci,
                    Module::<$standard>::new((2 * params.n) as u64),
                    true,
                    true,
                    0,
                    true,
                    6,
                );
            }
            #[test]
            #[ignore = "full CI bootstrapping preset at 2^15 slots"]
            fn ckks_ci_bootstrap_preset_n15() {
                let preset = $crate::presets::bootstrapping::ci_n15_d35_k720_p19_s2c().unwrap();
                let ci = Module::<$backend>::new(preset.n() as u64);
                $crate::test_suite::presets::ci_bootstrapping_preset_meets_precision(
                    ci,
                    Module::<$standard>::new((2 * preset.n()) as u64),
                    preset,
                    $params.base2k,
                );
            }
            #[test]
            #[ignore = "full CI bootstrapping preset at 2^16 slots"]
            fn ckks_ci_bootstrap_preset_n16() {
                let preset = $crate::presets::bootstrapping::ci_n16_d35_k720_p19_s2c().unwrap();
                let ci = Module::<$backend>::new(preset.n() as u64);
                $crate::test_suite::presets::ci_bootstrapping_preset_meets_precision(
                    ci,
                    Module::<$standard>::new((2 * preset.n()) as u64),
                    preset,
                    $params.base2k,
                );
            }
            #[test]
            fn ckks_ci_leveled() {
                let params = $params;
                let module = Module::<$backend>::new(params.n as u64);
                let host = Module::<HostBytesBackend>::new(params.n as u64);
                $crate::test_suite::conjugate_invariant::test_conjugate_invariant_leveled(params, &module, &host);
            }
        }
    };
}

#[allow(clippy::too_many_arguments)]
pub fn test_conjugate_invariant_bootstrapping<BE, STD>(
    mut params: CKKSTestParams,
    ci: Module<BE>,
    standard: Module<STD>,
    s2c_first: bool,
    encapsulate: bool,
    log_sparsity: usize,
    eval_round: bool,
    guard_bits: usize,
) where
    BE: TestContextBackend<Ring = poulpy_hal::layouts::ConjugateInvariant>,
    STD: TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    for<'a> STD::BufRef<'a>: HostDataRef,
    for<'a> STD::BufMut<'a>: HostDataMut,
    Module<STD>: TestContextModule<STD>
        + crate::api::CKKSEncodingOps<STD, f64>
        + crate::api::CKKSBootstrappingOps<STD>
        + crate::api::CKKSDFTMatrixOps<STD, f64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextModule<BE>
        + crate::api::CKKSEncodingOps<BE, f64>
        + crate::api::CKKSBootstrappingOps<BE>
        + crate::api::CKKSDFTMatrixOps<BE, f64>,
    Module<HostBytesBackend>: TestContextHostModule,
{
    use crate::{CoeffsMeta, layouts::*, polynomial::SplitStrategy};
    let layers = ci.n().ilog2() as usize;
    let schedule: Vec<_> = (0..layers).step_by(2).map(|i| ((layers - i).min(2), 2)).collect();
    let log_delta = 35;
    let log_msg_ratio = if s2c_first { 13 } else { 8 };
    let plan = BootstrappingPlan::new(
        if s2c_first {
            BootstrappingPipeline::S2CFirst
        } else {
            BootstrappingPipeline::C2SFirst
        },
        BootstrappingTechniques {
            sparse_secret_encapsulation: encapsulate.then_some(SparseSecretEncapsulation { hamming_weight: 32 }),
            eval_round_plus: eval_round.then(|| EvalRoundPlus {
                coeffs_to_slots_bypass: DFTPlan::new(
                    DFTType::Encode,
                    vec![(1, 1); layers],
                    DFTOutputFormat::SplitRealAndImag,
                    CoeffsMeta::from_delta_budget(96, 4),
                )
                .unwrap(),
            }),
        },
        DFTPlan::new(
            DFTType::Encode,
            schedule.clone(),
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(48, 3),
        )
        .unwrap(),
        EvalModPlan {
            eval_mod_type: EvalModType::CosHKEven,
            log_msg_ratio,
            f_mod_degree: 30,
            f_mod_interval: 16,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(42, 4),
            f_mod_log_delta: 58,
        },
        DFTPlan::new(
            DFTType::Decode,
            schedule,
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(28, 2),
        )
        .unwrap()
        .with_scaling(if s2c_first { 0.5 } else { 256.0 })
        .unwrap(),
    )
    .unwrap();
    let plan = if s2c_first {
        plan.with_c2s_guard_bits(guard_bits).unwrap()
    } else {
        plan
    };
    let output_k = 4 * params.base2k - 1;
    let input_k = plan.input_k(log_delta + log_msg_ratio);
    params.n = ci.n();
    params.k = plan.bootstrap_k(output_k + 1, log_delta);
    params.prec_meta = crate::CKKSMeta {
        log_delta,
        log_sparsity,
        slots: SlotsKind::Real,
    };
    params.prec_log_budget = 8;
    params.dsize = if params.base2k < 40 { 7 } else { 3 };
    params.hw = if encapsulate { 128 } else { 32 };
    let standard_params = CKKSTestParams {
        n: 2 * params.n,
        ..params
    };
    let keys_layout = CIBootstrappingKeysLayout {
        bootstrap_keys: BootstrappingKeysLayout {
            automorphism_key: standard_params.atk_layout().layout,
            tensor_key: standard_params.tsk_layout().layout,
            encapsulation: encapsulate.then(|| EncapsulationKeysLayout {
                dense_to_sparse: standard_params.ksk_layout(log_delta + log_msg_ratio).layout,
                sparse_to_dense: standard_params.ksk_layout(params.k).layout,
            }),
        },
        ci_to_standard: standard_params.ksk_layout(input_k).layout,
        standard_to_ci: standard_params
            .ksk_layout(
                output_k
                    + 1
                    + if s2c_first {
                        plan.c2s_guard_bits()
                    } else {
                        plan.eval_mod().f_mod_log_delta - log_delta
                    },
            )
            .layout,
    };
    let mut run = super::presets::CIBootstrappingRun::setup(ci, standard, &plan, params, keys_layout, input_k, output_k);
    for pair in [false, true] {
        run.bootstrap(pair);
        for stats in run.precision(pair) {
            assert!(
                stats.min_log2_prec > 17.0,
                "CI bootstrap pair={pair} s2c={s2c_first} sparse={log_sparsity}: {stats:?}"
            );
        }
    }
    use poulpy_core::{GLWENormalize, layouts::GLWESwitchingKeyPreparedFactory};
    for input in &mut run.inputs {
        let mut converted = run.ci.ckks_ciphertext_alloc((params.base2k - 1).into(), input.k());
        converted.set_meta(input.meta());
        run.ci.glwe_normalize(&mut converted, input, &mut run.scratch.borrow());
        *input = converted;
    }
    for output in &mut run.outputs {
        *output = run.ci.ckks_ciphertext_alloc((params.base2k - 1).into(), params.k.into());
    }
    run.bootstrap(true);
    for stats in run.precision(true) {
        assert!(stats.min_log2_prec > 17.0, "{stats:?}");
    }
    for output in &mut run.outputs {
        output.set_k(params.k.into());
    }
    let return_key = &run.keys.standard_to_ci;
    let return_capacity = return_key.gglwe_layout().gadget_k().as_usize();
    let mut too_wide = run.ci.ckks_ciphertext_alloc(
        params.base2k.into(),
        (return_capacity + 1 + run.context.standard.output_consumed_bits(log_delta)).into(),
    );
    let before_too_wide = too_wide.to_host_owned::<BE>();
    let err = crate::layouts::CIRingBridge::new(&run.ci, &run.standard)
        .unwrap()
        .bootstrap(
            &mut too_wide,
            &run.inputs[0],
            &run.context,
            &run.keys,
            &mut run.std_scratch.borrow(),
        )
        .unwrap_err();
    assert!(err.to_string().contains("standard-to-CI"), "{err}");
    assert_eq!(too_wide.data().data().as_ref(), before_too_wide.data().data().as_ref());
    assert_eq!(too_wide.meta(), before_too_wide.meta());
    assert_eq!(too_wide.k(), before_too_wide.k());
    let before = run.outputs[0].to_host_owned::<BE>();
    let before_right = run.outputs[1].to_host_owned::<BE>();
    let [left, right] = &mut run.outputs;
    let right_meta = run.inputs[1].meta();
    run.inputs[1].set_log_delta(right_meta.log_delta + 1);
    assert!(
        crate::layouts::CIRingBridge::new(&run.ci, &run.standard)
            .unwrap()
            .bootstrap_pair(
                left,
                right,
                &run.inputs[0],
                &run.inputs[1],
                &run.context,
                &run.keys,
                &mut run.std_scratch.borrow()
            )
            .is_err()
    );
    run.inputs[1].set_meta(right_meta);
    assert_eq!(left.data().data().as_ref(), before.data().data().as_ref());
    assert_eq!(right.data().data().as_ref(), before_right.data().data().as_ref());
    assert_eq!(left.k(), before.k());
    assert_eq!(right.k(), before_right.k());
    let short_schedule: Vec<_> = (0..layers - 1).step_by(2).map(|i| ((layers - 1 - i).min(2), 2)).collect();
    let short_plan = BootstrappingPlan::new(
        plan.pipeline(),
        BootstrappingTechniques {
            sparse_secret_encapsulation: plan.techniques().sparse_secret_encapsulation,
            eval_round_plus: None,
        },
        DFTPlan::new(
            DFTType::Encode,
            short_schedule.clone(),
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(48, 3),
        )
        .unwrap(),
        *plan.eval_mod(),
        DFTPlan::new(
            DFTType::Decode,
            short_schedule,
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(28, 2),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        CIBootstrappingContext::<STD, f64>::compile(
            &run.standard,
            params.base2k.into(),
            &short_plan,
            &mut run.std_scratch.borrow()
        )
        .is_err()
    );
    assert!(CIBootstrappingContext::<BE, f64>::compile(&run.ci, params.base2k.into(), &plan, &mut run.scratch.borrow()).is_err());
    let smaller_standard = Module::<STD>::new(params.n as u64);
    run.keys.standard_to_ci =
        smaller_standard.glwe_switching_key_prepared_alloc_from_infos(&poulpy_core::layouts::GLWESwitchingKeyLayout {
            n: params.n.into(),
            ..keys_layout.standard_to_ci
        });
    assert_eq!(run.keys.standard_to_ci.n().as_usize(), params.n);
    assert!(
        crate::layouts::CIRingBridge::new(&run.ci, &run.standard)
            .unwrap()
            .bootstrap(left, &run.inputs[0], &run.context, &run.keys, &mut run.std_scratch.borrow())
            .is_err()
    );
    assert_eq!(left.data().data().as_ref(), before.data().data().as_ref());
    assert_eq!(left.meta(), before.meta());
}
