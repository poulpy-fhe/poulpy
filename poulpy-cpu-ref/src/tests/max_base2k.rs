use poulpy_hal::{
    api::{
        SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPrepare, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
    },
    layouts::{
        HostBytesBackend, Module, PrepareHint, PrimeSet, ScalarZnxToBackendRef, SvpPPolToBackendMut, SvpPPolToBackendRef,
        VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxDftToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
};

use crate::{FFT64Ref, NTT4x30Ref, reference::ntt4x30::primes::Primes30};

#[test]
fn max_base2k_uses_the_requested_budget() {
    let radix: Option<usize> = Module::<NTT4x30Ref>::max_base2k(1 << 15, 32, 128);
    assert_eq!(radix, Some(54));
    assert_eq!(Module::<NTT4x30Ref>::max_base2k(1 << 16, 32, 128), Some(54));
    assert_eq!(Module::<NTT4x30Ref>::max_base2k(1 << 16, 32, 256), Some(53));
    assert_eq!(Module::<NTT4x30Ref>::max_base2k(1 << 16, 512, 128), Some(53));
    assert_eq!(Module::<HostBytesBackend>::max_base2k(8, 1, 128), None);
}

#[test]
fn max_base2k_fft_uses_the_requested_budget() {
    let radix: Option<usize> = Module::<FFT64Ref>::max_base2k(1 << 15, 32, 128);
    assert_eq!(radix, Some(19));
    assert_eq!(Module::<FFT64Ref>::max_base2k(1 << 16, 32, 128), Some(19));
    assert_eq!(Module::<FFT64Ref>::max_base2k(1 << 16, 32, 256), Some(18));
    assert_eq!(Module::<FFT64Ref>::max_base2k(1 << 16, 128, 128), Some(18));
    assert_eq!(Module::<FFT64Ref>::max_base2k(1 << 16, usize::MAX, usize::MAX), Some(0));
}

#[test]
fn max_base2k_fft_is_largest_radix_meeting_the_rounding_error_envelope() {
    for log_n in 3..=18 {
        let n = 1usize << log_n;
        for products in [1, 2, 3, 4, 8, 32, 128, 65_535, usize::MAX] {
            for failure_bits in [1, 40, 128, 256, 1024, usize::MAX] {
                let radix = Module::<FFT64Ref>::max_base2k(n, products, failure_bits).unwrap();
                // Evaluate the rounding threshold directly from the model's
                // standard deviation, independently of the selector's log formula.
                let d = products as f64;
                let scalar_mac = 2.0 / 3.0 + (d + 1.0) / 6.0 - 1.0 / (3.0 * d);
                let fused_mac = (d + 0.5) / 3.0;
                let error_factor = 5.0 * (log_n as f64 - 1.0) + scalar_mac.max(fused_mac);
                let log2_envelope = |k: usize| {
                    let sigma = 2.0_f64.powi(2 * k as i32 - 53) * (n as f64 * d * error_factor).sqrt() / 12.0;
                    let x = 0.5 / (std::f64::consts::SQRT_2 * sigma);
                    log_n as f64 - x * x * std::f64::consts::LOG2_E
                };
                if radix > 0 {
                    assert!(
                        log2_envelope(radix) <= -(failure_bits as f64),
                        "degree {n}, products {products}, failure bits {failure_bits}, radix {radix}"
                    );
                }
                if radix < 62 {
                    assert!(
                        log2_envelope(radix + 1) > -(failure_bits as f64),
                        "degree {n}, products {products}, failure bits {failure_bits}, radix {}",
                        radix + 1
                    );
                }
            }
        }
    }
}

#[test]
fn max_base2k_fft_decreases_with_degree_accumulation_and_failure_budget() {
    let counts = [1, 2, 3, 4, 8, 32, 128, 65_535, usize::MAX];
    let targets = [1, 40, 128, 256, 1024, usize::MAX];
    for log_n in 3..=18 {
        let n = 1usize << log_n;
        for target in targets {
            let mut previous = 62;
            for count in counts {
                let current = Module::<FFT64Ref>::max_base2k(n, count, target).unwrap();
                assert!(current <= previous);
                previous = current;
            }
        }
        for count in counts {
            let mut previous = 62;
            for target in targets {
                let current = Module::<FFT64Ref>::max_base2k(n, count, target).unwrap();
                assert!(current <= previous);
                previous = current;
            }
        }
    }
    for count in counts {
        for target in targets {
            let mut previous = 62;
            for log_n in 3..=18 {
                let current = Module::<FFT64Ref>::max_base2k(1usize << log_n, count, target).unwrap();
                assert!(current <= previous);
                previous = current;
            }
        }
    }
}

#[test]
fn max_base2k_is_largest_radix_meeting_the_gaussian_envelope() {
    let q = Primes30::Q.into_iter().map(u128::from).product::<u128>() as f64;
    for log_n in 3..=Primes30::MAX_LOG_N {
        let n = 1usize << log_n;
        for products in [1, 32, 97, 65_536] {
            for failure_bits in [40, 128, 256, 1024] {
                let radix = Module::<NTT4x30Ref>::max_base2k(n, products, failure_bits).unwrap();
                let log2_envelope = |k: usize| {
                    let p = (1u64 << (k - 1)) as f64;
                    let sigma = p * p * ((n as f64) * (products as f64)).sqrt() / 3.0;
                    let x = (q / 2.0) / (std::f64::consts::SQRT_2 * sigma);
                    log_n as f64 - x * x * std::f64::consts::LOG2_E
                };
                assert!(log2_envelope(radix) <= -(failure_bits as f64));
                if radix < 62 {
                    assert!(log2_envelope(radix + 1) > -(failure_bits as f64));
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "products must be positive")]
fn max_base2k_rejects_zero_products() {
    Module::<NTT4x30Ref>::max_base2k(1 << 16, 0, 128);
}

#[test]
#[should_panic(expected = "failure_bits must be positive")]
fn max_base2k_rejects_zero_failure_bits() {
    Module::<NTT4x30Ref>::max_base2k(1 << 16, 32, 0);
}

fn square_constant_ntt(module: &Module<NTT4x30Ref>, base2k: usize) -> VecZnxBigOwned<NTT4x30Ref> {
    let n = module.n();
    let coefficient = -(1i64 << (base2k - 1));
    let mut scalar = module.scalar_znx_alloc(n, 1);
    scalar.at_mut(0, 0).fill(coefficient);
    let mut input = module.vec_znx_alloc(n, 1, 1);
    input.at_mut(0, 0).fill(coefficient);
    let mut prepared = module.svp_ppol_alloc(n, 1, PrepareHint::Reuse);
    module.svp_prepare(
        &mut prepared.to_backend_mut(),
        0,
        &<_ as ScalarZnxToBackendRef<NTT4x30Ref>>::to_backend_ref(&scalar),
        0,
    );
    let mut product = module.vec_znx_dft_alloc(n, 1, 1);
    module.vec_znx_dft_apply(
        1,
        0,
        &mut product.to_backend_mut(),
        0,
        &<_ as VecZnxToBackendRef<NTT4x30Ref>>::to_backend_ref(&input),
        0,
    );
    module.svp_apply_dft_to_dft_assign(&mut product.to_backend_mut(), 0, &prepared.to_backend_ref(), 0);
    let mut result = module.vec_znx_big_alloc(n, 1, 1);
    module.vec_znx_idft_apply_tmpa(&mut result.to_backend_mut(), 0, &mut product.to_backend_mut(), 0);
    result
}

#[test]
fn ntt_log_n_15_structured_square_is_outside_the_uniform_input_model() {
    let n = 1 << 15;
    let module = Module::<NTT4x30Ref>::new(n as u64);
    let base2k = 52;
    let square = square_constant_ntt(&module, base2k);
    let coefficient_square = 1i128 << (2 * base2k - 2);
    for (i, &actual) in square.at(0, 0).iter().enumerate() {
        let expected = (2 * i as i128 + 2 - n as i128) * coefficient_square;
        assert_eq!(actual, expected, "coefficient {i}, radix {base2k}");
    }

    // The uniform-input recommendation does not cover this structured input.
    // At radix 53 its final coefficient is 2^119 > Q/2 and wraps negative.
    assert_eq!(Module::<NTT4x30Ref>::max_base2k(n, 32, 128), Some(54));
    let overflowing_square = square_constant_ntt(&module, 53);
    let q: i128 = Primes30::Q.into_iter().map(i128::from).product();
    let expected = 1i128 << 119;
    assert!(expected > q / 2);
    assert_eq!(overflowing_square.at(0, 0)[n - 1], expected - q);
    assert!(overflowing_square.at(0, 0)[n - 1] < 0);
}

#[test]
#[should_panic(expected = "n must be a power of two")]
fn max_base2k_rejects_zero() {
    Module::<FFT64Ref>::max_base2k(0, 32, 128);
}

#[test]
#[should_panic(expected = "n must be a power of two")]
fn max_base2k_rejects_non_power_of_two() {
    Module::<FFT64Ref>::max_base2k(24, 32, 128);
}

#[test]
#[should_panic(expected = "n is below the backend's minimum degree")]
fn max_base2k_rejects_unsupported_small_degree() {
    Module::<FFT64Ref>::max_base2k(4, 32, 128);
}
