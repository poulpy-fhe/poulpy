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
fn max_base2k_is_const_and_depends_on_degree() {
    const FFT_RADIX: usize = Module::<FFT64Ref>::max_base2k(1 << 16);
    const NTT_RADIX: usize = Module::<NTT4x30Ref>::max_base2k(1 << 16);
    assert_eq!((FFT_RADIX, NTT_RADIX), (19, 52));

    // Odd and even exponents exercise rounding with both DFT capacities.
    for (n, fft, ntt) in [
        (8, 25, 59),
        (16, 25, 58),
        (32, 24, 58),
        (1 << 10, 22, 55),
        (1 << 15, 19, 53),
        (1 << 16, 19, 52),
        (1 << 18, 18, 51),
    ] {
        assert_eq!(Module::<FFT64Ref>::max_base2k(n), fft, "FFT64 degree {n}");
        assert_eq!(Module::<NTT4x30Ref>::max_base2k(n), ntt, "NTT4x30 degree {n}");
    }
}

#[test]
fn max_base2k_for_failure_is_const_and_uses_the_requested_budget() {
    const RADIX: Option<usize> = Module::<NTT4x30Ref>::max_base2k_for_failure(1 << 15, 32, 128);
    assert_eq!(RADIX, Some(54));
    assert_eq!(Module::<NTT4x30Ref>::max_base2k_for_failure(1 << 16, 32, 128), Some(54));
    assert_eq!(Module::<NTT4x30Ref>::max_base2k_for_failure(1 << 16, 32, 256), Some(53));
    assert_eq!(Module::<NTT4x30Ref>::max_base2k_for_failure(1 << 16, 512, 128), Some(53));
    assert_eq!(Module::<FFT64Ref>::max_base2k_for_failure(1 << 16, 32, 128), None);
    assert_eq!(Module::<HostBytesBackend>::max_base2k_for_failure(8, 1, 128), None);
}

#[test]
fn max_base2k_for_failure_is_largest_radix_meeting_the_gaussian_envelope() {
    let q = Primes30::Q.into_iter().map(u128::from).product::<u128>() as f64;
    for log_n in 3..=Primes30::MAX_LOG_N {
        let n = 1usize << log_n;
        for products in [1, 32, 97, 65_536] {
            for failure_bits in [40, 128, 256, 1024] {
                let radix = Module::<NTT4x30Ref>::max_base2k_for_failure(n, products, failure_bits).unwrap();
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
fn max_base2k_for_failure_rejects_zero_products() {
    Module::<NTT4x30Ref>::max_base2k_for_failure(1 << 16, 0, 128);
}

#[test]
#[should_panic(expected = "failure_bits must be positive")]
fn max_base2k_for_failure_rejects_zero_failure_bits() {
    Module::<NTT4x30Ref>::max_base2k_for_failure(1 << 16, 32, 0);
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
fn ntt_log_n_15_structured_square_can_exceed_the_uniform_input_recommendation() {
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
    assert_eq!(Module::<NTT4x30Ref>::max_base2k(n), 53);
    let overflowing_square = square_constant_ntt(&module, 53);
    let q: i128 = Primes30::Q.into_iter().map(i128::from).product();
    let expected = 1i128 << 119;
    assert!(expected > q / 2);
    assert_eq!(overflowing_square.at(0, 0)[n - 1], expected - q);
    assert!(overflowing_square.at(0, 0)[n - 1] < 0);
}

#[test]
fn max_base2k_returns_zero_without_product_capacity() {
    const RADIX: usize = Module::<HostBytesBackend>::max_base2k(8);
    assert_eq!(RADIX, 0);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn max_base2k_saturates_when_degree_exhausts_capacity() {
    const RADIX: usize = Module::<FFT64Ref>::max_base2k(1 << 63);
    assert_eq!(RADIX, 0);
}

#[test]
#[should_panic(expected = "n must be a power of two")]
fn max_base2k_rejects_zero() {
    Module::<FFT64Ref>::max_base2k(0);
}

#[test]
#[should_panic(expected = "n must be a power of two")]
fn max_base2k_rejects_non_power_of_two() {
    Module::<FFT64Ref>::max_base2k(24);
}

#[test]
#[should_panic(expected = "n is below the backend's minimum degree")]
fn max_base2k_rejects_unsupported_small_degree() {
    Module::<FFT64Ref>::max_base2k(4);
}
