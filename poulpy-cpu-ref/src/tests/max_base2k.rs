use poulpy_hal::{
    api::{
        SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPrepare, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, HostBytesBackend, Module, PrepareHint, PrimeSet, ScalarZnxToBackendRef, SvpPPolToBackendMut,
        SvpPPolToBackendRef, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxDftToBackendMut, VecZnxToBackendRef, ZnxView,
        ZnxViewMut,
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
        (8, 25, 58),
        (16, 25, 58),
        (32, 24, 57),
        (1 << 10, 22, 55),
        (1 << 15, 19, 52),
        (1 << 16, 19, 52),
        (1 << 18, 18, 51),
    ] {
        assert_eq!(Module::<FFT64Ref>::max_base2k(n), fft, "FFT64 degree {n}");
        assert_eq!(Module::<NTT4x30Ref>::max_base2k(n), ntt, "NTT4x30 degree {n}");
    }
}

#[test]
fn max_base2k_ntt_is_largest_radix_with_centered_product_capacity() {
    let q: u128 = Primes30::Q.into_iter().map(u128::from).product();
    for log_n in NTT4x30Ref::MIN_DEGREE.ilog2()..=Primes30::MAX_LOG_N {
        let n = 1usize << log_n;
        let base2k = Module::<NTT4x30Ref>::max_base2k(n);
        // A normalized limb can contain -2^(base2k - 1). Squaring the
        // constant-coefficient polynomial attains this bound at X^(n - 1).
        let product_bound = (n as u128) << (2 * base2k - 2);
        assert!(2 * product_bound < q, "degree {n}, radix {base2k}");
        assert!(2 * (4 * product_bound) >= q, "degree {n}, radix {}", base2k + 1);
    }
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
fn max_base2k_ntt_log_n_15_square_reconstructs_without_wraparound() {
    let n = 1 << 15;
    let module = Module::<NTT4x30Ref>::new(n as u64);
    let base2k = Module::<NTT4x30Ref>::max_base2k(n);
    let square = square_constant_ntt(&module, base2k);
    let coefficient_square = 1i128 << (2 * base2k - 2);
    for (i, &actual) in square.at(0, 0).iter().enumerate() {
        let expected = (2 * i as i128 + 2 - n as i128) * coefficient_square;
        assert_eq!(actual, expected, "coefficient {i}, radix {base2k}");
    }

    // The formerly advertised radix 53 puts the final coefficient at
    // 2^119 > Q/2, so the centered CRT reconstruction makes it negative.
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
