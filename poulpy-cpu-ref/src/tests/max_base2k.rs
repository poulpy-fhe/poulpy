use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::{FFT64Ref, NTT4x30Ref};

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
