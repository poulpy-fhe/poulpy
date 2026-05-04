#[cfg(test)]
pub mod fft64_ref;

#[cfg(test)]
pub mod test_suite;

#[cfg(test)]
pub mod ntt120_ref;

#[cfg(test)]
#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub mod fft64_avx;

#[cfg(test)]
#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub mod ntt120_avx;

#[cfg(test)]
#[cfg(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f"))]
pub mod fft64_avx512;

#[cfg(test)]
#[cfg(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f"))]
pub mod ntt120_avx512;

#[cfg(test)]
#[cfg(all(
    feature = "enable-ifma",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_feature = "avx512vl"
))]
pub mod ntt126_ifma;
