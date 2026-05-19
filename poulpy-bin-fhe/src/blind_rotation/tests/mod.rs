#[cfg(test)]
#[cfg(not(any(
    all(
        feature = "enable-avx",
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ),
    all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f")
)))]
mod fft64_ref;

#[cfg(test)]
#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(all(feature = "enable-avx512f", target_feature = "avx512f"))
))]
mod fft64_avx;

#[cfg(test)]
#[cfg(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f"))]
mod fft64_avx512;

#[cfg(test)]
mod test_suite;
