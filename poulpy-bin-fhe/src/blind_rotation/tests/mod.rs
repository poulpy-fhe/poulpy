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
#[cfg(all(
    feature = "enable-avx-rayon",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod fft64_avx_rayon;

#[cfg(test)]
#[cfg(all(
    feature = "enable-avx-rayon",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod ntt4x30_avx_rayon;

#[cfg(test)]
#[cfg(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f"))]
mod fft64_avx512;

#[cfg(test)]
#[cfg(all(feature = "enable-avx512f-rayon", target_arch = "x86_64", target_feature = "avx512f"))]
mod fft64_avx512_rayon;

#[cfg(test)]
#[cfg(all(feature = "enable-avx512f-rayon", target_arch = "x86_64", target_feature = "avx512f"))]
mod ntt4x30_avx512_rayon;

#[cfg(test)]
#[cfg(all(feature = "enable-neon-rayon", target_arch = "aarch64"))]
mod fft64_neon_rayon;

#[cfg(test)]
#[cfg(all(feature = "enable-neon-rayon", target_arch = "aarch64"))]
mod ntt4x30_neon_rayon;

#[cfg(test)]
mod test_suite;
