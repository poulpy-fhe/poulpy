//! Which of this crate's backends the running CPU can use.

use poulpy_cpu_ref::capabilities::BackendCapability;

const AVX512F: &str = "+avx512f";
const IFMA: &str = "+avx512f,+avx512ifma,+avx512vl";

/// The AVX-512 and IFMA backends, with CPU support and whether this build
/// enabled them.
pub fn capabilities() -> Vec<BackendCapability> {
    let (avx512f, ifma) = cpu_supported();
    let f = cfg!(feature = "enable-avx512f");
    let rayon = cfg!(feature = "enable-rayon");
    let ifma_built = cfg!(feature = "enable-ifma");
    vec![
        entry("FFT64Avx512", "enable-avx512f", AVX512F, avx512f, f),
        entry("NTT4x30Avx512", "enable-avx512f", AVX512F, avx512f, f),
        entry("FFT64Avx512Rayon", "enable-rayon", AVX512F, avx512f, rayon),
        entry("NTT4x30Avx512Rayon", "enable-rayon", AVX512F, avx512f, rayon),
        entry("NTT3x42Ifma", "enable-ifma", IFMA, ifma, ifma_built),
        entry(
            "NTT3x42IfmaRayon",
            "enable-ifma,enable-rayon",
            IFMA,
            ifma,
            ifma_built && rayon,
        ),
    ]
}

/// `(avx512f, avx512f + ifma + vl)`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_supported() -> (bool, bool) {
    let avx512f = std::is_x86_feature_detected!("avx512f");
    let ifma = avx512f && std::is_x86_feature_detected!("avx512ifma") && std::is_x86_feature_detected!("avx512vl");
    (avx512f, ifma)
}

/// These backends only exist on x86.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn cpu_supported() -> (bool, bool) {
    (false, false)
}

fn entry(
    backend: &'static str,
    feature: &'static str,
    target_features: &'static str,
    supported: bool,
    compiled: bool,
) -> BackendCapability {
    BackendCapability {
        backend,
        krate: "poulpy-cpu-avx512",
        feature: Some(feature),
        target_features: Some(target_features),
        supported,
        compiled,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "diagnostic: prints this machine's backend support"]
    fn print_report() {
        let mut caps = poulpy_cpu_ref::capabilities::reference_backends();
        caps.extend(super::capabilities());
        println!("{}", poulpy_cpu_ref::capabilities::report(&caps));
    }
}
