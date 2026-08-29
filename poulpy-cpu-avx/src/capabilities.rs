//! Which of this crate's backends the running CPU can use.

use poulpy_cpu_ref::capabilities::BackendCapability;

/// The AVX2/FMA backends, with CPU support and whether this build enabled them.
pub fn capabilities() -> Vec<BackendCapability> {
    let supported = cpu_supported();
    let serial = cfg!(feature = "enable-avx");
    let rayon = cfg!(feature = "enable-rayon");
    vec![
        entry("FFT64Avx", "enable-avx", supported, serial),
        entry("NTT4x30Avx", "enable-avx", supported, serial),
        entry("FFT64AvxRayon", "enable-rayon", supported, rayon),
        entry("NTT4x30AvxRayon", "enable-rayon", supported, rayon),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_supported() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// These backends only exist on x86.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn cpu_supported() -> bool {
    false
}

fn entry(backend: &'static str, feature: &'static str, supported: bool, compiled: bool) -> BackendCapability {
    BackendCapability {
        backend,
        krate: "poulpy-cpu-avx",
        feature: Some(feature),
        target_features: Some("+avx2,+fma"),
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
