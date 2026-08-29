//! Which of this crate's backends the running CPU can use.

use poulpy_cpu_ref::capabilities::BackendCapability;

/// The NEON backends. AArch64 always implements NEON, so support is decided by
/// the target architecture alone.
pub fn capabilities() -> Vec<BackendCapability> {
    let supported = cfg!(target_arch = "aarch64");
    let neon = cfg!(feature = "enable-neon");
    let rayon = cfg!(feature = "enable-rayon");
    vec![
        entry("FFT64Neon", "enable-neon", supported, neon),
        entry("NTT4x30Neon", "enable-neon", supported, neon),
        entry("FFT64NeonRayon", "enable-rayon", supported, rayon),
        entry("NTT4x30NeonRayon", "enable-rayon", supported, rayon),
    ]
}

fn entry(backend: &'static str, feature: &'static str, supported: bool, compiled: bool) -> BackendCapability {
    BackendCapability {
        backend,
        krate: "poulpy-cpu-arm",
        feature: Some(feature),
        target_features: None,
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
