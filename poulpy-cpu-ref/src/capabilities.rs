//! Reporting which backends a machine can run.
//!
//! Backend availability is a build-time decision: a backend exists only if its
//! Cargo feature and its `target-feature` flags were passed. Each backend crate
//! declares its own entries as [`BackendCapability`] values; this module owns
//! the reference backends and the shared formatting.

/// One backend, whether this CPU can run it, and whether it is in this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapability {
    /// Backend marker type, e.g. `NTT3x42Ifma`.
    pub backend: &'static str,
    /// Crate providing it.
    pub krate: &'static str,
    /// Cargo feature enabling it, if any.
    pub feature: Option<&'static str>,
    /// `-C target-feature=` flags it needs, if any.
    pub target_features: Option<&'static str>,
    /// Whether the CPU running this code implements the required instructions.
    pub supported: bool,
    /// Whether the features enabling it were set for the current build.
    pub compiled: bool,
}

/// The portable backends, available everywhere.
pub fn reference_backends() -> Vec<BackendCapability> {
    ["FFT64Ref", "NTT4x30Ref"]
        .into_iter()
        .map(|backend| BackendCapability {
            backend,
            krate: "poulpy-cpu-ref",
            feature: None,
            target_features: None,
            supported: true,
            compiled: true,
        })
        .collect()
}

/// A printable table of the given capabilities.
///
/// Backend crates pass their own entries; combine them to describe a build.
pub fn report(caps: &[BackendCapability]) -> String {
    let mut out = String::from("backend              crate               cpu  built  build with\n");
    for c in caps {
        let recipe = match (c.feature, c.target_features) {
            (Some(f), Some(t)) => format!("--features {f}   RUSTFLAGS=\"-C target-feature={t}\""),
            (Some(f), None) => format!("--features {f}"),
            _ => String::new(),
        };
        out.push_str(&format!(
            "{:20} {:19} {:4} {:6} {recipe}\n",
            c.backend,
            c.krate,
            if c.supported { "yes" } else { "no" },
            if c.compiled { "yes" } else { "no" }
        ));
    }
    out.push_str("`-C target-cpu=native` enables every instruction set this CPU has.\n");
    out
}

/// One instruction set, and whether the CPU running this code implements it.
///
/// For the backends each set unlocks, see `docs/performance.md`; for what a
/// given build enabled, see each backend crate's [`BackendCapability`] report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructionSet {
    /// How the set is named in `-C target-feature`, e.g. `avx512ifma+vl`.
    pub name: &'static str,
    /// Whether this CPU implements it.
    pub present: bool,
    /// Flags a build needs to use it.
    pub target_features: &'static str,
}

/// `(avx2+fma, avx512f, avx512ifma+vl)` on x86, all false elsewhere.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_features() -> (bool, bool, bool) {
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let avx512f = std::is_x86_feature_detected!("avx512f");
    let ifma = avx512f && std::is_x86_feature_detected!("avx512ifma") && std::is_x86_feature_detected!("avx512vl");
    (avx2, avx512f, ifma)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn x86_features() -> (bool, bool, bool) {
    (false, false, false)
}

/// Every accelerated instruction set poulpy can use, and whether this CPU has it.
///
/// Sets from other architectures are listed as absent, so the table reads the
/// same everywhere.
pub fn instruction_sets() -> Vec<InstructionSet> {
    let (avx2, avx512f, ifma) = x86_features();
    vec![
        InstructionSet {
            name: "avx2+fma",
            present: avx2,
            target_features: "+avx2,+fma",
        },
        InstructionSet {
            name: "avx512f",
            present: avx512f,
            target_features: "+avx512f",
        },
        InstructionSet {
            name: "avx512ifma+vl",
            present: ifma,
            target_features: "+avx512f,+avx512ifma,+avx512vl",
        },
        InstructionSet {
            name: "neon",
            present: cfg!(target_arch = "aarch64"),
            target_features: "",
        },
    ]
}

/// A printable table of [`instruction_sets`].
pub fn report_instruction_sets() -> String {
    let mut out = String::from("instruction set   present  build with\n");
    for set in instruction_sets() {
        let flags = if set.target_features.is_empty() {
            String::new()
        } else {
            format!("RUSTFLAGS=\"-C target-feature={}\"", set.target_features)
        };
        out.push_str(&format!(
            "{:17} {:8} {flags}\n",
            set.name,
            if set.present { "yes" } else { "no" }
        ));
    }
    out.push_str("\nThe reference backends need none of these and run everywhere.\n");
    out.push_str("`-C target-cpu=native` enables every instruction set this CPU has.\n");
    out.push_str("For the backends each set unlocks, see docs/performance.md.\n");
    out
}

#[cfg(test)]
mod tests {
    use super::{instruction_sets, reference_backends, report_instruction_sets};

    #[test]
    fn reference_backends_are_always_available() {
        assert!(reference_backends().iter().all(|c| c.supported && c.compiled));
    }

    #[test]
    fn sets_from_other_architectures_are_listed_as_absent() {
        let sets = instruction_sets();
        assert_eq!(sets.len(), 4);
        let present = |name: &str| sets.iter().find(|s| s.name == name).unwrap().present;
        if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            assert!(!present("neon"));
        } else if cfg!(target_arch = "aarch64") {
            assert!(present("neon") && !present("avx2+fma") && !present("avx512f"));
        }
    }

    #[test]
    #[ignore = "diagnostic: prints what this machine can run"]
    fn print_report() {
        println!("{}", report_instruction_sets());
    }
}
