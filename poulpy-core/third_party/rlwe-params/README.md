# rlwe-params tables

`128bits_security.json` is the 128-bit security table of [rlwe-params](https://github.com/jdumezy/rlwe-params), copied unchanged from the release named in `VERSION`.
`SHA256` holds its digest, which matches the `digest` of the release asset: it shows the copy intact, not its values right.
rlwe-params is distributed under the Apache License 2.0.

`poulpy-core/src/security/rlwe_params_128.rs` holds the table as Rust, one line per cell, which `poulpy_core::security` reads.
The test `security::generate` rebuilds that file from the JSON, fails when they differ, and refuses a table whose format, target, error or samples differ from what the module assumes.
After updating the JSON, `POULPY_REGENERATE_SECURITY_TABLE=1 cargo test -p poulpy-core --lib security::generate` rewrites the file.
The workflow `.github/workflows/rlwe-params.yml` checks for a newer release every week and opens an issue listing the update steps, without changing the repository.
A maintainer updates the table by hand, and the regenerated file shows each changed cell in review.
The bounds of the validated presets may move with it, which the preset tests catch.
