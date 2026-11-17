# rlwe-params tables

`128bits_security.json` is the 128-bit security table of [rlwe-params](https://github.com/jdumezy/rlwe-params), copied unchanged from the release named in `VERSION`.
`SHA256` holds its digest, which matches the `digest` of the release asset.
rlwe-params is distributed under the Apache License 2.0.

`poulpy-core/src/security/rlwe_params_128.rs` holds the table as Rust, one line per cell, which `poulpy_core::security` reads.
The test `security::generate` rebuilds that file from the JSON, fails when they differ, and refuses a table whose format, target, error or samples differ from what the module assumes.
After updating the JSON, `POULPY_REGENERATE_SECURITY_TABLE=1 cargo test -p poulpy-core --lib security::generate` rewrites the file.
The workflow `.github/workflows/rlwe-params.yml` checks for a newer release every week, checks its digest against the release, pushes the JSON, `VERSION` and `SHA256` to a branch `rlwe-params-<tag>`, and opens an issue linking to a prefilled pull request.
If it cannot push, the issue lists the update steps instead.
The bounds of the certified presets may move with it, which the preset tests catch.
