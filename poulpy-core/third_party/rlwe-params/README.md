# rlwe-params tables

`128bits_security.json` is the 128-bit security table of [rlwe-params](https://github.com/jdumezy/rlwe-params), copied unchanged from the release named in `VERSION`.
Its sha256 is `9672dc7af7afaa75b04be306f677ba0d4acb9413320b523903e5ca766358c408`.
rlwe-params is distributed under the Apache License 2.0.

`build.rs` turns the table into the cells of `poulpy_core::security` and refuses a table whose format, target, error or samples differ from what the module assumes.
To update it, replace the JSON with the asset of a newer release and write that release's tag to `VERSION`.
