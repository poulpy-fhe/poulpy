# Poulpy CPU Oracle

Correctness-testing backends for Poulpy. Applications should use
`poulpy-cpu-portable` or a SIMD backend instead.

- `FFT64Oracle`: scalar radix-2 FFT with independently generated tables.
- `NTT4x30Oracle`: scalar negacyclic NTT with modular reduction after every
  butterfly, direct modular products, and independently computed CRT inverses.

The backends implement the required HAL primitives and inherit all optional
operations from HAL. Prepared products use ordinary transform order and direct
scalar loops. Normalization reconstructs each coefficient as an arbitrary-precision
integer, rounds once, and writes centered radix digits; it uses heap storage.

This crate does not import production CPU kernels or their generated tables.
Some scalar support routines share source ancestry with the portable backend
and are maintained independently. Transform tests use direct polynomial
evaluation and schoolbook negacyclic convolution as additional checks.

`enable-core` enables Core integration; `enable-ckks` adds CKKS. Generic
HAL/Core/CKKS compositions are intentionally shared. Cross-backend tests validate
backend implementations, not the correctness of a shared composition itself.
The existing independent expected-result, noise, and cleartext tests complement
these comparisons.

This crate is unpublished (`publish = false`). Use oracle backends through
path-only development dependencies within the workspace. Compare coefficient-domain
public results across layouts; use numerical tolerances for FFT operations.
Oracle types do not promise raw-buffer compatibility with production backends.
Backend-local IFMA scalar checks remain useful for internal representations and
ranges, but share production helpers and are a separate category of validation.

```toml
[dev-dependencies]
poulpy-cpu-oracle = { path = "../poulpy-cpu-oracle", features = ["enable-core"] }
```

```sh
cargo test -p poulpy-cpu-oracle --features enable-ckks --profile ci
cargo test -p poulpy-cpu-portable --features enable-ckks --profile ci tests::oracle
```

CI enables `poulpy-cpu-oracle/enable-ckks` explicitly in each backend feature set.
