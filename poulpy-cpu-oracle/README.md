# Poulpy CPU Oracle

A correctness oracle for Poulpy arithmetic. Tests run the same operations through
the oracle and production backends, then compare their results. The oracle keeps
its arithmetic simple and independently maintained so optimized kernels can be
checked against an inspectable implementation.

- `FFT64Oracle`: scalar radix-2 FFT with independently generated tables.
- `NTT4x30Oracle`: scalar negacyclic NTT with modular reduction after every
  butterfly, direct modular products, and independently computed CRT inverses.

These types implement HAL backend interfaces to participate in the shared test
suites. They implement the required HAL primitives and inherit all optional
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

This crate is unpublished (`publish = false`). Use the oracle through
path-only development dependencies within the workspace. Compare coefficient-domain
public results across layouts; use numerical tolerances for FFT operations.
Oracle types do not promise raw-buffer compatibility with production backends.
Backend-local IFMA scalar checks remain useful for internal representations and
ranges, but share production helpers and are a separate category of validation.

CKKS encoding has a separate [canonical floating-point contract](../docs/backends.md#ckks-encoding).
The oracle implements its butterfly graph independently and checks scalar bits exactly; its ring FFT retains the independent decomposition described above.

```toml
[dev-dependencies]
poulpy-cpu-oracle = { path = "../poulpy-cpu-oracle", features = ["enable-core"] }
```

```sh
cargo test -p poulpy-cpu-oracle --features enable-ckks --profile ci
cargo test -p poulpy-cpu-portable --features enable-ckks --profile ci tests::oracle
```

CI enables `poulpy-cpu-oracle/enable-ckks` explicitly in each backend feature set.
