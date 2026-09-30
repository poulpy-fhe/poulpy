# Poulpy CPU Oracle

A correctness oracle for Poulpy arithmetic.
Tests run the same operations through the oracle and a production backend, then compare their results.
The oracle keeps its arithmetic simple and independently maintained, so optimized kernels can be checked against an inspectable implementation.

- `FFT64Oracle`: scalar radix-2 FFT with independently generated tables.
- `NTT4x30Oracle`: scalar negacyclic NTT with modular reduction after every butterfly, direct modular products, and independently computed CRT inverses.

Both implement the HAL backend interfaces so they can take part in the shared test suites.
They implement the required HAL primitives and inherit every optional operation from HAL.
Prepared products use ordinary transform order and direct scalar loops.
Normalization reconstructs each coefficient as an arbitrary-precision integer, rounds once, and writes centered radix digits, using heap storage.
A sparse operand, whose degree divides the call degree, is materialized through its degree embedding before the dense kernel runs.

This crate does not import production CPU kernels or their generated tables.
Some scalar support routines share source ancestry with `poulpy-cpu-ref` and are maintained independently.
Transform tests use direct polynomial evaluation and schoolbook negacyclic convolution as additional checks.

`enable-core` registers the generic Core compositions.
Generic HAL and Core compositions are shared with the production backends, so cross-backend tests validate backend implementations, not the correctness of a shared composition itself.
The existing expected-result, noise, and cleartext tests complement these comparisons.

This crate is unpublished (`publish = false`).
Use the oracle through a path-only development dependency within the workspace.
Compare coefficient-domain public results across layouts, and use numerical tolerances for FFT operations.
Oracle types do not promise raw-buffer compatibility with production backends.

```toml
[dev-dependencies]
poulpy-cpu-oracle = { path = "../poulpy-cpu-oracle", features = ["enable-core"] }
```
