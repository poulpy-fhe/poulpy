# Poulpy CPU Oracle

A correctness oracle for Poulpy arithmetic.
Tests run the same operations through the oracle and a production backend, then compare their results.
The oracle keeps its arithmetic simple and independently maintained, so optimized kernels can be checked against an inspectable implementation.

- `FFT64Oracle`: scalar radix-2 FFT with independently generated tables.
- `NTT4x30Oracle`: scalar negacyclic NTT over four 30-bit primes, with modular reduction after every butterfly, direct modular products, and CRT reconstruction.

Both are instances of one implementation of the HAL backend interfaces, generic over the transform family, so they can take part in the shared test suites.
It implements only the required HAL operations, directly and with plain loops, and inherits every optional operation from HAL.
The one exception is the in-place multiplication by `X^k - 1`, whose derived body needs more scratch than the Core callers provide.
Temporaries live on the heap, so every scratch size is zero.
The implementation also takes the backend ring as a type parameter, the standard ring by default.
`FFT64CIOracle` and `NTT4x30CIOracle` are the conjugate-invariant instances.
Every ring-dependent operation on the conjugate-invariant ring of degree `n` is computed in the standard ring of degree `2n`, through the embedding of the subring and back.
Their transforms store the `n` evaluations of the image at one root of each conjugate pair.
Prepared products use ordinary transform order.
Normalization reconstructs each coefficient as an arbitrary-precision integer, rounds once, and writes centered radix digits.
A sparse operand, whose degree divides the call degree, is materialized through its degree embedding before the dense operation runs.

This crate does not import production CPU kernels or their generated tables.
Transform tests check against direct polynomial evaluation, and the prime set against its declared roots and modulus size.

`enable-core` registers the generic Core compositions.
`enable-ckks` adds CKKS, with the canonical encoding circuit of the CKKS reference over the oracle FFT, which is generic over the float precision, and `enable-bin-fhe` adds binary FHE.
Both register the generic scheme compositions and imply `enable-core`.
Generic HAL, Core and scheme compositions are shared with the production backends, so cross-backend tests validate backend implementations, not the correctness of a shared composition itself.
The existing expected-result, noise, and cleartext tests complement these comparisons.

This crate is unpublished (`publish = false`).
Use the oracle through a path-only development dependency within the workspace.
Compare coefficient-domain public results across layouts, and use numerical tolerances for FFT operations.
Oracle types do not promise raw-buffer compatibility with production backends.

```toml
[dev-dependencies]
poulpy-cpu-oracle = { path = "../poulpy-cpu-oracle", features = ["enable-ckks", "enable-bin-fhe"] }
```
