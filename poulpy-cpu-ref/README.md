# 🐙 Poulpy-CPU-REF

**Poulpy-CPU-REF** is the **reference (portable) CPU backend for Poulpy**.

It implements the Poulpy HAL extension traits without requiring SIMD or specialized CPU instructions, making it suitable for:

- all CPU architectures (`x86_64`, `aarch64`, `arm`, `riscv64`, …)
- development machines and CI runners
- environments without AVX or other advanced SIMD support

This backend integrates transparently with:

- `poulpy-hal`
- `poulpy-core`
- `poulpy-ckks`
- `poulpy-bin-fhe`

---

## When is this backend used?

The FFT64 and NTT4x30 reference HAL backends are always available and require
**no compilation flags and no CPU features**.

It is automatically selected when:

- the project does not request an optimized backend, or
- the target CPU does not support the requested SIMD backend (e.g., AVX), or
- portability and reproducibility are more important than raw performance.

No additional configuration is required to use it.

Higher-level backend wiring is feature-gated:

- `enable-core` wires the reference backends into the `poulpy-core` reference implementations.
- `enable-ckks` wires the reference backends into the `poulpy-ckks` reference implementations and
  also enables core support.
- `enable-test-suite` ships the portable comparison adapters other backend crates run their core
  parity suites against. Those adapters answer from a controlled-sampling scope and panic outside
  one, so enable this from `[dev-dependencies]` only, never in a shipping build.

Useful test commands:

```sh
# HAL/reference backend tests
cargo test -p poulpy-cpu-ref

# Core conformance tests on FFT64Ref and NTT4x30Ref
cargo test -p poulpy-cpu-ref --features enable-core

# CKKS conformance tests on FFT64Ref and NTT4x30Ref
cargo test -p poulpy-cpu-ref --features enable-ckks
```

---

## 🧪 Basic Usage

This crate exposes two backends:

```rust
use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend
let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << log_n);

// Q120 NTT backend (CRT over four ~30-bit primes)
let module: Module<NTT4x30Ref> = Module::<NTT4x30Ref>::new(1 << log_n);
```

Both work on **all supported platforms and architectures**.

---

## Performance Notes

`poulpy-cpu-ref` prioritizes:

* portability
* correctness
* ease of debugging

For maximum performance on x86_64 CPUs with AVX2 + FMA support, consider enabling the optional optimized backend:

```
poulpy-cpu-avx (feature: enable-avx)
```

For x86_64 CPUs with AVX-512 support, consider the AVX-512 backend:

```
poulpy-cpu-avx512 (features: enable-avx512f, enable-ifma)
```

Benchmarks and applications can freely switch between backends without changing source code — backend selection can be handled with feature flags, for example

```rust
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
use poulpy_cpu_ref::FFT64Ref as BackendImpl;
```

The same pattern applies to NTT4x30 backends (`NTT4x30Ref` / `NTT4x30Avx`).

---

## 🤝 Contributors

To implement your own backend (SIMD or accelerator):

1. Define a backend struct and implement the `Backend` trait from `poulpy-hal`.
2. Implement each required HAL OEP method and inherit or override its derived defaults.
3. Implement the core `*Impl` traits, or use family macros to select reference algorithms and derived defaults.
4. Optionally, do the same for `poulpy-ckks` behind a backend-owned `enable-ckks` feature using the `impl_ckks_*_reference!` macros or direct OEP trait implementations.

CPU backends share their common registrations through `impl_cpu_core_defaults!`
and `impl_cpu_ckks_defaults!`. Tensoring, strided digit products, encoding
transforms, and encapsulated ModUp remain explicit backend choices. Backends
that override other families can register the individual operation macros.

Use either a family macro or a handwritten implementation of the same core
`*Impl` trait. Reference helpers remain callable for methods you forward, while
derived defaults reuse selected backend operations. Validate the resulting
backend with the shared conformance tests; the [core backend guide](../poulpy-core/docs/core-contracts.md)
describes the contracts and test setup.

Your backend will automatically integrate with the backend-generic layers:

* `poulpy-hal`
* `poulpy-core`
* `poulpy-ckks`

No modifications to those crates are necessary — the HAL provides the extension points. Only the operations that need a faster implementation require explicit overrides, and each override is validated by the parity test against an attested backend (attestation is transitive back to `reference`: the portable backend runs it directly, and your backend may test against the portable backend or against any backend already attested), correct only when that test passes; everything else runs the `reference` layer, which is the implementation.

---

For questions or guidance, feel free to open an issue or discussion in the repository.

## Conjugate invariant rings

`Module::<FFT64CIRef>::new(n)` and `Module::<NTT4x30CIRef>::new(n)` select
an `n`-coefficient ring fixed by `X -> X^-1` inside `Z[X]/(X^(2n)+1)`.
The coefficient basis is `1, X^j + X^-j` for `1 <= j < n`, and the ambient
cyclotomic order is `4n`. `FFT64Ref` and `NTT4x30Ref` retain the standard ring.
Accelerated CPU and Rayon backends have corresponding CI siblings, such as
`FFT64CIAvx512`, `NTT4x30CIAvx`, and `NTT3x42CIIfmaRayon`.
NTT modules support invariant degrees up to `2^17` with the current prime sets.

The ring is selected by the backend type. Standard plans store no CI tables,
and CI plans own their required tables directly. The two rings have distinct
module handles and prepared-data types; layout compatibility only connects
implementations of the same ring.

Transforms, polynomial products, and automorphisms use this basis. Sparse
operands embed through `X -> X^(N/n)`. Arbitrary monomial multiplication is
not closed in this ring, so rotation and `X^p - 1` operations reject it.

## Binary-FHE integration

`enable-bin-fhe` selects the binary-FHE reference circuits and registers the
complete paired and same-backend lifecycle suites. Backend opt-in and test
registration share one declaration in `src/bin_fhe_impl.rs`. Custom operations
implement the binary-FHE `*Impl` contracts, including their scratch queries.

```sh
cargo test -p poulpy-cpu-ref --features enable-bin-fhe bin_fhe_parity
```
