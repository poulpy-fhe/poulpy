# Implementing a core backend

This guide explains what a backend must preserve when it replaces a
`poulpy-core` operation. The [public API](../src/api) defines the inputs and
outputs. For deterministic operations, the [HAL reference algorithms](../src/reference)
and [core-derived defaults](../src/oep/derived) define the integer results, rounding and mutations that a replacement must
reproduce.

A backend can choose its own kernels, prepared-key storage and scratch size.
The caller should get the same result through the public API.

## Choosing an implementation

OEP means **open extension point**: a trait through which the public API calls
backend code. Core uses the same distinction as HAL:

| Component | Purpose |
|---|---|
| `GLWERotate` | Public rotation API on `Module<BE>`. |
| `GLWERotateImpl` | Backend hook, implemented directly for `BE`. |
| `GLWERotateReference` | Reusable rotation algorithm built from HAL operations. |
| `oep::derived` | Defaults built from other core operations, preserving their backend dispatch. |

For example, rotating each GLWE polynomial with HAL's rotation operation is a
**reference** implementation. Rotating a GGSW by calling core GLWE rotation on
each row is **derived**. A backend that overrides GLWE rotation also changes the
rotation used by that derived GGSW operation.
Reverse subtraction and out-of-place multiplication by `X^p - 1` also compose
core operations. Some assign variants remain required HAL-based primitives
because their scratch contracts do not provide room for a whole ciphertext
copy.

To select the HAL-based rotation algorithm, a backend with the required HAL
operations can write:

```rust
poulpy_core::impl_glwe_rotate_reference_full!(MyBackend);
```

This macro implements `GLWERotateImpl` for `MyBackend`. To replace one method,
write that implementation yourself and forward unchanged methods to
`GLWERotateReference`. The reference methods remain callable directly.
The [compiled example](../../poulpy-cpu-ref/src/tests/delegating_backend.rs)
checks both reference forwarding and dispatch through custom backend methods.

Derived methods have default bodies on their `*Impl` traits. A backend can
inherit those defaults or override them, including their scratch queries.
A directly called helper checks its own budget; nested operations use their
selected backend queries. An override can therefore reserve private workspace
and pass the remaining arena to a helper.
Trace, packing, gadget row operations and encryption wrappers reuse core
operations this way. Polynomial evaluation composes caller-supplied `BSGSOps`:
the caller owns scheme arithmetic and precision, while the derived schedule
chooses the order of those calls.

`impl_core_reference_full!` selects the available reference algorithms and
derived defaults, except sampling. Use individual family macros when replacing
a family, so its `*Impl` is defined only once. Sampling is supplied through
`SamplingImpl`. Preparation and decompression helpers reuse selected operations;
see the [OEP documentation](../src/oep/mod.rs) for the available hooks and macros.

## Matching the result

Every OEP override must implement the same circuit as the reference
implementation and pass the corresponding [parity tests](../src/test_suite/parity).
The reference implementation defines the required behavior.

## Respecting storage and scratch

Inputs borrowed through `&T` must remain unchanged. An assign method reads the
destination's old value before overwriting it. Other mutable inputs, such as the
ciphertexts consumed by packing, follow their API's mutation rules. Preserve any
columns, storage and metadata that the operation says it leaves untouched.

A `*_tmp_bytes` query must return enough scratch for the matching operation and
input layouts. Scratch may contain arbitrary bytes on entry. For example, if a
backend reports 1,024 bytes, its test must run with that budget even if the
reference backend reports 4,096 bytes. Giving both implementations 4,096 bytes
would hide an underestimate.

Prepared keys and transformed buffers can have different layouts across
backends. Prepare each backend's objects independently and compare the integer
outputs of operations that use them. Host-only inspection and noise diagnostics
have additional host-access requirements; generic backend code must respect
opaque buffers.

The shared gadget-product and GLWE external-product algorithms need partial
views of DFT limbs. They require `Backend::DFT_LIMBS_CONTIGUOUS = true`. A backend
with another layout must provide its own `GGLWEProductDigitsStridedImpl` and
`GLWEExternalProductImpl` implementations. The incompatible shared bodies
are rejected at compile time.

## Testing a replacement

Select the comparison backend with `backend_ref` and the backend under test
with `backend_test` in `core_parity_test_suite!` or
`core_encryption_parity_test_suite!`. Neither suite is tied to `poulpy-cpu-ref`.
A validated backend can bootstrap another: parity is transitive for the
operations and parameter ranges covered by the tests.

Encryption parity requires identical sampled inputs. Backends with matching
random streams can be compared directly; otherwise use the optional
[controlled-sampling support](../src/test_suite/parity/controlled_sampling.rs).

## Running the tests

Run the portable core parity and encryption suites with the pinned toolchain:

```sh
cargo test -p poulpy-cpu-ref --lib --profile ci --features enable-core -- \
  core_parity core_encryption --test-threads=2
```

Backend crates register these suites for portable FFT/NTT, AVX, AVX-512/IFMA,
NEON and supported Rayon variants. CI runs them on native CPUs or under Intel
SDE, with an additional optional NEON run under QEMU. New operations need tests
and registrations for each supported backend.

Check public documentation with:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc -p poulpy-core --no-deps --features enable-core
```
