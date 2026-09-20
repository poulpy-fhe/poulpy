# Implementing a core backend

This guide explains what a backend must preserve when it replaces a
`poulpy-core` operation. The [public API](../src/api) defines the inputs and
outputs. For deterministic operations, the [portable implementations](../src/reference)
define the integer results, rounding and mutations that a replacement must
reproduce.

A backend can choose its own kernels, prepared-key storage and scratch size.
The caller should get the same result through the public API.

## Choosing an implementation

OEP means **open extension point**: a trait through which the public API calls
backend code. For GLWE rotation, the relevant traits are:

| Trait | Purpose |
|---|---|
| `GLWERotateImpl` | The backend implementation called by the public API. |
| `GLWERotateReference` | Where a backend can select or replace the rotation methods. |
| `GLWERotateComposition` | The shared rotation algorithm, built from HAL operations. |

For most operation families, implementing `*Reference` supplies the matching
`*Impl` automatically. The `Reference` suffix therefore also appears on traits
that a backend implements with its own optimized code.

To use the shared rotation algorithm, a backend that provides its required HAL
operations can write:

```rust
poulpy_core::impl_glwe_rotate_reference_full!(MyBackend);
```

To replace one rotation method, implement `GLWERotateReference<MyBackend>` for
`Module<MyBackend>` yourself. Put the replacement in that method and forward the
remaining methods, including scratch queries, to `GLWERotateComposition`.
The [compiled example](../../poulpy-cpu-ref/src/tests/delegating_backend.rs)
shows this and checks that a public call reaches the replacement.

`impl_core_reference_full!` selects all shared core families except sampling.
Use the individual family macros when customizing a family, to avoid defining
its implementation twice. A backend may also implement `*Impl` directly; in
that case, omit the corresponding `*Reference` implementation.

Sampling, tensoring and strided gadget products use direct implementation
traits: `SamplingImpl`, `GLWETensoringImpl` and `GGLWEProductDigitsStridedImpl`.
The latter two have reference forwarding macros. Preparation and decompression
helpers reuse backend operations; they do not each provide a separate override
trait. See the [OEP documentation](../src/oep/mod.rs) for the full list of traits
and macros.

## Matching the result

A GLWE layout specifies its polynomial degree `n`, mask count `rank`, radix
`base2k` and precision `k`. Storage can hold more bits than `k`. For example,
`base2k = 17` and `k = 50` need three limbs, but only 50 bits are live. Copying
that value into a 34-bit destination must apply the reference rounding rule.
Simply dropping the last limb can give a different result.

The required representation depends on the operation:

- **Add, subtract, negate and monomial rotations** operate on the stored limbs.
  They do not automatically propagate carries. Multiplication by `X^p - 1`
  also leaves raw limb results.
- **Normalize** propagates carries and rounds at the destination radix and
  precision. Its output digits must match the reference exactly.
- **Copy** preserves raw limbs when the radix matches and the destination has
  enough precision. A changed radix or reduced precision uses normalization;
  reducing precision also applies the reference rounding rule.
- **Shifts, products and operations using prepared keys** must match their
  reference algorithm's rounding, conversion offsets and output widths.
  A mathematically equivalent value with different required digits is a failure.

Trace divides by two before each automorphism-add stage, so it computes a
normalized trace. An empty trace copies its input. Packing consumes its input
ciphertexts and requires indices aligned to `2^log_gap_out`; its preconditions
are described in the [public operation traits](../src/api/operations.rs).

Reject incompatible dimensions, ranks and strides according to the operation's
preconditions. Preserve errors returned by key lookup or arithmetic callbacks.
For polynomial evaluation, the caller supplies arithmetic through `BSGSOps`;
core provides the evaluation order and error propagation.

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
`GLWEExternalProductReference` implementations. The incompatible shared bodies
are rejected at compile time.

## Testing a replacement

The shared [parity tests](../src/test_suite/parity) compare a backend with an
explicit `poulpy-cpu-ref` execution:

1. Give both implementations the same logical inputs.
2. Prepare their keys and other backend-specific objects independently.
3. Allocate each implementation's advertised scratch budget and fill scratch
   and outputs with nonzero data to expose missing initialization.
4. Compare the resulting integer coefficients and relevant metadata.

Include assign and accumulation variants, boundary precisions, different ranks
and digit sizes, and regions that must remain unchanged. The compiled override
example also checks dispatch: matching output alone cannot show that a custom
method was actually called.

Sampling needs a separate check. The same seed may produce different samples on
different backends. The [controlled-sampling fixture](../../poulpy-cpu-ref/src/test_suite/controlled_sampling.rs)
feeds the tested backend's actual samples into the reference encryption
algorithm. This lets tests compare the ciphertext calculations with identical
random inputs. [Sampling tests](../src/test_suite/sampling.rs) separately check
distributions, repeatability within a backend, seed consumption and untouched
columns.

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
