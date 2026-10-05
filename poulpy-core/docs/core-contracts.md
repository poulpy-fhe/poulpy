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
| `oep::derived` | Crate-private defaults built from other core operations, preserving their backend dispatch. |

For example, rotating each GLWE polynomial with HAL's rotation operation is a
**reference** implementation. Rotating a GGSW by calling core GLWE rotation on
each row is **derived**. A backend that overrides GLWE rotation also changes the
rotation used by that derived GGSW operation.
Reverse subtraction also composes core operations. Multiplication by `X^p - 1`
uses HAL reference algorithms in both forms, preserving raw-limb arithmetic.

To select the HAL-based rotation algorithm, a backend with the required HAL
operations can write:

```rust
poulpy_core::impl_glwe_rotate_reference_full!(MyBackend);
```

This macro implements `GLWERotateImpl` for `MyBackend`. To replace one method,
write that implementation yourself and forward unchanged methods to
`GLWERotateReference`. The reference methods remain callable directly.

Derived methods have default bodies on their `*Impl` traits. A backend can
inherit those defaults or override them, including their scratch queries.
The derived helpers themselves are crate-private.
A directly called reference helper checks its own budget; nested operations use their
selected backend queries. An override can therefore reserve private workspace
and pass the remaining arena to a helper.
Trace, packing, gadget row operations and encryption wrappers reuse core
operations this way. Polynomial evaluation composes caller-supplied `BSGSOps`:
the caller owns scheme arithmetic and precision, while the derived schedule
chooses the order of those calls.

`impl_core_reference_full!` selects the available reference algorithms and
derived defaults, except sampling and the monomial families (LWE conversion,
packing, GLWE/GGSW rotate, `mul_xp_minus_one`). Use individual family macros when replacing
a family, so its `*Impl` is defined only once. Sampling is supplied through
the mandatory `SamplingImpl`, including full-width flooding.
Preparation and decompression helpers reuse selected operations;
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
comparison backend reports 4,096 bytes. Giving both implementations 4,096 bytes
would hide an underestimate.

Public-key generation follows this contract: allocate the caller's arena using
`glwe_public_key_generate_tmp_bytes` and pass it to `glwe_public_key_generate`.
The derived default sizes and uses that arena through the selected secret-key
encryption implementation, without allocating an internal scratch buffer.

Prepared keys and transformed buffers can have different layouts across
backends. Prepare each backend's objects independently and compare the integer
outputs of operations that use them. Host-only inspection and noise diagnostics
have additional host-access requirements; generic backend code must respect
opaque buffers.

The shared gadget-product and GLWE external-product algorithms need partial
views of DFT limbs. They require `Backend::DFT_LIMBS_CONTIGUOUS = true`. A backend
with another layout must provide its own `GGLWEProductDigitsStridedImpl` and
`GLWEExternalProductImpl::glwe_external_product_dft` with its matching
`glwe_external_product_internal_tmp_bytes` query. The public reference external
product calls these selected methods after any radix conversion, so it can be
retained for another DFT layout. `GLWEExternalProductDftReference` exposes the
contiguous-limb fallback without dispatching back to the override. Incompatible
reference digit loops are rejected at compile time.

## Public keys

`GLWEPublicKey` defines the rank-`r` public key, `GLWEPublicKeyGenerate` its
generation and `GLWEEncryptPk` public-key encryption. A replacement must keep
their draw order from `source_xa`, `source_xe` and `source_xu`: the parity
tests compare the outputs and the source states. A ciphertext's phase is
`m + Sum_l u_l e_l + e_0 + Sum_j e_j s_j`. `GLWEPublicKey` stores its `r`
encryptions of zero as a GGLWE does, one matrix of one row, entry `l` at input
column `l`; `at` and `at_mut` view an entry as a GLWE. The entries are
canonical: generation writes them so, `read_from` trusts the stream, and a
writer through a mutable view must leave them so. `GLWEPublicKeyPrepared` is
that matrix prepared by one `vmp_prepare`, and the reference encryption
computes `Sum_l u_l pk_l` as one vector-matrix product, then adds each
column's error before its single normalization at the output's `k`.

Each mask column `c_j = Sum_l u_l a_{l,j} + e_j` is a rank-`r` module-LWE
sample in `(u_1, .., u_r)` with independent uniform masks, and the body is one
more with the pseudorandom `b_l`, so a ciphertext is MLWE(`n`, `r`), the
instance the key rests on. A single ephemeral would give `r + 1` ring-LWE
samples of degree `n` in one secret, so parameters sized for dimension `n r`
would not protect it.

## Encryption metadata

Encryption and protocol finalization derive `EncryptionMetadata` for their
outputs. Secret provenance records the base distribution and number of secret
contributors. The separate `FreshNoiseEstimate` records effective phase
variance in integer coefficient units at the stored creation precision `k`.
Its square root is the effective fresh sigma; multiplying that sigma by
`2^-k` gives its torus scale. `variance_at` and `std_dev_at` express the same
historical estimate on another precision grid without adding rounding error.
Positive infinity denotes an unbounded estimate, including numeric overflow.

Fresh ciphertexts derive their own phase-error estimate from the encryption
path and both precisions. Let the output precision be `k`, public-key precision
be `k + a`, rank be `r`, and degree be `n`. Write `V_pk` for a public-key entry's
variance on the key's grid, `q_u = E[u^2]` for the ephemeral second moment, and
`q_S = E[S^2]` for the destination secret's second moment. Ordinary public-key
encryption records the model, in output-grid coefficient units:

`V_ct = r*n*q_u*V_pk*2^(-2*a) + (1 + r*n*q_S)*sigma_fresh^2 + R`.

The first term is inherited key error. Extra key precision attenuates it before
normalization to `k`. The implementation samples fresh body and mask errors at
the output precision `k`, even though its accumulator has the key's precision.
Those fresh terms survive normalization. `R` models the new rounding error:
zero when `a = 0`, otherwise `(1 + r*n*q_S)/4`. Adding this half-ulp term as an
independent variance is a noise-model approximation.

For a centered base secret with coefficient variance `v`, a public key built
from `P` independent shares has `V_pk = P*sigma_fresh^2` and `q_S = P*v`;
its ephemeral still has `q_u = v`. Thus:

| Encryption path | Fresh ciphertext variance at output precision `k` |
| --- | --- |
| Secret key | `sigma_fresh^2` |
| Single-party public key at `k + a` | `(1 + r*n*v*(1 + 2^(-2*a))) * sigma_fresh^2 + R` |
| `P`-party public key at `k + a` | `(1 + r*n*P*v*(1 + 2^(-2*a))) * sigma_fresh^2 + R` |

The equal-precision formulas follow by setting `a = 0` and `R = 0`. For binary
secrets, use the second moments in the general formula, including the squared
collective mean. Smudging replaces only the fresh body variance with the chosen
flood's variance. Re-encryption replaces the ciphertext's previous metadata.

Encrypting entirely at `k + a` and then truncating would instead attenuate all
of that encryption's phase error by `2^(-2*a)` and introduce rounding at `k`.
That construction requires sampling the fresh body and mask errors at `k + a`.
The current public-key encryption API samples them at its destination's `k`.

The estimate accounts for inherited public-key error and amplification during
key generation or protocol finalization. It is a variance model, not an exact
distribution descriptor for sums or products of errors. A whole GGSW uses the
largest column estimate. Aggregation can sum independent error variances;
when binary ephemerals reuse one public key, a conservative sum of sigmas
accounts for possible covariance.

This metadata describes creation noise. Homomorphic evaluation does not make
it a running estimate of the output's current noise. Copies, compression,
preparation and backend transfers preserve the recorded estimate and its
creation precision, even when the destination's precision differs. Backend
views copy metadata by value, so an operation that records new metadata must
update the owner through its setter.

Equality includes the fresh estimate and its precision. The `PNM2` wire format
preserves them and rejects earlier metadata versions; `PNM1` does not contain
enough information to reconstruct amplified fresh noise.

## Testing a replacement

Select the comparison backend with `backend_ref` and the backend under test
with `backend_test` in `core_parity_test_suite!` or
`core_encryption_parity_test_suite!`.
A validated backend can bootstrap another: parity is transitive for the
operations and parameter ranges covered by the tests.

Encryption parity requires identical sampled inputs. Backends with matching
random streams can be compared directly; otherwise use the optional
[controlled-sampling support](../src/test_suite/parity/controlled_sampling.rs).

## Running the tests

The shared suites are generic over their backend types. Register and run them
in the crate that supplies those implementations, with its supported parameters.

Run core's own unit tests with:

```sh
cargo test -p poulpy-core
```

Check public documentation with:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc -p poulpy-core --no-deps --features enable-core
```

## Full-precision smudging

`VecZnxAddNoise` dispatches to the mandatory `SamplingImpl`. It adds
one independent integer sample per coefficient, scaled by `2^-k` at the
destination's precision `k`, to a canonical input column. Each sample is
decomposed across all necessary balanced limbs; drawing independent Gaussian
limbs or shifting a small machine-word sample is not equivalent. Other columns
remain untouched, and padding below the sampling precision contributes zero. Adding two bounded
balanced digits leaves enough headroom for normalization; the result is
unnormalized and callers must normalize before another smudging addition.

Ordinary encryption uses `Noise::ENCRYPTION` at the output precision. The same
`Noise` descriptor represents floods, with an exact dyadic Gaussian parameter
or a signed uniform width. `Noise::assert_valid_for` checks a flood against its
destination before protocol mutation. Sampling takes no scratch arena. The
production small-Gaussian path scans a 128-bit cumulative table; larger
Gaussians use exact integer rejection and may take variable time.
Distributional exactness assumes uniform private bits.

The delegate only derives a private child seed and dispatches. Backend
implementations validate their input before drawing randomness or mutating
the destination. Same-backend seeds reproduce samples; different
backends may select different exact sampling algorithms. Tests for replacements
must reconstruct complete integers, exercise widths beyond 128 bits, and check
low bits and partial-limb padding. A variance or histogram check alone is not a
proof of negligible statistical error.
