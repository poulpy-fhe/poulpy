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

Prepared-right tensor multiplication is a `GLWETensoring` operation. Its
`glwe_tensor_apply_prepared_right_tmp_bytes` query sizes the selected implementation;
`glwe_tensor_apply_prepared_right` preserves the left operand and prepared right
operand. Reference free functions remain independently callable. Tensor products
normalize diagonal and pairwise products to full output limbs before subtraction,
then round all tensor columns to the requested precision.

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

## Component noise

Only the metadata an operation leaves on its outputs is specified. Mutable
access to an output (`to_backend_mut`, row views, `data_mut`, narrowing
`set_k`) clears it, so writing an output leaves `None`; an output returned
unwritten or copied from an input is cleared with `set_noise(None)`. Secret-key encryption,
key generation included, records `ComponentNoise::from_secret_at` with the
encrypting secret's distribution and the output's `k` and rank (scalar LWE: its
dimension). Public-key encryption records `public_key_encryption_noise`.
Copies at equal or wider precision
(zero masks appended for a wider rank), preparation, compression, decompression
and transfers keep the source's estimate, recorded after their last write.
Every other operation leaves `None`.
Provenance checks run before any mutation, and delegates only forward calls.

Encryption and construction of fresh collective keys or ciphertexts derive
`ComponentNoise` for their outputs. Secret provenance records the base
distribution and number of secret contributors. The `LWEInfos::noise()` accessor exposes `rank + 1` terms in `[body, mask_0, ...]`
order. Each `FreshNoiseEstimate` records coefficient-noise variance before
secret weighting, in integer units at the common creation precision `k`.
Secret-key encryption records `sigma_fresh^2` in the body and zero in each
mask. Scalar LWE has one term per scalar mask coefficient; terms after the
last nonzero one are not stored.
The square root of each variance is its effective fresh sigma; multiplying that sigma by
`2^-k` gives its torus scale. `variance_at` and `std_dev_at` express the same
historical estimate on another precision grid without adding rounding error.
Positive infinity denotes an unbounded estimate, including numeric overflow.
`phase_noise(n)` reconstructs the standard negacyclic GLWE estimate as
`body + n*E[S^2]*sum(masks)`; `lwe_phase_noise()` uses scalar products and
the LWE secret dimension for fixed-weight and block distributions. After flattening
a rank greater than one fixed-weight GLWE secret, use
`lwe_phase_noise_with_block(n)` with its original polynomial dimension.
For conjugate-invariant products, `weighted_phase_noise(n, 4*n)` bounds
every coefficient, including coefficient zero; the average weight is about `2*n`.

Public-key encryption draws its fresh errors at the output's `k` and
normalizes the full-precision key product once. Its estimate follows from the
key's metadata. Let the output precision be `k`, public-key precision be
`k + a`, rank be `r`, and degree be `n`. Write `V_pk` for a public-key entry's
variance on the key's grid, `q_u = E[u^2]` for the ephemeral second moment, and
`q_S = E[S^2]` for the destination secret's second moment. In output-grid
coefficient units (replace the product weight `n` by `4*n` for the
conjugate-invariant bound, retaining `n` when computing the secret law's
coefficient moments):

- `I = r*n*q_u*V_pk*2^(-2*a)`, the inherited public-key error; this assumes centered laws or zero inherited mask errors. Noncentered laws with inherited mask error record an unbounded estimate;
- `F = (1 + r*n*q_S)*sigma_fresh^2`, the fresh body and mask error;
- `R = (1 + r*n*q_S)/4 + M^2` when `a > 0` and zero otherwise, the rounding of the key's extra bits. Ties round up, so a noncentered secret has the bias bound `M = 2^(-(a+1)) * (1 + r*n*abs(E[S]))`; centered secrets use `M = 0`.

The phase estimate is `V_ct = I + F + R`, adding rounding as an independent
variance as in the library's other noise models. Each component records its
inherited error, `sigma_fresh^2` and, when `a > 0`, `1/4 + M^2/(1 + r*n*q_S)`.
Smudged encryption replaces the fresh body error by the requested flood, added
after normalization at the output's `k`.

For a centered base secret with coefficient variance `v`, a public key built
from `P` independent shares has `V_pk = P*sigma_fresh^2` and `q_S = P*v`;
its ephemeral still has `q_u = v`. Thus:

| Encryption path | Fresh ciphertext variance at output precision `k` |
| --- | --- |
| Secret key | `sigma_fresh^2` |
| Single-party public key at `k + a` | `(r*n*v*2^(-2*a) + 1 + r*n*v) * sigma_fresh^2 + R` |
| `P`-party public key at `k + a` | `(r*n*P*v*2^(-2*a) + 1 + r*n*P*v) * sigma_fresh^2 + R` |

For binary secrets, use the second moments in the general formula, including
the squared collective mean. Re-encryption replaces the ciphertext's previous metadata.

The estimate accounts for inherited public-key error and amplification during
key generation or protocol finalization. It is a variance model, not an exact
distribution descriptor for sums or products of errors. A whole GGSW uses the
largest estimate for each component across its columns. Aggregation can sum
independent error variances;
when binary ephemerals reuse one public key, a conservative sum of sigmas
accounts for possible covariance.

This metadata describes fresh encryption. Homomorphic operations clear the
entire output metadata to `None`, even when the inputs have identical metadata or the
operation happens to leave the value unchanged. This includes arithmetic with
plaintexts, normalization, key switching, rotations, extraction and evaluation
conversions. Noise composition is deferred to a later release. Fresh encryption
and key-generation factories record their output metadata after any internal
arithmetic used during construction.

Copies at equal or wider precision, compression, preparation and backend
transfers preserve the recorded estimate and its creation precision. Narrowing
copies clear metadata because re-quantization adds error. Rank padding appends
zero-noise masks; preparation requires equal ranks. Copies also preserve absent
metadata on evaluated ciphertexts.

Equality includes all component estimates and their precision. The `PNM3` wire
format preserves them and validates the component count against the ciphertext
shape. The metadata block stores the lossless distribution tag and payload, contributor
count, precision, logical component count, and the prefix through the last
nonzero variance. Readers restore zero suffix terms after validating the shape.
Readers require the destination's existing component count even when metadata
is absent, and validate payload headers before changing that count. Compressed
readers also require the existing seed count and matrix entry shape. Failed
reads clear metadata while preserving the component count, so retrying a read
cannot turn a prior untrusted shape into a metadata allocation bound.
Unsupported metadata markers are rejected. ENCAPSULATED provenance is recorded
as NONE, retaining an unknown-law estimate without serializing a process-local tag.

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
destination's precision `k`, to a column with sufficient signed-word headroom. Each sample is
decomposed across all necessary balanced limbs; drawing independent Gaussian
limbs or shifting a small machine-word sample is not equivalent. Other columns
remain untouched, and padding below the sampling precision contributes zero. Each sum of an existing digit and the sampled balanced digit must fit its signed
word. The result is unnormalized; repeated additions are valid while that
headroom remains. Normalization restores balanced digits.

Secret-key encryption uses `Noise::ENCRYPTION` at the output precision;
public-key encryption uses the derived intermediate sampling precision before
its final normalization, as described above. The same
`Noise` descriptor represents floods, with an exact dyadic Gaussian parameter
`sigma >= 1`, truncated at six `sigma`, or a signed uniform width. `Noise::assert_valid_for` checks a flood against its
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
