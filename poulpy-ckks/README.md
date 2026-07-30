# poulpy-ckks

`poulpy-ckks` is the Poulpy crate implementing the CKKS (Cheon-Kim-Kim-Song)
scheme.

It is built explicitly on top of:

- `poulpy-hal` for backend-agnostic modules, layouts, scratch management, and
  low-level arithmetic dispatch
- `poulpy-core` for Module-LWE-oriented cryptographic building blocks used to assemble
  the CKKS evaluator

The crate exposes:

- CKKS-specific ciphertext and plaintext wrappers
- slot encoding/decoding helpers
- secret-key encryption and decryption
- leveled arithmetic implemented through traits on `Module<BE>`

## Toolchain

`poulpy-ckks` requires **nightly Rust** by default: the portable quad-precision scalar [`Quad`] is a newtype over the unstable primitive `f128` (`#![feature(f128)]`). The workspace pins a known-good nightly in `rust-toolchain.toml`.

On non-Apple x86_64, the optional `libquadmath` feature routes `Quad`'s transcendental math through libquadmath (via the `f128` crate) for faster on-the-fly FFT-table builds; the `Quad` type, its storage, and exact arithmetic are identical in every configuration. Elsewhere — other architectures, and macOS, whose toolchain does not ship libquadmath — the feature is a no-op.

## Tests And Backend Integration

`poulpy-ckks` exposes its public API as soon as the crate is imported. Backend
crates own the feature flags that wire concrete CKKS implementations into that
API.

```sh
cargo test -p poulpy-ckks
```

The full backend-generic CKKS conformance suite is instantiated by backend
crates. To run it against the portable reference backends:

```sh
cargo test -p poulpy-cpu-ref --features enable-ckks
```

To run the reference CKKS example:

```sh
cargo run -p poulpy-cpu-ref --example ckks_poly2 --features enable-ckks
```

Like the rest of Poulpy, the public API is backend-agnostic. `poulpy-ckks`
does not depend on any concrete backend crate. Default dispatches and fallback
implementations flow through `poulpy-hal` and `poulpy-core`, while
`poulpy-ckks` remains free to override behavior at the scheme level when
CKKS-specific semantics require it. Concrete execution comes from backend
crates such as `poulpy-cpu-ref` and `poulpy-cpu-avx`.

## Design Notes

This CKKS implementation uses a bivariate Torus representation rather than the
RNS representation used by many other libraries.

## Why Bivariate Instead of RNS?

The main user-visible consequence of the bivariate representation is that CKKS
precision and homomorphic capacity are managed at the bit level rather than at
the prime-chain level.

That changes the ergonomics in a few important ways:

- **Bit-level homomorphic consumption:** operations consume exactly the number of
  bits they need. For example, multiplying by `3 / 2^8` consumes `8` bits of
  capacity, rather than forcing a whole-prime level drop.
- **Trivial scale management:** scales and remaining capacity are tracked as powers
  of two, so rescaling and alignment are expressed directly in bits instead of
  through modulus-chain bookkeeping and rational scaling factors.
- **Easier parameterization:** users specify a modulus budget by size rather than by
  hand-picking an RNS prime chain. In that view, `logQ = 1000` means "about
  1000 bits of total modulus budget," and capacity is then consumed bit by bit.
- **Compact plaintexts:** plaintexts polynomials do not suffer any expansion unlike
  the RNS basis. They stay in an optimal compact representation instead of living
  across the full `logQ`.
- **Circuit-independent evaluation-key parameterization:** because capacity is
  granular at the bit level, evaluation keys are not tied to a specific level
  schedule or prime decomposition for a given circuit.

The goal of this representation is not just ergonomics. It is meant to provide
those advantages while remaining comparable in performance to state-of-the-art
RNS CKKS libraries.

Each ciphertext carries CKKS metadata:

- `log_delta`: base-2 logarithm of the plaintext precision
- `log_sparsity`: sparse-packing factor (`log2` of the slot replication; `0` is dense)
- `log_budget`: remaining homomorphic capacity (includes message integer part), derived from the wrapped GLWE's torus width `k` as `k - log_delta` rather than stored

That metadata is part of the evaluator state. User code should treat it as
scheme-managed information: encryption, rescale, multiplication, addition, and
the other evaluator methods update it automatically.

Another important design point is that cryptographic and arithmetic operations
are invoked through traits on `Module<BE>`, not through methods on the
ciphertext/plaintext types themselves. This matches the rest of Poulpy: data
lives in layouts, behavior lives in module traits, and backend-specific
overrides remain possible. Data-management methods (`.set_meta_checked()`,
`.to_host_owned()`) and typestate transitions (`.normalize()`) are the
exceptions: they live on the struct because they are inherently tied to the
type, not to the backend.

## Crate Organization

The crate is arranged in four interdependent modules (plus supporting
modules for encoding, data structures, testing, and error handling)
that follow the same pattern used throughout the Poulpy workspace:

```
   ┌─────────┐     ┌─────────┐     ┌─────────────┐     ┌────────────────┐
   │   api   │────►│   oep   │────►│  delegates  │◄────│    default     │
   └─────────┘     └─────────┘     └─────────────┘     └────────────────┘
```

**Overriding a method**: a backend replaces the default behavior for any
operation by implementing the corresponding `oep` trait directly instead
of relying on the blanket wiring to `default`.  Only hot-path operations
need explicit overrides; everything else is inherited for free.

### Layer descriptions

| Module | Visibility | Role |
|--------|-----------|------|
| `api` | public | Typed, ergonomic evaluator traits (`CKKSAddOps`, `CKKSMulOps`, `CKKSAffineOps`, …) that `Module<BE>` implements. These are what user code calls. |
| `delegates` | crate-private | Implements each `api` trait on `Module<BE>` by delegating to `oep`. Also owns composite operations (affine, mul-add, dot-product, etc.) that are built from two or more primitives and therefore live above the OEP layer. |
| `oep` | public | Operation Exposition Pattern. Each `CKKS*Impl<BE>` unsafe trait defines the raw dispatch surface: static methods taking `&Module<BE>` directly. A blanket `impl` wires every backend that satisfies the HAL bounds to the corresponding `default` method. Macros (`impl_ckks_*_defaults!`) are the only thing a backend crate needs to call to opt in. `CKKSImpl<BE>` is the aggregate supertrait required by composite ops. |
| `default` | public | One trait per operation family (e.g. `CKKSAddDefault<BE>`) holding the portable algorithm implementations as regular methods on `Module<BE>`. Backends that need to override an operation implement the corresponding `oep` trait directly instead of relying on this layer. |
| `layouts` | public | CKKS-level data wrappers: `CKKSCiphertext<D>`, `CKKSPlaintext<D>`, `UnnormalizedCKKSCiphertext<D>`, allocation helpers (`CKKSModuleAlloc`), and the `CKKSPlaintextVecHostCodec<F>` encoding trait. |
| `encoding` | public | Scheme-level encoding definitions shared by backends (e.g. the PaCo host reference `paco_coeff_encodings_host`). Slot/coefficient encoding itself is a backend-resident operation exposed by `api::CKKSEncodingOps` and dispatched through `oep::CKKSEncodingImpl`. |
| `test_suite` | public (feature `test-utils`) | Backend-agnostic test suite. Enable the `test-utils` feature (backend crates do so in dev-dependencies) and invoke `ckks_backend_test_suite!` in a backend crate's test module to run the full suite against that backend without duplicating test logic. |
| `error` | private | `CKKSError`, `CKKSResult`, and `CKKSCompositionError`, re-exported at the crate root, plus checked arithmetic helpers used by the default implementations. |

## Public Types

The main CKKS-facing types are:

- `CKKSCiphertext<D>` — encrypted CKKS value; wraps a core GLWE ciphertext
- `UnnormalizedCKKSCiphertext<D>` — typestate wrapper for ciphertexts produced
  by unnormalized linear operations; cannot be passed to DFT-domain primitives
  until `.normalize(module, scratch)` is called
- `CKKSPlaintext<D>` — quantized CKKS plaintext in the torus / ZNX domain
- `CKKSMeta` — semantic precision metadata
- `CKKSPlaintextVecHostCodec<F>` — trait for encoding/decoding host floats
  into/out of a `CKKSPlaintext`

`CKKSMeta` stores the logical precision metadata used by the scheme:

```rust
pub struct CKKSMeta {
    pub log_delta: usize,
    pub log_sparsity: usize,
}
```

`log_budget`, the remaining homomorphic capacity, is not stored here; it is
derived from the wrapped GLWE's torus width `k` as `log_budget = k - log_delta`.

## Encoding

Slot and coefficient encoding are backend-resident operations on `Module<BE>`,
exposed by `api::CKKSEncodingOps<BE, F>`:
`ckks_encode_slots_assign_into` / `ckks_decode_slots_into` operate on a
backend-resident `CKKSEncodingBuffer`, `ckks_encode_coeffs_into` /
`ckks_decode_coeffs_into` on raw coefficients, and
`ckks_slots_to_coeffs_assign` / `ckks_coeffs_to_slots_assign` convert a buffer
in place. `api::CKKSEncodingHostOps` layers host-slice convenience adapters on
top:

```rust,ignore
use poulpy_ckks::api::CKKSEncodingHostOps;

let m = 8;  // number of complex slots
let re = vec![0.0f64; m];
let im = vec![1.0f64; m];

// allocate a plaintext via the module, then encode
let mut pt = module.ckks_pt_vec_alloc(base2k.into(), prec);
module.ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch)?;

let mut re_out = vec![0.0f64; m];
let mut im_out = vec![0.0f64; m];
module.ckks_decode_reim_into(&pt, &mut re_out, &mut im_out, &mut scratch)?;
```

The scalar `F` (e.g. `f64`) is fixed by the backend's
`oep::CKKSEncodingImpl<BE, F>` implementation, and FFT plans are owned and
cached by the module itself, so user code never constructs an encoder object
or an FFT table.

## End-to-End Example: Chebyshev sine approximation

The crate includes a runnable example at
[`poulpy-cpu-ref/examples/ckks_poly2.rs`](../poulpy-cpu-ref/examples/ckks_poly2.rs)
that approximates `sin(x)` on `[-1, 1]` with a degree-31 Chebyshev interpolation,
`sin(x) ≈ Σ cᵢ·Tᵢ(x)`, and evaluates it homomorphically through the Baby-Step
Giant-Step polynomial evaluator. It follows the standard six-phase CKKS workflow:

1. **setup** — build the module, secret key, and relinearization (tensor) key
2. **encoding** — Chebyshev-interpolate `sin`, decompose it into BSGS form, and encode the input slots
3. **encryption** — encrypt the slot vector `x`
4. **evaluation** — populate the Chebyshev power basis and run the BSGS evaluation
5. **decryption** — decrypt and decode the result
6. **verification** — compare against the reference `f64::sin`

The polynomial is interpolated and decomposed on the host, then evaluated on a
power basis built from the encrypted input:

```rust,ignore
use poulpy_ckks::{
    api::CKKSPolynomialEvaluationOps,
    polynomial::{Basis, EncodeBSGS, Polynomial},
    power_basis::{PowerBasis, PowerBasisGen},
};

// host side: degree-31 Chebyshev interpolation of sin on [-1, 1], in BSGS form
let poly = Polynomial::chebyshev_interpolate(DEGREE, -1.0, 1.0, f64::sin)?;
let bsgs = poly.encode_bsgs(&host_module, BASE2K.into(), COEFF_META)?;

// encrypted side: populate the Chebyshev power basis, then evaluate
let mut pb = PowerBasis::new(Basis::Chebyshev, ct_x);
pb.populate(DEGREE, bsgs.log_split(), bsgs.parity(), &module, &tsk_prepared, &mut scratch)?;

let mut ct_sin = module.ckks_ciphertext_alloc(BASE2K.into(), CT_K.into());
module.ckks_eval_poly_real_const_coeffs_from_power_basis(
    &mut ct_sin, &bsgs, &pb, &tsk_prepared, &mut scratch,
)?;
```

That example is meant to showcase the intended user workflow end to end:
encoding, encryption, evaluation, decryption, and decoding.

## Evaluation Style

Leveled operations are invoked through traits implemented on
`poulpy_hal::layouts::Module<BE>`. All traits are defined in `crate::api`.

| Trait | Operations |
|-------|-----------|
| `CKKSEncryptOps` / `CKKSDecryptOps` | encryption and decryption |
| `CKKSEncodingOps` / `CKKSEncodingHostOps` | backend-resident slot/coefficient encoding and decoding, plus host-slice adapters |
| `CKKSAddOps` | normalized and unnormalized ciphertext and plaintext addition |
| `CKKSSubOps` | normalized and unnormalized subtraction |
| `CKKSNegOps` | negation |
| `CKKSMulOps` | ciphertext–ciphertext and ciphertext–plaintext multiplication |
| `CKKSMulAddOps` | fused `dst += a * b` variants |
| `CKKSMulSubOps` | fused `dst -= a * b` variants |
| `CKKSAffineOps` | fused affine: `dst = a * scale_coeff + offset_coeff` |
| `CKKSAddManyOps` | tree-reduction add over slices |
| `CKKSDotProductOps` | inner product of ciphertext or plaintext slices |
| `CKKSImagOps` | multiplication and division by `i` (imaginary unit rotation) |
| `CKKSCopyOps` | level-aware ciphertext copy |
| `CKKSRotateOps` | homomorphic slot rotation |
| `CKKSConjugateOps` | homomorphic conjugation |
| `CKKSPow2Ops` | multiplication and division by powers of two |
| `CKKSPlaintextVecOps` | plaintext ZNX operations |
| `CKKSPolynomialEvaluationOps` | Baby-Step Giant-Step polynomial evaluation (monomial and Chebyshev bases) |
| `CKKSLinearTransformationOps` | homomorphic matrix-vector product over the slots (BSGS diagonal method) |
| `CKKSDFTOps` / `CKKSDFTMatrixOps` | homomorphic DFT (`CoeffsToSlots` / `SlotsToCoeffs`) and its compiled plaintext matrices |
| `CKKSEvalModOps` | homomorphic modular reduction (`EvalMod`) |
| `CKKSBootstrappingOps` | bootstrapping pipeline (mod-raise, homomorphic DFT, `EvalMod`) |
| `CKKSPaCoOps` | PaCo bootstrapping (see [`docs/paco.md`](../docs/paco.md)) |
| `CKKSAllOpsTmpBytes` | scratch size queries for all operations |

For example, ciphertext addition uses `CKKSAddOps<BE>` and is called through
the module:

```rust,ignore
use poulpy_ckks::{
    api::CKKSAddOps,
    layouts::CKKSCiphertext,
};

module.ckks_add_into(&mut dst, &lhs, &rhs, scratch)?;
module.ckks_add_assign(&mut lhs, &rhs, scratch)?;
```

### Unnormalized Operations

The `*_unnormalized` methods on `CKKSAddOps` and `CKKSSubOps` (e.g.
`ckks_add_into_unnormalized`, `ckks_sub_assign_unnormalized`) write into an
`UnnormalizedCKKSCiphertext`. This type does not implement
`GLWEToBackendRef`/`GLWEToBackendMut`, so it cannot be accidentally passed to
any DFT-domain primitive (keyswitching, convolution, automorphisms). Call
`.normalize(module, scratch)` to propagate carries and recover a
`CKKSCiphertext`.

Note that `.normalize()` is an exception to the principle stated above:
it is a method on the struct rather than on `Module<BE>`. It lives there
because it must *consume* the `UnnormalizedCKKSCiphertext` by value as the
only typestate exit, which cannot be expressed as a module method. The
actual computation is still dispatched through the `module` argument it
receives.

## Backends

`poulpy-ckks` does not depend on any concrete backend crate. In practice, most
users will choose one of:

- `poulpy-cpu-ref` for portable reference execution
- `poulpy-cpu-avx` for optimized x86_64 execution when AVX2/FMA is available
- `poulpy-cpu-avx512` for AVX-512F and AVX-512-IFMA execution when those
  target features are available

Backend selection happens through the `BE` parameter of `Module<BE>`. Encoding
dispatches through the backend like every other operation
(`oep::CKKSEncodingImpl<BE, F>`); backend crates opt in with the
`impl_ckks_*` macros, so no backend-specific type appears in user code.

## Roadmap

The core leveled evaluator building blocks are now implemented:

- polynomial evaluation (Baby-Step Giant-Step / Paterson-Stockmeyer)
- linear transformations (matrix-vector products over the slots)
- homomorphic DFT (`CoeffsToSlots` / `SlotsToCoeffs`)
- homomorphic modular reduction (`EvalMod`)
- bootstrapping (mod-raise, homomorphic DFT, and `EvalMod`)
- PaCo bootstrapping (partial CoeffsToSlots, without ModUp or `EvalMod`; see [`docs/paco.md`](../docs/paco.md))

Planned evaluator work:

- conjugate invariant ring

Higher-level functionality on top of that foundation:

- discrete CKKS
- scheme switching
- additional higher-level circuit and application primitives built on top of the
  leveled and bootstrapped evaluator

The intent is to keep the low-level API modular and agnostic enough of the encoding
(for example to easily support the conjugate invariant ring) while progressively adding
these higher-level features without changing the backend-agnostic programming model.

## Where to Look Next

- `src/api/encoding.rs` for the slot/coefficient encoding API (the reference packing lives in `poulpy-cpu-ref/src/ckks_encoding.rs`)
- `src/layouts/` for CKKS data structures
- `src/api/` for evaluator trait definitions
- `src/test_suite/` for end-to-end usage patterns
- `poulpy-cpu-ref/examples/ckks_poly2.rs` for the full end-to-end runnable example
