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
- `log_budget`: remaining homomorphic capacity (includes message integer part)

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
| `api` | public | Typed, ergonomic evaluator traits (`CKKSAddOps`, `CKKSMulOps`, `CKKSAffineOps`, …) that `Module<BE>` implements. These are what user code calls. Re-exported under `leveled::api` for backwards compatibility. |
| `delegates` | crate-private | Implements each `api` trait on `Module<BE>` by delegating to `oep`. Also owns composite operations (affine, mul-add, dot-product, etc.) that are built from two or more primitives and therefore live above the OEP layer. |
| `oep` | public | Operation Exposition Pattern. Each `CKKS*Impl<BE>` unsafe trait defines the raw dispatch surface: static methods taking `&Module<BE>` directly. A blanket `impl` wires every backend that satisfies the HAL bounds to the corresponding `default` method. Macros (`impl_ckks_*_defaults!`) are the only thing a backend crate needs to call to opt in. `CKKSImpl<BE>` is the aggregate supertrait required by composite ops. |
| `default` | public | One trait per operation family (e.g. `CKKSAddDefault<BE>`) holding the portable algorithm implementations as regular methods on `Module<BE>`. Backends that need to override an operation implement the corresponding `oep` trait directly instead of relying on this layer. |
| `layouts` | public | CKKS-level data wrappers: `CKKSCiphertext<D>`, `CKKSPlaintext<D>`, `UnnormalizedCKKSCiphertext<D>`, allocation helpers (`CKKSModuleAlloc`), and the `CKKSPlaintextVecHostCodec<F>` encoding trait. |
| `encoding` | public | Slot encoder/decoder. `Encoder<T>` packs complex slot values into a `CKKSPlaintext` via a negacyclic FFT table `T` supplied by the backend crate. |
| `leveled` | public | Compatibility shim: re-exports `api::*` under `leveled::api::*` so existing code continues to compile. |
| `test_suite` | public | Backend-agnostic test suite. Invoke `ckks_backend_test_suite!` in a backend crate's test module to run the full suite against that backend without duplicating test logic. |
| `error` | private | `CKKSCompositionError` and checked arithmetic helpers used by the default implementations. |

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
    pub log_budget: usize,
}
```

## Encoding

The `encoding::Encoder<T>` helper packs user-provided real and imaginary slot
vectors into a `CKKSPlaintext`. `T` is the negacyclic FFT table implementation
provided by the backend crate (e.g. `FFT64ReimTable<f64>` from
`poulpy-cpu-ref`).

```rust,ignore
use poulpy_ckks::encoding::Encoder;
use poulpy_cpu_ref::FFT64ReimTable;

let m = 8;  // number of complex slots
let re = vec![0.0f64; m];
let im = vec![1.0f64; m];

let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m)?;

// allocate a plaintext via the module, then encode
let mut pt = module.ckks_pt_vec_alloc(base2k.into(), prec);
encoder.encode_reim(&mut pt, &re, &im)?;

let mut re_out = vec![0.0f64; m];
let mut im_out = vec![0.0f64; m];
encoder.decode_reim(&pt, &mut re_out, &mut im_out)?;
```

If you already have a concrete FFT table instance (e.g. one shared across
encoders), use `Encoder::from_table(table, m)` instead of `new`.

## End-to-End Example: Evaluate `(a + b*x) + (c + d*x) * x^2`

The crate includes a runnable example at
[`poulpy-cpu-ref/examples/ckks_poly2.rs`](../poulpy-cpu-ref/examples/ckks_poly2.rs) that:

1. encodes complex slots into a CKKS plaintext
2. encrypts `x`
3. evaluates `(a + b*x) + (c + d*x) * x^2`
4. decrypts and decodes the result

Polynomial coefficients `a`, `b`, `c`, `d` are packed as consecutive scalar
entries inside a single `CKKSPlaintext`. Complex linear forms (e.g. `a + b*x`
for complex `a`, `b`) are built by computing two real affine evaluations and
combining them via `ckks_mul_i_assign`:

```rust,ignore
use poulpy_ckks::{
    CKKSInfos,
    api::{CKKSAddOps, CKKSAffineOps, CKKSImagOps, CKKSMulOps},
    layouts::{CKKSCiphertext, CKKSMaintainOps, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
};

// squaring: consumes one log_delta chunk of homomorphic capacity
let mut ct_x2 = module.ckks_ciphertext_alloc(BASE2K.into(), ct_x.log_budget().into());
module.ckks_square(&mut ct_x2, &ct_x, &tsk_prepared, scratch.borrow())?;
module.ckks_compact_limbs(&mut ct_x2)?;

// build left branch a + b*x and right branch c + d*x
// using packed_coeffs: a at index 0, b at 2, c at 4, d at 6 (real parts)
//                      and at indices 1, 3, 5, 7 (imaginary parts)
let linear_k = ct_x.effective_k() - PREC_PT.log_delta;
let left_linear  = build_complex_affine(&module, &ct_x, &packed_coeffs,
                                        COEFF_A, COEFF_B, linear_k)?;
let right_linear = build_complex_affine(&module, &ct_x, &packed_coeffs,
                                        COEFF_C, COEFF_D, linear_k)?;

// multiply right branch by x^2 and add the two branches
let right_branch_k = ct_x2.effective_k() - ct_x2.log_delta();
let mut right_branch = module.ckks_ciphertext_alloc(BASE2K.into(), right_branch_k.into());
module.ckks_mul(&mut right_branch, &right_linear, &ct_x2, &tsk_prepared, scratch.borrow())?;
module.ckks_compact_limbs(&mut right_branch)?;

let mut poly = module.ckks_ciphertext_alloc(BASE2K.into(), right_branch.max_k());
module.ckks_add_into(&mut poly, &right_branch, &left_linear, scratch.borrow())?;
```

Where `build_complex_affine` combines two real affine forms:

```rust,ignore
// real part: dst = x * scale_re + offset_re
module.ckks_affine_pt_const_into(&mut part0, x, packed_coeffs, offset.re, scale.re, scratch)?;
// imaginary part: part1 = x * scale_im + offset_im, then rotate by i
module.ckks_affine_pt_const_into(&mut part1, x, packed_coeffs, offset.im, scale.im, scratch)?;
module.ckks_mul_i_assign(&mut part1, scratch)?;
module.ckks_add_assign(&mut part0, &part1, scratch)?;
```

That example is meant to showcase the intended user workflow end to end:
encoding, encryption, evaluation, decryption, and decoding.

## Evaluation Style

Leveled operations are invoked through traits implemented on
`poulpy_hal::layouts::Module<BE>`. All traits are defined in `crate::api`.
The historical `crate::leveled::api` path remains available as a backwards-compat alias.

| Trait | Operations |
|-------|-----------|
| `CKKSEncrypt` / `CKKSDecrypt` | encryption and decryption |
| `CKKSAddOps` | normalized ciphertext and plaintext addition |
| `CKKSAddOpsUnnormalized` | unnormalized add (result is `UnnormalizedCKKSCiphertext`) |
| `CKKSSubOps` | normalized subtraction |
| `CKKSSubOpsUnnormalized` | unnormalized subtraction |
| `CKKSNegOps` | negation |
| `CKKSMulOps` | ciphertext–ciphertext and ciphertext–plaintext multiplication |
| `CKKSMulAddOps` | fused `dst += a * b` variants |
| `CKKSMulSubOps` | fused `dst -= a * b` variants |
| `CKKSAffineOps` | fused affine: `dst = a * scale_coeff + offset_coeff` |
| `CKKSAddManyOps` / `CKKSMulManyOps` | tree-reduction add/multiply over slices |
| `CKKSDotProductOps` | inner product of ciphertext or plaintext slices |
| `CKKSImagOps` | multiplication and division by `i` (imaginary unit rotation) |
| `CKKSCopyOps` | level-aware ciphertext copy |
| `CKKSRotateOps` | homomorphic slot rotation |
| `CKKSConjugateOps` | homomorphic conjugation |
| `CKKSPow2Ops` | multiplication and division by powers of two |
| `CKKSRescaleOps` | rescaling and level alignment |
| `CKKSPlaintextVecOps` | plaintext ZNX operations |
| `CKKSMaintainOps` | limb reallocation and compaction |
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

`CKKSAddOpsUnnormalized` and `CKKSSubOpsUnnormalized` write into an
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

Backend selection happens through the `BE` parameter of `Module<BE>`. Note that
the `encoding::Encoder<T>` requires a concrete FFT table type (e.g.
`FFT64ReimTable<f64>`) which comes from the chosen backend crate.

## Roadmap

Planned work for `poulpy-ckks` includes both lower-level evaluator building
blocks and higher-level CKKS-based functionality.

Near- and mid-term evaluator work:

- linear transformations
- polynomial evaluation
- homomorphic DFT
- state-of-the-art bootstrapping
- conjugate invariant

Higher-level functionality on top of that foundation:

- discrete CKKS
- scheme switching
- additional higher-level circuit and application primitives built on top of the
  leveled and bootstrapped evaluator

The intent is to keep the low-level API modular and agnostic enough of the encoding
(for example to easily support the conjugate invariant ring) while progressively adding
these higher-level features without changing the backend-agnostic programming model.

## Where to Look Next

- `src/encoding/reim.rs` for slot packing
- `src/layouts/` for CKKS data structures
- `src/api/` for evaluator trait definitions
- `src/test_suite/` for end-to-end usage patterns
- `poulpy-cpu-ref/examples/ckks_poly2.rs` for the full end-to-end runnable example
