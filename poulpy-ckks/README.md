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
- reusable minimax fitting, degree selection, composite sign generation, and
  interval-mapped BSGS approximation plans

## Toolchain

`poulpy-ckks` requires **nightly Rust** by default: the portable quad-precision scalar [`Quad`](src/scalar.rs) is a newtype over the unstable primitive `f128` (`#![feature(f128)]`). The workspace pins a known-good nightly in `rust-toolchain.toml`.

The optional `libquadmath` feature changes the backing math library for `Quad`
on supported targets; its storage and arithmetic interfaces stay the same.
See [`Cargo.toml`](Cargo.toml) for the target conditions.

## Tests and backend integration

The public API is available when the crate is imported. Backend crates select
implementations by explicitly implementing the corresponding `oep::*Impl`
contracts, usually through the `impl_ckks_*_reference!` macros.

```sh
cargo test -p poulpy-ckks
```

The `test-utils` feature exposes independent mathematical conformance tests and
paired tests that accept a caller-selected comparison backend. Backend crates
register the supported operation/scalar combinations. See the
[repository README](../README.md) for concrete commands, examples, and CI
coverage, and [Implementing CKKS operations](docs/ckks-contracts.md) for the
replacement and parity contracts.

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
lives in layouts, behavior lives in module traits, and a backend may override
an operation with a faster route to the same result, validated by the parity
test. Data-management methods (`.set_meta_checked()`,
`.to_host_owned()`) are the exception: they live on the struct because they
are inherently tied to the type, not to the backend.

## Crate organization

Public calls follow `api → delegates → oep`. A backend's explicit `*Impl`
implementation selects the operation. Lower-layer algorithms are public
`reference` implementations composed from core and HAL operations. Simple
compositions of CKKS operations are crate-private derived defaults on the OEP
traits; they call the selected constituent operations.

```text
core + HAL operations → CKKS reference algorithms
                                 ↓ explicit backend wiring
public API → delegates → OEP *Impl contracts
                                 ↳ derived defaults over CKKS operations
```

To replace a method, implement its OEP family directly and call the public
reference methods for the unchanged members. Do not also invoke that family's
reference macro. Other families can retain their own reference wiring. An
override must compute the same circuit and pass parity against a validated
implementation chosen by the caller. This validation is transitive for the
operations and parameter ranges tested; passing finite tests is not a proof for
all parameters.

| Module | Role |
|---|---|
| `api` | Public evaluator traits implemented on `Module<BE>`. |
| `delegates` | Crate-private dispatch, validation, and API compositions such as affine operations and dot products. |
| `oep` | Explicit backend `*Impl` contracts; no blanket selection of operation implementations. `CKKSImpl` aggregates capabilities. |
| `oep::derived` | Crate-private same-layer defaults: add/subtract one, one-shot polynomial evaluation, and the six DFT format/direction wrappers. |
| `reference` | Callable lower-layer algorithms and canonical encoding definitions. Their implementation bounds do not constrain an unrelated override. |
| `layouts` | Ciphertexts, plaintexts, metadata, allocation, prepared plans, and backend-resident encoding buffers. |
| `encoding` | PaCo and SHIP scheme embeddings and explicitly named host helpers, also exposed through `reference::encoding`. |
| `test_suite` | Independent mathematical tests and caller-selected paired tests, enabled with `test-utils`. |
| `error` | Checked errors re-exported as `CKKSError`, `CKKSResult`, and `CKKSCompositionError`. |

## Public Types

The main CKKS-facing types are:

- `CKKSCiphertext<D>` — encrypted CKKS value; wraps a core GLWE ciphertext
- `CKKSPlaintext<D>` — quantized CKKS plaintext in the torus / ZNX domain
- `CKKSMeta` — semantic precision metadata
- `CKKSPlaintextVecHostCodec<F>` — trait for encoding/decoding host floats
  into/out of a `CKKSPlaintext`

`CKKSMeta` stores the logical precision metadata used by the scheme:

```rust
pub struct CKKSMeta {
    pub log_delta: usize,
    pub log_sparsity: usize,
    pub slots: SlotsKind,
}
```

`log_budget`, the remaining homomorphic capacity, is not stored here; it is
derived from the wrapped GLWE's torus width `k` as `log_budget = k - log_delta`.

## Evaluation Keys and `k_aux`

CKKS tensor (relinearization), automorphism, and switching keys use the core
GGLWE gadget layout. These key layouts no longer take a total `k` field.
Instead, they expose `dnum`, `dsize`, and the auxiliary guard `k_aux`; their
total torus precision is derived as

```text
key.k() = dnum * dsize * base2k + k_aux
```

The first term is the gadget-decomposition region. `k_aux` is the region below
the gadget digits that absorbs noise and prevents operations from truncating in
the middle of a digit. It must cover at least one complete digit:

```text
k_aux >= dsize * base2k
```

For example, a conservative tensor-key layout can cover the ciphertext width
with gadget digits and reserve one digit plus the ring-growth allowance as the
auxiliary guard:

```rust,ignore
let digit_bits = DSIZE * BASE2K;
let dnum = CT_K.div_ceil(digit_bits);
let k_aux = digit_bits + N.ilog2() as usize;

let tsk_layout = GLWETensorKeyLayout {
    n: N.into(),
    base2k: BASE2K.into(),
    dnum: dnum.into(),
    dsize: DSIZE.into(),
    k_aux: k_aux.into(),
    rank: Rank(1),
};
```

When migrating an old limb-aligned key with `Dnum(d)` and no auxiliary guard,
use `Dnum(d - 1)` with `k_aux = dsize * base2k`: this preserves the total key
width while making the formerly implicit final digit an explicit guard. Gadget
operations now derive their working width from the input ciphertext and the
key's `(dsize, k_aux)`; they no longer take a caller-supplied output size.

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
use poulpy_ckks::{
    CKKSMeta, SlotsKind,
    api::CKKSEncodingHostOps,
    layouts::CKKSModuleAlloc,
};
use poulpy_hal::api::ModuleN;

let m = 8;  // number of complex slots
let re = vec![0.0f64; m];
let im = vec![1.0f64; m];

let mut pt = module.ckks_pt_vec_alloc(base2k.into(), 50usize.into());
pt.set_meta_checked(CKKSMeta {
    log_delta: 40,
    log_sparsity: (module.n() / (2 * m)).ilog2() as usize,
    slots: SlotsKind::Complex,
})?;
// The host adapter needs module.ckks_reim_tmp_bytes(m) scratch bytes.
module.ckks_encode_reim_into(&mut pt, &re, &im, &mut scratch)?;

let mut re_out = vec![0.0f64; m];
let mut im_out = vec![0.0f64; m];
module.ckks_decode_reim_into(&pt, &mut re_out, &mut im_out, &mut scratch)?;
```

The backend implements `oep::CKKSEncodingImpl<F>` for each supported scalar
precision. Its module owns and caches the transform plans; callers pass only
encoding buffers and plaintexts. The canonical slot ordering, normalization,
and quantization live in [`reference::encoding`](src/reference/encoding.rs).
Quantization rejects non-finite and out-of-range values before modifying the
plaintext. Host slices appear in the explicitly named host helpers and
convenience adapters; the resident OEP signatures require no host access.

## End-to-End Example: Chebyshev sine approximation

For applications that choose a polynomial from an accuracy target instead of
fixing its degree, `poulpy_ckks::approximation` provides Remez minimax fitting,
precision/depth-based degree selection, composite sign coefficients, and
`PolynomialApproximation`. The latter owns both the prepared BSGS polynomial
and its interval map; `CKKSApproximationOps::ckks_eval_approximation` applies
that map and evaluates the polynomial in one reusable operation.
The `_with` selectors accept explicit `RemezOptions` for downstream planners.
Composite-sign generation can propagate a CKKS error margin between factors.
`minimax_multi_interval` and the matching degree/depth selectors fit a single
polynomial on an ordered union such as `[-1, -τ] ∪ [τ, 1]`; its Chebyshev
input map spans the union's convex hull, while the excluded gaps do not
contribute to the reported sup-norm error.

The [repository README](../README.md) links a runnable example that
interpolates `sin(x)` on `[-1, 1]`, encodes the polynomial and input, evaluates
with the Baby-Step Giant-Step method, and checks the decrypted result.

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
| `CKKSAddOps` | ciphertext and plaintext addition |
| `CKKSSubOps` | ciphertext and plaintext subtraction |
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
| `CKKSApproximationOps` | interval mapping and evaluation of a prepared `PolynomialApproximation` |
| `CKKSPolynomialEvaluationOps` | Baby-Step Giant-Step polynomial evaluation (monomial and Chebyshev bases) |
| `CKKSLinearTransformationOps` | homomorphic matrix-vector product over the slots (BSGS diagonal method) |
| `CKKSDFTOps` / `CKKSDFTMatrixOps` | homomorphic DFT (`CoeffsToSlots` / `SlotsToCoeffs`) and its compiled plaintext matrices |
| `CKKSEvalModOps` | homomorphic modular reduction (`EvalMod`) |
| `CKKSBootstrappingOps` | bootstrapping pipeline (mod-raise, homomorphic DFT, `EvalMod`) |
| `CKKSPaCoOps` | PaCo bootstrapping (see [`docs/paco.md`](../docs/paco.md)) |
| `CKKSShipOps` | SHIP half bootstrapping (see [`docs/ship.md`](../docs/ship.md)) |
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

### Lazy normalization

Additions, subtractions, plaintext additions and `ckks_double_into` do not
normalize their result: they clear the wrapped GLWE's canonical flag.
Negation, copies and multiplication by `±i` keep their operand's flag. The next
operation that reads the digits through a DFT (products, rotations,
conjugation, keyswitching, decryption) normalizes a flag-clear operand first,
so a chain of linear steps costs one normalization. Normalize a value that
several such operations read once, with `glwe_normalize_assign`.

## Backend selection

The `BE` parameter of `Module<BE>` selects execution. The CKKS crate contains no
concrete backend dependency or hardware feature selection. Backend crates own
their implementation wiring and supported scalar precisions; see the
[repository README](../README.md) for available implementations.

## Roadmap

The core leveled evaluator building blocks are now implemented:

- polynomial evaluation (Baby-Step Giant-Step / Paterson-Stockmeyer)
- linear transformations (matrix-vector products over the slots)
- homomorphic DFT (`CoeffsToSlots` / `SlotsToCoeffs`)
- homomorphic modular reduction (`EvalMod`)
- bootstrapping (mod-raise, homomorphic DFT, and `EvalMod`)
- PaCo bootstrapping (partial CoeffsToSlots, without ModUp or `EvalMod`; see [`docs/paco.md`](../docs/paco.md))
- SHIP half bootstrapping (mux blind rotations over a sparse secret, without ModUp or `EvalMod`; see [`docs/ship.md`](../docs/ship.md))

Planned evaluator work:

- conjugate invariant ring

Higher-level functionality on top of that foundation:

- scheme switching
- additional higher-level circuit and application primitives built on top of the
  leveled and bootstrapped evaluator

The intent is to keep the low-level API modular and agnostic enough of the encoding
(for example to easily support the conjugate invariant ring) while progressively adding
these higher-level features without changing the backend-agnostic programming model.

## Where to look next

- [Implementing CKKS operations](docs/ckks-contracts.md) for reference ownership, overrides, scratch, and parity.
- [`src/api/encoding.rs`](src/api/encoding.rs) and [`src/reference/encoding.rs`](src/reference/encoding.rs) for resident encoding interfaces and canonical scheme math.
- [`src/layouts/`](src/layouts/) for CKKS data structures.
- [`src/api/`](src/api/) for evaluator traits.
- [`src/test_suite/`](src/test_suite/) for independent and paired conformance tests.
- [Repository README](../README.md) for runnable examples and backend test commands.
