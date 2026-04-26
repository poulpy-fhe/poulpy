# poulpy-ckks

`poulpy-ckks` is the Poulpy crate implementing the CKKS (Cheon-Kim-Kim-Song)
scheme.

It is built explicitly on top of:

- `poulpy-hal` for backend-agnostic modules, layouts, scratch management, and
  low-level arithmetic dispatch
- `poulpy-core` for RLWE-oriented cryptographic building blocks used to assemble
  the CKKS evaluator

The crate exposes:

- CKKS-specific ciphertext and plaintext wrappers
- slot encoding/decoding helpers
- secret-key encryption and decryption
- leveled arithmetic implemented through traits on `Module<BE>`

Like the rest of Poulpy, the public API is backend-agnostic. `poulpy-ckks`
does not implement raw backend arithmetic by itself; instead, it composes
`poulpy-hal` and `poulpy-core`. Default dispatches and fallback implementations
flow through those lower layers, while `poulpy-ckks` remains free to override
behavior at the scheme level when CKKS-specific semantics require it. Concrete
execution still comes from backend crates such as `poulpy-cpu-ref` and
`poulpy-cpu-avx`.

## Design Notes

This CKKS implementation uses a bivariate Torus representation rather than the
RNS representation used by many other libraries.

## Why Bivariate Instead of RNS?

The main user-visible consequence of the bivariate representation is that CKKS
precision and homomorphic capacity are managed at the bit level rather than at
the prime-chain level.

That changes the ergonomics in a few important ways:

- Bit-level homomorphic consumption: operations consume exactly the number of
  bits they need. For example, multiplying by `3 / 2^8` consumes `8` bits of
  capacity, rather than forcing a whole-prime level drop.
- Trivial scale management: scales and remaining capacity are tracked as powers
  of two, so rescaling and alignment are expressed directly in bits instead of
  through modulus-chain bookkeeping and rational scaling factors.
- Easier parameterization: users specify a modulus budget by size rather than by
  hand-picking an RNS prime chain. In that view, `logQ = 1000` means “about
  1000 bits of total modulus budget,” and capacity is then consumed bit by bit.
- Compact plaintexts: plaintexts do not expand over a full RNS basis. They stay
  in a compact representation instead of living across the full `logQ`.
- Circuit-independent evaluation-key parameterization: because capacity is
  granular at the bit level, evaluation keys are not tied to a specific level
  schedule or prime decomposition for a given circuit.

The goal of this representation is not just ergonomics. It is meant to provide
those advantages while remaining comparable in performance to state-of-the-art
RNS CKKS libraries.

Each ciphertext carries CKKS metadata:

- `log_decimal`: base-2 logarithm of the plaintext precision
- `log_hom_rem`: remaining homomorphic capacity

That metadata is part of the evaluator state. User code should treat it as
scheme-managed information: encryption, rescale, multiplication, addition, and
the other evaluator methods update it for you.

Another important design point is that the public operational API lives on
`Module<BE>`, not on the ciphertext/plaintext structs themselves. This matches
the rest of Poulpy: data lives in layouts, behavior lives in module traits, and
backend-specific overrides remain possible.

## Crate Organization

| Module | Role |
|--------|------|
| `encoding` | CKKS encoding helpers, currently including real/imaginary slot packing |
| `layouts` | CKKS wrappers around core GLWE layouts (`CKKSCiphertext`, `CKKSPlaintextVecZnx`, `CKKSPlaintextVecRnx`, `CKKSPlaintextCstRnx`, `CKKSPlaintextCstZnx`) |
| `leveled` | Encryption, decryption, leveled arithmetic, and rescaling |

## Public Types

The main CKKS-facing types are:

- `CKKSCiphertext<D>`
- `CKKSPlaintextVecZnx<D>`
- `CKKSPlaintextVecRnx<F>`
- `CKKSPlaintextCstRnx<F>`
- `CKKSPlaintextCstZnx`
- `CKKSMeta`

`CKKSMeta` stores the logical precision metadata used by the scheme:

```rust
pub struct CKKSMeta {
    pub log_decimal: usize,
    pub log_hom_rem: usize,
}
```

## Encoding Example

The `encoding::Encoder` helper packs user-provided real and imaginary slot
vectors into an RNX plaintext:

```rust
use anyhow::Result;
use poulpy_ckks::{
    encoding::Encoder,
    layouts::CKKSPlaintextVecRnx,
};

fn main() -> Result<()> {
    let m = 8;
    let re = vec![0.0; m];
    let im = vec![1.0; m];

    let encoder = Encoder::<f64>::new(m)?;
    let mut pt = CKKSPlaintextVecRnx::<f64>::alloc(2 * m)?;

    encoder.encode_reim(&mut pt, &re, &im)?;

    let mut re_out = vec![0.0; m];
    let mut im_out = vec![0.0; m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out)?;

    Ok(())
}
```

## End-to-End Example: Evaluate `(a + b*x) + (c + d*x) * x^2`

The crate includes a runnable example at
[`examples/poly2.rs`](./examples/poly2.rs) that:

1. encodes complex slots into a CKKS plaintext
2. encrypts `x`
3. evaluates `(a + b*x) + (c + d*x) * x^2`
4. decrypts and decodes the result

The evaluation phase follows the same operation order as the runnable example:

```rust,ignore
use poulpy_ckks::{
    CKKSInfos,
    layouts::{CKKSCiphertext, CKKSMaintainOps},
    leveled::{CKKSAddOpsUnsafe, CKKSMulAddOps, CKKSMulOps},
};
use poulpy_core::GLWENormalize;

let mut ct_x2 = CKKSCiphertext::alloc(N.into(), ct_x.log_hom_rem().into(), BASE2K.into());
module.ckks_square(&mut ct_x2, &ct_x, &tsk_prepared, scratch.borrow())?;
module.ckks_compact_limbs(&mut ct_x2)?;

let linear_k = ct_x.effective_k() - PREC_PT.log_decimal;

let mut right_linear = CKKSCiphertext::alloc(N.into(), linear_k.into(), BASE2K.into());
module.ckks_mul_pt_const_rnx(&mut right_linear, &ct_x, &cst_d, PREC_PT, scratch.borrow())?;
unsafe {
    module.ckks_add_pt_const_rnx_assign_unsafe(&mut right_linear, &cst_c, PREC_PT, scratch.borrow())?;
}
module.glwe_normalize_assign(&mut right_linear, scratch.borrow());

let right_branch_k = ct_x2.effective_k() - ct_x2.log_decimal();
let mut right_branch = CKKSCiphertext::alloc(N.into(), right_branch_k.into(), BASE2K.into());
module.ckks_mul(&mut right_branch, &right_linear, &ct_x2, &tsk_prepared, scratch.borrow())?;
module.ckks_compact_limbs(&mut right_branch)?;

let mut poly = CKKSCiphertext::alloc(N.into(), right_branch.effective_k().into(), BASE2K.into());
unsafe {
    module.ckks_add_pt_const_rnx_unsafe(&mut poly, &right_branch, &cst_a, PREC_PT, scratch.borrow())?;
}
module.ckks_mul_add_pt_const_rnx(&mut poly, &ct_x, &cst_b, PREC_PT, scratch.borrow())?;
```

That example is meant to showcase the intended user workflow end to end:
encoding, encryption, evaluation, decryption, and decoding.

## Evaluation Style

Leveled operations are invoked through traits implemented on
`poulpy_hal::layouts::Module<BE>`.

For example, ciphertext addition uses `CKKSAddOps<BE>` and is called through the
module:

```rust,ignore
use poulpy_ckks::{
    layouts::CKKSCiphertext,
    leveled::CKKSAddOps,
};

module.ckks_add(&mut dst, &lhs, &rhs, scratch)?;
module.ckks_add_assign(&mut lhs, &rhs, scratch)?;
```

The same pattern is used for:

- encryption/decryption
- plaintext-ciphertext arithmetic
- multiplication and squaring
- rotation and conjugation
- rescaling and level alignment

## Backends

`poulpy-ckks` does not hard-code a concrete backend. In practice, most users
will choose one of:

- `poulpy-cpu-ref` for portable reference execution
- `poulpy-cpu-avx` for optimized x86_64 execution when AVX2/FMA is available

Backend selection happens through the `BE` parameter of `Module<BE>`.

## Roadmap

Planned work for `poulpy-ckks` includes both lower-level evaluator building
blocks and higher-level CKKS-based functionality.

Near- and mid-term evaluator work:

- linear transformations
- polynomial evaluation
- homomorphic DFT
- state-of-the-art bootstrapping

Higher-level functionality on top of that foundation:

- discrete CKKS
- scheme switching
- additional higher-level circuit and application primitives built on top of the
  leveled and bootstrapped evaluator

The intent is to keep the low-level API modular while progressively adding these
higher-level features without changing the backend-agnostic programming model.

## Where to Look Next

- `src/encoding/reim.rs` for slot packing
- `src/layouts/` for CKKS data structures
- `src/leveled/` for evaluator traits
- `src/leveled/tests/test_suite/` for end-to-end usage patterns
