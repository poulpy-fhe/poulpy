# Getting Started

This guide is a map of the Poulpy codebase.
It describes what each crate contains and where to look, how the layers are organized, how to build, test, and benchmark, and how the parameters in the code relate to the usual FHE notation.

## Library organization

Poulpy is a stack of crates with a strict dependency direction.

```
poulpy-hal                 hardware abstraction: layouts and operation traits
└── poulpy-core            scheme-agnostic (Module-)LWE arithmetic (LWE, GLWE, GGSW)
    ├── poulpy-ckks        leveled CKKS evaluator
    └── poulpy-bin-fhe     binary and gate-level FHE

poulpy-cpu-ref             portable reference backend
poulpy-cpu-avx             AVX2 / FMA backend
poulpy-cpu-avx512          AVX-512 / IFMA backend
```

### poulpy-hal

The hardware abstraction layer.
It defines the data layouts and the operation traits, but no arithmetic of its own beyond the layouts.
Its design deliberately mirrors the C library [spqlios-arithmetic](https://github.com/tfhe/spqlios-arithmetic), and the reference transform kernels in the backends are direct Rust ports of it, so the layouts and the operation API match spqlios closely.

The layouts split into three domains.
In the coefficient domain, `VecZnx` is the central type, a base-`2^K` vector of polynomials in `Z[X]/(X^N + 1)` stored as `size` limbs of `N` coefficients, and `ScalarZnx` is its single-limb specialization used for plaintexts and secret keys.
`MatZnx` is the matrix form behind GGSW.
In the transform domain, `VecZnxDft` holds polynomials after the backend's DFT or NTT, where multiplication becomes pointwise, and the prepared operands `SvpPPol` (a scalar polynomial) and `VmpPMat` (a polynomial matrix) live here too.
The large-coefficient `VecZnxBig` is the wide accumulator that holds products before they are normalized back into base-`2^K` limbs.

A `Module<B>` pairs a ring degree `N` with a backend handle that carries any precomputed state such as DFT twiddle tables, and it is the entry point for every operation.
The operation traits are grouped by the type they act on: `vec_znx` for coefficient-domain arithmetic, `vec_znx_dft` for transform-domain arithmetic, `vec_znx_big` for the accumulator, `svp` for scalar-vector products, `vmp` for vector-matrix products, plus convolution and scratch management.

### The backends

A backend supplies the actual low-level arithmetic behind the HAL: the Fourier or number-theoretic transform, the vector-matrix and scalar-vector products, and the coefficient-domain operations.
Each backend crate exposes one or more zero-sized marker types that you pass as the `B` in `Module<B>`.

`poulpy-cpu-ref` is the portable reference.
It implements the full HAL operation set in plain scalar Rust with no intrinsics, runs on any target, and acts as the correctness oracle the other backends are checked against.
It provides `FFT64Ref` and `NTT4x30Ref`.

`poulpy-cpu-avx` and `poulpy-cpu-avx512` do not reimplement the whole HAL.
They hand-vectorize only the hot paths, the transform butterflies and the matrix-vector products, and delegate every other operation to the reference implementation through shared macros.
`poulpy-cpu-avx` adds AVX2 and FMA kernels and provides `FFT64Avx` and `NTT4x30Avx`.
`poulpy-cpu-avx512` adds AVX-512 and IFMA kernels and provides `FFT64Avx512`, `NTT4x30Avx512`, and `NTT3x42Ifma`, the last of which reconstructs its CRT output with an AVX-512 IFMA kernel.

Results are deterministic and bit-identical across backends, since the NTT families are exact and the FFT family is held within correct rounding.
This means you can develop and test against `FFT64Ref` and switch to an accelerated backend with no change in output.
See [backends.md](backends.md) for the three arithmetic families and how to pick one.

### poulpy-core

Scheme-agnostic Module-LWE arithmetic built on the HAL, and the largest crate.
It defines the ciphertext, plaintext, and key types under `layouts/`, and their operations under `api/`.

The ciphertext family is `LWE` (a scalar ciphertext), `GLWE` (its polynomial-ring generalization with `rank` mask polynomials), and the gadget-carrying `GGLWE` and `GGSW` used as the left operands of external products.
Plaintexts are `LWEPlaintext` and `GLWEPlaintext`, and secret keys are `LWESecret` and `GLWESecret`.
On top of these sits a family of evaluation keys, each a thin wrapper over a `GGLWE`: `GLWESwitchingKey` for key-switching, `GLWEAutomorphismKey` for Galois automorphisms, `GLWETensorKey` for relinearization after a tensor product, `GGLWEToGGSWKey` for promoting a GGLWE to a GGSW, and the bridges `GLWEToLWEKey` and `LWEToGLWEKey`.

Two cross-cutting variants exist for most of these types.
A *prepared* variant such as `GLWEPrepared` or `GGSWPrepared` stores the data in the backend's transform domain so repeated products are cheap, and is produced from the standard form by a `prepare` step.
A *compressed* variant stores only a 32-byte seed for the uniform mask, which it regenerates on decompression, to cut serialized size.

The operations under `api/` cover secret-key and public-key encryption, decryption, the external product, key-switching, Galois automorphisms, the trace (a sum of automorphisms), GLWE arithmetic (add, subtract, normalize, multiply by a plaintext or a constant, rotate), the tensoring and relinearization used by ciphertext-ciphertext products, the baby-step/giant-step engines for polynomial evaluation and linear transformations (matrix-vector products), noise measurement, and conversions such as LWE sample extraction from a GLWE.
Encryption takes explicit randomness streams through `Source`, one per role, and a `NoiseInfos` carrying the sigma, with `DEFAULT_SIGMA_XE = 3.2` as the default.
Secret-key encryption takes two, one for the Gaussian error and one for the uniform mask, while public-key encryption takes a third for the public-key randomness.
No operation allocates on the heap: the caller passes a scratch arena, and every operation has a companion `*_tmp_bytes` method that reports how large that arena must be.
The crate also ships a generic conformance suite under `test_suite/` that any backend can run to prove it implements the operations correctly.

### poulpy-ckks

The leveled CKKS evaluator built on core.
It uses the same bivariate base-`2^K` representation rather than an RNS one, and exposes precision through `CKKSMeta`, which tracks `log_delta`, the base-2 log of the encoding scale, and `log_sparsity`, the sparse-packing factor. The remaining homomorphic headroom, `log_budget`, is not stored in `CKKSMeta`; it is derived from the ciphertext's torus width `k` as `k - log_delta`.
The `encoding/` folder maps complex slots to and from a plaintext polynomial through a negacyclic FFT packing of real and imaginary parts.
The `leveled/` arithmetic tracks the `CKKSMeta` precision metadata for you: addition and subtraction align operands by budget and do not consume capacity, while multiplication does in rescaling, which is a bit shift rather than a division by a prime.
The public operations include add, subtract, negate, multiply, fused multiply-add and multiply-subtract, affine maps, slot rotation, conjugation, and multiplication or division by `i` or by powers of two.
Higher-level evaluators build on these: polynomial evaluation, linear transformations (matrix-vector products over the slots), the homomorphic DFT (`CoeffsToSlots` / `SlotsToCoeffs`) built as a chain of those linear transformations, homomorphic modular reduction (`EvalMod`), and a bootstrapping pipeline that composes mod-raise, the homomorphic DFT, and `EvalMod` to refresh a ciphertext's budget.

### poulpy-bin-fhe

The binary and gate-level FHE crate built on core, organized as three layered building blocks.

`blind_rotation` evaluates a programmable lookup table under a GLWE ciphertext, driven by an LWE ciphertext that acts as the rotation index, producing a fresh GLWE whose constant term decrypts to the looked-up value.
This is the gate bootstrapping primitive the rest of the crate is built on.
The available algorithm is `CGGI`, the Chillotti-Gama-Georgieva-Izabachene blind rotation from TFHE, which dispatches at runtime between a classic path, a block-binary path that batches several coefficients per product, and an extended path that spans the table across several polynomials for more precision.

`circuit_bootstrapping` turns a GLWE ciphertext encrypting a small plaintext into a GGSW ciphertext, which can then act as a CMux selector, letting Boolean circuits run on ciphertexts without per-gate noise growth.
It composes three steps and so bundles three sub-keys: a blind rotation key (`brk`), a set of automorphism keys for the trace and packing step (`atk`), and a GGLWE-to-GGSW tensor-switching key (`tsk`).

`bdd_arithmetic` provides word-level arithmetic on encrypted unsigned integers.
A plaintext integer is packed bit by bit into a single GLWE (`FheUint`), each bit is then circuit-bootstrapped into a GGSW (`FheUintPrepared`) so it can drive a CMux, and a statically compiled Binary Decision Diagram circuit for the chosen operation is evaluated bit by bit with those CMux gates.
Addition, subtraction, logical and arithmetic shifts, signed and unsigned comparison, and bitwise and, or, and xor over `u32` come out of the box, alongside an oblivious `GLWEBlindSelection` that picks one ciphertext from many by an encrypted index.
Key preparation and circuit evaluation both offer multi-threaded variants, and the keys are `Sync`.

### poulpy-bench

The consolidated Criterion benchmark suite for the workspace.
It is an internal crate and is not published.

## Inside a crate: api, oep, default, delegates

Every layer (`poulpy-hal`, `poulpy-core`, `poulpy-ckks`) follows the same internal four-folder shape.

| Folder | Role |
|--------|------|
| `api` | Public traits user code calls |
| `oep` | Open extension points, the unsafe backend dispatch traits |
| `default` | Portable fallback algorithms every backend gets for free |
| `delegates` | Wires the `api` traits onto `Module<B>` through `oep` |

The reason for this split is that it decouples scheme code from the arithmetic backend.
You write a scheme once against the `api` traits, name a backend as the `B` in `Module<B>`, and the same code runs on the reference backend, on AVX, on AVX-512, or on a future GPU or FPGA backend with no change.
A new backend implements the `Backend` trait and the `oep` traits, and inherits every algorithm from `default` for free, so it is correct from the first day.
It then overrides only its hot paths by implementing the relevant `oep` trait directly instead of taking the `default`, and each override is independent per operation and per layer.
This is what lets the portable reference prove correctness once while the accelerated backends add speed incrementally without ever forking the scheme logic.
Note that `poulpy-hal` has no `default` folder, since it defines the layouts and the dispatch but leaves the algorithms to the backends.

## Building, testing, and benchmarking

Build the default workspace.

```sh
cargo build
```

The scheme APIs are available as soon as the crates are imported, but backend-owned integration tests stay feature-gated so default builds stay light.
Run them with the matching features.

```sh
cargo test -p poulpy-core
cargo test -p poulpy-ckks
cargo test -p poulpy-cpu-ref --features enable-core
cargo test -p poulpy-cpu-ref --features enable-ckks
cargo test -p poulpy-bin-fhe --features enable-bin-fhe
```

Benchmark targets are split by family through the `hal-bench`, `core-bench`, `bin-fhe-bench`, and `ckks-bench` features.

```sh
cargo bench -p poulpy-bench --features hal-bench,core-bench,bin-fhe-bench,ckks-bench
```

Run a single benchmark binary, optionally with acceleration.

```sh
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench
RUSTFLAGS="-C target-feature=+avx2,+fma" \
  cargo bench -p poulpy-bench --bench vec_znx --features hal-bench,enable-avx
```

## Parameters and the bivariate representation

Poulpy follows the paper [Revisiting Key Decomposition Techniques for FHE](https://eprint.iacr.org/2023/771) and represents a torus polynomial in base `2^K`.
A coefficient is split into limbs, each carrying `K` bits.
This base-`2^K` decomposition is the gadget decomposition itself, so there is no separate RNS layout to manage.
The user only specifies the bit size of the modulus, rescaling is a bit shift rather than a division by a prime, and all schemes share this one plaintext space, which is what lets Poulpy bridge between them.

The paper sets `N` and `K` once at key generation.
The values `L` and `ℓ` vary during a homomorphic computation as the noise budget is consumed.
The genuine paper symbols map to Poulpy names as follows.

| Paper | Poulpy name | Type | Meaning |
|-------|-------------|------|---------|
| `N` | `n` | `Degree` | Ring degree of `X^N + 1`, a power of two |
| `K` | `base2k` | `Base2K` | Bit size of one limb in the base-`2^K` representation |
| `L` | `k` | `TorusPrecision` | Total bits of torus precision, the working ciphertext modulus size |
| `ℓ` (limbs per coefficient) | `size` | `usize` | Number of limbs, always derived as `ceil(k / base2k)` |

Three further layout parameters describe how a ciphertext is shaped.
They do not map to a single paper symbol.

- `rank` (`Rank`) is the number of mask polynomials in a GLWE ciphertext.
  A GLWE ciphertext stores `rank + 1` polynomials, one body and `rank` masks.
  `rank = 1` is standard RLWE.
- `dnum` (`Dnum`) is the number of digits in the gadget decomposition of a GGLWE or GGSW ciphertext.
- `dsize` (`Dsize`) is the number of base-`2^K` limbs grouped into one gadget digit.

Setting the ciphertext modulus is simple in practice.
You pick `base2k` once, then express precision through `k`, which fixes the working modulus at `2^k`.
The limb count `size` is always derived from `k` and `base2k`, never set by hand.
For gadget ciphertexts you additionally choose `dnum` and `dsize` to set the decomposition granularity.

These names appear as fields of the layout structs.

```rust
let glwe_layout = GLWELayout {
    n: Degree(1024),
    base2k: Base2K(17),
    k: TorusPrecision(34),
    rank: Rank(1),
};

let ggsw_layout = GGSWLayout {
    n: Degree(1024),
    base2k: Base2K(17),
    k: TorusPrecision(51),
    rank: Rank(1),
    dnum: Dnum(3),
    dsize: Dsize(1),
};
```

## Where to go next

- For a GLWE encrypt and decrypt roundtrip, read `poulpy-cpu-ref/examples/core_encryption.rs`.
- For the gate and encrypted integer API, read `poulpy-bin-fhe/examples/bdd_arithmetic.rs`.
- For CKKS, read `poulpy-cpu-ref/examples/ckks_poly2.rs`.
- For CKKS polynomial evaluation and homomorphic linear transformations, read [polynomial_evaluation.md](polynomial_evaluation.md) and [linear_transformation.md](linear_transformation.md).
- For the choice of arithmetic backend, read [backends.md](backends.md).
