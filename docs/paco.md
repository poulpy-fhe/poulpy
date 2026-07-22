# PaCo bootstrapping

`poulpy-ckks` implements PaCo (Coron and Seure, [ePrint
2025/886](https://eprint.iacr.org/2025/886)) as a native CKKS operation. PaCo
refreshes selected polynomial coefficients without ModUp or EvalMod. It uses a
structured, low-weight secret to express decryption as four encrypted packing
vectors, then evaluates modular addition through multiplication on the unit
circle.

The implementation uses Poulpy's generator-5 DFT embedding. The mid-pipeline
slot convention is selected by `PaCoSlotOrder` on the plan (`Natural` is the
default; `PaCoPlan::with_slot_order` selects `BitRevLow`); the choice changes
the BSGS diagonal offsets and hence the Galois key set, so it must be kept
stable per persisted key bundle. The exact convention and its relationship to
the paper's reference code are recorded in
[the PaCo DFT specification](spec/paco_dft_convention.md).

## Public API

Invariant-bearing data is exported from `poulpy_ckks::layouts`:

- `PaCoPlan` and `PaCoDFTPlan` describe dimensions, factor schedules, scales,
  plaintext budgets, and BSGS giant steps.
- `PaCoContext` is the compiled, backend-resident set of plaintext linear
  transformations. It contains no encrypted key material.
- `PaCoSecretSpec` samples and validates the structured PaCo secret.
- `PaCoKeySet` is validated, unprepared key material; `PaCoKeysPrepared` is its
  eager backend-prepared form. `PaCoKeys` is the operation-facing access trait
  for eager, lazy, or streamed key stores. `PaCoKeyParameters` fingerprints the
  key-defining `(N, h, C, q, Delta_bsk)` parameters independently of an
  evaluation schedule.

Evaluation is exposed through `poulpy_ckks::api::CKKSPaCoOps` on `Module<BE>`.
All outputs are caller-allocated:

- `ckks_paco_bootstrap_direct_into`: sequential evaluation when the input is
  already under the structured PaCo secret.
- `ckks_paco_bootstrap_into`: one dense-to-PaCo key switch followed by the same
  sequential evaluation.
- `ckks_paco_bootstrap_parallel_direct_into` and
  `ckks_paco_bootstrap_parallel_into`: the corresponding bounded-parallel
  variants.

Each method takes `kappa`. `kappa = 1` is seqPaCo and recovers `C` coefficient
classes; larger powers of two recover `kappa*C` classes. Parallel evaluation
uses the caller plus a borrowed slice of reusable `PaCoWorker` contexts. Each
worker owns a separately configured backend module handle and scratch arena;
at most `1 + workers.len()` branches run concurrently, and branches are
recombined in increasing order.

The only PaCo-specific backend hook is coefficient encoding, the
input-dependent conversion of public ciphertext residues into the four beta
plaintexts. A backend opts in by implementing
`poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl`, which supplies the scratch
bound and the encoding itself; the `CKKSPaCoOps` methods
`ckks_paco_coeff_encodings` and `ckks_paco_coeff_encodings_tmp_bytes`
dispatch to it. The trait imposes no FFT engine, encoder, or host codec: a
backend with a native encoder may implement the whole step as one fused
kernel from the ciphertext residues. The complete scheme definition of the
step is exported as `poulpy_ckks::encoding::paco_coeff_encodings_host`, and a
CPU backend with host-accessible buffers adopts it wholesale with
`poulpy-cpu-ref`'s `impl_ckks_paco_coeff_encoding!` macro, which routes the
staged host routine through the backend's own CKKS encoding implementation.
The rest of PaCo composes existing CKKS multiplication, automorphism,
trace/fold, linear-transformation, allocation, transfer, and metadata APIs.

## Construction outline

```rust,ignore
use poulpy_ckks::{
    api::CKKSPaCoOps,
    layouts::{PaCoContext, PaCoDFTPlan, PaCoPlan},
};
use poulpy_core::layouts::Base2K;

let coeffs_to_slots = PaCoDFTPlan::new(c2s_depths, c2s_giant_steps,
                                       c2s_log_delta, log_budget, c2s_scaling)?;
let slots_to_coeffs = PaCoDFTPlan::new(stc_depths, stc_giant_steps,
                                       stc_log_delta, log_budget, stc_scaling)?;
let plan = PaCoPlan::new(log_n, h, c, log_q)?
    .with_evaluation(log_delta_bsk, log_beta_budget, coeffs_to_slots, slots_to_coeffs)?;

let context = PaCoContext::<MyBackend, f64>::compile(
    &module, Base2K(base2k), plan.clone(), &mut scratch,
)?;

// Obtain a validated PaCoKeys implementation from the application's key
// manager. Context compilation does not create or certify key material; the
// required secret-key relationships are stated below.

let mut output = module.ckks_ciphertext_alloc(Base2K(base2k), k_boot.into());
module.ckks_paco_bootstrap_into(
    &mut output, &exhausted_input, &context, &keys, kappa, &mut scratch,
)?;
```

The outline deliberately leaves `keys` application-supplied: this API has no
integrated PaCo key-generation factory. A key manager derives the structured
secret and four `sigma_t` slot vectors from `PaCoSecretSpec`, creates their
ciphertexts and the required core evaluation keys with the standard encryption
and key-generation operations, then assembles them with `PaCoKeySet::new` and
optionally `PaCoKeySet::prepare` or the clone-free consuming
`PaCoKeySet::into_prepare`. The cryptographic provenance conditions in the next
section are part of that construction and cannot be inferred by the bundle
constructors.

`PaCoPlan::k_boot(base2k, headroom)` computes a limb-aligned output capacity.
Ask the module for the exact caller-arena bound with
`ckks_paco_bootstrap_direct_tmp_bytes` or `ckks_paco_bootstrap_tmp_bytes`; the
latter also covers the one-time dense-to-PaCo switch. Every parallel worker
arena must provide the direct bound reported by its own module. Reuse the same
`PaCoWorker` values across calls. For each worker actually used, preflight
checks the ring degree and the scratch bound computed by that worker's module.
The backend type system does not identify a device, stream, allocator, or
runtime context, so it cannot prove that two `Module<BE>` handles can access
the same buffers. A multi-device backend must therefore construct workers in
the context that owns the supplied ciphertext, plaintext, compiled-context,
and key buffers, or explicitly provide the peer access required by that
backend.

## Key and ciphertext contract

The exhausted input has ring degree `N`, rank one, dense metadata, a valid limb
radix, and effective torus width `log_q`. It is accepted through a generic
backend-readable bound rather than a concrete buffer type, so owned
ciphertexts and scratch-carved views are equally valid inputs; the output is
always a backend-owned ciphertext, since it must hold the full bootstrap width
and outlive the call. Its radix may differ from the
context's: coefficient extraction decodes the input in its own radix, while
encapsulation normalizes the structured ciphertext to the context radix. The
context radix fixes the bootstrapping keys, compiled plaintexts, and output. In
encapsulated mode the input starts under the application's dense secret and is
switched once, at the small input modulus, to the structured PaCo secret.
Direct mode skips this switch.

The four bootstrapping ciphertexts are `Enc_app(sigma_t)`: their plaintexts
contain the structured secret, but the ciphertexts themselves are encrypted
under the application/output key. Consequently the blind rotation transfers
the computation back to the application key and no PaCo-to-dense switch is
required. All four ciphertexts must have the same degree, rank, radix, width,
scale, dense metadata, and enough budget for the complete plan.

`PaCoKeySet::new`, `PaCoKeysPrepared::new`, and operation preflight validate
layouts, metadata, Galois labels, gadget dimensions, storage capacity, and the
`PaCoKeyParameters` fingerprint. They cannot decrypt ciphertexts or inspect
the secrets from which keys were generated. The application key manager is
therefore responsible for all cryptographic provenance and must enforce these
relationships:

- the direct input, or the destination of the optional switching key, uses the
  structured secret represented by `PaCoSecretSpec`;
- bootstrapping ciphertext `t` encrypts that same specification's `sigma_t`,
  and all four are encrypted under one application/output secret;
- every automorphism key and the tensor key is generated for that same
  application/output secret; and
- in encapsulated mode, the switching key maps the input's dense application
  secret to the structured PaCo secret. To return under the original
  application key, that dense secret must also be the application/output
  secret used by the bootstrapping, automorphism, and tensor keys.

The automorphism map must contain every element returned by
`plan.galois_elements()`, with each map label equal to the key's own Galois
element. The tensor and optional switching keys must match the ring, radix,
ranks, and required storage sizes. Constructors and operation preflight reject
structurally incompatible material before evaluation; passing structurally
valid material with incorrect provenance instead produces a cryptographically
invalid result.

After recombination the output metadata is:

```text
log_sparsity = log2(N / (kappa*C))
log_delta    = bootstrap_scale - (log_q - 2 - input_scale - extra_scale_log2)
```

The relabel is budget-neutral: the effective torus width is unchanged. Signed
scale arithmetic is checked before evaluation, so a negative scale, budget
underflow, overflow, or insufficient output capacity is returned as an error.

## Security and parameter selection

PaCo relies on a non-standard structured secret distribution: a binary key of
weight exactly `h`, with one nonzero in each residue class modulo `h`. The
paper's security discussion maps this distribution to a reduced-dimension,
reduced-weight sparse-secret estimate and studies parameters such as
`N >= 2^15` and `h >= 64`. Treat that mapping as an explicit deployment
assumption; do not infer a security level from the ring degree alone. Sparse
encapsulation keeps the application's long-lived key dense, but it does not
remove the PaCo structured-key assumption from the bootstrap key material.

The circle embedding also imposes a precision contract. The selected scalar
must represent residues modulo `q` exactly, the bootstrap scale must exceed
`log_q`, and useful messages must remain small compared with `q`. The `f64`
path therefore accepts at most 52 residue bits; wider exact scalar/backend
combinations may use up to the implementation's 63-bit residue limit. For a
coefficient of magnitude `m`, the leading small-angle error is approximately
`|m| * (2*pi*|m|/q)^2 / 6`, before homomorphic noise. Size `headroom` for the
recovered coefficient magnitude plus a safety margin, and validate precision
and security with application-scale parameters rather than the small test
instances.

## Cost and retained PaCo-specific code

For a validated plan, budget consumption is

```text
bootstrap_scale * (1 + log2(h))
  + c2s_factor_count * c2s_scale
  + stc_factor_count * stc_scale
```

The trace and product folds remain small PaCo-specific CKKS compositions. The
general GLWE trace is not equivalent: PaCo folds a periodic slot layout through
specific automorphism-add and ciphertext-product schedules. Factor generation
also remains PaCo-specific, while factor evaluation uses the standard BSGS
linear-transformation engine.

The implementation fuses the psi/mu map into the last partial CoeffsToSlots
factor and eta/pair packing into the first SlotsToCoeffs factor. The psi/mu
fusion takes whichever form the schedule makes cheapest. When psi shares its
last schedule group with butterfly layers, the map is antilinear in the factor
output and is evaluated as the conjugation-augmented pair `A*w + B*conj(w)`:
one plain conjugation keyswitch and two diagonal matrices at one level, which
erases the mu level at roughly twice that factor's diagonal work. When psi is
scheduled alone (last `factorization_depth` entry `1`) the pair would
degenerate to two one-diagonal matrices, so the evaluator instead emits the
operation-lean fast tail: the pairing as a single fused conjugate-rotate
keyswitch (Galois element `-5^C`) followed by one mu-mask multiplication,
which then costs its own level — the paper's layout. Both forms compute the
identical map; the schedule therefore selects the speed/depth trade per
instance: end the c2s schedule in `1` to prioritize throughput, merge psi
deeper to buy a level with diagonals.

`PaCoKeySet::prepare` is preprocessing, not transfer: its storage type must
already be the host-accessible `BE::OwnedBuf`. The borrowed form clones the
four bootstrapping ciphertexts; `PaCoKeySet::into_prepare` consumes the
unprepared set and moves them instead. Neither form uploads cross-backend
material or verifies device/runtime-context residency beyond what the
backend's buffer types express. Cross-backend upload, serialization, and any
device-placement checks belong in the application's key manager; already
prepared material can be structurally checked with `PaCoKeysPrepared::new`.

## Validation

The reusable backend suite covers plan rejection, structured-secret packing,
coefficient encoding, individual trace/product folds, both linear transforms,
direct and encapsulated bootstrap, ordered parallel recombination, and output
scale/budget/sparsity. Its independent cleartext oracle and intermediate gates
live in `poulpy_ckks::test_suite`, not in the production operation surface.

Run the reference backend gates with:

```bash
cargo test -p poulpy-cpu-ref --features enable-ckks paco
```
