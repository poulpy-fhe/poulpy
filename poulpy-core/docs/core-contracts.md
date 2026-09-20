# Core operation contracts and executable inventory

The normative portable computations are the callable bodies in `reference`.
[`core-contracts.json`](core-contracts.json) records every trait method declared
in `oep`, including scratch queries, and the reference, key-policy, preparation
and decompression interfaces exposed elsewhere. Each record identifies its
source, public or internal dispatch, portable body or explicit exception, and
executable contract suite. Backend registrations identify the concrete portable
CPU oracle and the libtest modules that must run.

The inventory distinguishes implementation boundaries from conveniences:

- Abstract `*Reference` traits are explicit backend choices. A backend can
  implement one family while forwarding the remaining families with macros.
  The corresponding `*Composition` traits carry the HAL requirements for using
  the portable bodies; their blanket implementations do not occupy the override
  boundary.
- Sampling and strided gadget products are direct backend extension points.
  Tensoring has a direct `*Impl` boundary with callable reference compositions.
- Prepared factories and decompression traits include blanket convenience
  methods and metadata/size adapters. Their inclusion records required behavior;
  it does not advertise those blanket methods as independent override points.
- `BSGSOps` and key providers are caller policies. The BSGS engine owns sequencing;
  the policy owns arithmetic, precision accounting and key lookup. Data-view and
  metadata accessors (`*ToBackendRef`, `*Infos`, `BabyStep`, `PowerBasisHelper`,
  and similar layout adapters) describe operands rather than additional backend
  arithmetic kernels. Their behavior is exercised by the operation consumers.
- `GLWEKeyswitchInternal` and `GGLWEProductReference` are shared composition
  plumbing, not extra blanket-free specialization surfaces.

## Common observable contract

Logical coefficients live in the integer/torus layouts described by HAL.
Ring operations use `Z[X]/(X^N + 1)`; `rank` counts mask polynomials, while gadget
rows are indexed by `dnum`, `dsize`, auxiliary precision and row stride. The
operation's input/output metadata determines the live precision and radix.
Allocated capacity may exceed live precision. Raw addition, subtraction and
negation return limb arithmetic without carry normalization; operations with a
normalization/conversion stage instead require its exact canonical output,
rounding and tail behavior. Mathematically equivalent noncanonical limbs do not
replace a required canonical result.

Out-of-place inputs are read-only. Assign forms consume the previous destination
as an input while preserving the metadata/storage not explicitly transformed by
the operation. Safe Rust borrows govern aliasing; a backend must not introduce
additional aliases through raw storage. Shape errors must be rejected according
to the public contract and the checks in the reference path; implementations
must not reinterpret incompatible ranks, dimensions or strides. APIs returning
`Result` preserve error propagation from key providers and arithmetic policies.

A `*_tmp_bytes` result advertises sufficient arena capacity for that backend's
implementation with the supplied metadata. Tests allocate the reference and
candidate budgets separately and poison scratch. Equal counts between backends
are not required. Preparation and allocation helpers similarly report storage
for their own backend's opaque representation; prepared bytes are not a portable
comparison format.

The partial DFT views in portable digit-product/external-product compositions
require `Backend::DFT_LIMBS_CONTIGUOUS`. Incompatible instantiations fail at
compile time with the required override named. A backend with another layout
must replace those operations; this does not justify a fused backend-specific
operation in HAL. Host coefficient statistics and diagnostic decryption remain
host-oriented utilities where their bounds require host access.

## Family contracts

| Family | Result and metadata contract | Executable oracle |
|---|---|---|
| Add, subtract, negate | Raw componentwise signed-limb arithmetic, without carry normalization. Add/subtract require equal radix, support plaintext rank-zero operands where specified, and truncate/zero-extend to destination limb capacity. Assign forms use the old destination; normalize explicitly when canonical digits are needed. | Portable-reference integer equality over boundary widths and ranks; explicit assign variants. |
| Copy, zero, normalize | Copy converts the declared radix/precision; zero clears the represented ciphertext; normalization returns the canonical rounded value at destination precision/radix. At equal precision it preserves the represented torus value. Capacity/tail behavior follows the reference. | Poisoned destinations, precision boundaries, canonical output and metadata comparison. |
| Rotation, `X^p - 1`, shifts | Rotation and `X^p - 1` apply their negacyclic permutation/sign or subtraction operations to raw limbs. Binary shifts include saturating counts and use the reference’s precision-dependent normalization/rounding. | Reference equality, assign variants, exact per-backend scratch. |
| Constant/plain multiplication | Multiply by the selected plaintext coefficient or polynomial with the specified conversion offset; apply the reference's truncation/rounding and output precision. | Independent preparation and logical integer comparison, including assign variants. |
| Tensoring/relinearization | Produce the triangular mask-product layout, then key-switch its quadratic terms to the destination key/rank. Auxiliary guard and spill widths are part of the result. | Tensor/square and independent prepared-key relinearization equality; tensor decryption. |
| Keyswitching/digit products | Decompose canonical input at the requested digit size/stride and accumulate the selected prepared gadget rows. The output width includes the reference's carry/guard requirements. | Explicit CPU reference, differing ranks/digit sizes, assign variants, strided prepared-key tests. |
| Automorphism | Apply the requested odd Galois map and key-switch into the declared destination key; add/subtract/sub-negate variants compose with the prior destination exactly as specified. | All public GLWE variants; GGSW and automorphism-key variants; independently prepared keys. |
| External product | Contract the ciphertext's digit decomposition with the prepared GGSW operand; canonicalize at destination precision. Internal DFT fill obeys the same product and layout capability. | GLWE, GGLWE and GGSW results, assign variants, zero prepared multiplier. |
| Conversion/expansion | Preserve the specified logical ciphertext or sample while changing LWE/GLWE/gadget layout; sample indices, flattened dimensions and rank constraints are observable. | Independent reference outputs and metadata; prepared conversion keys. |
| Trace/packing | Apply the normalized trace/projection: each stage shifts right by one before automorphism-add, with the reference’s fixed-point rounding at each stage. Packing embeds the selected input slots through these projections; key precision and input mutations are observable. | Reference equality with independently prepared key maps; exact scratch, partial/empty traces and sparse packing input mutations. |
| Linear transformation | Prepare the requested plaintext diagonals/baby rotations and evaluate their scheduled products. Prepared and streamed RHS modes represent the same linear map. | Resident/streamed reference results using independent caches and exact scratch for each phase. |
| BSGS phases | Select even/odd/full terms, seed at the highest required power, propagate policy errors, and combine baby steps with one prepared right operand per repeated giant-step power. | Exact integer backend-buffer policy plus portable reference; fused/fallback paths, metadata, hoisting and errors. |
| Encryption/decryption | Produce the reference ciphertext composition from the same logical plaintext, key and realized sampled values; decryption recovers the same canonical integer phase. Compressed forms retain the seed needed to regenerate their masks. | Controlled-sampling portable reference, logical ciphertext/phase and metadata equality, compressed expansion. |
| Sampling | Respect distribution support/weight, Gaussian scale/precision, untouched columns and per-backend seeded reproducibility/source advancement. Cross-backend seeded bytes are deliberately unspecified. | Distribution bounds and replay/source-state checks on each backend separately. |
| Preparation/allocation helpers | Preserve the source's relevant shape/key/distribution metadata; size queries describe actual allocated backend storage and preparation scratch is sufficient. | Direct/from-info allocation and byte counts; independently budgeted preparations; logical consumers compare against reference. |

## Enforcing the inventory

Generate metadata with the repository's pinned nightly. Hidden/private items are
included so contracts mentioned through public bounds cannot silently disappear
from the inventory merely because a module is not reexported:

```sh
cargo rustdoc -p poulpy-core --lib --features enable-core -- \
  --document-private-items --document-hidden-items -Z unstable-options --output-format json
python3 tools/check_core_contracts.py
```

`--rustdoc-json` selects the JSON under a target-specific Cargo documentation
directory when `CARGO_BUILD_TARGET` is set. The checker compares the explicit
method set to rustc's resolved traits, validates dispatch and executable
reference/test bodies, and rejects missing supported registrations. Adding a
method requires reviewing and adding its contract mapping; metadata generation
does not generate or automatically approve the manifest.

Pass captured successful libtest output to also check execution. This rejects
ignored, filtered, unregistered or failed contract tests:

```sh
set -o pipefail
cargo test -p poulpy-cpu-ref --lib --profile ci --features enable-core -- \
  core_parity core_encryption --test-threads=2 | tee /tmp/core-ref-tests.log
python3 tools/check_core_contracts.py --run-package poulpy-cpu-ref /tmp/core-ref-tests.log
```

`bash tools/verify_core_execution.sh PACKAGE LOG` regenerates metadata in the
active Cargo target directory and validates that package’s execution log.
`python3 tools/test_check_core_contracts.py` tests rejection of inventory drift,
missing registrations, wrong backend prefixes and skipped/failed tests.

Use `--run BACKEND LOG` for selected backend combinations, or repeat
`--run-package PACKAGE LOG` for a workspace log. The required matrix covers
portable FFT/NTT, AVX, AVX-512 including IFMA, NEON and their supported Rayon
variants. Native and emulator CI invocations must include core contract modules;
HAL-only filters do not establish core coverage. The native/SDE lane also runs
the AVX core groups, guaranteeing coverage if the separate AVX runner lacks
AVX2/FMA. No core Rayon implementation is
advertised by the portable CPU-reference crate. Unsupported combinations must be
recorded with a reason in the test entry rather than silently omitted.

Public documentation is validated separately, without private-item visibility:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc -p poulpy-core --no-deps --features enable-core
```
