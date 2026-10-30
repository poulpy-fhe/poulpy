# Binary-FHE operation contracts

Public `api` traits follow the [layering contract](../../../docs/guides/layering-contract.md). The `CGGI` marker
selects the blind-rotation algorithm, independently of storage and scheduling.

## Public API organization

User-facing contracts have separate operation files under
[`api/blind_rotation`](../src/api/blind_rotation.rs),
[`api/circuit_bootstrapping`](../src/api/circuit_bootstrapping.rs), and
[`api/bdd`](../src/api/bdd.rs), private modules whose traits `api` and the crate
root re-export. Execution and its matching scratch queries share a trait.
All public operation traits follow `API -> delegate -> OEP`, including
`LookupTableFactory`, whose delegate requires only its backend hook. LUT
algorithms live in independently callable reference functions; a backend opts
in with `impl_lookup_table_factory_reference_full!`, which the full
blind-rotation opt-in includes, or provides its own implementation. Key
allocation and sizing (`<object>_alloc_from_infos`,
`<object>_bytes_of_from_infos`) are direct on the module, with no OEP method:
`BlindRotationKeyCompressedFactory` is implemented by every module, and the
prepared-key factories compose the core prepared factories, each allocator
refusing a component whose degree differs from the module degree. Existing
domain import paths remain compatibility re-exports.

## Operation map

OEP contracts end in `Impl`. The table groups public API families by their
reference bodies and test coverage.
Every operation that uses caller workspace has a selected scratch query.

| API family | Reference / derived implementation | Paired coverage |
|---|---|---|
| `BlindRotationExecute`, `BlindRotationModSwitch` | [`reference/blind_rotation`](../src/reference/blind_rotation/) | Both directions, standard/block/extended domains, distributions and rounding boundaries. |
| `BlindRotationKeyEncryptSk`, `BlindRotationKeyCompressedEncryptSk`, `BlindRotationKeyDecompress`, `BlindRotationKeyPreparedFactory` | Same directory | Raw/compressed encryption, decompression, independent preparation and reused prepared keys. |
| `LookupTableFactory` | [`reference/blind_rotation`](../src/reference/blind_rotation/) | LUT correctness and independent opaque-storage dispatch overrides. |
| `BlindRotationKeyCompressedFactory` | Direct on the module, no OEP family | Compressed key lifecycle, byte counts, degree refusals and allocation on opaque storage. |
| `CircuitBootstrappingExecute` | [`reference/circuit_bootstrapping.rs`](../src/reference/circuit_bootstrapping.rs); one-shot constant/exponent wrappers are derived | Constant/exponent outputs, prepared plans, repeated execution, differing radices and invalid shapes. |
| `CircuitBootstrappingKeyEncryptSk`, `CircuitBootstrappingKeyPreparedFactory` | Same reference module | Shared raw key, independently prepared components and exact query-sized workspace. |
| `GLWECmux`, `GLWECswap` | [`reference/bdd`](../src/reference/bdd/) | Assignment variants, differing precisions, output capacity and operand preservation. |
| `GLWEBlindRotation`, `GGSWBlindRotation`, `GLWEBlindSelection`, `GLWEBlindRetrieval` | Same reference directory; same-layer wrappers in [`oep/derived/bdd`](../src/oep/derived/bdd/) | Rotation/selection variants, sparse branches and reusable retrieval state. |
| `GLWEBlindRetriever` | Stateful streaming helper composed from selected `GLWECmux`, `GLWECopy` and `GLWEZero` operations | Batch/incremental parity, borrowed selectors, empty/singleton streams, capacity limits, reuse and exact layout-dependent workspace. |
| `BDDCircuitExecute`, `BDDCircuitExecute1WTo1W`, `BDDCircuitExecute2WTo1W` | Same directories | Single/multiple outputs, serial/parallel calls and worker counts. |
| `FheUintPrepare`, `FheUintPreparedEncryptSk`, `BDDKeyEncryptSk`, `BDDKeyPreparedFactory` | Same directories | Preparation and key lifecycle, including optional switching keys. |

Layout accessors, serialization, plain container constructors, and generated
BDD descriptions remain data helpers. Key allocation and sizing are direct on
the module; key preparation and LUT construction dispatch through their OEP
contracts. Convenience evaluators route through the selected public operation
traits.

`GLWEBlindRetriever` and `GLWEBlindRetrieval` serve different purposes. The
retriever accumulates inputs without changing them and returns one selected
ciphertext, using logarithmic persistent storage. Its `add`/`flush` methods
compose the selected CMux/core operations; the container has no separate
whole-stream backend hook. `GLWEBlindRetrieval` rearranges a vector in place
through conditional swaps and provides an inverse permutation. Both paths
have caller-selected parity tests.

## Replacing an operation

An override reproduces the reference as the [layering contract](../../../docs/guides/layering-contract.md) states. Parity
compares coefficient-domain outputs,
precision, rank, radix and allocated tails exactly. Prepared representations
belong to each backend: compare their observable results, not their bytes.

`impl_bin_fhe_reference_full!` selects every family, through the group
umbrellas `impl_blind_rotation_reference_full!`,
`impl_circuit_bootstrapping_reference_full!` and `impl_bdd_reference_full!`.
Each OEP trait has its own opt-in, `impl_<family>_reference_full!` (or
`impl_<family>_derived_full!` when every method has a default), `<family>`
being the trait in snake case without `Impl`: circuit execution, key
encryption and key preparation have
`impl_circuit_bootstrapping_execute_reference_full!`,
`impl_circuit_bootstrapping_key_encrypt_sk_reference_full!` and
`impl_circuit_bootstrapping_key_prepared_factory_reference_full!`, allowing a backend
to replace execution while retaining the key lifecycle implementations.

BDD opt-ins are also available per operation contract. For example, keep the
reference CMux, blind selection, and derived retrieval while implementing a
custom GLWECswap:

```rust,ignore
poulpy_bin_fhe::impl_glwe_cmux_reference_full!(MyBackend);
poulpy_bin_fhe::impl_glwe_blind_selection_reference_full!(MyBackend);
poulpy_bin_fhe::impl_glwe_blind_retrieval_derived_full!(MyBackend);
// Implement GLWECswapImpl for MyBackend with the selected execution and scratch query.
```

Select the other required contracts through their individual macros in
[`oep/bdd.rs`](../src/oep/bdd.rs). `impl_bdd_reference_full!` composes all of
these opt-ins; omit that aggregate when providing a custom BDD implementation.
The integer-parameterized opt-ins still implement their contracts for every
supported `UnsignedInteger` type, and key/preparation opt-ins take the algorithm
marker explicitly.

The backend's executor picks the blind-rotation schedule. The reference and
its query run the products of a block in parallel exactly when
`BE::TaskExecutor::IS_PARALLEL` holds, reserving one worker scratch per product
of the block, and add their contributions in index order, so every executor and
pool width returns the same bytes. No opt-in selects a schedule.

## Workspace

Use the selected query for the exact layouts, capacities and execution
parameters. A replacement may require more workspace than the reference and
must replace its query accordingly. Compound budgets include selected
constituent queries, live intermediate storage and alignment between workers.
Out-of-place GLWE/GGSW blind-rotation queries take both source and destination
layouts; assignment uses the same query with the destination as the source. Selection takes the destination and
all candidate input infos, retaining their actual allocation capacities. Its
reference budget deduplicates equal shapes before checking the ordered
constituent combinations. Do not substitute a single maximum-width layout:
a selected implementation's workspace can depend on either operand.

Queries for reusable per-worker storage return bytes rounded by
`worker_scratch_bytes` and must be multiplied using the documented worker-count
convention; do not assume all execution variants share a budget. An arena below
an operation's query panics at entry, before any write.

Blind-selection scratch queries take the output, a slice describing every input
in the map, and the selector. Pass actual input capacities: the reduction can
combine different precisions and cannot assume a backend's workspace grows
monotonically with width. Equal layouts are deduplicated before checking pairs.

Circuit scratch queries share the selected prepared-execution query through
`CircuitBootstrappingPlanLayout`, so overriding prepared execution also changes
the budgets used by its direct constant/exponent wrappers. The legacy constant
query has no plaintext-domain parameter: `log_domain = None` requests a
conservative bound for every valid domain at that output layout. A concrete
plan supplies its domain and gaps. Querying requires only metadata, without
constructing or uploading a LUT.

Circuit key encryption includes the retained prepared secret in the BRK phase
and reuses caller scratch for all three component encryptions. No additional
scratch arena is allocated inside the operation.

For streaming retrieval, take the maximum of the retriever's `add_tmp_bytes`
queries for each actual input layout and `flush_tmp_bytes` for the destination.
These include the persistent accumulator's compact layout. The static
`retrieve_tmp_bytes` convenience query applies when input/output layouts and
capacities match and the accumulator was allocated from those same infos.

The parity harness allocates exactly the advertised capacity, initializes it
with nonzero bytes, and checks surrounding guards after execution. Heap-owned
keys, plans and temporary ciphertexts are separate from caller scratch.

## Host boundaries

Operation contracts and reference functions do not require `'static` backend,
key-provider or plaintext integer types. A key provider may borrow prepared
ciphertexts, and an `UnsignedInteger` implementation may borrow plaintext data.
Borrows must remain valid for their uses.

Execution contracts use backend-owned buffers and views without
`HostDataRef`/`HostDataMut` bounds. Fixture transfers are explicit; the tested
backend need not provide host-readable storage. LUT construction and generated
circuit descriptions are host data. Serialization, debugging and decryption
convenience helpers retain their explicit host requirements.

Modulus switching exposes `BlindRotationModSwitchImpl`: the reference downloads
LWE coefficients and emits host rotation indices, while an override can select
its staging strategy. Secret-dependent key construction uses explicit transfer
boundaries where host control data is needed. This does not impose host views
on ciphertext storage. Prepared keys provide component constructors/accessors
for independent implementations.

## Testing a replacement

Register `bin_fhe_parity_test_suite!` with caller-selected `backend_ref` and
`backend_test`. A validated backend can validate another for the same
operations and parameter range. Both sides invoke their selected public APIs;
reference-body/override regressions are separate tests.
`bin_fhe_reference_test_suite!` registers same-backend randomized lifecycle
comparisons and requires the lower-layer capabilities of those reference bodies.

Fixtures upload identical coefficients and raw key material and prepare each
side independently. Equal random seeds do not imply equal draws across
samplers: encryption reference checks use one backend's sampler on both paths,
and cross-backend execution shares realized key material. The independent
correctness suite remains necessary alongside differential checks.

Register the entire shared suite beside every supported backend opt-in, so new
suite cases automatically run for every registered backend. Runtime selection
and concrete CI commands are repository/backend concerns.

## Migration

Backend authors must explicitly implement the scheme OEP families; importing
the crate no longer installs blanket reference implementations. Operation
traits live in `api`, re-exported at the crate root and, for traits that moved
there, from their domain modules (`blind_rotation`, `circuit_bootstrapping`,
`bdd_arithmetic`); the `api` family modules are private. Generic callers
should bound the public operation they need, not its reference dependencies.
Use the selected operation's scratch query, including the prepared integer
encryption query, the blind-rotation queries, which size the assignment forms
with the destination as the source, and the source infos required by rotation
and selection queries. Backend crates expose scheme opt-in through their own
`enable-bin-fhe` feature; the scheme crate's historical feature does not select
a backend.

`BlindRotationKeyCompressedFactory<BRA, BE>` is implemented by every module
for each algorithm with a compressed key. Generic callers bound the module by
it and call `module.blind_rotation_key_compressed_alloc_from_infos(infos)`, as
the `BlindRotationKeyCompressed::alloc(module, infos)` convenience call does.
Importing the crate alone no longer installs LUT construction from a backend's
HAL arithmetic capabilities.
