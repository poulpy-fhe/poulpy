# Binary-FHE operation contracts

Public `api` traits delegate to backend `oep::*Impl` contracts. A backend
explicitly selects a reference family or implements a replacement. Reference
algorithms compose lower-layer core/HAL operations; simple same-layer
compositions are crate-private defaults in `oep::derived`. The `CGGI` marker
selects the blind-rotation algorithm, independently of storage and scheduling.

## Public API organization

User-facing contracts have separate operation files under
[`api/blind_rotation`](../src/api/blind_rotation.rs),
[`api/circuit_bootstrapping`](../src/api/circuit_bootstrapping.rs), and
[`api/bdd`](../src/api/bdd.rs). Each family re-exports its traits, as do `api`
and the crate root. Execution and its matching scratch queries share a trait.
All public operation traits follow `API -> delegate -> OEP`, including
`LookupTableFactory` and `BlindRotationKeyCompressedFactory`. Their delegates
require only the corresponding backend hook. LUT algorithms and compressed-key
allocation live in independently callable reference functions; a backend opts
in with `impl_bin_fhe_lookup_table_reference!` and
`impl_bin_fhe_blind_rotation_key_compressed_factory_reference!`, or provides its
own implementation. The full blind-rotation opt-in includes both hooks.
Existing domain import paths remain compatibility re-exports.

## Operation map

OEP contracts end in `Impl`. The table groups public API families by their
reference bodies and test coverage.
Every operation that uses caller workspace has a selected scratch query.

| API family | Reference / derived implementation | Paired coverage |
|---|---|---|
| `BlindRotationExecute`, `BlindRotationModSwitch` | [`reference/blind_rotation`](../src/reference/blind_rotation/) | Both directions, standard/block/extended domains, distributions and rounding boundaries. |
| `BlindRotationKeyEncryptSk`, `BlindRotationKeyCompressedEncryptSk`, `BlindRotationKeyDecompress`, `BlindRotationKeyPreparedFactory` | Same directory | Raw/compressed encryption, decompression, independent preparation and reused prepared keys. |
| `LookupTableFactory`, `BlindRotationKeyCompressedFactory` | [`reference/blind_rotation`](../src/reference/blind_rotation/) | LUT correctness, compressed key lifecycle, and independent opaque-storage dispatch overrides. |
| `CircuitBootstrappingExecute` | [`reference/circuit_bootstrapping.rs`](../src/reference/circuit_bootstrapping.rs); one-shot constant/exponent wrappers are derived | Constant/exponent outputs, prepared plans, repeated execution, differing radices and invalid shapes. |
| `CircuitBootstrappingKeyEncryptSk`, `CircuitBootstrappingKeyPreparedFactory` | Same reference module | Shared raw key, independently prepared components and exact query-sized workspace. |
| `Cmux`, `Cswap` | [`reference/bdd`](../src/reference/bdd/) | Assignment variants, differing precisions, output capacity and operand preservation. |
| `GLWEBlindRotation`, `GGSWBlindRotation`, `GLWEBlindSelection`, `GLWEBlindRetrieval` | Same reference directory; same-layer wrappers in [`oep/derived/bdd`](../src/oep/derived/bdd/) | Rotation/selection variants, sparse branches and reusable retrieval state. |
| `ExecuteBDDCircuit`, `ExecuteBDDCircuit1WTo1W`, `ExecuteBDDCircuit2WTo1W` | Same directories | Single/multiple outputs, serial/parallel calls and worker counts. |
| `FheUintPrepare`, `FheUintPreparedEncryptSk`, `BDDKeyEncryptSk`, `BDDKeyPreparedFactory` | Same directories | Preparation and key lifecycle, including optional switching keys. |

Layout accessors, serialization, plain container constructors, and generated
BDD descriptions remain data helpers. Prepared and compressed key factories
and LUT construction dispatch through their OEP contracts. Convenience
evaluators route through the selected public operation traits.

## Replacing an operation

An override must compute the same circuit as the reference and pass the parity
tests. The source reference defines the result and mutation semantics; this
guide does not duplicate each algorithm. Compare coefficient-domain outputs,
precision, rank, radix and allocated tails exactly. Prepared representations
belong to each backend: compare their observable results, not their bytes.

Select the reference families needed by the backend, then implement the
remaining `*Impl` traits. `impl_bin_fhe_reference_full!` is a convenience opt-in
for all families; omit it when replacing one of those families. The reference
functions remain directly callable independently of the public dispatch path.
Derived defaults dispatch to the selected same-layer operations and queries.
Circuit execution, key encryption and key preparation also have separate
`impl_bin_fhe_circuit_bootstrapping_*_reference!` opt-ins, allowing a backend to
replace execution while retaining the key lifecycle implementations.

The ordinary blind-rotation reference has a fixed schedule. Explicit
`scheduling = parallel` opt-in selects the reusable parallel implementation;
the canonical circuit does not inspect an executor flag to choose a different
implementation.

## Workspace

Use the selected query for the exact layouts, capacities and execution
parameters. A replacement may require more workspace than the reference and
must replace its query accordingly. Compound budgets include selected
constituent queries, live intermediate storage and alignment between workers.
Out-of-place GLWE/GGSW blind-rotation queries take both source and destination
layouts; assignment has a separate query. Selection takes the destination and
all candidate input infos, retaining their actual allocation capacities. Its
reference budget deduplicates equal shapes before checking the ordered
constituent combinations. Do not substitute a single maximum-width layout:
a selected implementation's workspace can depend on either operand.

Queries for reusable per-worker storage must be multiplied using the documented
worker-count convention; do not assume all execution variants share a budget.

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

The parity harness allocates exactly the advertised capacity, initializes it
with nonzero bytes, and checks surrounding guards after execution. Heap-owned
keys, plans and temporary ciphertexts are separate from caller scratch.

## Host boundaries

Operation contracts and reference functions do not require `'static` backend
or key-provider types. A key provider may borrow prepared ciphertexts; those
borrows only need to remain valid for the operation.

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
the crate no longer installs blanket reference implementations. Existing
public trait paths are re-exported from their former modules. Generic callers
should bound the public operation they need, not its reference dependencies.
Use the selected operation's scratch query, including the prepared integer
encryption query, separate blind-rotation assignment queries, and the source
infos required by rotation and selection queries. Backend crates expose scheme opt-in through their own
`enable-bin-fhe` feature; the scheme crate's historical feature does not select
a backend.

`BlindRotationKeyCompressedFactory<BRA, BE>` is now a module operation with an
`&self` receiver. Generic callers use a bound on the module and call
`module.blind_rotation_key_compressed_alloc(infos)`; the existing
`BlindRotationKeyCompressed::alloc(module, infos)` convenience call also uses
that selected factory. Importing the crate alone no longer installs LUT
construction from a backend's HAL arithmetic capabilities.
