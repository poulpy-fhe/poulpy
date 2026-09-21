# Implementing CKKS operations

Public evaluator calls follow `api → delegates → oep`. Implement the relevant
`*Impl` trait on the backend type to select its behavior. The CKKS crate does not
select a concrete backend or require host storage in these dispatch contracts.

## Reference and derived implementations

A **reference** implementation defines a CKKS operation by composing core or HAL
operations. These routines are public so an override can reuse them. A
**derived** implementation is a simple composition of CKKS operations at the
same layer. Its helper is `pub(crate)` and is reached through the default method
on an explicit OEP contract.

Use `impl_ckks_*_reference!` to install a reference family. To replace one of its
methods, write that family's `*Impl` implementation directly and forward the
unchanged required methods to their public reference routines. The other
families retain their own wiring. Derived defaults call the selected constituent
operations, so their overrides remain effective.

| OEP family | Canonical implementation |
|---|---|
| `CKKSAddImpl`, `CKKSSubImpl` | [`reference/add.rs`](../src/reference/add.rs), [`reference/sub.rs`](../src/reference/sub.rs); adding/subtracting one are derived plaintext compositions. |
| `CKKSCopyImpl`, `CKKSNegImpl`, `CKKSPow2Impl`, `CKKSImagImpl` | Corresponding [`reference`](../src/reference/) modules; multiplication and division by `i` use direct core monomial rotations by `N/2` and `-N/2`. |
| `CKKSMulImpl` | [`reference/mul.rs`](../src/reference/mul.rs), including ordinary/prepared and plaintext products. |
| `CKKSRotateImpl`, `CKKSConjugateImpl` | [`reference/rotate.rs`](../src/reference/rotate.rs), [`reference/conjugate.rs`](../src/reference/conjugate.rs). |
| `CKKSPlaintextZnxImpl` | [`reference/plaintext.rs`](../src/reference/plaintext.rs). |
| `CKKSEncryptionImpl` | [`reference/encryption.rs`](../src/reference/encryption.rs), for encryption and decryption. |
| `CKKSEncodingImpl<F>` | [`reference/encoding.rs`](../src/reference/encoding.rs): slot ordering, transform normalization, and quantization; backend-owned plans and cache. |
| `DFTMatrixImpl<F>`, `DFTImpl` | [`reference/dft`](../src/reference/dft/): matrix generation, preparation, and evaluation; the six standard/split/repack direction wrappers are derived defaults. |
| `CKKSPolynomialEvaluationImpl` | [`reference/polynomial_evaluation.rs`](../src/reference/polynomial_evaluation.rs) for evaluation from a power basis; one-shot real/complex evaluation derives from CKKS input mapping and power-basis construction. |
| `CKKSEvalModImpl` | [`reference/eval_mod.rs`](../src/reference/eval_mod.rs), including the selected scratch query. |
| `CKKSEncapsulatedModUpImpl` | [`reference/bootstrapping.rs`](../src/reference/bootstrapping.rs). |
| `CKKSPaCoCoeffEncodingImpl`, `CKKSShipCoeffEncodingImpl` | Scheme embeddings exposed by [`reference::encoding`](../src/reference/encoding.rs), with explicitly named host reference helpers. |

`CKKSImpl` aggregates capabilities; it does not choose operation
implementations. Affine operations, dot products, linear transformations, and
complete bootstrap pipelines are API compositions with integration tests, not
additional OEP families.

## What an override must preserve

Compute the same circuit as the reference and validate it with paired tests.
The reference and API contracts define the result, metadata, supported shapes,
normalization, and failure/mutation behavior; a second algorithm description is
unnecessary.

In particular, CKKS metadata is observable: `k`, radix, rank, `log_delta`,
`log_sparsity`, and `SlotsKind` must agree, and `log_budget` remains derived from
`k - log_delta`. Preserve the normalized/unnormalized typestate boundary and the
core canonical-at-`k` representation. Prepared objects and encoding plans belong
to their backend; compare their operation results rather than opaque storage.

The resident encoding contracts take backend scalar buffers. Host reference
helpers carry their own host-access requirements; a native override does not
inherit those bounds. Quantization rejects non-finite or out-of-range values
before writing the plaintext, including when the scalar's integer casts would
otherwise saturate.

## Scratch

Use the selected implementation's scratch query for the same layouts and
parameters passed to execution. In particular, `ckks_copy_tmp_bytes(dst, src)`
receives both layouts because narrowing, radix conversion, and intermediate
storage depend on their relationship. Likewise, `ckks_decrypt_tmp_bytes(pt, ct)`
includes the plaintext allocation width even when its effective precision is
lower. Add/subtract-one calls use `ckks_add_one_tmp_bytes` and
`ckks_sub_one_tmp_bytes`; their derived queries follow the selected
plaintext-constant add/subtract implementation. An override may report a
different size from the reference. EvalMod's query dispatches through
`CKKSEvalModImpl::ckks_eval_mod_tmp_bytes_impl`, so replacing its evaluation and
workspace policy does not pin execution to the reference budget.

DFT and polynomial evaluation retain the shared `CKKSAllOpsTmpBytes` workflow
bounds. They do not expose separate per-method scratch APIs. Reference and
derived bodies must account for the selected constituent operations within
those bounds. PaCo and SHIP coefficient-encoding hooks report their own
workspace requirements.

The paired harness gives each implementation its own advertised scratch
capacity, poisons it, and checks surrounding guards through explicit transfers.
Input preservation and failure behavior are checked separately from numerical
parity.

## Testing a replacement

Enable `test-utils` and register the relevant helpers from
[`test_suite::parity`](../src/test_suite/parity/) with
`ckks_parity_test_suite!`. Both the comparison backend and tested backend, the
scalar, and the test parameters are caller-selected. A previously validated
implementation can validate another: this is transitive for the same operations
and supported parameter ranges. No particular backend is a mandatory oracle.

The paired helpers upload identical logical fixtures and prepare keys, plans,
and transformed objects independently. They compare canonical coefficients and
metadata, using numerical tolerances for encoding/transform rounding where
appropriate. Compact diagonals are embedded into the common ring for comparison,
so different minimum allocation degrees do not require identical storage. The tested buffers remain opaque; the harness uses explicit
transfers instead of `HostDataRef`/`HostDataMut` bounds.

Encryption parity needs identical realized random draws. Matching seeds alone
do not guarantee this when samplers differ. Supply matching streams or a
controlled-sampling comparison adapter; retain the separate sampling contract
tests and independent mathematical conformance suite.

Register each supported operation/scalar combination in its backend crate.
Encoding and DFT matrix generation accept any scalar satisfying their generic
bounds when the backend implements that precision; paired coverage includes
`f32` where supported. PaCo and SHIP coefficient encoding instead require the
sealed `PaCoScalar` and `ShipScalar` traits, which permit only `f64` and `Quad`.
Their residue modulus must also fit exactly below the scalar's mantissa limit.

Existing paired cases exercise arithmetic and plaintext variants, encoding
plans/codecs, DFT formats, polynomial/EvalMod evaluation, ModUp branches,
PaCo/SHIP embeddings, metadata, and mutation boundaries. The independent suite
also tests complete pipelines. Passing these cases establishes evidence for the
covered parameters, not a proof for every possible circuit or layout. Concrete
registrations, runtime selection, and commands belong in the
[repository README](../../README.md) and backend test modules.
