# Implementing CKKS operations

Public evaluator calls follow the [layering contract](../../../docs/guides/layering-contract.md). The CKKS crate does not
select a concrete backend or require host storage in these dispatch contracts.

## Reference and derived implementations

Each family's opt-in macro is `impl_<family>_reference_full!`, or
`impl_<family>_derived_full!` when every method is a derived default, `<family>` being the
OEP trait in snake case without `Impl` (`CKKSAddImpl`: `impl_ckks_add_reference_full!`,
`DFTImpl`: `impl_dft_reference_full!`, `CKKSAffineImpl`: `impl_ckks_affine_derived_full!`).
`impl_ckks_reference_full!` selects every scalar-independent family that serves both
rings, the families `CKKSImpl` bundles minus the standard-only ones; a standard backend
adds the standard-only opt-ins,
and every backend implements the scalar-generic seams (`CKKSEncodingImpl<F>`,
`DFTMatrixImpl<F>`) and the PaCo and SHIP coefficient encodings itself. A backend
replacing one of the umbrella's families selects the others with their own opt-ins.
The table gives the reference of each family.

| OEP family | Canonical implementation |
|---|---|
| `CKKSAddImpl`, `CKKSSubImpl` | [`reference/add.rs`](../src/reference/add.rs), [`reference/sub.rs`](../src/reference/sub.rs); adding/subtracting one are derived plaintext compositions. |
| `CKKSCopyImpl`, `CKKSNegImpl`, `CKKSPow2Impl`, `CKKSImagImpl` | Corresponding [`reference`](../src/reference/) modules; multiplication and division by `i` use direct core monomial rotations by `N/2` and `-N/2`. |
| `CKKSMulImpl` | [`reference/mul.rs`](../src/reference/mul.rs), including the left and right preparations, ordinary/prepared and plaintext products. The prepared operands follow poulpy-core's lifecycle: `CKKSModuleAlloc::ckks_prepared_{left,right}_alloc_from_infos` allocate one for a ciphertext layout under a `PrepareHint` and `ckks_prepared_{left,right}_bytes_of_from_infos` give its bytes, directly on `Module` with no OEP method, and `ckks_prepare_{left,right}_into` fill it with no allocation, rejecting a source of another layout with `InvalidOperand` before any write; the allocating `ckks_prepare_{left,right}` are derived defaults, an allocation then a preparation into it. |
| `CKKSDotProductImpl` | [`reference/dot_product.rs`](../src/reference/dot_product.rs): the ciphertext and prepared dot products, relinearized once where the pairs share one product rule; the sum of prepared terms by prepared plaintexts over poulpy-core's `glwe_mul_prepared_plain_sum`, each plaintext read through an odd Galois element, summed in the transform domain and normalized once, every operand checked before `dst` is written; and their scratch. The plaintext dot products are derived defaults, a product then an addition per pair. |
| `CKKSLinearTransformationImpl` | [`reference/linear_transformation.rs`](../src/reference/linear_transformation.rs): the preparations, the evaluation over a caller's baby cache and the summed evaluation; `_assign` and the self-caching evaluations are derived defaults. The preparation and every evaluation refuse diagonals of more than one scale, width or slot kind with `InvalidOperand` before any write, and the preparation a destination allocated for another layout. |
| `CKKSRotateImpl`, `CKKSConjugateImpl` | [`reference/rotate.rs`](../src/reference/rotate.rs), [`reference/conjugate.rs`](../src/reference/conjugate.rs). |
| `CKKSInnerSumImpl` | Derived defaults only, selected by `impl_ckks_inner_sum_derived_full!`: `ckks_copy`, poulpy-core's summed trace over a window of levels, forward for the inner sum and reversed for the replicate, and, for a mean, `ckks_div_pow2_assign`; the Galois elements are poulpy-core's `glwe_trace_galois_elements` over those levels at the ciphertext's degree. |
| `CKKSPlaintextZnxImpl` | [`reference/plaintext.rs`](../src/reference/plaintext.rs). |
| `CKKSEncryptionImpl` | [`reference/encryption.rs`](../src/reference/encryption.rs), for encryption and decryption. |
| `CKKSEncodingImpl<F>` | [`reference/encoding.rs`](../src/reference/encoding.rs): slot ordering, transform normalization, and quantization behind the four required kernels `ckks_{quantize,dequantize}_coeffs_into` and `ckks_{slots_to_coeffs,coeffs_to_slots}_assign_with_plan`; backend-owned plans and cache. The six API operations are derived defaults that check the buffer and plaintext shapes and read the plans from the cache. |
| `DFTMatrixImpl<F>`, `DFTImpl` | [`reference/dft`](../src/reference/dft/): matrix generation, preparation, and evaluation; the six standard/split/repack direction wrappers are derived defaults. A prepared matrix is allocated by `CKKSModuleAlloc::ckks_dft_matrix_prepared_alloc_from_infos` from the unprepared one and filled by `ckks_prepare_dft_matrix` with no allocation, which refuses a matrix of another layout with `InvalidOperand` before any write; the allocating `ckks_prepare_dft_matrix_alloc` is a derived default. |
| `CKKSPolynomialEvaluationImpl`, `CKKSComplexPolynomialEvaluationImpl` | [`reference/polynomial_evaluation.rs`](../src/reference/polynomial_evaluation.rs) for evaluation from a power basis; one-shot real/complex evaluation derives from CKKS input mapping and power-basis construction. The complex family is standard-only. |
| `CKKSEvalModImpl` | [`reference/eval_mod.rs`](../src/reference/eval_mod.rs), including the selected scratch query. Serves both rings; the conjugate-invariant ring refuses `ExpCmplx`. |
| `CKKSEncapsulatedModUpImpl` | [`reference/bootstrapping.rs`](../src/reference/bootstrapping.rs). |
| `CKKSBootstrappingImpl` | [`reference/bootstrapping.rs`](../src/reference/bootstrapping.rs): the ModUp stages and the identity and functional pipelines. |
| `CKKSBootstrappingCIImpl` | [`reference/bootstrapping.rs`](../src/reference/bootstrapping.rs): `ckks_bootstrap_ci_reference`, registered by `impl_ckks_bootstrapping_ci_reference_full!($be, $ci)`, `$ci` its conjugate-invariant twin. |
| `CKKSFoldLayoutImpl`, `CKKSFoldImpl<R>` | [`reference/fold.rs`](../src/reference/fold.rs): `CKKSFoldLayoutReference` and `CKKSFoldReference<BE, R>`, a crate-private per-ring step around ring packing; `impl_ckks_fold_layout_reference_full!` selects the layout and scratch query, and `impl_ckks_fold_reference_full!` registers every input ring the reference folds. |
| `CKKSPaCoCoeffEncodingImpl`, `CKKSShipCoeffEncodingImpl` | Scheme embeddings exposed by [`reference::encoding`](../src/reference/encoding.rs), with explicitly named host reference helpers. |
| `CKKSAddManyImpl`, `CKKSMulAddImpl`, `CKKSAffineImpl`, `CKKSMulSubImpl` | Derived defaults ([`oep/derived/composite.rs`](../src/oep/derived/composite.rs)): sums, multiply-accumulate and affine forms over the selected products, additions and subtractions. |
| `CKKSApproximationImpl` | Derived defaults ([`oep/derived/approximation.rs`](../src/oep/derived/approximation.rs)): the interval map through the selected power-of-two, copy, addition or affine operation, then the selected real polynomial evaluation; sized by `ckks_all_ops_tmp_bytes`. |
| `CKKSRingMapImpl` | Standard-only derived defaults ([`oep/derived/ring_map.rs`](../src/oep/derived/ring_map.rs)): the core conjugate-invariant embedding and trace composed with the selected key switch. |
| `CKKSPaCoImpl` | Standard-only derived defaults ([`oep/derived/paco`](../src/oep/derived/paco/)): the sequential and parallel PaCo drivers, their scratch queries and the slot folds `ckks_slot_{trace,product}_assign` with their queries, over the selected CKKS and core operations and `CKKSPaCoCoeffEncodingImpl`. |
| `CKKSShipImpl` | Standard-only: the SHIP drivers and their query are derived defaults ([`oep/derived/ship`](../src/oep/derived/ship/)); the masking accumulation and the hoisted mux rotation are required hooks with references in [`reference/ship`](../src/reference/ship/) (`CKKSShipReference`), registered by `impl_ckks_ship_reference_full!`. |

`CKKSImpl` aggregates capabilities; it does not choose operation
implementations. Every API method dispatches through a same-named OEP method;
compositions of selected operations are that method's derived default.

## What an override must preserve

An override reproduces the reference as the [layering contract](../../../docs/guides/layering-contract.md) states. CKKS
metadata is observable: `k`, radix, rank, `log_delta`,
`log_sparsity`, and `SlotsKind` must agree, and `log_budget` remains derived from
`k - log_delta`. Preserve the core canonical flag and
canonical-at-`k` representation. Prepared objects and encoding plans belong
to their backend; compare their operation results rather than opaque storage. A
preparation's `PrepareHint` may select its representation, never the bits of the
products that read it.

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
`CKKSEvalModImpl::ckks_eval_mod_tmp_bytes`, so replacing its evaluation and
workspace policy does not pin execution to the reference budget.

DFT matrix generation uses `ckks_new_dft_matrix_tmp_bytes`, preparation
`ckks_prepare_dft_matrix_tmp_bytes`; raw evaluation and format wrappers use
`ckks_dft_tmp_bytes`. Bootstrap sizing includes the selected
DFT queries. Their reference budgets include selected Core and CKKS operations.
Bootstrap compilation is staged for the same reason: the preparation budget depends on the generated matrices, so `BootstrappingContext::compile_unprepared` returns them unprepared, `compile_unprepared_tmp_bytes` reports the budget of the first stage, `prepare_tmp_bytes` the selected budget of the second, and `prepare` produces the context.
`compile` runs both stages on one arena, which must then also cover both budgets.
Polynomial evaluation retains the shared `CKKSAllOpsTmpBytes` bounds. PaCo and
SHIP coefficient-encoding hooks report their own workspace requirements and refuse
a smaller arena themselves.

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
appropriate. Diagonals are embedded into the common ring for comparison,
so different minimum allocation degrees do not require identical storage. The tested buffers remain opaque; the harness uses explicit
transfers instead of `HostDataRef`/`HostDataMut` bounds.

Encryption parity needs identical realized random draws. Matching seeds alone
do not guarantee this when samplers differ. Supply matching streams or a
controlled-sampling comparison adapter; retain the separate sampling contract
tests and independent mathematical conformance suite.

Register `ckks_determinism_test_suite!(mod <name>, backend = <type>)` for each
standard-ring type. It checks the encoding and the setup constants, in `f64` and
`Quad`, against the frozen fixtures of
[`test_suite/determinism.txt`](../src/test_suite/determinism.txt), which every
backend reproduces byte for byte.

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
[repository README](../../../README.md) and backend test modules.

