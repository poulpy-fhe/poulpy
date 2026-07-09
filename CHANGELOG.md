# CHANGELOG

## [Unreleased]

This release adds **PaCo** (Coron & Seuré, [ePrint 2025/886](https://eprint.iacr.org/2025/886)) — bootstrapping for CKKS via **Pa**rtial **Co**effToSlot — as a native, backend-generic CKKS operation with a bounded multi-threaded evaluator; moves CKKS encoding off the host-side codec onto a backend-resident op family backed by a new HAL plan cache and multi-ring FFT/NTT plan sets; and lands a full production-readiness pass over `poulpy-ckks` (typed errors, constructor-validated plans, scratch-carved intermediates, binary128-exact table generation, and a broad consolidation of the four-layer api/oep/delegates/default wiring).

### `poulpy-hal`
- Add `ModulePlanCache` — a per-`Module`, heterogeneous, typed cache of immutable plan families (keyed by a logical ZST type via `TypeId`) with a closure-based `with_or_create` accessor that initializes each entry exactly once under concurrency — plus the `unsafe trait ModulePlanCacheProvider` implemented by backend handles that own the cache, keeping plan location and destruction order a backend concern (important for device-resource plans). This generalizes the FFT-table cache previously private to `poulpy-cpu-ref` into a neutral HAL ownership contract.
- **Breaking:** `Backend` gains two required methods, `copy_view_to_host` and `copy_host_to_view`, for byte transfers between arena views (not owned buffers) and host slices; device backends implement them as device↔host copies. External `Backend` implementations must add both.
- Re-document `Module` as a multi-ring execution context: `n()` is now specified as the **maximum** ring degree and the new alias `max_n()` makes dimension-aware call sites explicit, so one module can serve every power-of-two sub-dimension (used by the new encoding plan sets).

### `poulpy-core`
- Add `GLWESecret::fill_binary_coeffs(col, coeffs)`, installing a caller-provided binary `{0,1}` coefficient vector into one secret column and tagging the distribution `BinaryFixed(hamming_weight)` — the entry point for structured secrets such as the PaCo bootstrapping key, complementing the existing random `fill_binary_*` samplers.
- Add `GLWEPlaintextReborrowBackendRef` / `GLWEPlaintextReborrowBackendMut` for reborrowing a scratch/view-backed plaintext as a shared or mutable backend view; the existing `GLWEToBackendRef`/`GLWEToBackendMut` impls now delegate to them (needed by the backend-resident encoding paths).
- Export the `view_wrapper!` and `impl_glwe_infos!` macros so downstream crates can generate nominal backend-view wrapper types with forwarded `LWEInfos`/`GLWEInfos` (used by `poulpy-ckks` for its plaintext and encoding-buffer views).

### `poulpy-ckks`: PaCo bootstrapping
- Add PaCo as a native CKKS operation: it refreshes selected polynomial coefficients **without ModUp or EvalMod**, using a structured low-weight secret to rewrite decryption as four encrypted packing vectors and evaluating modular addition through multiplication on the unit circle (`ψ(a) = exp(2πi·a/q)`). The pipeline is blind rotation (one slot-wise ct product covering the `h·k` small-ring products) → partial CoeffsToSlots (a truncated DFT stopping after `log 2C` butterfly levels) → product fold (`Pr_{n→2C}`, `log h` ct×ct levels — the EvalMod substitute) → SlotToCoeff′.
- Evaluation is exposed through `api::CKKSPaCoOps` on `Module<BE>`, all outputs caller-allocated: `ckks_paco_bootstrap_direct_into` (sequential, input already under the structured PaCo secret), `ckks_paco_bootstrap_into` (one dense→PaCo key switch first), the bounded-parallel `ckks_paco_bootstrap_parallel_direct_into` / `ckks_paco_bootstrap_parallel_into`, the scratch-sizing queries `ckks_paco_bootstrap[_direct]_tmp_bytes`, and `ckks_paco_coeff_encodings` (+ `_tmp_bytes`), which builds the four input-dependent β plaintexts. Each bootstrap takes `kappa` (a power of two): `kappa = 1` is seqPaCo recovering `C` coefficient classes, larger values recover `kappa·C` classes. Every method validates degree, rank, radix, modulus, sparse metadata, output capacity, key layouts, required Galois elements, and scale/budget arithmetic before mutating anything.
- Invariant-bearing data lives in `layouts`: `PaCoPlan` (dimension-only `new(log_n, h, c, log_q)` plus `with_evaluation(log_delta_bsk, log_beta_budget, c2s, stc)`; memoized `galois_elements()`, `consumed_bits`, `k_boot`) and the per-chain `PaCoDFTPlan` (explicit or `uniform` factor schedules over `FactorSchedule`, including the fused `ψ` and `pack` pseudo-layers); `PaCoContext::compile` lowers a plan into the backend-resident plaintext linear transformations (no encrypted key material); `PaCoSecretSpec` samples and validates the structured secret (redacting `Debug`) and packs the four σ vectors; `PaCoKeySet` (validated unprepared bundle: four bootstrapping keys, rotation keys, tensor key, optional dense→PaCo encapsulation key) with `prepare`/`into_prepare` into the eager `PaCoKeysPrepared`; the `PaCoKeys<BE>` access trait abstracts eager, lazy, or streamed key stores; `PaCoKeyParameters` fingerprints the key-defining `(N, h, C, q, Δ_bsk)` independently of any evaluation schedule, so persisted keys are reusable across schedules.
- Parallel evaluation uses the caller thread plus a borrowed slice of reusable `PaCoWorker` contexts (each owning its own `Module` handle and scratch arena): at most `1 + workers.len()` branches run concurrently under `std::thread::scope` with bounded per-worker channels (O(workers) peak branch storage), branches recombine in exact sequential order, every worker is validated (degree, cyclotomic order, per-worker scratch bound) before any output mutation, and worker panics surface as `CKKSError::Internal`. An empty worker slice degenerates to the sequential path. Preflight validation and conservative scratch sizing (`BranchScratchLayout`) are factored apart from the concurrency code.
- The only PaCo-specific backend hook is coefficient encoding — the input-dependent conversion of public ciphertext residues into the four β plaintexts — via the `oep::CKKSPaCoCoeffEncodingImpl<BE>` extension point; everything else composes existing CKKS multiplication, automorphism, trace/fold, linear-transformation, allocation, transfer, and metadata APIs. The scheme-defining host reference `encoding::paco_coeff_encodings_host` is public, and `poulpy-cpu-ref` exports `impl_ckks_paco_coeff_encoding!` wiring it for a backend; all nine CPU backends (FFT64/NTT4x30 across ref/avx/avx512/arm, plus NTT3x42Ifma) opt in, while a device backend can implement the OEP directly with a fused kernel.
- `PaCoSlotOrder` (`Natural`, default, or `BitRevLow`) selects the mid-pipeline slot-layout convention: identical budget, levels, and recovered coefficients, but different BSGS diagonal offsets and hence a different Galois key set (up to ~95 fewer automorphism keys and 156 fewer plaintext multiplies at `C = 128`), so persisted key bundles must pin both the schedule and the slot order.
- Scalar precision is sealed behind `api::PaCoScalar` (`f64`, 53-bit mantissa, and `Quad`, 113-bit); context compilation rejects plans with `log_q ≥ F::MANTISSA_BITS`, and the implementation reuses Poulpy's official generator-5 DFT embedding in natural slot order rather than the paper's private factor generators — the divergence, and why packed vectors are validated by structural identities instead of bit-comparison against the reference implementation, is specified in `docs/spec/paco_dft_convention.md`.
- Backend-generic conformance tests cover plan rejection, structured-secret packing, coefficient encoding, the individual trace/product folds, both partial linear transforms, direct and encapsulated bootstraps, ordered parallel recombination, and output scale/budget/sparsity — including a paper-scale run and an independent cleartext oracle (`seq_paco_reference`) the homomorphic pipeline is checked against.
- Security note (documented in `docs/paco.md`): PaCo assumes a non-standard structured sparse secret (weight-`h` binary with one nonzero per residue class mod `h`); sparse-secret encapsulation keeps the application key dense but does not remove the structured-key assumption on the bootstrap key material.

### `poulpy-ckks`: backend-resident encoding
- **Breaking:** replace the host-side `Encoder<T>` (deleted, along with its `encoding::reim` module) with the backend-resident op family `api::CKKSEncodingOps<BE>` on `Module<BE>`: `ckks_encode_coeffs_into` / `ckks_decode_coeffs_into` (quantize/dequantize, no transform) and `ckks_slots_to_coeffs_assign` / `ckks_coeffs_to_slots_assign` (in-place planar `[re|im]` FFT/permutation), plus provided compositions `ckks_encode_slots_assign_into` / `ckks_decode_slots_into`. Encoding is now dispatched through the module so device backends can compose a native FFT with the plaintext codec without a device↔host round trip.
- Add `CKKSEncodingHostOps` (blanket over `CKKSEncodingOps`) with the host-slice adapters `ckks_encode_reim_into` / `ckks_decode_reim_into` and `ckks_encode_coeffs_host_into` / `ckks_decode_coeffs_host_into` (staged through the standard scratch arena, sized by `ckks_reim_tmp_bytes`), and the backend seam `oep::CKKSEncodingImpl<BE, F>` — scalar-generic over the new `api::CKKSEncodingScalar` (`CKKSScalar + FloatConst + Pod + Send + Sync`), with an opaque per-precision `Plans` family resolved through the HAL plan cache so `f64` and `Quad` encoders coexist on one module.
- Add the `CKKSEncodingBuffer<D, F>` layout family — a backend-resident planar `[re…|im…]` scalar workspace (deliberately not a host slice, so device memory works) with backend ref/mut views, scratch-arena carving (`take_ckks_encoding_buffer_scratch`), and host transfer helpers.
- **Breaking:** `CKKSPlaintextVecHostCodec` drops the separate sparse codec (`encode_host_floats_sparse` / `decode_host_floats_sparse`); `encode_host_floats` / `decode_host_floats` are now the unified stride-aware path, with sparsity realized through the coefficient gap carried by `CKKSMeta::log_sparsity`.
- **Breaking:** `EvalMod::from_literal` (host-backend-only) is replaced by the free generic `compile_eval_mod` / `compile_eval_mod_exp`, which encode the approximation coefficients directly on any destination backend through the new encoding ops and take a scratch arena.

### `poulpy-ckks`: hardening, typed errors, and plan validation
- **Breaking:** every fallible op-trait method across all four layers now returns `CKKSResult<T>` (= `Result<T, CKKSError>`) instead of `anyhow::Result<T>`. `CKKSError` is `#[non_exhaustive]` with `Composition(CKKSCompositionError)` and `Internal(anyhow::Error)` variants, a `composition()` accessor for recoverable conditions, and a downcasting `From<anyhow::Error>` bridge; `CKKSCompositionError` is now `#[non_exhaustive]` and gains `InvalidPlan` and `PreparedOperandLayoutMismatch`. Callers using `?` into `anyhow` keep working; explicit `anyhow::Result` signatures must be updated.
- **Breaking:** naming sweep on the api layer: `CKKSEncrypt` → `CKKSEncryptOps`, `CKKSDecrypt` → `CKKSDecryptOps`, `DFTOps` → `CKKSDFTOps`, aligning every op family on the `CKKS*Ops` convention.
- **Breaking:** `ckks_encrypt_sk` (and `BootstrappingContext::generate_keys`) now take the randomness sources in the order `(source_xe, source_xa)` — a positional swap of the two `&mut Source` parameters; call sites must be updated by hand since the types are identical.
- **Breaking:** `ckks_decrypt` now decrypts into the **destination plaintext's preset** `(log_delta, log_budget)` frame instead of stamping the ciphertext's metadata onto it, so a caller can extract at a different precision; preset `pt.set_meta(ct.meta())` for the previous behavior. Frame mismatches surface as typed `PlaintextAlignmentImpossible` / `PlaintextBase2KMismatch` errors.
- **Breaking:** `DFTPlan` is now constructor-validated: fields are private, plans are built with `DFTPlan::new(kind, schedule, format, meta)` + `with_scaling` (rejects non-finite/non-positive values) + `with_bit_reversed`, the per-factor schedule is the new `FactorSchedule` of `FactorStep { depth, giant_step }` (making the parallel-array length invariant unrepresentable), the `meta` field is renamed `coeffs_meta` and carries the new lightweight `CoeffsMeta` (`k` + `CKKSMeta`) instead of a full `CKKSLayout`, `DFTPlan::check` is removed, and the schedule-introspection queries (`galois_elements`, `diagonal_indexes`, `num_diagonals`) are infallible instead of panicking on malformed plans.
- **Breaking:** `BootstrappingPlan` is constructor-validated (stage directions checked at `new`, power-of-two `f_mod_interval` enforced at plan/context compile — configurations that previously produced silent precision garbage now error at setup) and drops its `ephemeral_secret_weight` parameter: `EncapsulationKeysLayout` is the single source of truth for sparse-secret encapsulation, and the pipeline toggles on key presence.
- Bootstrapping, eval_mod, and the linear-transformation wrappers now carve every working ciphertext from scratch instead of heap-allocating (up to seven owned ciphertexts per bootstrap previously); the new `ckks_bootstrap_tmp_bytes` sizing entry point is wired through all four layers and charges the full pipeline, and `ckks_eval_mod_tmp_bytes` charges its working copy. `ensure_uniform_diagonal_scale` rejects hand-built linear transformations with heterogeneous diagonal scales.
- Generalize DFT-matrix and eval-mod table generation over the scalar via the new `DftScalar` trait: per-factor scale roots use `nth_root_scalar` (f64 seed + Newton–Raphson refinement in `F`), the 256-bit CosHK solve lands through the mantissa-exact `fbig_to_scalar` triple-double decomposition, and `approximate_cos` is scalar-generic — so `Quad` matrices are binary128-exact instead of f64-rounded. `Quad` is now `bytemuck::Pod` and the `libquadmath` feature routes only its transcendental methods (exact ops stay on the primitive), with new full-mantissa parity tests.
- **Breaking:** the `crate::leveled::*` backwards-compatibility re-export shim is removed; use the canonical `api::` / `oep::` / `delegates::` / `default::` / `layouts::` paths.
- **Breaking:** the backend-conformance `test_suite` module is now gated behind the new `test-utils` feature (backends enable it from dev-dependencies; production dependents no longer compile ~17k lines of test code), and the dead `enable-ckks` / `enable-avx` / `enable-neon` feature flags are removed from `poulpy-ckks` itself (backend crates keep their own opt-in features). Dependencies: `rand` / `rand_distr` dropped, `num-traits` and `paste` added.
- The bootstrapping conformance test now asserts a measured precision floor (`MIN_AVG_LOG2_PREC = 24.0` bits, ~4 bits under the observed 27.5–28.3) on every configuration, and the retired host reim encoder survives only as the test-suite reference oracle (`test_suite/reference_encoder.rs`).

### `poulpy-ckks`: four-layer consolidation
- `CKKSInfos` is now a supertrait of `LWEInfos` with provided `log_delta` / `log_budget` / `log_sparsity` derived from `meta()` and `k()` — implementors supply only `fn meta()`; ~140 `LWEInfos + CKKSInfos` bound pairs collapse to `CKKSInfos`. New bound-alias traits `CKKSCtBounds` (ct operands) and `CKKSAtkBounds<BE>` (automorphism-key operands) replace the spelled-out multi-trait clusters across api/oep signatures.
- The 16-method add and sub Default/OEP families are generated from one shared body by the new `ckks_carry_verb_default!` / `ckks_carry_verb_oep!` macros; the five tensor-multiply variants share the `tensor_mul_core` driver; the eight fused multiply-then-accumulate composite bodies share `mul_then_combine`; and the `impl_ckks_infos!` macro generates the full metadata/view impl bundle for ciphertext and plaintext views. Net effect: one place to fix per algorithm family, with byte-identical behavior.
- The DFT family joins Pattern A: `DFTDefault` / `DFTMatrixDefault` methods carry default bodies (per-method bounds + `Self: Borrow<Module<BE>>`) forwarding to the reference implementations, and `impl_ckks_dft_defaults!` shrinks from 221 lines of copied signatures to two one-line marker impls — no call-site change for backends, which now also get single-method partial overrides for free. `ckks_new_dft_matrix` moves onto the dedicated scalar-generic `DFTMatrixImpl<BE, F>` / `DFTMatrixDefault` OEP pair (backends wiring DFT natively must account for the split).
- `CKKSModuleAlloc` is fully default-bodied over `ModuleCoreAlloc` (backends get it from an empty blanket impl) and gains `ckks_ciphertext_alloc_with_rank`, with `ckks_ciphertext_alloc` as the rank-1 convenience. The `CKKSImpl` one-stop bundle now also covers `DFTImpl`, `CKKSEvalModImpl`, and `CKKSBootstrappingImpl`, and `oep`'s module doc records the wiring-pattern taxonomy (opt-in marker defaults, unconditional composite blankets, scalar-generic backend-side seams, no-OEP composite families).
- `UnnormalizedCKKSCiphertextRefMut` is sealed — public because it appears in OEP signatures, but without a public constructor, since one would void the `Normalized` compile-time guard. Shared layout validation helpers (`validate_gadget_key`, `validate_storage_capacity`, and backend-view variants) are promoted to `layouts/validation.rs`.

### `poulpy-cpu-ref` / `poulpy-cpu-avx` / `poulpy-cpu-avx512` / `poulpy-cpu-arm`
- Add `poulpy-cpu-ref/src/ckks_encoding.rs`, the production CPU encoding implementation behind three exported wiring macros: `impl_ckks_encoding_fft64_f64!` (f64 on an FFT64 backend, borrowing the backend's native ring FFT plans), `impl_ckks_encoding_owned_for!` (one concrete precision with self-contained plans), and `impl_ckks_encoding_owned!` (all precisions). Every backend wires `f64` + `Quad` encoding (FFT64 family via the native-FFT path, NTT families via owned plan sets).
- **Breaking:** backend ring handles now hold plan **sets** covering every power-of-two sub-dimension instead of a single fixed-N table pair: `FFT64PlanSet` / `FFT64Plan` with `FFTModuleHandle::get_fft_plan(n)` replace `get_fft_table` / `get_ifft_table`, and `NttPlanSet` / `NttPlan` with `get_ntt_plan(n)` replace `get_ntt_table` / `get_intt_table`. Handles also carry the new `ModulePlanCache` (exposed through `ModuleTableCacheAccess`); the old `ModuleTableCache` / `ModuleTableCacheProvider` names remain as re-export aliases of the HAL types.
- Add `poulpy-cpu-ref/src/ckks_paco.rs` with the `impl_ckks_paco_coeff_encoding!` macro (stages the exhausted ciphertext to host and runs the shared coefficient-encoding reference); all four backend crates invoke it for their FFT64/NTT4x30 (and NTT3x42Ifma) backends.

### Build & Docs
- Add `docs/paco.md` (user-facing PaCo guide: API, construction outline, key contract, security discussion) and `docs/spec/paco_dft_convention.md` (the DFT-convention specification relating Poulpy's generator-5 embedding to the paper's reference implementation, including the `Natural` vs `BitRevLow` cost tables); link them from `docs/README.md`.
- Update `docs/bootstrapping.md` for the constructor-validated plans (corrected `1/K` CoeffsToSlots pre-scale documentation, encapsulation weight moved to `EncapsulationKeysLayout`, power-of-two `f_mod_interval` requirement) and add a Toolchain section to the `poulpy-ckks` README documenting the nightly requirement (`#![feature(f128)]`) and the x86_64-only `libquadmath` feature.

## [0.7.0] - 2026-07-09

This release builds the full leveled-CKKS evaluation stack on top of the backend-generic core: scale-agnostic Baby-Step/Giant-Step polynomial evaluation, slot-domain linear transformations, the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs), EvalMod, and the complete CKKS bootstrapping pipeline (`ModUp → CoeffsToSlots → EvalMod → SlotsToCoeffs`). It also adds a new AArch64 NEON backend crate (`poulpy-cpu-arm`), lands a broad round of NTT/convolution/VMP performance work across the AVX / AVX-512 / IFMA backends, and renames the multi-prime NTT backend families to the `<primes>x<bits>` scheme (`NTT120` → `NTT4x30`, `NTT126` → `NTT3x42`; entries below use the name current when the change landed).

### Renamed
- Rename the multi-prime NTT backend families to a `<primes>x<bits>` scheme, leaving room for future variants such as `NTT2x60`: `NTT120` → `NTT4x30` (CRT over four ~30-bit primes) and `NTT126` → `NTT3x42` (three ~42-bit primes). This updates the public backend marker types (`NTT4x30Ref` / `NTT4x30Avx` / `NTT4x30Avx512` / `NTT4x30Neon`, and `NTT3x42Ifma`), their modules and test/bench references, the CKKS test parameters (`NTT4X30_PARAMS_F64` / `NTT4X30_PARAMS_F128`), and the documentation. The `Q120` / `Q126` moduli and `Primes30` / `Primes42` prime sets keep their names.

### `poulpy-hal`
- Add HAL APIs for scalar automorphisms and packed matrix helpers: `ScalarZnxAutomorphismBackend`, `ScalarZnxAutomorphismAssignBackend`, `VecZnxTransposeBackend`, and `VecZnxBigColWeightedSum`.
- Add reusable DFT-domain automorphism planning and application via `VecZnxDftAutomorphismPlan` and `VecZnxDftAutomorphism`, with backend-specific plan types wired through `HalVecZnxDftImpl`.
- Add the accumulating convolution apply `Convolution::cnv_apply_dft_accumulate` (`res += a (x) b` in the DFT domain), wired through the api/oep/delegate layers and implemented by every backend; it is bit-identical to `cnv_apply_dft` followed by a DFT-domain add (asserted raw-byte-exact by a new cross-backend conformance test), leaves limbs beyond the convolution bound untouched, and reuses the `cnv_apply_dft_tmp_bytes` scratch contract.
- Add `VecZnxLshAddCoeffToCoeffBackend` and `VecZnxLshSubCoeffToCoeffBackend` hooks plus portable reference implementations so coefficient-level plaintext accumulation can handle left-shift alignment.

### `poulpy-core`
- Add a backend-generic, fully **scale-agnostic** Baby-Step/Giant-Step polynomial-evaluation engine and public `GLWEPolynomialEvaluation` Module API (baby-step / giant-step methods) wired through the api/oep/delegates/default layers. The engine owns only the combinatorial schedule (parity loop, giant-step pairing tree, and the per-level `X^{gsp}` hoisting) and delegates every arithmetic operation — with all precision bookkeeping, normalization and compaction — to the scheme through the `BSGSBabyOps` (accumulator seed, `ct×pt` terms, compaction) and `BSGSGiantOps` (associated `Prepared` right operand, hoisted `ct×ct` multiply, `ct+ct` add, final copy) traits; it never references `log_delta` / `log_budget` / `k` itself.
- Add scheme-agnostic polynomial-evaluation layouts `Polynomial`, `BSGSPolynomial`, `PowerBasis`, `Basis`, `Parity`, and `SplitStrategy`, with monomial and Chebyshev bases, Chebyshev interpolation, arbitrary-interval support (`with_interval` / `interval` / `change_of_basis` / `evaluate_on_interval`, so a polynomial approximated on `[a, b]` carries its domain through to evaluation), BSGS/Paterson-Stockmeyer decomposition planned by the `MinDepth` / `MinMult` split strategies, and ahead-of-time budget accounting via the closed-form `bsgs_consumed_bits` (validated against a reference walk of the schedule) and `bsgs_eval_depth`.
- Factor the GLWE tensor product into a shared `glwe_tensor_apply_loop` and add the public hoisted-operand primitives `glwe_prepare_right` / `glwe_tensor_apply_prepared_right` (+ `_tmp_bytes`), which prepare a ciphertext once into a reusable right convolution operand — letting a scheme prepare each `X^{gsp}` once and share it across the baby-step pairs of a giant-step run.
- Allow GLWE plaintext add/sub alignment to left-shift when the encoded plaintext precision exceeds the ciphertext budget, enabling the coefficient-indexed plaintext constants used by polynomial evaluation.
- Add the scheme-agnostic GLWE-level linear-transformation (matrix–vector product over the slots) engine: the unprepared `LinearTransformation` / `LinearTransformationGiantStep` / `LinearTransformationDiagonal` (encoded diagonals bucketed by giant step), the integer-level BSGS schedule types (`LinearTransformationLayout`, `LinearTransformationPlan`, `LinearTransformationStrategy` (`Bsgs { giant_step }` / `Direct`), `optimal_bsgs_giant_step`), and the `GLWELinearTransformations` trait. It evaluates `M·v = Σ_k diag_k ⊙ rot(v, k)` via a baby-step/giant-step double loop over raw GLWE ciphertexts, carries no CKKS scale notion, and receives only base2k-level alignment integers from the caller (docs/linear_transformation.md).
- Add the convolution-domain prepared operand `PreparedDiagonal` (a `CnvPVecR` paired with its diagonal's plaintext layout) so the resident transform is just `LinearTransformation<PreparedDiagonal>` (aliased `LinearTransformationPrepared`) — the same generic container as the streamed `LinearTransformation<CKKSPlaintext>`, differing only in the diagonal representation. The left operand stays the separate `LinearTransformationBabySteps` (hoisted baby rotations); both caches are allocated up front and populated separately so a giant-step schedule reuses one baby cache across factors and evaluations. The prepared evaluator keeps the per-giant product, giant rotations, and cross-giant accumulation in the DFT / `VecZnxBig` domain with a single final normalization.
- Add a streamed (unprepared-RHS) evaluation path that prepares each diagonal on the fly through scratch instead of materializing the full prepared right operand, trading recompute for lower resident memory (for bandwidth-bound backends). The per-giant product is dispatched by the diagonal type through the `DiagonalProd` trait (one impl per concrete diagonal type: fused for `PreparedDiagonal`, stream-prepare for plaintext diagonals), so the resident and streamed transforms share one giant-step driver over a single generic `LinearTransformation<P>`. The `GLWELinearTransformations` evaluation surface is correspondingly a single `glwe_eval_linear_transformation_into` generic over `P: DiagonalProd` — there is no separate prepared-only eval method; `P = PreparedDiagonal` is just the resident instantiation.
- Add the scheme-agnostic clear evaluator `Diagonals<T>` with the `Evaluate` / `DiagonalArithmetic` traits (and an in-place `transpose`), used as the plaintext reference the homomorphic engine matches bit-for-bit up to scheme precision.
- Add packed LWE matrix support with `LWEMatrix`, `LWEMatrixLayout`, `LWEMatrixInfos`, `BackendLWEMatrix`, backend ref/mut adapters, and `ModuleCoreAlloc::{lwe_matrix_alloc,lwe_matrix_alloc_from_infos}`.
- Add core APIs for packed LWE matrix workflows: `GLWEExpandLWEMatrix` expands a GLWE into a matrix of LWE samples, and `LWEMatrixDecrypt` decrypts packed LWE rows into a GLWE plaintext-shaped result.
- Add `GLWEMaskFill` and `LWEFillMask` traits for backend-generic mask generation from a `Source` or deterministic seed; compressed LWE/GLWE decompression now uses those mask-fill defaults, and `GLWECompressed` exposes `data()` / `data_mut()` accessors for its stored ciphertext data.
- Accumulate the `dsize > 1` keyswitch digits in place: `gglwe_product_dft` now folds every digit past the first directly into the result through `vmp_apply_dft_to_dft_accumulate`, dropping the per-digit scratch `VecZnxDft` buffer and the separate per-column `vec_znx_dft_add_assign` pass.

### `poulpy-ckks`
- Add a thin scale-aware polynomial-evaluation bridge over the core engine: the `PolynomialEvaluation` trait and its `ckks_eval_poly_real_const_coeffs_from_power_basis` driver, the `EncodeBSGS` / `PowerBasisGen` extension traits, re-exports of the core layout types under CKKS module paths, and `CKKSBSGSOps` — the implementation of the core `BSGSBabyOps` / `BSGSGiantOps` traits as a thin dispatch onto the CKKS API (`ckks_prepare_right` / `ckks_mul_prepared_assign` / `ckks_add_assign` / `ckks_copy` / `ckks_*_pt_const`), so all scale (`log_delta` / `log_budget` → `cnv_offset`) math lives in the CKKS multiply ops rather than the BSGS glue.
- Add complex-coefficient polynomial evaluation (monomial and Chebyshev) via the `ckks_eval_poly_complex_const_coeffs_from_power_basis` driver, which takes the polynomial as a single `&ComplexBSGSPolynomial` (the aligned real/imag BSGS decompositions held together), combines the matched real/imag baby steps as `re + i·im` through `CKKSImagOps`, and folds the trailing complex constant through the highest power when present.
- Add the one-shot `ckks_eval_poly_real_const_coeffs` / `ckks_eval_poly_complex_const_coeffs` convenience entry points, which build the power basis internally from the input ciphertext before evaluating.
- Add the hoisted `ct×ct` multiply to the CKKS mul API: `CKKSMulOps::ckks_prepare_right` hoists a ciphertext's forward transform into the backend-resident `CKKSPreparedRight` right operand, and `ckks_mul_prepared_assign` multiplies `dst *= prepared` against it; both are wired through the api/oep/delegates/default layers and share `ckks_mul`'s metadata rule and scratch bound (`ckks_mul_tmp_bytes`).
- Add backend-generic CKKS `eval_mod` (homomorphic `x mod 1`): `EvalModPlan` over `EvalModType` `SinContinuous` / `CosContinuous` / `CosDiscrete` / `Exp`, with double-angle composition, optional arcsine post-composition, and a caller-selected BSGS `split_strategy`. EvalMod runs at its own evaluation scale — `ckks_eval_mod` reinterprets the working ciphertext to the plan's `f_mod_log_delta` on entry and restores the input scale on exit through a budget-neutral set-scale round-trip, keeping the approximation's `ct×ct` precision independent of the caller's scale — and `EvalModPlan::consumed_bits` charges the multiplicative budget deterministically at `f_mod_log_delta` (folding the `1/f_mod_interval` range normalization and the optional arcsine into the count).
- Add the complex-exponential `Exp` variant: `exp(2i·pi·x)` via complex Chebyshev, doubled by squaring into a complex ciphertext; the polynomial is held by the new `EvalModBsgs::{Real, Complex}` enum.
- Add eval_mod OEP/delegate wiring (`CKKSEvalModImpl`, `CKKSEvalModOps`) and conformance tests for every variant on FFT64/f64, NTT120/f64, and NTT120/f128.
- Generalize fused CKKS multiply-add/plaintext paths over backend-owned ciphertext/plaintext layouts.
- Add conformance tests for CKKS polynomial evaluation: monomial and Chebyshev evaluation, power-basis generation, interpolation, metadata errors, split-strategy behavior, and complex-coefficient evaluation (even/odd parity and the trailing-constant fold).
- Add the CKKS linear-transformation (matrix–vector product over the slots) API `LinearTransformationOps` (with `LinearTransformation` / `GiantStep` / `Diagonal` and the prepared caches re-exported from `poulpy-core`). The CKKS layer owns the scale (`log_delta` / `log_budget`) math: it derives the convolution alignment and result metadata and delegates evaluation to the core engine. Evaluation is a **single** entry point generic over the diagonal representation `P` — `ckks_eval_linear_transformation_into` / `_assign` (caller-supplied baby cache) and `_self_into` / `_self_assign` (self-allocating) — where `P = PreparedDiagonal` is the resident path and `P = CKKSPlaintext` the streamed path: the prepared-vs-streamed choice is which `P` you pass, not a separate method (the per-`P` scale/key-size bookkeeping is read uniformly off the first diagonal via the CKKS-local `LtDiagonalScale` trait, so `poulpy-core`'s `DiagonalProd` engine trait stays scheme-agnostic and carries no scale concept). `ComplexDiagonals<T>` plus `ckks_encode_linear_transformation_from_diagonals` build the encoded transform (with optional transpose for the `a·B` orientation) from a raw complex diagonal map. A backend-generic conformance test validates both the resident and streamed results against the plaintext `ComplexDiagonals::evaluate`.
- Strengthen the test-suite `assert_decrypt_precision[_at_log_delta]` to decrypt once and run two complementary checks off the single decryption: a ring-domain noise-`std` bound over the full-width plaintext coefficients (catches corruption above the plaintext head-room that decoding would clip) and the canonical-embedding per-slot precision assertion (decoded slots vs. expected at `log_delta`).
- Add the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs) via the `DFTOps` trait. `ckks_new_dft_matrix` builds a factorized (I)DFT from a `DFTPlan` (per-factor `factorization_depth` schedule, per-factor BSGS `factor_giant_steps`, `bit_reversed`, `scaling`, plus the schedule-introspection queries `galois_elements(log_n, cyclotomic_order)` / `diagonal_indexes(log_n)` / `num_diagonals(log_n)` that report exactly the rotation keys a CoeffsToSlots / SlotsToCoeffs — and hence a bootstrap — needs) as the host, unprepared plaintext form `DFTMatrix`; `ckks_prepare_dft_matrix` promotes that to the convolution-domain resident form `DFTMatrixPrepared` for faster repeated evaluation (build → prepare, mirroring the rest of poulpy). `ckks_coeffs_to_slots` / `ckks_slots_to_coeffs` (plus `_split` and sparse `_repack` variants) evaluate either form by chaining one linear transformation per factor with no explicit rescale between them (the plaintext-multiply realigns to the input scale). `DFTMatrix<BE, Dir, Fmt, R>` carries its transform **direction** (`Encode`/`Decode`) and output **format** (`Standard`/`Split`/`Repack`) as compile-time type-state markers, so each eval entry requires the exact matrix — e.g. `ckks_coeffs_to_slots_repack` only accepts a `DFTMatrix<BE, Encode, Repack, _>` — making a direction/format mismatch a **compile error** rather than a runtime `ensure()` check. The markers are established once, at `ckks_new_dft_matrix` (which errors only when `Repack` is requested for dense parameters, the single runtime format resolution); the orthogonal factor-storage axis (the diagonal representation `P`; `DFTMatrix` = plaintext, `DFTMatrixPrepared` = resident) is preserved across `ckks_prepare_dft_matrix`, and both share one evaluator: each factor is applied by the *same* unified `ckks_eval_linear_transformation_assign` (generic over `P`), so the DFT carries no bespoke prepared/streamed factor dispatch of its own — the resident-vs-streamed split lives entirely in the linear-transformation layer. `DFTOps` is wired through the `api → oep → delegates ← default` layers (`DFTImpl` / `DFTDefault` / `impl_ckks_dft_defaults!`), so a backend can override the whole-DFT evaluation — every factor plus the inter-factor rotations/conjugations and split/repack glue — with a single fused kernel. The BSGS schedule is caller-supplied per factor — the library applies no implicit optimum, since the cost-optimal width is backend-dependent. Backend-generic conformance tests cover CoeffsToSlots/SlotsToCoeffs (Standard, split, and sparse repack) as **directional oracle** checks: each transform's output is compared (via `GLWENoise`, bounding the log2 of the residual std) against an *independent* plaintext reference built from the dual encoding — not an `Encode∘Decode` round trip — so a basis/permutation/scale error that a round trip would cancel is caught.
- Add base-`2^base2k` budget/compaction maintenance for leveled pipelines: `SetCKKSInfos::compact_in_place` (an O(1) `set_size` to the limbs spanning the live `k`, on ciphertexts, view-muts and plaintexts) and `ckks_set_log_delta` (re-interpret the encoding scale, preserving the message and `log_budget` — widening reallocates the buffer, narrowing compacts it). Every budget-consuming op (`ct×ct` / `ct×pt` multiply, each homomorphic-DFT factor, and EvalMod) compacts its result, so each pipeline stage drops the sub-precision low limbs and the next stage operates on a tight ciphertext.
- Add CKKS bootstrapping (`CKKSBootstrappingOps`): the `ModUp → CoeffsToSlots → EvalMod → SlotsToCoeffs` pipeline. The only new primitive is **ModUp** (`ckks_mod_up_into`), the base-`2^base2k` modulus raise — a digit shift, not an RNS prime-basis extension: it right-shifts the MSB-aligned digits into a wider destination so the secret-dependent term stops wrapping (cleartext `I(X)·q + Δ·m`), exactly the input EvalMod expects, with message ratio `q/Δ = 2^{src.log_budget()}`. The other three stages are re-exported as supertraits (`DFTOps` + `CKKSEvalModOps`) so one bound drives the whole pipeline; the crate ships **no orchestrator**, keeping the stages composable. `BootstrappingPlan` bundles the per-stage parameterization (the CoeffsToSlots / SlotsToCoeffs `DFTPlan`s, the `EvalModPlan`, the sparse-secret-encapsulation weight, and `consumed_bits` / `galois_elements` accounting), and `BootstrappingContext::compile` lowers it once into resident DFT matrices (with the `1/f_mod_interval` scaling folded into CoeffsToSlots) plus the uploaded, encoded EvalMod, reused across bootstraps. Wired through the api/oep/delegates/default/layouts layers, with an end-to-end conformance test exercising the full `ModUp → CoeffsToSlots(split) → EvalMod×2 → SlotsToCoeffs(split)` composition.

### `poulpy-cpu-ref` / `poulpy-cpu-avx` / `poulpy-cpu-avx512`
- Add the `bootstrap_trace` example (`poulpy-cpu-ref`, `--features enable-ckks`): the end-to-end CKKS bootstrapping pipeline on the `ntt120_f64` reference backend compiled into a standalone binary for sampling profilers (`samply`), with `BOOTSTRAP_ITERS` to repeat the run.
- Implement the new transpose, weighted-sum, scalar/DFT automorphism, and packed LWE matrix defaults across the reference backend, with AVX and AVX-512 overrides for the accelerated automorphism paths.
- Rework the FFT64 convolution applies into fused column kernels behind the new `Reim4Convolution::reim4_convolution_apply` / `reim4_convolution_pairwise_apply` hooks (the reference default keeps the previous per-block path): the AVX/AVX-512 kernels tile 3/4 output limbs over a zero-padded sliding window of the left operand and stage outputs per 16-block group so each destination cache line is written exactly once — the limb stride is a multiple of 4 KiB, so the previous per-limb half-line stores all aliased a single L1 set. `cnv_apply_dft` and `cnv_pairwise_apply_dft` are 2.3-3.5x faster on `FFT64Avx512`/`FFT64Avx` across n = 2^13..2^15.
- Interleave the `NTT120Avx512` VMP prepared-matrix prime planes per (block-pair, output-column) chunk so the apply streams the matrix as one sequential run instead of four planes hundreds of MB apart (which defeated the hardware prefetcher): `vmp_apply_dft_to_dft` improves up to 3.4x (77 ms to 23 ms at 16384x(1x31)x(2x32)), and `glwe_keyswitch` at n = 2^15 drops from 41.5 ms to 24.7 ms.
- Switch the NTT-family prepared convolution operands (`CnvPVecL` / `CnvPVecR`) to block-major rows with the right operand in reversed limb order, deleting the per-apply pack/gather passes; `CnvPVecL` now stores the canonical kernel-ready encoding so the per-apply `% q` reduction of the left operand happens once at prepare time (ntt120 family), and the prepare NTT writes its final normalised blocks straight into the prepared rows (`NTT126Ifma`). The applies tile four output limbs per pass over a zero-padded window (`NttMulBbc1ColX2::ntt_mul_bbc_tile4_x2`, defaulted for any backend, with canonical-x AVX-512/AVX2 kernels that skip the identically-zero `x_hi` product path), reduce once per output, and group-stage their output flush. Measured at 32768x14: `NTT126Ifma` apply -36% / pairwise -62%, `NTT120Avx512` CKKS `mul_ct` -15%.
- Fuse the `NTT126Ifma` forward NTT level-0 twist into the first butterfly level and run the last three levels (`nn = 8, 4, 2`) as a single radix-8 register pass mirroring the existing inverse head, making the forward transform ~5% faster; add n = 2^12 cross-backend idft conformance tests, since the breadth-first levels above `NTT_BLOCK` were previously uncovered by the n = 2^8 suites.

### `poulpy-cpu-avx`
- Fuse NTT levels in the `NTT120Avx` (q120) by-level phase. Fold the forward level-0 twist (`a[i] *= ω^i`) into the first butterfly level and the inverse final normalization (`a[i] *= ω^{-i}/n`) into the last butterfly level, so neither needs its own full-array sweep. Above `FUSE_MIN_N = 2^15`, additionally fuse pairs of forward (DIF) and inverse (GS) butterfly levels into single load/store radix-4 passes (the twist/normalization folds into the first/last radix-4 pass); below it the radix-2 path is kept, since radix-4 register pressure (16 ymm) only pays off once the working set spills out of cache. All kernels are bit-identical to the prior radix-2 passes (same arithmetic and lazy-reduction schedule, verified against `ntt120-ref` up to n = 2^16). Measured forward/inverse NTT ~34%/~26% faster at n = 2^16 and ~9%/~5% at n = 2^15 (radix-4 + fold), with the radix-2 twist-fold giving a further ~2–5% on the forward NTT at n = 2^12…2^14.
- Batch the q120b → i128 CRT reconstruction (`b_to_znx128_avx2`, the `NttToZnx128` consume path) four coefficients at a time: transpose each 4×4 (coefficient × prime) residue block in registers so the CRT weighted sum `Σ_k t[k]·(Q/Q[k])` accumulates vertically across the four prime lanes, replacing the per-coefficient horizontal reductions with one transpose amortized over four outputs (a scalar `nn % 4` tail handles the remainder). Bit-identical to the prior per-coefficient path and ~15% faster on the reconstruction kernel (≈5% of the full iDFT-consume).
- Vectorize the `b_to_znx128_avx2` final CRT reduction as well, with a planar 4-wide fold mirroring the AVX-512 path: carry-propagate the per-lane `(acc_lo, acc_mid, acc_hi)` accumulators to 128-bit, fold mod `TOTAL_Q` via a Barrett `q ≈ v >> 120 ∈ {0,1,2,3}` table-lookup plus one conditional subtract, all across four lanes with `blendv`-selected masked 128-bit add/sub and sign-flipped unsigned 64/128-bit compares (AVX2 has no mask registers), leaving only the symmetric sign lift scalar. Bit-identical to the prior per-coefficient finalize.

### `poulpy-cpu-avx512`
- Migrate the `NTT126Ifma` DFT-domain representation from the 4-lane array-of-structs layout (one `[u64; 4]` per coefficient, with three CRT residues and a padding lane) to a planar 3-prime layout (three contiguous residue planes per limb), removing the wasted fourth lane and the hand-written assembly CRT kernel (`vec_znx_dft_asm.s`).
- Optimize the planar transforms for pass efficiency: mask the sub-8 butterfly remainders in the upper levels, process the radix-8 tail and head as eight transposed blocks per pass, fuse pairs of upper levels into single load/store radix-4 passes (forward DIF, inverse DIT), and fold the forward final normalization and the inverse level-0 untwist into adjacent butterfly stores. Together these roughly halve the raw NTT/iNTT cost at log n = 15 and let the planar backend outperform the previous 4-lane layout on CKKS multiplication.
- Vectorize the planar 3-prime CRT-to-i128 reconstruction in the iNTT consume path (`simd_b_ntt126_ifma_to_znx128`), replacing the per-lane scalar Garner reconstruction with an AVX-512-IFMA base-2^52 limb accumulation plus a scalar symmetric-range fix.
- Replace the `NTT126Ifma` forward/inverse NTT kernels with radix-2 passes (superseding the prior radix-8 forward register pass and the radix-4/radix-8 planar passes): the forward is a Cooley-Tukey transform (natural-order input → bit-reversed output), the inverse a Gentleman-Sande transform (bit-reversed → natural, `1/n` folded in), each transforming the three contiguous prime planes in place. Butterfly values are kept under a lazy reduction (`[0, 4q)` forward, `[0, 2q)` inverse) and the difference feeds straight into the Harvey/Shoup modular multiply without a pre-reduction (the 52-bit IFMA product absorbs inputs up to `2^52`); distances `t = n/2 … 8` use a broadcast-twiddle form while the `t = 4, 2, 1` tail uses sub-vector interleaved loads with duplicated tail roots. Above `BASE_NTT_SIZE` each plane is split depth-first — one top broadcast stage then recursion into the two halves, indexing the shared full-`n` root tables by `(depth, half)` — so large-`n` transforms stay cache-resident instead of sweeping the whole plane per stage.
- Add a `lazy_output` mode to the `NTT126Ifma` forward NTT (`ntt_avx512`) that leaves the result in `[0, 4q)` instead of fully reducing to `[0, q)`, skipping the final reduction pass; the prepare paths (convolution `cnv_prepare_left` / `_right`, SVP prepare, VMP prepare) opt in since their consumers re-reduce — the BBC product and `c_from_b`, whose `2^44` bound exceeds `4q` for the 42-bit primes — while the public DFT contract keeps the full `[0, q)` reduction. A conformance test asserts the lazy output stays in `[0, 4q)` and equals the fully-reduced forward mod `q`.
- Vectorize the `NTT120Avx512` (q120) `b_to_znx128` final CRT reduction with a planar 8-wide fold: transpose four `reduce_b_and_apply_crt_512` outputs (8 coefficients) into prime-planar vectors, accumulate the CRT weighted sum vertically across the four prime lanes, carry-propagate to 128-bit, and fold mod `TOTAL_Q` via a Barrett table-lookup plus one masked conditional subtract, processing 8 coefficients per iteration (with a 2-coefficient + odd-coefficient scalar tail). Byte-identical to `b_to_znx128_ref` for every `nn`; wired into the iNTT-consume compaction path.
- Replace the scalar per-element modular accumulate in the `NTT120Avx512` VMP output save (`save_blk_add`) with a lazy AVX-512 fold: both operands hold q120b residues in `[0, 2·Q_SHIFTED)`, so one SIMD compare + masked subtract (`lazy_reduce_512`) per operand reproduces the scalar `%` byte-for-byte and the sum stays in range for the downstream iNTT/normalize.
- Pack the `NTT3x42Ifma` prepared VMP matrix: the three 42-bit CRT residues of each coefficient are packed into two u64 words at prepare time (down from three) and unpacked in-register by the VMP apply kernels, cutting the prepared-matrix footprint and streamed bandwidth by a third on the bandwidth-bound apply path.

### `poulpy-cpu-arm`
- Add the `poulpy-cpu-arm` NEON/ASIMD CPU backend for AArch64, exposing `FFT64Neon` and `NTT120Neon`. Hand-written NEON kernels cover every accelerated `poulpy-hal` family — the f64 Reim FFT/iFFT and frequency-domain `ReimArith`, the `Reim4` mat-vec and convolution kernels behind VMP, the i64 big-coefficient convolution (`I64Ops`), the Q120 NTT/iNTT, CRT `b`/`c` conversions, the bbc mat-vec and VMP kernels, and the coefficient-domain `Znx*` arithmetic and base-2^k normalization — while the remaining families inherit the portable `poulpy-cpu-ref` defaults through the `hal_impl_*` macros, and the `poulpy-core` / `poulpy-ckks` (behind `enable-ckks`) scheme wiring is inherited through `impl_*_defaults_full!` / `impl_ckks_*_defaults!`. The backend is opt-in via `enable-neon` and `compile_error!`s on non-AArch64 targets; integer / `Ntt*` operations are bit-identical to `poulpy-cpu-ref` and FFT operations match within ULP. A `neon` CI lane type-checks, lints, and runs the backend suite natively on `aarch64` GitHub-hosted runners.
- Add the canonical-x NEON convolution-apply kernel `NttMulBbc1ColX2::ntt_mul_bbc_tile4_x2` (`vec_mat_tile2_bbc_canonical_neon`) — the NEON analog of the AVX2/AVX-512 tiled kernels — so the NTT120 `cnv_apply_dft` / `cnv_apply_dft_accumulate` / `cnv_pairwise_apply_dft` route through the shared block-major reference tiling onto fused NEON inner products that skip the identically-zero `x_hi` product path.

### `poulpy-bench`
- Add the `ckks_poly_eval` Criterion benchmark, sweeping polynomial degree and `MinDepth` / `MinMult` BSGS split strategies on `ntt120-ref` while reporting baby-step size and observed log-budget/level consumption.
- Add the `ckks_eval_mod` Criterion benchmark, timing the `EvalModType` variants (sin continuous with optional arcsine, cos discrete under both `MinDepth` and `MinMult` split strategies, cos continuous) on `ntt120-ref` and reporting level consumption and predicted CT–CT mul depth.
- Enable the `poulpy-cpu-avx` CKKS implementations in the `ckks-bench` feature so the CKKS benchmarks compile with `enable-avx`.
- Add the `cnv_apply_dft_accumulate` sweep to the convolution Criterion benchmark.
- Add a `glwe_tensor_relinearize` benchmark covering the relinearization (keyswitch) phase of CKKS multiplication.
- Extend the `ckks_poly_eval` benchmark to also run on `NTT120Avx` (feature `enable-avx`) and `NTT126Ifma` (feature `enable-ifma`) via a backend-parameterized macro, alongside the existing `ntt120-ref` sweep.

### Build & Docs
- Add `docs/linear_transformation.md` (+ `docs/img/linear_transformation.png`), the design note for the baby-step/giant-step linear transformation: the diagonal decomposition, the prepared convolution-domain caches, the hoisted-baby / lazy-giant evaluation, and the CKKS scale accounting.
- Add `docs/polynomial_evaluation.md` describing the Baby-Step/Giant-Step method, the `MinDepth` / `MinMult` strategies, the supported polynomial flavors, and a measured table of modulus consumption per degree.
- Refresh the `ckks_poly2` example to use Chebyshev interpolation, `PowerBasis`, and the new BSGS evaluator pipeline.
- Overhaul the documentation set under `docs/`: add `docs/README.md` (index), `docs/backends.md` (backend overview including the NEON backend and the AVX-512 feature requirements), `docs/bootstrapping.md` (the CKKS bootstrapping design note), `docs/grafting-vs-bivariate.md` (comparison note), and the `docs/spec/` linear-transformation specification (`lt_bsgs.md` + implementation walkthrough `lt_bsgs_impl.md`); refresh `docs/getting-started.md` to surface the poly-eval / linear-transform / DFT subsystems, consolidate the LT docs and move images to `docs/img/`, link the CKKS docs from the main README, and trim the root README.

## [0.6.0] - 2026-05-18

This release completes the migration from the legacy host-oriented HAL/backend plumbing to backend-generic HAL and core layers, so backends can now own buffers, scratch space, and transfer paths explicitly, and adds a new AVX-512 backend crate (`poulpy-cpu-avx512`) exposing three accelerated backends (`FFT64Avx512`, `NTT120Avx512`, `NTT126Ifma`).

### `poulpy-hal`
- Refactor `VecZnx`, `ScalarZnx`, `MatZnx`, `VecZnxDft`, `VecZnxBig`, `SvpPPol`, `VmpPMat`, `CnvPVecL`, and `CnvPVecR` to store private shape metadata snapshots instead of exposing mutable layout fields directly.
- Add explicit shape/getter APIs (`shape()`, `n()`, `cols()`, `size()`, `max_size()`) and metadata-only resizing helpers (`with_size`, `set_size`) so backend views remain cheap value descriptors without encouraging field mutation on temporaries.
- Make `Module` the canonical allocation entrypoint for raw coefficient-domain layouts (`VecZnx`, `ScalarZnx`, `MatZnx`) and `VecZnxBig`; migrate workspace call sites to `Module::*_alloc[_n]` and restrict the old host-owned `alloc` constructors to crate-private visibility.
- Remove the public explicit-`n` raw allocator surface entirely; allocation is now expressed only as `module.scalar_znx_alloc(...)`, `module.vec_znx_alloc(...)`, `module.vec_znx_alloc_with_max_size(...)`, and `module.mat_znx_alloc(...)`. Special-degree cases now build a module carrying the desired `n` first, then allocate through that module.
- Remove the temporary host-allocation helper traits again and migrate tests/bench/core staging to explicit `Module::<HostBytesBackend>::new(...).<alloc>(...)` calls, keeping host-owned allocation a direct module concern rather than a separate HAL abstraction.
- **Breaking:** HAL compute traits now take backend-native borrows and scratch explicitly: `Scratch` becomes `ScratchArena<'_, BE>`, `*ToRef` / `*ToMut` become `*ToBackendRef<BE>` / `*ToBackendMut<BE>`, and public trait names move to backend-explicit forms such as `VecZnxAddIntoBackend`, `VecZnxRotateAssignBackend`, and `VecZnxRshSubBackend`.
- Add backend-owned/layout interop APIs: `Backend::{OwnedBuf, BufRef, BufMut}`, `HostBackend` / `DeviceBackend`, `TransferFrom<From>`, backend view aliases/reborrow traits for all major layouts, and allocator traits `ScalarZnxAlloc`, `VecZnxAlloc`, `MatZnxAlloc`, plus `api::reim::{NegacyclicFFT, NegacyclicFFTNew}`.
- Add `VmpApplyDftToDftAccumulate` (+ `*_tmp_bytes`) for a fused `res += a · pmat` with limb-offset shift, replacing the scattered `vmp_apply_dft_to_dft` + per-column `vec_znx_dft_add_assign` fold in `gglwe_product_dft`.
- Fix the convolution API by renaming the output-shift parameter to `cnv_offset`, moving it to the front of the apply calls, and updating delegates and conformance tests to match the corrected calling convention.
- **Breaking:** `Convolution::cnv_by_const_apply` no longer takes a raw coefficient slice; it now takes a backend `VecZnx` plus `(b_col, b_coeff)` selectors, matching the rest of the backend-native convolution surface.
- Replace the legacy monolithic `oep::HalImpl` entrypoint with per-family OEP traits (`HalModuleImpl`, `HalVecZnxImpl`, `HalVecZnxBigImpl`, `HalVecZnxDftImpl`, `HalSvpImpl`, `HalVmpImpl`, and `HalConvolutionImpl`) so backends can opt into and override only the families they own.
- Add family defaults for `vec_znx`, `vec_znx_big`, `vec_znx_dft`, `svp_ppol`, `vmp_pmat`, and `convolution` to reduce backend boilerplate and make per-family overrides explicit.
- Remove the aggregate `HalImpl` dispatch surface; update delegates to route through the per-family OEP traits and simplify dispatch.
- Update layouts and encoding helpers to match the new dispatch surface.
- Generalize scratch/layout plumbing around backend-owned buffers and views so HAL families no longer assume host-resident storage.
- Refresh HAL test suites to align with the new defaults and dispatch.
- Add family-level module/scratch defaults to cut backend boilerplate and centralize scratch sizing.
- Make `WriterTo` for `MatZnx` and `VecZnx` emit the canonical logical byte length from layout metadata, write only that prefix, and error when backing storage is shorter than the coefficient span.
- Fix `ScalarZnx::write_to` to emit the full `n * cols` coefficient byte span (aligned `i64` layout).
- **Breaking:** Remove `ReaderFrom` / `WriterTo` for prepared DFT layouts (`SvpPPol`); remove `SvpPPolFromBytes`, `VmpPMatFromBytes`, and `from_bytes` on the corresponding prepared types. Document that `SvpPPol` / `VmpPMat` DFT alignment assumes a power-of-two ring degree.

### `poulpy-core`
- Fix #158: `VecZnxScalarProduct` semantics corrected — the default implementation now computes element-wise Hadamard products `res[limb][k] = a[limb][k] * b[k]` stored in `VecZnxBig`; callers that need the inner sum must follow up with `VecZnxBigInnerSumBackend`. LWE encryption updated accordingly (adds a `vec_znx_big_inner_sum_backend` step after `vec_znx_scalar_product`).
- Remove the redundant `for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>` bound from the `LWEEncryptSk::lwe_encrypt_sk` user-facing API method; the blanket impl always satisfies this constraint, so it was unnecessary noise on call sites.
- Add rank>1 support to `GLWEExpandLWE::glwe_expand_lwe`; add `glwe_expand_lwe_tmp_bytes` to the trait.
- Add rank>1 support to `SecretConversion::lwe_secret_from_glwe_secret`, producing the concatenated LWE key `(s_0(-X) || ... || s_{k-1}(-X))` needed to decrypt sample-extracted ciphertexts from rank-k GLWEs.
- Update `test_glwe_expand_lwe` to exercise both rank=1 and rank=2, verifying correctness of the secret derivation and decryption round-trip for each rank.
- Update layout wrappers, encryption/conversion paths, and tests to consume backend-view getters/constructors instead of reaching into HAL layout fields directly.
- Add `ModuleCoreAlloc` and `ModuleCoreCompressedAlloc`, then migrate workspace allocation sites so standard and compressed `poulpy-core` wrappers are allocated through `Module` instead of direct static `alloc*` constructors.
- Restrict standard and compressed wrapper `alloc` / `alloc_from_infos` constructors to crate-private visibility now that `Module` is the canonical public allocation entrypoint.
- Route the remaining host-owned raw-layout construction inside `poulpy-core` wrappers through `crate::layouts::host_module(...)`, so wrapper internals follow the same module-first allocation rule as HAL-facing code.
- **Breaking:** Core traits and helpers now follow the HAL backend-native calling convention: scratch arguments are `ScratchArena<'_, BE>`, layout bounds use `GLWEToBackendRef<BE>` / `GLWEToBackendMut<BE>` and related prepared-key `...ToBackendRef<BE>` traits, and backend-generic extension points no longer assume host-slice views.
- Add `api::ModuleTransfer` for typed upload/download of `LWE`, `GLWE`, `GGLWE`, `GGSW`, plaintexts, secrets, and prepared keys across backends; downstream code can now move full core objects without reaching into raw buffers.
- Thread the corrected convolution-offset semantics through GLWE constant/plaintext multiply and tensoring paths so scratch sizing, truncation, and normalization all use the same convention.
- Pass explicit effective-k information into convolution-backed multiply/tensor routines and mask partial bottom limbs correctly instead of assuming every input uses its full stored limb width.
- Refresh GLWE tensor tests to cover the updated convolution API and the corrected effective-width handling.
- Fix tensoring noise blowup when output operand had a smaller size than the input operand.
- Split public APIs into `api` trait modules backed by `delegates` and `oep` layers to separate user-facing traits from backend hooks and dispatch.
- Reorganize encryption, decryption, conversions, keyswitching, external products, and operations to match the new API structure.
- **Breaking:** Backend-default wiring is now exported per family (`impl_encryption_defaults_full!`, `impl_glwe_trace_defaults_full!`, `impl_glwe_packing_defaults_full!`, etc.) instead of only through the old monolithic core default macro; add `impl_glwe_rotate_impl_from!` as an explicit delegation helper for backends that forward rotation to another backend.
- Move backend conformance suites into `src/test_suite` and keep unit tests separate.
- Refresh layouts, noise helpers, and utilities to align with the new API surface.
- Re-export top-level modules to preserve public API ergonomics while routing through the new `api` traits.
- Standardize prepared allocations on `DeviceBuf` for backend-owned buffers to make data ownership explicit.
- Add explicit backend-to-backend transfer APIs for ciphertexts, plaintexts, secrets, and prepared keys to support upload/download flows across devices.
- Rename Module allocation/prepare helpers to struct-first names (e.g. `gglwe_prepared_alloc`, `glwe_secret_prepare`) to match the rest of the API.
- **Breaking:** Remove `ReaderFrom` / `WriterTo` for `LWESecret` and `GLWESecret`; secret material should use seeds or application-level transfer, not library binary I/O.

### `poulpy-cpu-ref` / `poulpy-cpu-avx`
- Refresh FFT64/NTT120 references and backend glue for the new private-shape HAL layouts, including explicit `from_data[_with_max_size]` rebuilding where host helpers reinterpret backend buffers.
- Update FFT64 and NTT120 convolution implementations, references, and tests to the corrected `cnv_offset` API.
- Optimize NTT120 convolution on the AVX backend by wiring the prep paths to backend-specific kernels and restructuring `cnv_apply_dft` / `cnv_pairwise_apply_dft` around prepacked x2 blocks, substantially reducing GLWE tensoring time on large `ntt120-avx` workloads.
- Add a row-prime-major prepared-matrix layout for the `NTT120Avx` VMP (`vmp_prepare_avx_pm`, `vmp_apply_dft_to_dft_avx`, `vmp_apply_dft_to_dft_accumulate_avx`); the hot apply path streams one prime plane at a time and reuses extracted input rows across the output-column loop.
- Reorganize backend implementations around `hal_impl` modules and `hal_defaults` to mirror the new per-family HAL extension surface and reduce duplication.
- Remove legacy per-family FFT64/NTT120 modules; route implementations through the new HAL defaults to keep a single source of truth.
- Update FFT64/NTT120 reference kernels, normalization, and shift helpers to keep behavior aligned with the new dispatch path.
- Flatten AVX test module paths to remove redundant crate prefixes.
- Split backend code into family-specific `hal_impl/*` modules (module/scratch/vec_znx/vmp/svp/convolution) for clearer override points.
- Export FFT-table types needed by the new CKKS encoder API: `poulpy-cpu-ref::FFT64ReimTable` and `poulpy-cpu-avx::FFT64AvxReimTable`.
- Move the runnable CKKS `poly2` example and reusable CKKS backend tests into `poulpy-cpu-ref`; add `poulpy-cpu-avx`'s opt-in `enable-ckks` feature so accelerated backends can wire in the CKKS layer without making it an unconditional dependency.

### `poulpy-cpu-avx512`
- Add `FFT64Avx512` — f64 complex-FFT backend gated on `enable-avx512f`; mixes AVX-512F REIM butterflies with AVX2+FMA REIM4 vec-mat kernels.
- Add `NTT120Avx512` — Q120 NTT backend with CRT over four ~30-bit primes (Primes30), gated on `enable-avx512f`; AVX-512F NTT butterflies with `nn=4` cross-block pair-pack and 2× unrolled NTT / mat-vec kernels. The row-prime-major VMP and AVX-512F convolution kernels override the cpu-ref defaults at the `HalVmpImpl` / `HalConvolutionImpl` level, and the `vec_znx_idft_apply_tmpa` hook uses a fused iNTT + Garner CRT compaction kernel.
- Add `NTT126Ifma` — Q126 NTT backend with CRT over three ~42-bit primes (Primes42), accelerated with AVX-512-IFMA; gated on `enable-ifma` and requires AVX-512F + AVX-512-IFMA + AVX-512-VL + BMI2 + ADX. Implements every HAL family directly against IFMA-specialized kernels (NTT/INTT, BBC mat-vec, VMP including row-prime-major + `vmp_apply_dft_to_dft_accumulate`, SVP, VecZnxDft, convolution). The post-iNTT 3-prime CRT-to-i128 reconstruction is a hand-written assembly kernel fusing IFMA Garner reduction with a BMI2/ADX scalar carry chain; the `vec_znx_idft_apply_tmpa` hook uses the same fused kernel.
- Build configuration: `enable-avx512f` and `enable-ifma` fail the build with a clear `compile_error!` if the matching CPU target features are not enabled, rather than emitting binaries that SIGILL at runtime. When neither feature is set, the crate compiles to an empty shell so non-AVX-512 hosts (including macOS ARM) keep building.
- Opt-in `enable-ckks` feature mirrors `poulpy-cpu-avx`: the three AVX-512 backends pick up the CKKS evaluator defaults via `impl_ckks_*_defaults!` and run the full `poulpy-ckks::test_suite` against `FFT64Avx512`, `NTT120Avx512` (f64), and `NTT126Ifma` (f64 and f128).

### `poulpy-ckks`
- Implement a fully backend-generic leveled CKKS evaluator: all operations (add, sub, mul, rotate, conjugate, rescale, encryption, decryption, and plaintext-polynomial ops) are now generic over any backend implementing `poulpy-hal`, including `FFT64Ref`, `FFT64Avx`, `NTT120Ref`, and `NTT120Avx`.
- Organize the public interface into the same four-module layered architecture as `poulpy-core`: `api` (user-facing traits), `oep` (backend extension points), `delegates` (dispatch), and `default` (portable reference implementations). Backends opt into portable defaults via `impl_ckks_*_defaults!` macros or can override individual operations directly through OEP.
- **Breaking:** Collapse the previous plaintext family split (`CKKSPlaintextVecZnx`, `CKKSPlaintextVec`, `CKKSPlaintextConstZnx`, `CKKSPlaintextConst`, and the old conversion traits) into a unified `CKKSPlaintext<D>` plus `CKKSPlaintextVecHostCodec<F>` for host float encode/decode and `CKKSModuleAlloc` for module-first plaintext/ciphertext allocation.
- Add first-class `api` trait families for CKKS copy, affine/composite helpers, and imaginary-unit operations, so backends can inherit or override those evaluator entrypoints independently.
- Add CKKS-local normalization typestate: `CKKSCiphertext<D, S = Normalized>` now tags normalized versus unnormalized values with `PhantomData`, and `UnnormalizedCKKSCiphertext<D>` aliases `CKKSCiphertext<D, Unnormalized>` for fused linear operations before an explicit `normalize`.
- Add `CKKSCiphertextViewMut` for in-place write patterns that avoid hot-path allocations in composite operations.
- Add scratch/layout helpers for the new evaluator surface: `ScratchArenaTakeCKKS`, backend-view aliases `CKKSCiphertextRef` / `CKKSCiphertextMut`, and new user-facing traits `CKKSAddOpsUnnormalized`, `CKKSSubOpsUnnormalized`, `CKKSAffineOps`, `CKKSImagOps`, and `CKKSCopyOps`.
- Move the CKKS conformance test suite into `poulpy-ckks/src/test_suite/` and wire it into `poulpy-cpu-ref` via `ckks_backend_test_suite!`; CI gains a dedicated focused step that runs the CKKS suite against every available backend.
- Remove direct concrete-backend dependencies from `poulpy-ckks` itself; backend crates now opt into CKKS integration from their side, keeping the crate package-level dependencies backend-agnostic as well as the API.
- Preserve the historical `crate::leveled::api` import paths as a backwards-compatible re-export shim; canonical paths are now `crate::api`, `crate::oep`, `crate::delegates`, and `crate::default`.
- **Breaking:** CKKS backend override wiring also moves to per-family traits/macros (`CKKSAddImpl`, `CKKSMulImpl`, `CKKSRotateImpl`, `impl_ckks_add_defaults!`, etc.) instead of the older aggregate default macro/export pattern.
- Document unnormalized operations with signed-digit behavior, worst-case O(n) growth, Irwin–Hall O(√n) typical growth, and the `n ≤ 2^(63 − base2k)` safety bound against i64 overflow.

### `poulpy-bin-fhe`
- Remove unnecessary `BE: 'static` bounds from blind-rotation and BDD-arithmetic trait signatures, and spurious `BE: 's` lifetime bounds from scratch-arena generic methods.
- Remove redundant `for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>` where clauses from public API methods, examples, and macros; the blanket impl always satisfies this constraint.
- Update bin-FHE BDD arithmetic, blind rotation, and test suites for the new core/HAL APIs.
- Refresh blind-rotation / circuit-bootstrapping staging helpers for the new `ScalarZnx` view API.
- Refresh scheme examples and library wiring to match the crate split and the new backend-generic APIs.
- **Note:** `poulpy-bin-fhe` is not yet backend-agnostic: it still depends unconditionally on `poulpy-cpu-ref` and exposes host `Vec<u8>` / `HostBackend` bounds in several public APIs. Full backend-agnosticity for this crate is deferred to a follow-up.
- **Breaking:** Bin-FHE traits and helpers now follow the backend-owned core/HAL surface: methods take `ScratchArena<'_, BE>`, use `...ToBackendRef<BE>` / `...ToBackendMut<BE>` bounds for ciphertexts and prepared keys, and many generic entrypoints now require `BE: Backend<OwnedBuf = Vec<u8>>` plus `ModuleCoreAlloc`.
- Move public constructors/allocation helpers to module-first forms across the crate: `FheUint::alloc[_from_infos](module, ...)`, `LookupTable::alloc(module, ...)`, `GLWEBlindRetriever::alloc(module, ...)`, and `CircuitBootstrappingKey::alloc_from_infos(module, ...)`.
- Add `LookupTable::to_backend` for explicit backend transfer of LUT storage and keep prepared blind-rotation / circuit-bootstrapping factories on backend-owned output types via `ScratchArena`.
- Align bin-FHE key/prepared layouts and circuit helpers with the refactored core layouts.
- Add `ReaderFrom` / `WriterTo` for `CircuitBootstrappingKey` and `BDDKey<Vec<u8>>` (optional `ks_glwe` encoded with a presence tag), with stable ATK map serialization (sorted Galois keys).

### `poulpy-bench`
- Fix `VecZnxAutomorphismAssignBackend` rename in automorphism benchmark helpers (was `VecZnxAutomorphismAssign`).
- Remove redundant `for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>` where clauses from CKKS and scheme benchmark helpers.
- Fix `lwe.fill_uniform` call site (was `lwe.data_mut().fill_uniform`) in blind-rotation bench staging.
- Update core and HAL convolution benchmarks to the new convolution API.
- Align benchmark suites with the new HAL/core APIs and update parameter examples.
- Remove remaining direct layout-field assumptions from benchmark staging helpers.
- Add shared host-upload/randomization helpers and `ModuleTransfer`-based typed uploads so benchmark fixtures can be staged on arbitrary backends without reaching into raw layout internals.
- Make CKKS benchmarks opt-in behind a new `ckks-bench` feature and gate the CKKS bench targets with `required-features`, so default bench runs do not pull CKKS support unless requested.
- Split benchmark opt-ins by family (`hal-bench`, `core-bench`, `bin-fhe-bench`, and `ckks-bench`) instead of gating every benchmark target behind one monolithic bench feature.
- Add `enable-avx512f` and `enable-ifma` bench features that pull `poulpy-cpu-avx512` into the workspace bench targets.
- Export new benchmark-support helpers for backend-generic staging: `upload_host_*`, `random_host_*`, `random_backend_*`, and `*_backend_ref/mut` adapters for raw HAL/core objects.

### Build & Docs
- Refresh root and crate READMEs (naming, examples, links, and architecture guidance); document the shared `api` / `oep` / `delegates` / `default` layering and backend-integration flow across the workspace.
- Extend CI with dedicated CKKS-focused `poulpy-cpu-ref` test steps in both AVX-enabled and portable configurations.
- Add a dedicated `avx512` CI lane that type-checks and clippies the AVX-512F + IFMA configuration on every push; tests are NOT run there (GitHub-hosted runners lack AVX-512 silicon, so executing the AVX-512 suite requires a self-hosted runner).
- Add a hugepage hint in the aligned allocator: on Linux, allocations ≥ 2 MB issue `madvise(MADV_HUGEPAGE)` before the zero-fill, reducing TLB pressure on large NTT/VMP working sets (~5% measured on `ntt120-avx` at large rings; FFT64 paths within noise). The threshold is overridable via the `POULPY_HUGEPAGE_MIN_BYTES` environment variable.
- Move higher-level feature gating to backend-owned integration features while keeping scheme crate APIs available when their crates are imported; update CI to enable the backend and benchmark feature set explicitly.
- Add acknowledgements for PZ, EF, and ENS in the root README.

### Fixes
- Avoid under-allocating scratch space in bin-FHE scheme tests via new FheUint/BDD tmp-bytes helpers.
- Make AVX backend optional (`enable-avx`) to prevent build failures on non-AVX machines.

### Migration (before/after)

**HAL backend wiring** moved from a single monolithic OEP trait to per-family OEP traits (`HalModuleImpl`, `HalVecZnxImpl`, `HalVmpImpl`, `HalConvolutionImpl`, `HalSvpImpl`, `HalVecZnxBigImpl`, `HalVecZnxDftImpl`). `poulpy-cpu-ref` exposes per-family `hal_impl_*!` macros and `*Defaults` traits so accelerated backends opt into the reference scalar path for cold methods and override only the hot ones.

Before (single OEP entrypoint):

```rust
use poulpy_hal::oep::HalImpl;

unsafe impl HalImpl<FFT64Avx> for FFT64Avx {
    hal_impl_vec_znx!();
    hal_impl_module_fft64!();
    // ...
}
```

After (per-family OEPs with shared defaults):

```rust
use poulpy_hal::oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl};
use poulpy_cpu_ref::hal_defaults::{FFT64ModuleDefaults, FFT64VmpDefaults, /* ... */};

unsafe impl HalVecZnxImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}
unsafe impl HalModuleImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefaults);
}
unsafe impl HalVmpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefaults);
}
// ...one impl per family; override individual methods inline when a backend has a faster kernel.
```

**Core / CKKS backend wiring**: the old monolithic `CoreImpl` + `impl_core_default_methods!` macro is replaced by per-family `impl_*_defaults_full!` macros under `poulpy-core` (and `impl_ckks_*_defaults!` under `poulpy-ckks`).

Before (single core entrypoint):

```rust
use poulpy_core::oep::{CoreImpl, impl_core_default_methods};

unsafe impl CoreImpl<MyBackend> for MyBackend {
    impl_core_default_methods!(MyBackend);
}
```

After (per-family `_defaults_full!` macros):

```rust
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full,
    impl_gglwe_keyswitch_defaults_full, impl_ggsw_automorphism_defaults_full,
    impl_ggsw_external_product_defaults_full, impl_ggsw_keyswitch_defaults_full,
    impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full,
    impl_glwe_trace_defaults_full, impl_lwe_keyswitch_defaults_full,
};

impl_glwe_automorphism_defaults_full!(MyBackend);
impl_glwe_keyswitch_defaults_full!(MyBackend);
// ...one macro per family; override any method by writing it after the macro call.
```

## [0.5.0] - 2026-03-31

### `poulpy-bench` (new crate)
- Consolidate all benchmark suites into a single `poulpy-bench` crate; remove `bench_suite` modules from `poulpy-hal`, `poulpy-core`, and `poulpy-schemes`.
- Organize bench suite under three namespaces: `bench_suite::hal`, `bench_suite::core`, `bench_suite::schemes`.
- Add `standard` binary: one representative run across all layers with fixed parameters, intended for version-to-version regression tracking.
- Add JSON-configurable benchmark parameters via the `POULPY_BENCH_PARAMS` environment variable (file path or inline JSON). All sweep ranges and layout constants are overridable; any omitted field falls back to its built-in default.
  - `hal.sweeps` — `[log_n, cols, size]` points for `vec_znx_big`, `vec_znx_dft`, `svp`
  - `cnv.sweeps` — `[log_n, size]` points for `convolution`
  - `vmp.sweeps` — `[log_n, rows, cols_in, cols_out, size]` points for `vmp`
  - `svp_prepare.log_n` — ring degrees for SVP prepare
  - `core.{n, base2k, k, rank, dsize}` — layout for all core / scheme / standard benchmarks
- Add `run` JSON field: list of bench binary names or function names to execute; binary names run the whole binary, function names are applied as a Criterion regex filter across the default binary set.
- Add `backends` JSON field: list of backend labels (`fft64-ref`, `ntt120-ref`, `fft64-avx`, `ntt120-avx`) to restrict which backends are benchmarked; listing an AVX backend automatically enables `--features enable-avx` and sets `RUSTFLAGS="-C target-feature=+avx2,+fma"`.
- Replace per-group `measurement_time` overrides with a shared `criterion_config()` (100 samples, 5 s measurement budget).
- Add `examples/custom_params.json` and `examples/run_custom_params.sh`: runnable example demonstrating JSON-configurable parameters, backend selection, operation filtering, and baseline comparison.

### `poulpy-hal`
- Remove `VmpApplyDftToDftAdd` and `SvpApplyDftToDftAdd` traits; merge additive variant into `VmpApplyDftToDft` / `SvpApplyDftToDft` via a new `limb_offset` parameter.
  These traits accumulated VMP results directly into a scattered output buffer, causing severe cache misses. Writing into a contiguous temporary buffer and folding with `VecZnxDftAddAssign` is ~2× faster.
- Remove all associated OEP (`VmpApplyDftToDftAddImpl`, `VmpApplyDftToDftAddTmpBytesImpl`, `SvpApplyDftToDftAddImpl`), delegate, and bench-suite plumbing.
- Add family defaults for `vec_znx_big`, `vec_znx_dft`, `svp_ppol`, `vmp_pmat`, and `convolution`.
- Add portable defaults for `scratch` and `vec_znx` in `HalImpl`, reducing backend boilerplate.
- Remove legacy OEP traits for `vec_znx`, `vec_znx_big`, `vec_znx_dft`, `svp_ppol`, `vmp_pmat`, and `convolution`; use `HalImpl` + defaults instead.

### `poulpy-cpu-ref` / `poulpy-cpu-avx`
- Update FFT64 and NTT120 `vmp_apply_dft_to_dft` implementations to accept `limb_offset` directly, replacing the separate `_add` codepath.
- NTT120 AVX2 (`arithmetic_avx.rs`): add `reduce_b_and_apply_crt` that fuses the CRT multiply into the Barrett reduction pass, using new compile-time constants `POW32_CRT` and `POW16_CRT`; apply to `compact_all_blocks` to reduce instruction count by a factor of ~2x.
- Drop legacy backend-specific VMP/Convolution OEP impl modules; rely on HAL family defaults.
- Drop legacy backend-specific `scratch`/`vec_znx` impl modules and FFT64 `vec_znx_big` impls; NTT120 `vec_znx_big` now only provides the i128 ops hooks for HAL defaults.
- Drop legacy backend-specific `svp` impl modules; rely on HAL family defaults.
- Remove legacy `vec_znx_dft` OEP traits; use `HalImpl` family defaults instead.

### `poulpy-core`
- Rewrite external product (`glwe_external_product_internal`) and GLWE keyswitching inner loops to write intermediate per-digit VMP results into a dedicated temporary buffer before accumulating with `VecZnxDftAddAssign`, avoiding scattered-write cache thrashing. `where` bounds updated accordingly.
- Add `bench_suite::keyswitch::gglwe` module and `keyswitch_glwe` criterion benchmark targeting the NTT120 backend; remove the old FFT64-specific `keyswitch_glwe_fft64` benchmark.

## [0.4.4] - 2026-02-28

### `poulpy-hal`
- Add NTT120 reference primitives: primes, types, arithmetic, NTT butterfly, mat-vec, SVP, VMP, `VecZnxBig`, `VecZnxDft`, and convolution.
- Refactor byte size helpers: centralize scratch/layout size computations into `Module`.
- Consolidate FFT64 trait implementations to eliminate duplication between ref and AVX.

### `poulpy-cpu-ref`
- Add `NTT120Ref` backend: scalar Q120 NTT over CRT of four ~30-bit primes.
  - Full OEP coverage: `VecZnx`, `VecZnxBig`, `VecZnxDft`, `SvpPPol`, `VmpPMat`.
- Reorganize FFT64 sources into `fft64/` submodule.
- Add NTT120 benchmarks.

### `poulpy-cpu-avx`
- Add `NTT120Avx` backend: AVX2-accelerated NTT120.
  - AVX2 NTT butterfly with variable-shift accumulation.
  - AVX2 BBC mat-vec (`NttMulBbc`, x2, 2-column variants).
  - AVX2 arithmetic primitives: `b_from_znx64`, `c_from_b` (Barrett), `vec_mat1col_product_bbb`, `b_to_znx128` (hybrid AVX2/scalar CRT).
  - AVX2 `VecZnxBig` accumulation and normalization.
- Reorganize FFT64 sources into `fft64/` submodule.
- Add NTT120 benchmarks and unit tests for all AVX subroutines.

### `poulpy-core`
- Add NTT120 backend support across all operations: encryption, decryption, automorphisms, external products, keyswitching, and noise analysis.
- Extend test suite to cover `NTT120Ref` and `NTT120Avx`.

## [0.4.3] - 2026-01-16

- Fix [#131](https://github.com/poulpy-fhe/poulpy/issues/131)
- Fix [#130](https://github.com/poulpy-fhe/poulpy/issues/130)

## [0.4.2] - 2025-12-21

### `poulpy-core`
- Add `GLWEMulPlain` trait:
  - `glwe_mul_plain_tmp_bytes`
  - `glwe_mul_plain`
  - `glwe_mul_plain_assign`
- Add `GLWEMulConst` trait:
  - `glwe_mul_const_tmp_bytes`
  - `glwe_mul_const`
  - `glwe_mul_const_assign`
- Add `GLWETensoring` trait:
  - `glwe_tensor_apply_tmp_bytes`
  - `glwe_tensor_apply`
  - `glwe_tensor_relinearize_tmp_bytes`
  - `glwe_tensor_relinearize`
- Add method tests:
  - `test_glwe_tensoring`

### `poulpy-hal`
- Removed `Backend` generic from `VecZnxBigAllocBytesImpl`.
- Add `CnvPVecL` and `CnvPVecR` structs.
- Add `CnvPVecBytesOf` and `CnvPVecAlloc` traits.
- Add `Convolution` trait, which regroups the following methods:
  - `cnv_prepare_left_tmp_bytes`
  - `cnv_prepare_left`
  - `cnv_prepare_right_tmp_bytes`
  - `cnv_prepare_right`
  - `cnv_by_const_apply`
  - `cnv_by_const_apply_tmp_bytes`
  - `cnv_apply_dft_tmp_bytes`
  - `cnv_apply_dft`
  - `cnv_pairwise_apply_dft_tmp_bytes`
  - `cnv_pairwise_apply_dft`
- Add the following Reim4 traits:
  - `Reim4Convolution`
  - `Reim4Convolution1Coeff`
  - `Reim4Convolution2Coeffs`
  - `Reim4Save1BlkContiguous`
- Add the following traits:
  - `i64Save1BlkContiguous`
  - `i64Extract1BlkContiguous`
  - `i64ConvolutionByConst1Coeff`
  - `i64ConvolutionByConst2Coeffs`
- Update signature `Reim4Extract1Blk` to `Reim4Extract1BlkContiguous`.
- Add fft64 backend reference code for 
  - `reim4_save_1blk_to_reim_contiguous_ref`
  - `reim4_convolution_1coeff_ref`
  - `reim4_convolution_2coeffs_ref`
  - `convolution_prepare_left`
  - `convolution_prepare_right`
  - `convolution_apply_dft_tmp_bytes`
  - `convolution_apply_dft`
  - `convolution_pairwise_apply_dft_tmp_bytes`
  - `convolution_pairwise_apply_dft`
  - `convolution_by_const_apply_tmp_bytes`
  - `convolution_by_const_apply`
- Add `take_cnv_pvec_left` and `take_cnv_pvec_right` methods to `ScratchTakeBasic` trait.
- Add the following tests methods for convolution:
  - `test_convolution`
  - `test_convolution_by_const`
  - `test_convolution_pairwise`
- Add the following benches methods for convolution:
  - `bench_cnv_prepare_left`
  - `bench_cnv_prepare_right`
  - `bench_cnv_apply_dft`
  - `bench_cnv_pairwise_apply_dft`
  - `bench_cnv_by_const`
- Update normalization API and OEP to take `res_offset: i64`. This allows the user to specify a bit-shift (positive or negative) applied to the normalization. Behavior-wise, the bit-shift is applied before the normalization (i.e. before applying mod 1 reduction). Since this is an API break, opportunity was taken to also re-order inputs for better consistency.
  - `VecZnxNormalize` & `VecZnxNormalizeImpl`
  - `VecZnxBigNormalize` & `VecZnxBigNormalizeImpl`
  This change completes the road to unlocking full support for cross-base2k normalization, along with arbitrary positive/negative offset. Code is not ensured to be optimal, but correctness is ensured. 

### `poulpy-cpu-ref`
- Implemented `ConvolutionImpl` OPE on `FFT64Ref` backend.
- Add benchmark for convolution.
- Add test for convolution.

### `poulpy-cpu-avx`
- Implemented `ConvolutionImpl` OPE on `FFT64Avx` backend.
- Add benchmark for convolution.
- Add test for convolution.
- Add fft64 AVX code for
  - `reim4_save_1blk_to_reim_contiguous_avx`
  - `reim4_convolution_1coeff_avx`
  - `reim4_convolution_2coeffs_avx`

## [0.4.1] - 2025-11-21
- Default backend set to `poulpy-cpu-ref`, `poulpy-cpu-avx` is not anymore built and compiled by default.
- To build & use `poulpy-cpu-avx` user must use feature flag, see `poulpy-cpu-ref` and `poulpy-cpu-avx` READMEs.

## [0.4.0] - 2025-11-20

### Summary
- Full support for base2k operations.
- Many improvements to BDD arithmetic.
- Removal of **poulpy-backend** & spqlios backend.
- Addition of individual crates for each specific backend.
- Some minor bug fixes.

### `poulpy-hal`
- Add cross-base2k normalization

### `poulpy-core`
- Add full support for automatic cross-base2k operations & updated tests accordingly.
- Updated noise helper API.
- Fixed many tests that didn't assess noise correctly.
- Fixed decoding function to use arithmetic rounded division instead of arithmetic right shift.
- Fixed packing to clean values correctly.

### `poulpy-schemes`
- Renamed `tfhe` crate to `bin_fhe`.
- Improved support & API for BDD arithmetic, including multi-thread acceleration.
- Updated crate to support cross-base2k operations.
- Add additional operations, such as splice_u8, splice_u16 and sign extension.
- Add `GLWEBlindRetriever` and `GLWEBlindRetrieval`: a `GGSW`-based blind reversible retrieval (enables to instantiate encrypted ROM/RAM like object).
- Improved Cmux speed
- Added `sign` argument to GGSW-based blind rotation, which enables to choose the rotation direction of the test vector.

### `poulpy-cpu-ref`
- A new crate that provides the reference CPU implementation of **poulpy-hal**. This replaces the previous **poulpy-backend/cpu_ref**.

### `poulpy-cpu-avx`
- A new crate that provides an AVX/FMA accelerated CPU implementation of **poulpy-hal**. This replaces the previous **poulpy-backend/cpu_avx**.

## [0.3.2] - 2025-10-27

### `poulpy-hal`
- Improved convolution functionality

### `poulpy-core`
 - Rename `GLWEToLWESwitchingKey` to `GLWEToLWEKey`.
 - Rename `LWEToGLWESwitchingKey` to `LWEToGLWEKey`.
 - Add `GLWESecretTensor` which stores the flattened upper right of the tensor matrix of the pairs  `sk[i] * sk[j]`.
 - Add `GGLWEToGGSWKey`, `GGLWEToGGSWKeyPrepared`, `GGLWEToGGSWKeyCompressed`, which encrypts the full tensor matrix of all pairs `sk[i] * sk[j]`, with one `GGLWE` per row.
 - Update `GGLWEToGGSW` API to take `GGLWEToGGSWKey` instead of the `GLWETensorKey`
 - Add `GLWETensor`, the result of tensoring two `GLWE` of identical rank.
 - Changed `GLWETensorKey` to be an encryption of `GLWESecretTensor` (preliminary work for `GLWEFromGLWETensor`, a.k.a relinearization). 

### `poulpy-schemes`
 - Add `GLWEBlindRotation`, a `GGSW`-based blind rotation that evaluates `GLWE <- GLWE * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.` (`k` = `FheUintBlocksPrepared`).
 - Add `GGSWBlindRotation`, a `GGSW`-based blind rotation that evaluates `GGSW <- (GGSW or ScalarZnx) * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.` (`k` = `FheUintBlocksPrepared`).

## [0.3.1] - 2025-10-24

### `poulpy-hal`
 - Add bivariate convolution (X, Y) / (X^{N} + 1) with Y = 2^-K

### `poulpy-core`
 - Fix typo in the shared backend-view impl for `GLWEAutomorphismKey` that incorrectly required mutable data.

## [0.3.0] - 2025-10-23

- Fixed builds on MACOS

### Breaking changes
 - The changes to `poulpy-core` required to break some of the existing API. For example the API `prepare_alloc` has been removed and the trait `Prepare<...>` has been broken down for each different ciphertext type (e.g. GLWEPrepare). To achieve the same functionality, the user must allocated the prepared ciphertext, and then call prepare on it.

### `poulpy-hal`
 - Added cross-base2k normalization

### `poulpy-core`
 - Added functionality-based traits, which removes the need to import the low-levels traits of `poulpy-hal` and makes backend agnostic code much cleaner. For example instead of having to import each individual traits required for the encryption of a GLWE, only the trait `GLWEEncryptSk` is needed.

### `poulpy-schemes`
 - Added basic framework for binary decision circuit (BDD) arithmetic along with some operations.

## [0.2.0] - 2025-09-15

### Breaking changes
 - Updated the trait `FillUniform` to take `log_bound`.

### `poulpy-hal`
 - Added pure Rust reference code for `vec_znx` and `fft64` backend.
 - Added cross-backend generic test suite along with macros.
 - Added benchmark generic test suite.

### `poulpy-backend`
 - Added `FFTRef` backend, which provides an implementation relying on the reference code of `poulpy-hal`.
 - Added `FFTAvx` backend, which provides a pure Rust AVX/FMA accelerated implementation of `FFTRef` backend.
 - Added cross-backend tests between `FFTRef` and `FFTAvx`.
 - Added cross-backend tests between `FFTRef` and `FFT64Spqlios`.

### `poulpy-core`
 - Removed unsafe blocks.
 - Added tests suite for `FFTRef` and `FFTAvx` backends.

### Other
 - Fixed a few minor bugs.

## [0.1.0] - 2025-08-25
 - Initial release.
