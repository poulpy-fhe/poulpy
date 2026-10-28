# poulpy-bin-fhe review rules

## 1. Layering and files

- **What a new operation needs.** A new operation also brings:
  - when it samples, a `bin_fhe_reference_test_suite!` body;
  - its API trait in the operation map of `schemes/poulpy-bin-fhe/docs/bin-fhe-contracts.md`;
  - for an execution API, its calls in the `OpaqueBackend<'env>` contract.
- **Operation traits live in `api`.** Every replaceable operation, LUT construction included, is an API trait in a private `api/<family>/` module re-exported from `api` and the crate root, re-exported by a domain module (`blind_rotation`, `circuit_bootstrapping`, `bdd_arithmetic`) only to keep a path older than its move to `api`. Blanket `Module<BE>` impls outside `delegates` are only data helpers over public API traits (allocation, debug and noise inspection, serialization).
- **No unsafe blocks.** `unsafe` appears only on the `pub unsafe trait` OEP declarations and the `unsafe impl`s of opt-ins and backend overrides.

### 1.1. Opt-in macros

- **Group umbrellas.** Each OEP trait's opt-in is listed in its group umbrella `impl_<group>_reference_full!`, which `impl_bin_fhe_reference_full!` composes.
- **Marker and integer opt-ins.** Over the algorithm marker an opt-in takes `($be:ty, $algo:ty)` and implements `XImpl<$algo>`, over the integer type one `unsafe impl<T: UnsignedInteger> XImpl<T> for $be`, never per width, and a keyed blind-rotation opt-in implements `XImpl<CGGI>`.
- **Expansions add no names.** An opt-in compiles without imports at the expansion site, its `use` declarations and `unsafe impl` scoped in `const _: () = { .. };` or every path spelled in full.

### 1.2. Compositions

- **Compositions are derived defaults.** Serial forms and their queries (calling the `_multi_thread` form with `threads = 1`), per-GGSW-row loops and direct circuit-bootstrapping entry points (prepare, execute) are each a `<api method>_derived` behind a one-line default body.
- **Default-method bounds.** The default methods of an OEP trait whose defaults dispatch to another poulpy-bin-fhe family carry only their API method's where-clauses.
- **Reference stays behind the macros.** Only `reference`, the opt-in macro bodies and tests call into `crate::reference`.
- **Convenience methods forward.** A convenience method on a bin-fhe container or `define_bdd_*_trait!` trait forwards in one line to a public API trait bound as `M: <ApiTrait>`, passing `self` in the operand position, and multi-operation helpers (`FheUint::get_bit_lwe`) call public API traits only.
- **Stateful helpers compose.** A stateful helper (`GLWEBlindRetriever`) composes selected public operations without an OEP hook and sizes its query methods from theirs at its persistent layout, valid before any input.

## 2. Naming and signatures

- **Generic parameter order.** API traits take the algorithm marker or integer type first and `BE` last (`X<BRA, BE>`), host-storage traits storage and word (`LookupTableFactory<D: Data, W: ZnxWord>`), and layouts `<D: Data, Marker, W: ZnxWord>`, or `<D: Data, Marker, BE: Backend>` when prepared.
- **Pin `i64` only where needed.** `Backend<ZnxWord = i64>` bounds only items handling message bits or rotation indices as `i64`, at method level in a word-generic trait.
- **No `'static`, shared data `Sync`.** Contracts, reference functions and tests require no `'static` on the backend, key providers or `UnsignedInteger`; worker-read data is `Sync` (`GetGGSWBit`, `BitCircuitInfo`, `GetBitCircuitInfo`, `UnsignedInteger: Copy + Sync + Send`, `Module<BE>`), worker outputs are `Send`, and a new provider or circuit trait keeps these supertraits.
- **Operand forms.** Ciphertexts are backend views (`R: GLWEToBackendMut<BE> + GLWEInfos`, `A: GLWEToBackendRef<BE>`, GGSW forms with `GGSWAtView*`), keys, LUTs and plans owned bin-fhe types, a one-bit selector a `&GGSWPreparedBackendRef<'_, BE>` and a multi-bit one a `K: GetGGSWBit<BE>`.
- **Argument order.** `docs/review/workspace.md` section 12 (argument order), with these orders within its groups:
  - BDD scalars `sign, bit_rsh, bit_mask, bit_lsh` (blind rotations all four, selection and retrieval `bit_rsh, bit_mask`);
  - `_multi_thread` variants: `threads: usize` first, then the serial parameters unchanged;
  - circuit-bootstrapping exponent scalars `log_gap_out, log_domain, extension_factor`;
  - the selector as the last operand (CMux and Cswap: the prepared selector), a circuit as a plan.
- **Key-encryption signatures.** A key encryption takes exactly `(res, sk_lwe, sk_glwe, source_xe, source_xa, scratch)`, its compressed form a root `seed_xa: [u8; 32]` before `source_xe` instead of `source_xa`.

## 3. Scratch and queries

- **Query shapes.** A query takes the scalar parameters (`threads`, `log_gap_out`, `log_domain`, `block_size`, `extension_factor`) before operand infos:
  - blind rotations: `(res, a, k)`, shared by the `_assign` form with `res` as `a`;
  - `cmux`, `cmux_assign` and `cmux_assign_neg`: one shared `cmux_tmp_bytes(res, a, selector)`;
  - selection: the destination, every candidate's infos at actual capacity, the selector;
  - word-level BDD executors: `(res_infos, ggsw_infos, circuit, key)`;
  - streaming retrieval: the maximum of `add_tmp_bytes` per input layout and `flush_tmp_bytes` for the destination.
- **Queries at the actual layouts.** A query sizes caller buffers at actual capacity, temporaries as carved, both directions of each copy and alternating CMux, and every distinct ordered pair of selection inputs, deduplicated by `n`, `base2k`, `k`, `rank`, `max_size` and `size`.
- **Queries take metadata only.** A query takes infos, a `CircuitBootstrappingPlanLayout`, a circuit description or a key provider read only for its key layouts, never a LUT, plan or ciphertext, allocates nothing, and receives `block_size` and `extension_factor` as arguments, never via a generic `CircuitBootstrappingKeyInfos::block_size()`, a body reading `block_size` from the prepared key.

## 4. Keys, LUTs and containers

- **Bundle components move together.** A key-bundle component change also updates in one diff encryption, the encryption and preparation queries (maxima over all components), and `from_parts` with the part accessors.
- **Fields private, parts exposed.** Only `*Layout` descriptors, `BitCircuit` and `Circuit` have `pub` fields, and a container an OEP method fills or reads exposes each new field via `from_parts` with `parts` and `parts_mut`, or `x()` with `x_mut()` or `set_x`, getters alone only on an object immutable after construction (`CircuitBootstrappingPlan`).
- **Fresh keys are unfilled.** Every blind-rotation key allocator returns a key with no derived tables, only encryption, decompression, preparation and deserialization set `dist`, and execution rejects `NONE` with a static panic.
- **Re-preparation leaves no stale state.** Re-preparing an object rebuilds each optional table only when the source needs it, else `None` (the block-binary monomial table exists if and only if `dist` is `BinaryBlock`), and a new optional field gets a parity case re-preparing one destination from a source needing it, then one not.
- **Transfers carry every field.** `TransferInto` shape checks cover counts, optional parts, Galois set, LUT `base2k` and `k`; decompression and preparation also carry every scalar field (`dist`, LUT `drift` and direction), a new one tested like `test_lut_transfer_into_carries_scalars`.
- **Sorted Galois maps.** A loop over a `HashMap<i64, _>` of automorphism keys that draws, prepares, serializes or picks a layout runs in ascending element order (`keys().sorted()`, `keys().min()` for a layout read), over elements only from `trace_galois_elements`.
- **LUT construction.** A `LookupTableFactory` uploads host-staged samples with `BE::copy_from_host`, normalizes and rotates through HAL operations, splits an `extension_factor * n` domain over module-degree polynomials and keeps `f.len()` within the module degree, `LookupTable::alloc` and every consumer require a nonzero power-of-two extension factor, and blind rotation asserts that the LUT radix equals the key's and the output's. Only the factory allocates its own setup arena.
- **LUT drift and direction.** A factory override records the half-step pre-rotation with `set_drift` and keeps the caller's rotation direction, which consumers read back (`drift`, `rotation_direction()`) instead of recomputing.

### 4.1. Serialization

- **Serialized types.** Only raw ciphertexts and the raw keys `BlindRotationKey`, `BlindRotationKeyCompressed`, `CircuitBootstrappingKey` and `BDDKey` implement `WriterTo` and `ReaderFrom`.
- **Key wire format.** An optional component is a `u8` tag 0 or 1 then its payload, and a Galois map a count then `(element, key)` pairs in ascending order.

### 4.2. Key encryption

- **Key encryption workspace.** A key encryption's query is its component queries' maximum plus the live prepared GLWE secret, and it carves every secret (prepared GLWE secret, LWE-secret plaintexts, bridge secret) from the arena.
- **Composite draw order.** Composite key encryptions draw in this order, checked in `bin_fhe_reference_test_suite!`:
  - circuit-bootstrapping keys: automorphism keys by ascending Galois element, blind-rotation key, tensor key;
  - BDD keys: optional bridge `ks_glwe` (its intermediate secret first, from `source_xe`), GLWE-to-LWE key, circuit-bootstrapping key.

## 5. Blind-rotation algorithms and scheduling

- **Algorithms are type markers.** Each blind-rotation algorithm is a zero-sized `BlindRotationAlgo` marker with its own `alloc_key` and opt-ins, the `BRA` parameter of keys and keyed traits, selecting the algorithm only, never backend, storage or scheduling, with no runtime switch in the `CGGI`-typed bodies.
- **Fixed-order reduction.** Both blind-rotation schedules share one body that reduces worker contributions in fixed index order.
- **Multi-thread operations.** A BDD operation over independent items has a `_multi_thread` form as its only reference form, with `<op>_multi_thread_tmp_bytes(threads, ..)` where callers need a total, and outputs bit-identical for every `threads` and executor.
- **Per-worker scratch.** A per-worker query takes no `threads` and returns its bytes rounded by `worker_scratch_bytes::<BE>`.
- **Shared constants built once.** Data every worker or output needs (the BDD trivial one, a plan) is built once before `for_each_with_scratch` and shared by reference.

## 6. BDD arithmetic and generated circuits

- **Generated circuits are regenerated.** Circuit data in `bdd_arithmetic/circuits/u<w>/<op>_codegen.rs` changes only as regenerated, crate-private output shaped like its siblings, the PR stating its provenance. Hand edits there are only mechanical (imports, trait boilerplate, rustfmt).
- **Adding a word operation.** A new BDD word operation adds in one diff:
  - `circuits/u<w>/<op>_codegen.rs` with its `pub(crate) mod` line in `circuits/u<w>/mod.rs`;
  - a `define_bdd_2w_to_1w_trait!` or `define_bdd_1w_to_1w_trait!` invocation with a doc;
  - one `impl_bdd_*_trait!(Trait, method, u<w>, ..::AnyBitCircuit, ..::OUTPUT_CIRCUITS)` line per supported width;
  - `test_bdd_<op>` in `src/test_suite/bdd_arithmetic/<op>.rs`, declared and glob re-exported in its `mod.rs`, with `#[test] fn bdd_<op>()` in `bin_fhe_backend_test_suite!`;
  - an entry in the `bdd_arithmetic` module doc.
- **Node-array invariants.** A circuit fed to the BDD evaluator has levels of `max_inter_state` entries (0 for a zero output, else at least 2), an initial state holding constant 0 in slot 0 and constant 1 in slot 1, `Cmux(sel, hi, lo)` previous-level indices below `max_inter_state` and selectors below `input_size()`, `Copy` forwarding its own slot, a last level `[Cmux, None, ..]`, and two-word inputs `a` in `[0, BITS)`, `b` in `[BITS, 2 * BITS)`.
- **Bit placement.** Bit `i` of an `FheUint<T>` sits at coefficient `T::bit_index(i) << (module.log_n() - T::LOG_BITS)`, never `i`, byte `b` is reached through `T::bit_index(b << 3)`, byte-isolating traces start at `T::LOG_BITS - T::LOG_BYTES`, and the ring degree is a multiple of `T::BITS`.
- **Gates normalize once.** A CMux or Cswap style gate computes `glwe_external_product_dft` and adds the other branch in its one final normalization, never `glwe_external_product` plus `glwe_add`, reading an immutable branch in place.

## 7. Circuit bootstrapping

- **Plans record their parameters.** Plan preparation checks domain and gaps against the output layout and LUT coefficient overflow before encoding and records the output layout, key layouts and block size, an override building the plan with `CircuitBootstrappingPlan::from_lut` (which checks that LUT, key and output degrees agree, a nonzero block size and a power-of-two extension factor); every plan execution asserts its output and key against these records, and a query sizing it checks the key first.
- **Setup stays out of execution.** Prepared-plan execution uses only the plan, prepared key, operands and scratch, only the one-shot `circuit_bootstrapping_execute_to_constant` and `circuit_bootstrapping_execute_to_exponent` (and their prepared-key convenience forms) build or upload a LUT or plan, and a loop over inputs sharing output layout, domain and key prepares one plan (`prepare_to_constant`, `prepare_to_exponent`) before it.
- **One query path.** The direct queries derive from `circuit_bootstrapping_execute_prepared_tmp_bytes` over a `CircuitBootstrappingPlanLayout`, which a new execution mode extends, and `log_domain = None` bounds every domain at that output layout.

## 8. Host data versus execution

- **No host bounds on execution.** No item under `api/`, `delegates/`, `oep/` or `reference/`, nor an execution path they reach, takes an `AsRef<[u8]>`/`AsMut<[u8]>` bound, assumes `DFT_LIMBS_CONTIGUOUS` or stages through `HostBytesBackend`, host bounds staying on layout host helpers (plaintext packing, `noise_stats`, `FheUintPreparedDebug`, `FheUintPrepareDebug`, `Debug`, `Display`, serialization) and LUT samples and circuit descriptions being host data.
- **Host steps behind a hook.** A host step on an execution path (reading ciphertext coefficients) is its own OEP operation called through the module (`module.blind_rotation_mod_switch(..)`), never a direct `blind_rotation_mod_switch_reference` call.
- **The opaque-backend contract compiles.** Execution and factory traits, delegates, reference bodies and the core queries they reach stay callable on the borrowed, opaque, non-contiguous `OpaqueBackend<'env>` of `backends/poulpy-cpu-portable/src/tests/bin_fhe_overrides.rs`.

## 9. Degree, gadget products and validation

- **Operands at the module degree.** A body asserts that GLWE, GGSW, LUT and key operands have degree `module.n()` and LWEs the key's `n_lwe`, unless its first core call does, and allocators take the module and infos (`T::alloc[_from_infos](module, infos)`, `module.<object>_alloc[_from_infos](infos)`), asserting `infos.n() == module.n()`.
- **Narrow before a gadget product.** A body, CGGI included, narrows a gadget product's input with `set_k(min(k, max_input_k))` or lays it out at that `k` and sizes its query on the narrowed layout.
- **Composites check every component.** Preparation, decompression and execution of a composite check every part: layouts (`res.brk_infos() == other.brk_infos()`), counts, optional parts, Galois sets, LWE dimension, secret distribution, bit ranges, helper capacity, plans and LUTs.

## 10. Canonical-flag bypasses and metadata

- **A bypass feeds one reader.** A bypass sets the flag only on a local temporary or borrowed view fed straight to an external product or key switch.
- **A bypassed value never escapes.** The flag is set after any narrowing `set_k`, the value is normalized, or overwritten by a normalizing write, before anything copies it out, and a new bypass is tested through `glwe_decrypt_checked`.
- **Evaluation outputs are untagged.** Every evaluation output, the retriever flush and `FheUint` helpers included, gets `set_noise(None)` only where an input copy can reach the output or a data-dependent path can skip an element.

## 11. Tests and registration

- **Three suite macros.** Bodies register 1:1:
  - correctness bodies, in `src/test_suite/<domain>/`, in `bin_fhe_backend_test_suite!`;
  - `test_<op>_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)` bodies of `src/test_suite/parity/<family>.rs` in `bin_fhe_parity_test_suite!`;
  - same-backend, same-seed comparisons of randomized operations with their reference in `bin_fhe_reference_test_suite!`.

  Checks every backend type must run go in the latter two, which every opted-in type registers in its `bin_fhe_parity_*` module (backend side: `docs/review/backends.md`); only a host wire-format round trip is called directly, from `backends/poulpy-cpu-portable/src/tests.rs`.
- **Parity harness.** A parity body bounds both backends only by `ParityBackend` and its API traits, builds fixtures on `HostBytesBackend` (`fixture_glwe`, `fixture_ggsw`) and keys once on the reference side, transfers both, and compares `GlweSnapshot` or `GgswSnapshot` with the canonical flag after `assert_untagged`.
- **Parity sweeps.** A new operation's parity body sweeps the axes its query and circuit depend on: mixed radix and precision, spare capacity, sparse and empty inputs, assign and out-of-place forms, `BorrowedBits`, both rotation signs or directions, several thread counts and object reuse.
- **Correctness bodies.** Correctness bodies decrypt with `glwe_decrypt_checked`, refill setup arenas before each use (`source.fill_bytes(scratch.data.as_mut())`), and in BDD use the shared `&TestContext<BRA, BE>` and `TEST_*` layouts, each key at its own `base2k`.
- **Rejection tests.** A rejection test runs the refused call, `with_scratch` included, in `catch_unwind(AssertUnwindSafe(..))` and compares the destination (every prepared-key component, a retriever's state and output) with a prior snapshot, `#[should_panic(expected = "..")]` serving pure helpers only.
- **Unit tests for host helpers.** A new backend-free host helper's unit tests sweep its edge cases (signed lower limbs, exact ties, overflow bounds), anything needing a `Module` going into a registered suite.
