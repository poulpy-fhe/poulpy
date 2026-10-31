# poulpy-hal review rules

## 1. Layering and the op classes

- **No hal reference layer.** A new basis or variant op gets a portable kernel (`hal_defaults`, `hal_impl_*!`) and an independent poulpy-cpu-oracle implementation calling no portable kernel; poulpy-hal adds no `reference` module, `*_reference` function or reference macro.
- **Four op classes.** Every API method has one class fixing its OEP form:
  - `basis`: independently defined, required, no default body, tested across backends (samplers and the raw complex transform against their definitions);
  - `variant`: another storage or mutation form of a defined op (`_assign`, `ScalarZnx`, conjugate-invariant), required, except a layout conversion that re-views its operand as another layout and calls the defined op (`scalar_znx_automorphism`, `scalar_znx_ci_embed`), which is `derived`;
  - `derived`: a composition within its signature's scratch, defaulted over `oep::derived`;
  - `support`: allocation, `*_bytes_of`, `_tmp_bytes`, scratch, plans, module queries, no arithmetic.
- **Default bodies follow the class.** Only `derived` ops, their `_tmp_bytes` and `HalModuleImpl::synchronize` have default bodies.

## 2. API traits, naming and signatures

- **Backend parameter only when used.** An API trait takes a backend type parameter only when its signature names a backend type, never on `_tmp_bytes` or `<Layout>BytesOf` traits.
- **Declared where the checker sees.** An API trait is a plain `pub trait Name` at column 0 of a file under `poulpy-hal/src/api`, not macro-generated, `pub(crate)`, `pub unsafe` or cfg-gated, with methods `fn` or `unsafe fn`.
- **Operands are backend views.** API and OEP methods take layout operands as `&XBackendRef<'_, B>` or `&mut XBackendMut<'_, B>` (`R: XToBackendMut<B>` for a whole destination), and only `ReimFFT` and `ReimIFFT` take host slices.
- **Radix and precision are arguments.** Ops that read digits take `base2k` as a parameter; HAL layouts carry no radix or precision.
- **Degree first on allocation calls.** Functions with no polynomial operand to read a degree from (`<layout>_alloc`, `<layout>_bytes_of`, `Backend::<layout>_bytes_of`, `take_<layout>_scratch`, plan constructors, `Module::max_base2k`) take `n` first, then `rows, cols_in, cols_out, size` (matrices) or `cols, size` (vectors), then the `PrepareHint` of a prepared layout.
- **Clobbered sources.** A clobbered source says so in its sentence and definition.
- **Argument order.** A new method copies its closest sibling:
  - normalization `(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch)`, in place `(base2k, k, a_offset, a, a_col, scratch)`; shifts `(base2k, k, res, res_col, a, a_col, scratch)`;
  - ring maps `(p, res, res_col, a, a_col)`; binary column ops `(res, res_col, a, a_col, b, b_col)`; DFT limb selection `(step, offset, res, res_col, a, a_col)`;
  - convolutions `(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff | b_galois, scratch)`; VMP `(res, a, pmat, limb_offset, scratch)`, row selection `(res, a, first_row, row_step)`; samplers `(base2k, k, res, res_col, source)`;
  - outside VMP, op-wide parameters first, a one-operand selector (`res_limb`, `res_coeff`, `b_coeff`, `b_galois`) after that operand's column, shared column selectors after their operands (`cnv_pairwise_apply_dft(cnv_offset, res, res_col, a, b, i, j, scratch)`).
- **Queries take limb counts.** A HAL `_tmp_bytes` takes `module`, the family's leading parameter (`cnv_offset`) and `usize` sizes named after their operand (`res_size`, `b_rows`), keeping its siblings' list even where its default ignores one; a shape-independent query takes none and its `ensures` says so.
- **Plan-based ops.** Plan-based ops follow `VecZnxDftAutomorphismPlan`: an `<Op>Plan` supertrait with `Plan` and `<op>_plan(n, ..)`, a `Send + Sync` plan type in the OEP family, `plan: &Self::Plan` first in `_with_plan` forms, and a derived plan-free form (build, apply, drop).
- **Sampler draws.** A sampler has its definition `draw(reseed(source), ..)` by limb then coefficient, never by column, stating how far `source` advances, and its multi-column form derived, one seed per column in order.

## 3. Contract blocks

### 3.1. Grammar

- **One block per method.** A block is shared only by word-for-word identical contracts (` / ` or `*`).
- **Keys and values.** `mutation` is `none` exactly for `support`, and `fallback` and `override` appear exactly on `derived` blocks and `variant` blocks whose `requires` names a `_tmp_bytes`, plus a defaulted `support` hook (`synchronize`).
- **Op line spells the signature.** An `op` line lists the declared parameter names in order without `self`, except in wildcard blocks.
- **Test line names suite tests.** The `test` line lists the `pub fn test_*` suite functions exercising the op (a `_tmp_bytes` its op's, a new computing op its own, `test none` only for an untested `support` block).
- **Mutation follows the form.** `_assign` forms are `in-place` (`vec_znx_add_scalar_assign`, adding into one limb: `accumulate`), forms adding or subtracting their transformed or multiplied operand into the destination (`vec_znx_lsh_add`) are `accumulate`, other computing ops `out-of-place`, the raw complex transform `in-place`.
- **Grammar changes move together.** A grammar change (key, class, mutation value, `op` matching) updates the unit tests of `poulpy-hal/tests/contracts.rs` in one PR, never relaxing a check for one block.

### 3.2. Content

- **Complete definitions.** A `basis` definition is a closed formula fixing every visible output; a `variant` or `derived` one may compose HAL ops with every argument (columns, `tmp, 0`, scratch), reading each destination as its value just before that call; each is written in full, only an identical `domain` or `embed` line reading `as for <sibling op>`.
- **What definitions leave unchanged.** Definitions end with `other columns of res are unchanged` when column-selected, state preserved limbs or coefficients on the definition line when in place or accumulating, and quantify every column when whole-object; input limbs past `a.size()` read as zero, an out-of-range input column only where stated, and every index but the destination limb `j` is bound on its line.
- **Notation only.** Definitions and `domain`, `requires`, `ensures` and `embed` lines use only symbols of `# Value model`, `# Notation` and `# Canonical form` in `poulpy-hal/src/api/mod.rs`, a new symbol entering `# Notation` in the same diff with its arguments' meaning.
- **Domain lists every precondition.** `domain` or `requires` states every precondition a kernel asserts (ranges, degree relations, `res_k <= res.size() * res_base2k`, `step >= 1`, odd exponents and Galois elements, density) except column bounds and those the API module docs state once (the module serves the call degree, the scratch meets its query), and numeric bounds as unchecked caller obligations.
- **Ensures states the tail.** `ensures` states what destination limbs past the inputs' hold, never weakening or contradicting the definition, and `the digits are not renormalized` wherever digits can leave the canonical range (sums, differences, accumulating shifts, `X^p - 1` products, CI maps, entrywise matrix sums).
- **Rounding through rnd and canon.** A normalizing, rounding or decomposing op takes explicit `res_k` or `k`, is defined through `rnd(x, k)` (`canon(v, b, k, L)` for a canonical column), states digit ranges and congruence modulo 1 to `rnd(..)` in `ensures`, never an equality or a value in `[-1/2, 1/2)`, and the radix range and input bound (`[-2^62, 2^62]`, big coefficients `abs <= 2^126`) in `domain`; no new rounding mode is added.
- **Exact integer results.** A formula whose intermediates may leave the coefficient type writes `wrap(..)` or bounds them in `domain`.
- **DFT results through idft.** Contracts state DFT results through `idft(..)` and prepared operands through the polynomial they read as, never a layout, slot order, word type, byte representation or cross-backend byte equality (the raw complex transform alone fixes `rev_m`, `omega(m,r)`); slot-order helpers live in backend crates, poulpy-hal keeping only `check_galois` and `check_degree`.
- **Degree embedding.** Only slots an `embed` line names take a dense degree-`n` operand (`n` a power-of-two divisor of `N`), matching its `switch_ring_{n->N}` bit for bit, other operands at degree `N` and windows there as wide as `res`:
  - a derived op declares a slot only if every op its fallback feeds accepts it;
  - in the convolution only the prepared right operand embeds, prepares are not embedding-aware (a degree-`n` operand is prepared at `n`), and an embedded degree is at least `MIN_DEGREE`;
  - `embed` ops name their family's embedding test, and a new or extended slot updates `# Degree embedding` in `poulpy-hal/src/api/mod.rs`.
- **Support blocks.** A new allocation, sizing, carving, plan or module trait's `support` block bounds dimensions and degree in `domain` (`n: a power of two, MIN_DEGREE <= n <= the module's degree` for allocations and sizes, at most the module's degree for carves, a take fed to a kernel obeying that kernel's floor) and leaves contents unspecified in `ensures`.
- **Scratch lines.** A scratch-taking op `requires` `scratch >= <query>(<arguments>)` with its own query (`vec_znx_lsh_tmp_bytes(res.size())`) and a scratch-free op has neither, a query's `ensures` names the ops and sizes it serves, never a byte formula or the fallback's carve.
- **Fallback and override lines.** A derived op's `fallback` describes its body exactly, step by step (temporaries' limbs, columns and degree, nested calls in order, checks), updated with every body change, and its `override` reads `allowed, with <its query>`, `allowed, scratch-free` or, for a sampler, `allowed, drawing the same values`.
- **Notation changes are behaviour changes.** Changing the meaning of a Notation symbol or the canonical form (the `rnd` tie rule) updates `poulpy-hal/src/api/mod.rs`, the oracle and every backend kernel in one PR with a `**Breaking, behaviour:**` entry, never by regenerating fixtures alone.

## 4. Delegates

- **Delegates use the family macro.** Ring-arithmetic and query delegates are one `impl_<family>_delegate!` invocation per trait, fenced `Backend<ZnxWord = i64> + Hal<Family>Impl` (standard-only ops via the `monomial` and `ci` arms of `impl_vec_znx_delegate!`), and a new family macro keeps the fence.
- **Support delegates are plain.** Only allocation, `*_bytes_of`, plan-constructor, module and scratch delegates are plain `impl` blocks, bounded only by `B: Backend` (plus `HalModuleImpl` for module hooks, the OEP family for a plan constructor), forwarding to the owned `alloc` or inherent `Module` allocator and to `Backend::<layout>_bytes_of`.

## 5. OEP families and derived defaults

- **Family trait shape.** A family is re-exported by `oep`, and its `# Safety` cites the backend safety contract and the aspects it touches, or reads "Same contract as [`HalVecZnxImpl`]".
- **The composition graph.** A new family or edge updates the graph and topological order of the `poulpy-hal/src/oep/mod.rs` docs, and a new family-wide `_tmp_bytes` joins their exception paragraph.
- **File layout per family.** `api/<family>.rs` imports only `layouts` and `source`.
- **Default bodies are one call.** A default body is exactly `crate::oep::<op>_derived::<Self>(module, <its arguments in order>)` (`::<Self, R>` with a layout generic), a `_tmp_bytes` default `crate::oep::<op>_tmp_bytes_derived::<Self>(module, ..)`.
- **Derived functions.** A derived op is `#[doc(hidden)] pub fn <op>_derived<BE>(module: &Module<BE>, <the OEP method's arguments>) where BE: <its OEP family>` (plus its layout generic `R`), with a `<op>_tmp_bytes_derived` twin when it takes scratch (one per family-wide query), and no caller above poulpy-hal but a gated override's fallback.
- **What a derived body uses.** A derived body uses only `BE::<oep method>` calls (never a sibling `<op>_derived`), backend views with `to_backend_ref`/`to_backend_mut`, the hal view helpers (`*_backend_ref_from_mut`, `vec_znx_reborrow_backend_mut`, `mat_znx_at_backend_*`, `scalar_znx_as_vec_znx_backend_ref_from_ref`, `window_limbs`), shared checkers (`check_galois`), `take_*_scratch`, `Backend::*_bytes_of`, `Backend::scratch_aligned` and a sampler's `source.new_seed()`, and never narrows a DFT or prepared view.
- **Derived body defines the op.** A derived body's doc states its composition and every contract it fixes (limb selection, offsets, zeroed or untouched limbs, empty inputs, failures, column alignment, caller preconditions), which overrides reproduce bit for bit, through `idft` and normalization for DFT results and as the same canonical value for non-canonical digits.
- **In-place forms use a temporary.** An `_assign` derived body calls the in-place OEP form or computes from `vec_znx_backend_ref_from_mut` into a scratch temporary copied back by `vec_znx_copy`, never one buffer as source and destination, and a body writing `res` before reading states "`res` must not alias `a` or `b`" in its doc.
- **Composites validate up front.** A looping derived body asserts, before dispatching, every dimension the loop relies on with a static message.

## 6. Backend, Module and degree

- **Module constructors enforce the cap.** `HalModuleImpl::new` returns its module through `Module::from_nonnull` or `Module::from_raw_parts`, which assert `n` a power of two at most `MAX_RING_DEGREE`.
- **The degree floor.** `Backend::MIN_DEGREE` defaults to 8, and a backend with fixed-step kernels raises it to the smallest degree they handle.
- **Standard-only families.** A standard-ring op says so in its `domain` and lives in `HalVecZnxCIImpl` or `HalVecZnxMonomialImpl`.
- **Forward every Backend override.** `impl_backend_from!` forwards every defaulted `Backend` constant (`MIN_DEGREE`, `DFT_LIMBS_CONTIGUOUS`, `SCRATCH_ALIGN`) and overridden sizing method, new ones in the same diff, and every layout marker both ways, other traits (`MaxBase2k`, `ScratchWorkers`, `HalModuleImpl`, OEP families) implemented beside it.
- **Host module stays minimal.** `HostBytesBackend<R>` implements only `Backend` and `HalModuleImpl`, with no OEP compute family, `MaxBase2k` or table state.
- **Radix queries need a model.** `MaxBase2k::max_base2k(n, products, failure_bits, squaring)` and `Module::<BE>::max_base2k` return a `usize` (0 if none fits, at most the word width minus 2), and only a backend with a failure model implements it, once, as `impl<R: Ring> MaxBase2k for X<R>` over `max_base2k_ntt` or `max_base2k_fft64`, or by forwarding to its source.
- **Module handles are immutable.** `Module<B>` stays `Send + Sync`, a backend handle changing after `HalModuleImpl::new` only through internally synchronized state; scheme plans in its `ModulePlanCache` take one zero-sized key type per plan family covering every degree, built in `with_or_create` by an initializer never requesting its own key.
- **Deferred execution.** Every backend call is logically synchronous: a backend defers execution only under the synchronization contract of `poulpy-hal/docs/backend_safety_contract.md` (submission order, transfer waits, borrowed host memory, storage release), overriding `HalModuleImpl::synchronize`, detecting the conditions it reports on the host before returning, from layout metadata only, and panicking on a later device failure by the next transfer or `synchronize`; library code never relies on `synchronize`.

## 7. Layouts, views and memory

- **Aligned storage.** Host-owned `AlignedVec<T>` (`AlignedBuf` for bytes) comes from `alloc_aligned`, `alloc_aligned_custom` or `AlignedVec::zeroed`, keeps its `T: Copy + bytemuck::Zeroable` bound, frees with its own `Layout` and gives an empty buffer an aligned dangling address, not `NonNull::dangling`.
- **Owned buffers may be padded.** Owned buffer lengths are checked against `padded_bytes(*_bytes_of)` or as at least the logical length, never for equality with `*_bytes_of`; host-to-backend copies use the length-tolerant transfer hooks, and `transfer_buf_into` gets both buffers from one allocation path.
- **Byte constructors check lengths.** A constructor taking ownership of caller bytes (`from_bytes`) asserts a length of `padded_bytes(<bytes_of>(..))`, callers of the unchecked `from_data` and `from_shape` validate, and `raw`/`raw_mut` assert once per call that the element view fits.
- **Checked size arithmetic.** Layout byte sizes, element counts and window shapes, `Backend::*_bytes_of`, allocation and carving use `checked_product(&[..], "<what>")` in poulpy-hal and `checked_mul`/`checked_add` with a static `expect` in backend overrides.
- **Zeros come from zeroing paths.** Zeros come from `Backend::alloc_zeroed_bytes`, `vec_znx_alloc_zeroed::<BE>`, inherent `Module` allocators, documented zero-initialized constructors or explicit zeroing, never from `*Alloc`, `Backend::alloc_bytes` or `ScratchOwned::alloc`.
- **No mutating resize.** A HAL layout's `size` never changes after construction (no `set_size`, layout `with_size` or capacity), fewer limbs going through borrowed views (`vec_znx_backend_mut_with_size`, `with_size_mut`, `with_limb_range_mut`, `window_limbs`).
- **Where windows are accepted.** Windows exist only on `VecZnx` and `VecZnxBig` (`VecZnxDft::from_shape` and `with_shape` reject one); coefficient-wise ops (normalize, shifts) accept them, and ring ops (rotate, automorphism, `mul_xp_minus_one`, CI maps), DFT-domain ops and prepares reject them.
- **Whole-buffer helpers assert density.** Every whole-buffer helper of a window-capable layout (`raw`, `raw_mut`, `as_ptr`, `as_mut_ptr`, `digest_u64`, `to_owned_deep`, `to_host_owned`, `into_data`, `read_from`/`write_to`, `into_backend`, `into_big`, equality, hashing) asserts density.
- **Narrowed DFT views need capability.** Every strict `VecZnxDft` narrowing (`with_size_mut`, `with_limb_range_mut`) or indexed access (`zero_at`) panics unless `Backend::DFT_LIMBS_CONTIGUOUS`, the full range working on every layout.
- **Element views of backend-sized storage.** Host element access (`raw`, `at`, `zero`, `Display`) on DFT, big and prepared buffers needs byte size equal to element span, which a new `ZnxView` over such storage asserts in `validate_element_view`, a helper also serving packed buffers branches on the element span, and a container without flat (col, limb) indexing exposes only `raw`.
- **Hints select representations only.** A prepared layout's `PrepareHint` lives in its shape and every wrap call: `prepare` writes the destination hint's representation, copies and row extractions state and assert equal hints, and single-representation backends size hints equally (`test_word_compat_prepare_hint_sizes`).
- **The byte-level exception.** Only the word-compat suite compares `VecZnxDft`, `SvpPPol` and `VmpPMat` bytes, between backends declaring the layout marker over an exact word, and opaque `from_bytes` carries only value-agnostic operands.
- **Layout-compatibility markers.** `into_backend::<B2>()`, bounded on `*LayoutCompatible<B2>`, is the only zero-copy move, asserting equal `*_bytes_of` (and density on `VecZnxDft` and `VecZnxBig`) first and keeping the storage type; a marker, an `unsafe` claim of byte-identical buffers for every shape, is declared per family, never from a shared `DftWord`, with word-compat tests, a new opaque family adding the reflexive blanket impl and `into_backend`.
- **Byte moves through buffer traits.** Raw bytes move by `transfer_buf_into(src, dst)` into a caller-allocated buffer, new buffer types implement `CopyToHost`/`CopyFromHost` (`as_host_bytes*` `Some` only for host memory) and never name a backend pair, typed layouts move with core `TransferInto`, and only coefficient-domain layouts move by bytes, host-staged ones bounding `HostStaged` with `ZnxWord = i64`.
- **Equality.** Opaque layouts (`VecZnxDft`, `VecZnxBig`, `SvpPPol`, `VmpPMat`, `CnvPVecL`, `CnvPVecR`, new ones) compare, and hash where provided, as same shape and bytes, with no bound on the word.
- **A new layout's pieces.** A new HAL layout lands with:
  - a `#[repr(C)]` `*Shape` with `const` accessors, `ZnxInfos` (explicit `poly_count`) and `VecZnxInfos` (one flat `cols`) or `MatZnxInfos` (`rows`, `cols_in`, `cols_out`);
  - `DataView`/`DataViewMut`, the `*Owned`, `*BackendRef` and `*BackendMut` aliases, `*ToBackendRef`/`*ToBackendMut` on owned and borrowed forms, and the `*Reborrow*` traits;
  - `<Layout>Alloc`, `<Layout>BytesOf` (if opaque) and a `ScratchArenaTake<Layout>::take_<layout>_scratch` wrapping `take_region` with `from_data` into a `*ViewMut`, all sized by one `Backend::<layout>_bytes_of`.
- **Serialized layouts.** Only `ScalarZnx`, `VecZnx` and `MatZnx` implement `WriterTo`/`ReaderFrom`, as `u64` shape fields, a `u64` byte length and exactly the logical bytes, never padding; readers check shape product times word size against the length and the receiver's capacity.

## 8. Scratch

- **Derived queries mirror their bodies.** `<op>_tmp_bytes_derived` reports exactly what `<op>_derived` carves at the module degree, a lone temporary unrounded.
- **Derived carves and zeroing.** A derived body carves each temporary at `ZnxInfos::n` of the operand the nested op reads and zeroes every temporary column it reads or accumulates into unless fully written first, even where the next kernel zeroes the tail.
- **Family-wide shift queries.** `vec_znx_lsh_tmp_bytes` and `vec_znx_rsh_tmp_bytes` cover their whole family (out of place, `_add`, `_sub`, `_assign`, new ops included); an override of one body may leave the family query on its default, an override of the query still covers the bodies left on default.

## 9. What every backend kernel must guarantee

Every kernel of a HAL family `unsafe impl` guarantees these, beyond `poulpy-hal/docs/backend_safety_contract.md`.

- **Degree check first.** Every kernel whose `domain` involves the module degree calls `check_degree::<BE>(module.n(), n)` once at entry, `n` from `res` (or the operand setting the call degree), before any plan or table fetch, then checks other operands against `n` ("operand degree differs from the result degree"), or an `embed` slot as a power of two dividing `n`, at or above the floor only where its contract says so (the convolution's right operand).
- **Reject exactly the domain.** A kernel rejects before any write exactly what its `domain`, `requires` and the API module docs exclude, plus out-of-range columns and only limits its contract names (an embedded right convolution operand below `MIN_DEGREE`).
- **Window refusals.** A window-rejecting kernel calls `poulpy_hal::layouts::assert_dense(v, "<name>")` at entry; coefficient-wise kernels honour window offsets and strides.
- **Convolution terms are checked.** Every kernel or derived body applying convolution terms calls `poulpy_hal::layouts::check_cnv_dft_term` on each product it reads (`check_cnv_dft_terms` on a sum) at entry, before its first write, which covers degrees, columns and `check_galois` ("convolution: b_galois must be odd").
- **Embedded operands cost no more.** A kernel reads an embedded operand through the backend's slot correspondence (no staging gather), bit for bit the dense result, at no more cost than the dense apply.
- **Static messages on secret paths.** In poulpy-hal and its kernels the paths a secret or plaintext reaches include the `encode_*`/`decode_*` codecs, `svp_ppol_prepare`, the DFT of a secret and the `AlignedVec` sizing reached from secret-key size helpers, whose messages format nothing from that operand, length, shape and degree included (`check_degree` too).

## 10. Documentation and CHANGELOG

- **API prose is one sentence.** In `poulpy-hal/src/api` a trait doc is one sentence and its block (the sentence alone when its blocks sit on its methods), and a method doc one sentence on what the destination receives or the method returns (then its block in a grouped trait), without `# Arguments`, `# Panics` or `# Examples`, diagrams, restated `domain` or `ensures`, rationale or cost claims.
- **OEP method docs.** A defaulted HAL OEP method documents what it computes and its default body, `<op>_derived`, with the composition in words (a query: what its default sizes), a variant kept required states the temporary its signature cannot supply and why, a family-wide query names the bodies it covers and says a partial override leaves it on the default or returns enough, and a basis method has no doc.
- **Docs name roles, not backends.** Hal rustdoc, contract blocks, suite docs and `poulpy-hal/docs/backend_safety_contract.md` name roles ("a comparison backend"), never a device API or a backend-internal representation.
- **Safety contract holds obligations.** A new backend capability or obligation (`DFT_LIMBS_CONTIGUOUS`, deferred execution, a transfer or aliasing rule) enters `poulpy-hal/docs/backend_safety_contract.md` with its hook.
- **Review is the only check.** Every new private or `pub(crate)` hal item has a caller and every new intra-doc link resolves (methods as ``[`name`](Trait::method)``).
- **CHANGELOG classification.** A new required OEP method or family, or a new argument or field an overridable method reads (`CnvDftAccTerm`), is `**Breaking:**` and says what out-of-tree backends must add, explicitly when an override ignoring it still compiles and computes wrong results.

## 11. Tests and registration

- **Where HAL tests live.** HAL tests are declared on a line starting `pub fn ` (not macro-generated), in `poulpy-hal/src/test_suite/<family>.rs` (`svp.rs` and `vmp.rs` for `svp_ppol` and `vmp_pmat`), default-body pins in `derived.rs` unless the op's cross-backend test runs the derived function, and cross-cutting checks in `window.rs`, `embed.rs`, `word_compat.rs`, `transfer.rs` and `serialization.rs`, a new file declared `pub mod` in `test_suite/mod.rs`.
- **Test names.** HAL tests are `test_<op>`, or `test_<family>_<case>` for a family-wide check, embedding checks `test_<family>_embed[_<ops>]`, registered under the same name (`test_x => poulpy_hal::test_suite::<module>::test_x`).
- **Signatures follow the macros.** A HAL test takes `(params: &TestParams, module: &Module<BE>)` for `backend_test_suite!` or `(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)` for `cross_backend_test_suite!`, another signature only for an object the macros cannot build, called from a backend `#[test]`; its bounds are `TestBackend`, its API traits and, for an `oep::*_derived` call outside `HalVecZnxImpl`, that family.
- **Suites use the given modules.** A suite function builds no `Module`, runs ops as `module.<op>(..)`, and below the API calls only the pinning tests' `oep::<op>_derived` and module-free `Backend` items (transfer hooks, `*_bytes_of`, `MIN_DEGREE`, capability flags).
- **Degree sweep.** Suite tests compute at `params.n` (`params.size` is the module degree, not a limb count), adding a `module.n()` leg when needed and filtering lower degrees by `BE::MIN_DEGREE` (a module-independent op: every degree its `domain` admits), and HAL registrations set `n` to `MIN_DEGREE` below `size`, except statistical sampling, named large-ring and core noise suites.
- **Register on every backend.** A new HAL test joins, in the same PR, the `backend_test_suite!` or `cross_backend_test_suite!` lists of every implementing backend type in every crate (the oracle's conjugate-invariant tables and the FFT64 and NTT families included, where they exist), a default-body pin joining each `derived` suite, and a Rayon twin registering only the ops it parallelizes or overrides, at sizes past its parallel-work floors.
- **Check the contract independently.** A new op's test also checks a backend-independent expectation (host computation or API recomposition) on the edge inputs its domain admits: rounding ties, extreme and non-canonical digits, empty and mismatched sizes.
- **Sweep sizes, columns and windows.** A new op's test sweeps destination sizes below, at and above the inputs, every column and parameter edge (offsets, steps, `cnv_offset` past the product), accepted windows in `poulpy_hal::test_suite::window` (a coefficient and a strided limb window, matching the op on the materialized window, the outside untouched), and `embed` slots against the dense `switch_ring` embedding in `test_suite/embed.rs` or the convolution embedding test.
- **Window rejections are tested.** A new window-rejecting op gets a panic leg in `test_vec_znx_window_rejected_by_ring_ops` beside the accepting coefficient-wise control leg.
- **Cross-backend protocol.** HAL cross-backend tests compare coefficient results bit for bit, `VecZnxBig` results after `vec_znx_big_normalize`.
- **Dirty destinations.** Destinations start from random digits (differing per side out of place), a sentinel or a poisoned template, multi-column ones written at a non-zero column, tails and unselected columns compared after the call.
- **Input preservation on the backend.** An input-unchanged check compares the backend operand downloaded after the call (or its `digest_u64`) with a pre-call copy, or repeats the call, never a pre-upload host digest.
- **Rejection tests.** HAL rejection tests use `std::panic::catch_unwind(AssertUnwindSafe(..))`, pin only shared-helper messages (`check_galois`, `check_cnv_dft_term`), check the destination unchanged on the same backend (bytes, or `idft` and normalization) and cover every kernel check (a column equal to `cols`, a degree mismatch, empty operands, and for a composite an operand mismatch and an invalid term past a fused group).
- **Testing a derived op.** A derived op's test checks `<op>_derived::<BE>` and `module.<op>` bit for bit against an oracle of other API calls (never the op under test) or host computation, DFT and big results after normalization, canonical forms where decompositions differ.
- **Sampler tests.** A sampler test pins the stream against a backend-independent expectation from the seed (radix endpoints, a partial last limb, poisoned tails, unselected columns) and checks that a call consumes exactly one 32-byte seed.
- **Capability legs are gated.** A leg needing an optional capability runs only when every backend involved declares it (`if BR::DFT_LIMBS_CONTIGUOUS && BT::DFT_LIMBS_CONTIGUOUS`), the op staying tested everywhere.
- **Unsafe code gets Miri tests.** New or changed `unsafe` code in poulpy-hal gets a Miri-sized `#[cfg(test)]` unit test inside `poulpy-hal/src` that reaches it, foreign calls compiling out under `cfg(miri)` (`advise_hugepage`).
