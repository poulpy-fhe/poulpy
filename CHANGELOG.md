# CHANGELOG

## [Unreleased]

### `poulpy-hal`

- **Breaking:** add `VecZnxCanonicalize` and its scratch query to restore the canonical representation at a requested precision; `HalVecZnxImpl` gains the matching hooks, implemented by every built-in CPU backend through its coefficient-domain shift operations.
- **Breaking:** uniform `VecZnx` sampling now takes the target precision `k`; the sampler masks the unused low bits of the last live limb and clears limbs above `k`.
- The cross-backend `test_vmp_apply_dft_to_dft_accumulate` now sweeps `res` sizes that differ from the prepared matrix size and non-zero `limb_offset`, so the output limb window is compared across transform families.

### `poulpy-core`

- Canonicalize linear-transformation inputs and outputs through the HAL so inactive partial-limb bits cannot affect the transform and result storage remains canonical.
- Apply the gadget-product limb window on every backend. The window was gated on `DFT_IS_EXACT` because FFT64 lost precision with it; the loss was the FFT64 VMP bug below, not a property of approximate transforms, so FFT64 backends now materialize the same reduced key region as the NTT backends.
- Add a cross-family parity suite (NTT4x30 reference against FFT64) to `poulpy-cpu-ref`, at a radix where FFT64 products round exactly.
- Fix precision loss at non-`base2k`-aligned ciphertext widths. Masks and Gaussian noise are sampled at `k`-bit precision, avoiding redundant post-encryption rounding for both secret- and public-key ciphertexts.

Adds opt-in intra-operation Rayon scheduling to every accelerated CPU arithmetic family, backed by a shared HAL execution and scratch-allocation model. Fused primitives reduce intermediate traffic in core and CKKS paths, bin-FHE gains backend-driven parallel evaluation, and CKKS gains an even-Chebyshev EvalMod variant.

### `poulpy-hal`

- **Breaking:** `Backend` gains the required `TaskExecutor` associated type. Add `TaskExecutor`, `SerialTaskExecutor` and backend-declared `ScratchWorkers` limits, so default algorithms can schedule independent work without depending on Rayon.
- Add aligned per-worker scratch sizing and arena splitting helpers. Scratch reservations depend on the backend's fixed worker caps rather than the ambient pool width, keeping `*_tmp_bytes` stable across Rayon pools.
- **Breaking:** `HalConvolutionImpl` and `HalVecZnxDftImpl` gain fused constant-convolution accumulation, DFT automorphism accumulation, and consuming IDFT-plus-normalization operations. The public HAL API exposes the same primitives and every CPU backend implements them.
- Add `VmpExtractSelectedRows`: copies rows `first_row + i * row_step` of a `VmpPMat` into a smaller one, reading only the selected cells. The delegate validates the selection before dispatch.

### CPU backends

- Fix the FFT64 `vmp_apply_dft_to_dft` limb window when `res` is narrower than the prepared matrix: output limb `c` reads matrix limb `c + limb_offset`, but the window was clamped at `res.size()` instead of `res.size() + limb_offset`, dropping the top `limb_offset` limbs of every narrowed accumulating gadget digit. Shared by every FFT64 backend (reference, AVX2, AVX-512, NEON and their Rayon variants).
- Fix the NTT4x30 reference `vmp_apply_dft_to_dft_accumulate_tmp_bytes` under-reporting scratch when `res` is wider than the prepared matrix.
- Add `poulpy-cpu-rayon`, which provides the shared Rayon executor, nested-parallelism guard, scheduling thresholds, FFT64 kernels, coefficient normalization, and tuning utilities used by the accelerated CPU crates.
- Add the opt-in `FFT64AvxRayon`, `NTT4x30AvxRayon`, `FFT64Avx512Rayon`, `NTT4x30Avx512Rayon`, `NTT3x42IfmaRayon`, `FFT64NeonRayon` and `NTT4x30NeonRayon` backends. `enable-rayon` exposes them while retaining the serial backend types.
- Pack the four NTT4x30 transform-domain residues into `u32` words on AVX2 and AVX-512, halving DFT and prepared-key storage from 32 to 16 bytes per coefficient; update the serial and Rayon transform, convolution, SVP and VMP kernels for the packed layout.
- Parallelize the transform, convolution, VMP, normalization and coefficient-domain kernels that scale within one operation. A one-thread pool follows the serial path, and nested Rayon operations serialize their inner level rather than oversubscribing the pool.
- Fuse adjacent gadget digits in strided VMP key-switch kernels for AVX2 NTT4x30, AVX-512 NTT4x30 and AVX-512-IFMA NTT3x42, reusing each prepared key column for both products; the Rayon variants distribute the fused block work across workers.
- Declare the Rayon variants layout-compatible with their serial/reference families and wire them into the HAL, core, CKKS and bin-FHE operation surfaces. Backend parity suites cover the parallel types against the corresponding serial/reference result.
- Add instruction-set and compiled-backend capability reports, plus an ignored thread-scaling diagnostic that measures VMP, convolution, IDFT and coefficient work across pool widths. `docs/performance.md` documents backend selection, key-traffic tradeoffs and thread-count tuning.

### `poulpy-core`

- Refactor GLWE key switching to consume inverse-DFT results directly into normalized outputs, and use DFT automorphism accumulation in lazy/prepared-giant linear transformations. These paths avoid temporary big-polynomial copies while preserving the serial/reference behavior.
- Add an optional one-pass baby-step linear-combination hook to generic polynomial evaluation, falling back to the existing multiply-add sequence when an operation family does not override it. Extend the HAL parity suites for the new fused operations.
- `GGLWEInfos` gains `stride`, the row map a coarser read uses (digit `i` is stored row `(i + 1) * stride - 1`, defaulting to 1), plus `gglwe_layout_at_dsize` and `valid_dsizes`: the layout a key reports at a coarser `dsize` and the decompositions it admits.
- `GGLWEPrepared` carries its own `dnum` and `stride`, and `with_dsize` re-tags a prepared key as one read at a coarser `dsize`, or fails if the key does not admit it. It returns the same backend view every operation already takes; no new key type.
- A GGLWE product over a view gathers the selected rows once, then runs the ordinary kernels on them; its scratch query adds that gather. Query it with the key the operation will run through, coarsened or not.
- Add `GetAutomorphismKey` and `GetTensorKey`: a caller names a function and the precision it will use the key at, and the source answers with the backend view of a prepared key. A map and a bare key implement them as stored, so which key and which decomposition a precision gets is entirely the source's rule. Implementors write `lookup_automorphism_key`; `get_automorphism_key` checks the answer is a key for the element asked for, since operations rotate by the element they were given.
- **Breaking:** every operation that consumes an evaluation key takes that key's backend view rather than a generic prepared-key parameter: `&GGLWEPreparedBackendRef`, `&GLWEAutomorphismKeyPreparedBackendRef`, `&GGSWPreparedBackendRef` or `&GGLWEToGGSWKeyPreparedBackendRef`, written `&key.to_backend_ref()` at the call site. A view returned by `GetAutomorphismKey`/`GetTensorKey` or by `with_dsize` passes straight through.
- **Breaking:** `GLWEAutomorphismKeyHelper` and its `automorphism_key_infos()` are removed, along with the key type parameter every automorphism-consuming signature carried: no single layout describes a key set whose rotations resolve independently.
- Linear transformations resolve each rotation's key at the precision that rotation actually works at: baby steps at the source, giant steps at the post-product destination.
- Fix cross-radix tensor relinearization: the DFT operand width came from the storage precision rounded to the key's radix instead of from `a.k()`, handing the product one limb too many.
- Add the `error` module, exporting `CoreError` and `Result`.

### `poulpy-bin-fhe`

- Make FHE integer preparation and BDD evaluation honor their requested worker count through the backend executor and disjoint scratch arenas; serial backends continue to execute them on one worker.
- Parallelize the independent VMP contributions in block-binary CGGI blind rotation before deterministic accumulation, with serial-versus-Rayon parity coverage for the accelerated backend families.
- **Breaking:** multithreaded BDD execution requires its output buffer to implement `Send`.

### `poulpy-ckks`

- Add `EvalModType::CosHKEven`, a centred Han–Ki approximation folded through `T₂` when it reduces multiplication cost without increasing the modulus budget.
- **Breaking:** `BootstrappingPlan::new` now validates the EvalMod plan and derives the CoeffsToSlots input scaling from it, replacing any scaling already present on the supplied CoeffsToSlots plan.
- Fuse each BSGS baby-step linear combination into one accumulator and use constant-convolution accumulation to avoid repeated ciphertext temporaries.
- Run the real and imaginary EvalMod halves concurrently on parallel backends, with per-half scratch arenas; serial backends retain the existing order.
- **Breaking:** the ciphertext-ciphertext operations (`ckks_mul_*`, `ckks_square_*`, the `mul_add`/`mul_sub`/`dot_product` composites, polynomial evaluation, approximation, EvalMod and the PaCo slot product) take a tensor-key source rather than one key, and resolve it at the precision they work at.
- **Breaking:** `ckks_coeffs_to_slots_{split,repack}` drop their separate `conj_key`: one source answers every element, `-1` included. `BootstrappingKeys::conjugation_key` and its `AutomorphismKey` type are removed, and generation puts the conjugation key in `rotation_keys`.
- **Breaking:** add `ckks_conjugate_rotate_into(dst, src, k, ..)` for the fused conjugate-and-rotate PaCo's psi tail needs, which was previously expressed by handing `ckks_conjugate_into` a different key. It takes the rotation like `ckks_rotate` and resolves `-galois_element(k)` itself; `k = 0` is plain conjugation. `PaCoPsiTailMaterial::Mask` carries that rotation rather than the derived element.
- **Breaking:** `CKKSAtkBounds` is removed: a key type is now constrained as `GetAutomorphismKey<BE>`.
- **Breaking:** `CKKSCompositionError::MissingAutomorphismKey` carries the precision the lookup was made at; add `MissingRelinearizationKey`.
- `ckks_mul_tmp_bytes` sizes its tensor intermediate from the operands' precision; `res`'s pre-call precision no longer widens it.


## [0.8.2] - 2026-08-22

### `poulpy-hal`

- Add `reference::znx`, the portable scalar kernels over `[i64]`, moved out of `poulpy-cpu-ref` so that a crate can reach them without depending on a backend. `poulpy_cpu_ref::reference::znx` re-exports them, so existing paths are unchanged.

### `poulpy-bin-fhe`

- **Breaking:** the crate no longer depends on any backend, in `dependencies` or `dev-dependencies`, matching `poulpy-core` and `poulpy-ckks`. The `enable-avx`, `enable-avx512f`, `enable-ifma` and `enable-neon` features are removed (`enable-neon` was inert: nothing referenced `poulpy-cpu-arm`), and `enable-bin-fhe` no longer enables anything.
- **Breaking:** the per-backend test modules are replaced by public `blind_rotation::test_suite`, `circuit_bootstrapping::test_suite` and `bdd_arithmetic::test_suite`, instantiated by a backend crate through the new `bin_fhe_backend_test_suite!`. The `bdd_arithmetic` / `circuit_bootstrapping` / `max_array` examples move to `poulpy-cpu-ref` and run on `FFT64Ref`.

### `poulpy-core`

- Add the missing `GGLWEPreparedToBackendRef` impl for `GLWETensorKeyPrepared`, the twin of the `Mut` one. The newtype field and every `GGLWEPrepared` field are `pub(crate)`, so no downstream crate could supply it. The relinearization path (`GLWETensoringImpl::glwe_tensor_relinearize` and the `poulpy-ckks` `mul` / `polynomial_evaluation` / `composite` chains that reach it) carries the bound alongside `GLWETensorKeyPreparedToBackendRef`, so a backend override can read the key's `VmpPMat` without naming the newtype field.

### `poulpy-ckks`

- ModUp lifts `Δ·m` to `k - log_msg_ratio`, then to `f_mod_log_delta` fused into the widening shift, both before the sparse-to-dense switch. Recovers `f_mod_log_delta - k` bits (19.7 → 26.9 at `LogN=16`, `Δ=2^40`, ratio `2^8`). C2S-first only.
- **Breaking:** `ckks_mod_up_into` takes `&EvalModPlan` and returns a labelled ciphertext; new `ckks_bootstrap_mod_up` does the whole raise step; `CKKSEncapsulatedModUpImpl::ckks_encapsulated_mod_up` takes `scale_up`.

### `poulpy-bin-fhe`

- **Breaking:** drop the `enable-avx` / `enable-avx512f` / `enable-ifma` / `enable-neon` features and the backend dependencies; only `enable-bin-fhe` remains. The tests are backend-generic and instantiated by backend crates through `bin_fhe_backend_test_suite!`; the examples move to `poulpy-cpu-ref/examples`.

### CPU backends

- `mod tests` is gated on the backend feature alone, not on `enable-ckks`, so the core-parity and bin-FHE suites run without it.

## [0.8.1] - 2026-08-20

- Update `dashu-float` to 0.6 and `astro-float-num` to 0.3.7, and refresh the `bytemuck`, `serde`, `serde_json` and `anyhow` pins. The dashu 0.6 context operations return a `Result`, which the internal binary128 helpers now unwrap; no public API changes.
- Publish from CI through crates.io trusted publishing (`.github/workflows/release.yml`, triggered by a `v*` tag). The internal workspace dependencies carry a `version` requirement so `cargo publish` can package them.

## [0.8.0] - 2026-08-20

Adds two native CKKS bootstrapping families, **PaCo** (Coron & Seuré, [ePrint 2025/886](https://eprint.iacr.org/2025/886)) and **SHIP** (Cheon, Hanrot, Kim & Stehlé, [ePrint 2025/784](https://eprint.iacr.org/2025/784)), moves CKKS encoding onto a backend-resident op family, and lands a production-readiness pass over `poulpy-ckks` (typed errors, constructor-validated plans, scratch-carved intermediates, binary128-exact tables, four-layer consolidation). Keys every polynomial layout by an explicit word type (`ZnxWord` / `BigWord` / `DftWord`) and the DFT/big containers by their backend, collapses the limb-width model to a claimed precision plus an allocation, and reparameterizes evaluation keys by an auxiliary guard `k_aux`. Trims gadget products to their live limbs on exact-DFT backends, opens the strided GGLWE product and tensoring to backend overrides, and lands a round of AVX2/AVX-512 NTT, VMP and convolution optimization. Opens the layouts, key containers and benchmarks to non-host device backends, and adds a reference-vs-backend byte-parity test suite.

### `poulpy-hal`: word-keyed layouts

- Add `layouts::word` with the word traits `ZnxWord` (coefficient domain), `BigWord` (accumulator) and `DftWord` (prepared/DFT). A word names a byte-layout convention and governs sizing, element views, and serialization interpretation.
- **Breaking:** `Backend::ScalarPrep` → `DftWord` and `ScalarBig` → `BigWord` (with `size_of_scalar_prep`/`size_of_scalar_big` → `size_of_dft_word`/`size_of_big_word`), and every backend declares a coefficient word `type ZnxWord`.
- **Breaking:** the DFT/big layouts are re-keyed by word and backend: `VecZnxDft<D, W: DftWord, B>`, `VecZnxBig<D, W: BigWord, B>`, `SvpPPol`, `VmpPMat`, `CnvPVecL/R`. Containers of different backends are distinct types, so cross-backend interchange is a compile error. The `*Owned` / `*BackendRef` / `*BackendMut` aliases absorb the change.
- `VecZnx` / `ScalarZnx` / `MatZnx` gain a defaulted word parameter; `ReaderFrom`/`WriterTo` become word-generic over the unchanged wire format.
- **Breaking:** backend-bound constructors spell the backend through the owned alias (`VecZnxDftOwned::<B>::alloc`), and the DFT→big re-tag is `x.into_big()`.
- `x.to_backend_ref()` / `x.to_backend_mut()` infer the backend everywhere: no turbofish.
- Add `layouts::layout_compat`: per-container `unsafe` markers a backend pair declares to assert byte-identical layouts, unlocking the zero-copy `x.into_backend::<B2>()`. Validated by the new `test_suite::word_compat`.
- `ZnxView::raw()`/`at()` assert the element view fits the buffer instead of reading out of bounds.
- **Breaking:** `Backend` requires `PartialEq + Eq`. The DFT/big containers implement `PartialEq`/`Eq` on the representation.

### `poulpy-cpu-ref` / CPU backends: prime-set-parameterized NTT words

- **Breaking:** `PrimeSet` is lane-generic (`type PrimeElem`, GAT `type Lanes<T>`), and it, `LaneElem`, `LaneArray` and `CrtWord` move to `poulpy_hal::layouts::crt` (re-exported from `ntt4x30` for compatibility). CRT reconstruction constants move to the family extensions `PrimeSetCrt4` and `PrimeSetNtt3x42Ifma`.
- **Breaking:** `Q120bScalar` becomes an alias of the new block type `CrtWord<P: PrimeSet, T: LaneElem>`; tuple construction must spell `CrtWord([…])`. `Q126Scalar` stays a sizing-only `DftWord`.
- All CPU backends declare `type ZnxWord = i64`, and every backend marker derives `PartialEq, Eq, Hash`.
- Each accelerated crate declares `layout_compat` markers against its reference sibling (FFT64 and NTT4x30 families), validated by `word_compat`. `VmpPMat` and `CnvPVec` markers are intentionally absent; `NTT3x42Ifma` pairs with nobody.
- `poulpy-cpu-arm`: the NEON ntt4x30 modules are repaired and a native aarch64 `neon` CI job checks, lints, and tests the crate.

### `poulpy-bench`

- All 19 `core` runners drop `OwnedBuf = Vec<u8>`, the host-view bounds and `HostBackend`, so a device backend can run them. Non-measured operands are built on a host staging module and transferred in (`core::fill`) rather than encrypted, so the tested backend needs no sampling kernel.
- Fixed: the operation and tensor runners timed on zeroed buffers, which can hit float-FFT denormals.

### `poulpy-core`

- The prepared layouts follow the HAL re-key transparently; their derived `Eq` is dropped in favor of `PartialEq`.
- **Breaking:** remove `GGLWEPreparedVmpPMatRef`; use the inherent `GGLWEPrepared::data()`, which is generic over the buffer.
- Remove 47 dead `#[allow(private_bounds)]` from `oep`.
- **Breaking:** remove the exported `impl_glwe_rotate_impl_from!`; the blanket `GLWERotateImpl` impl supersedes it.
- **Breaking:** `GLWEKeyswitchInternal` and `GGLWEProductDefault` become public; they appear in the bounds of the public `glwe_keyswitch*_default` functions.
- The `*Default` override surfaces are no longer `#[doc(hidden)]`; `oep`'s module docs describe the `*Impl` / `*Default` split with a worked override.
- The gadget-digit width rule moves to the `GLWEKeyswitchDefault` contract and shared `gglwe_product_digit_output_size` helper; `GGLWEProductDefault` passes its accumulation count through `GGLWEProductDigitsStridedImpl` as a shape-derived `product_limbs` spill window instead of a fixed two limbs.
- Gadget products on an exact-DFT backend trim their working width to the limbs that can still affect the result, through the single `gadget_product_output_size`. It takes both operand precisions, so a cross-radix product is sized from its own shape. Approximate-DFT backends keep the full `key_work_size` width.
- **Breaking:** remove `GGLWEInfos::work_size` and `GGSWInfos::work_size`; the width is derived by `gadget_product_output_size` from the operand precisions, which `work_size` did not see.
- CKKS encapsulated ModUp composes ordinary dense-to-sparse and sparse-to-dense key switches around ModUp; backends can override `CKKSEncapsulatedModUpImpl` to fuse it.
- Relax to `D: Data`: `GGLWE` / `GLWEPlaintext` / `LWEPlaintext` / `GLWETensor` `data()` / `data_mut()`, `GLWESwitchingKeyDegrees(Mut)`, `Get`/`SetGaloisElement`, `GetDistribution(Mut)`, `SetBase2k`.
- **Breaking:** `GGLWEAtBackendRef`/`Mut` and `GGSWAtBackendRef`/`Mut` become public; `GLWESwitchingKey`, `GLWEAutomorphismKey` and `GLWETensorKey` delegate them.
- **Breaking:** remove `ModuleTransfer` and the inherent `Layout::to_backend` methods (29 in total) in favour of `api::TransferInto`, which writes into a destination the caller allocates: `src.transfer_into(&mut dst)`. Checks shape, not just byte length.
- **Breaking:** `test_suite` splits into `test_suite::noise` (the existing scheme-correctness suite, moved wholesale) and `test_suite::parity`. `core_backend_test_suite!` is unchanged.
- Add `BackendGLWESwitchingKey<BE>` / `BackendGLWEAutomorphismKey<BE>`.
- Add `test_suite::parity` and `core_parity_test_suite!`: runs one operation on a reference and a tested backend over identical uniform inputs and asserts byte equality. Covers key-switch (GLWE, assign, GGLWE), the strided GGLWE product hook, automorphism, external product, tensoring and the other keyless GLWE operations. Needs no secrets, encryption or noise model. An optional `shapes = ParityShapes { .. }` restricts the rank and `dsize` sweep for a backend with a narrower envelope.
- `poulpy-cpu-avx`, `-avx512` and `-arm` run the parity suite against their `poulpy-cpu-ref` sibling, for the FFT64 and NTT4x30 families.

### `poulpy-hal`

- **Breaking:** `Backend` gains `len_bytes_ref` / `len_bytes_mut`.
- **Breaking:** replace `TransferFrom` with the buffer traits `CopyToHost` / `CopyFromHost` and the free `transfer_buf_into(src, dst)`; the ~44 hand-written per-backend-pair impls are removed. `HostStaged` becomes `Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost>`.
- Add `layouts::vec_znx_backend_ref` / `vec_znx_backend_mut` / `vec_znx_reborrow_backend_mut`; `test_suite` re-exports them.
- Add `HalModuleImpl::Config` (defaulting to `()`) and `new_with(n, config)`, mirrored on `ModuleNew`, for device selection at construction. Requires `#![feature(associated_type_defaults)]`.
- Add `ModulePlanCache`, a per-`Module` typed cache of immutable plan families with a `with_or_create` accessor, plus `unsafe trait ModulePlanCacheProvider` for backend handles that own it.
- **Breaking:** `Backend` gains two required methods, `copy_view_to_host` and `copy_host_to_view`.
- **Breaking:** remove the public `set_size` / `with_size` resizing API from `VecZnx`, `VecZnxDft` and `VecZnxBig`; temporary compute widths are scoped views.
- **Breaking:** remove `max_size` from the layout family. `size` is both the working and the allocated width, fixed at construction, and narrowing is exclusively a borrowed view (`with_size_mut`, and the new `vec_znx_backend_mut_with_size`). Removed: `VecZnxShape::max_size`, the inherent `max_size()`, `from_data_with_max_size`, `vec_znx_alloc_with_max_size`; `VecZnxShape::new` loses its fourth argument. The `VecZnx` wire format drops one `u64` field.
- **Breaking:** split `ZnxInfos` into a shape trait plus `VecZnxInfos` (adding `cols()`) and `MatZnxInfos` (adding `rows()`, `cols_in()`, `cols_out()`); `poly_count()` becomes required. `ZnxView` now requires `VecZnxInfos` and the matrix containers expose inherent `raw()` / `raw_mut()` instead.
- Re-document `Module` as a multi-ring execution context: `n()` is the maximum ring degree, with the new alias `max_n()`.
- Add `Backend::DFT_IS_EXACT` (default `false`), declaring that the DFT round-trip is exact rather than floating-point. Consumers use it to decide whether a working width may be trimmed.
- Add `VecZnxDft::with_limb_range_mut(start, end)`, a borrowed view of a limb window, replacing manual `region_mut` / `from_data` carving.

### `poulpy-core`

- **Breaking:** the limb-width model collapses to two quantities, the claimed precision `k()` (with derived `size()`) and the allocation `max_size()`/`max_k()`. `LWEInfos::max_size()` now means the allocation everywhere. The `Compact` and `SetSize` traits are removed: narrowing a result means allocating the destination at the `k` you want. Plaintext operands are the one typed exception, declaring their consuming width through the new `IntPolyInfos::encoded_k()`, which every plaintext-consuming op now bounds.
- Fixed, uncovered by the width-model collapse: the inherent `GLWE::max_size()` shadowed the `LWEInfos` impl with different semantics; `GLWETensor` reported its allocation where siblings reported the stored width; the linear-transformation scratch budget read the stored width while claiming capacity; `SetSize::set_size` wrote a width `size()` did not read back.
- `LWEInfos::max_size()` is read off the payload for buffer-backed types; the two `LWE::validate_shape` checks asserting `size <= max_size` are dropped as no longer representable.
- Add `GLWESecret::fill_binary_coeffs(col, coeffs)`, installing a caller-provided binary coefficient vector and tagging `BinaryFixed(hamming_weight)`, for structured secrets.
- Add `GLWEPlaintextReborrowBackendRef` / `GLWEPlaintextReborrowBackendMut`.
- Export the `view_wrapper!` and `impl_glwe_infos!` macros for downstream nominal view types.

### `poulpy-core`: evaluation-key parameterization (`k` → `k_aux`)

- **Breaking:** every gadget-key layout stores an auxiliary guard `k_aux` in place of the total precision `k`, derived as `k() = dnum·dsize·base2k + k_aux` and enforcing `k_aux ≥ dsize·base2k`. A former `Dnum(d)` key with an implicit zero guard migrates to `Dnum(d−1)` with `k_aux = dsize·base2k`.
- **Breaking:** the gadget-key operations (`glwe_keyswitch`, `glwe_external_product`, `glwe_automorphism`, `glwe_tensor_relinearize`, and their compositions) no longer take an output/working size argument; it is derived from the input's `k` and the key's `(dsize, k_aux)`.
- **Breaking:** key allocation and byte-sizing signatures move to a uniform `(…, dnum, dsize, k_aux, rank…)` order.
- The noise model follows: `GGLWENoiseModel` / `GGSWNoiseModel`, blanket-implemented over the layouts, take the operand and the error variances only (`ksk.log2_std_noise_keyswitch(&ct_in, …)`). Evaluated at the key's real precision, the bounds gain the operand's carried error and the uncovered decomposition residue.

### `poulpy-ckks`: reusable approximation planning
- Add the `approximation` module with host-side single- and multi-interval Remez minimax fitting, precision/depth-based degree selection, and composite sign coefficient generation. Add `PolynomialApproximation` and `CKKSApproximationOps` for reusable interval mapping and prepared BSGS evaluation, including exact power-of-two scale and modulus-consumption accounting.

### `poulpy-ckks`: PaCo bootstrapping

- Add PaCo as a native CKKS operation refreshing selected polynomial coefficients without ModUp or EvalMod: blind rotation → partial CoeffsToSlots → product fold (the EvalMod substitute) → SlotToCoeff′, driven by a structured low-weight secret.
- Evaluation is `api::CKKSPaCoOps` on `Module<BE>`, all outputs caller-allocated: `ckks_paco_bootstrap_direct_into` (input already under the PaCo secret), `ckks_paco_bootstrap_into` (dense→PaCo switch first), the bounded-parallel variants, the `_tmp_bytes` queries, and `ckks_paco_coeff_encodings`.
- The direct entry points produce a leveled output: the caller allocates at any `k ≤ PaCoContext::max_output_k`, and the branch runs the whole circuit at the correspondingly narrower width.
- Invariant-bearing data lives in `layouts`: `PaCoPlan` and `PaCoDFTPlan`, `PaCoContext::compile` (plaintext material only), `PaCoSecretSpec`, `PaCoKeySet` → `PaCoKeysPrepared`, the `PaCoKeys<BE>` access trait, and `PaCoKeyParameters` fingerprinting the key-defining dimensions independently of any schedule.
- Parallel evaluation uses the caller plus a borrowed slice of reusable `PaCoWorker` contexts under `std::thread::scope`; branches recombine in sequential order, workers are validated before any output mutation, and an empty slice degenerates to the sequential path.
- The only PaCo-specific backend hook is coefficient encoding, via `oep::CKKSPaCoCoeffEncodingImpl<BE>`. The host reference `encoding::paco_coeff_encodings_host` is public and `poulpy-cpu-ref` exports `impl_ckks_paco_coeff_encoding!`, which every CPU backend invokes.
- `PaCoSlotOrder` (`Natural` or `BitRevLow`) selects the mid-pipeline slot convention: same budget and recovered coefficients, different Galois key set, so persisted bundles must pin both schedule and slot order.
- Scalar precision is sealed behind `api::PaCoScalar` (`f64`, `Quad`); compilation rejects `log_q ≥ F::MANTISSA_BITS`. The DFT-convention divergence from the paper's reference implementation is specified in `docs/spec/paco_dft_convention.md`.
- Backend-generic tests cover plan rejection, secret packing, coefficient encoding, the folds, both partial transforms, direct and encapsulated bootstraps, ordered parallel recombination, and output metadata, against an independent cleartext oracle.
- Security note (`docs/paco.md`): PaCo assumes a non-standard structured sparse secret; encapsulation keeps the application key dense but does not remove that assumption on the bootstrap key material.

### `poulpy-ckks`: SHIP bootstrapping

- Add SHIP as a native CKKS half bootstrap refreshing a one-limb bottom ciphertext into a slots-domain ciphertext without ModUp or EvalMod: dense→sparse encapsulation, then per support slot theta-column masking and hoisted base-B mux blind rotations, closed by a binary product tree over the `h + 1` factors and one conjugation.
- Evaluation is `api::CKKSShipOps<BE, F>`: `ckks_ship_bootstrap_into`, `ckks_ship_bootstrap_complex_into` (requires the `omega_2` mask set), `ckks_ship_bootstrap_tmp_bytes`, and `ckks_ship_coeff_encodings`. The scalar is sealed behind `api::ShipScalar`.
- Invariant-bearing data lives in `layouts`: `ShipPlan`, `ShipSecretSpec`, `ShipKeySet` → `ShipKeysPrepared`, fingerprinted by `ShipKeyParameters`. Key generation is integrated (`ShipKeySet::generate`): the mux keys are non-standard rank-2 → 1 switching keys no core keygen produces.
- The only SHIP-specific backend hook is coefficient encoding, via `oep::CKKSShipCoeffEncodingImpl<BE>`, with the public host reference `encoding::ship_coeff_encodings_host` and the `impl_ckks_ship_coeff_encoding!` macro.
- Backend-generic tests cover a cleartext replica of the paper's Algorithm 1, the HMuxRot primitive, and end-to-end real and complex bootstraps on every CPU backend at `f64` and `Quad`.

### `poulpy-ckks`: backend-resident encoding

- **Breaking:** the host-side `Encoder<T>` (and `encoding::reim`) is replaced by `api::CKKSEncodingOps<BE>` on `Module<BE>`: `ckks_encode_coeffs_into` / `ckks_decode_coeffs_into` and `ckks_slots_to_coeffs_assign` / `ckks_coeffs_to_slots_assign`, plus the provided `ckks_encode_slots_assign_into` / `ckks_decode_slots_into`.
- Add `CKKSEncodingHostOps` with the host-slice adapters, and the scalar-generic backend seam `oep::CKKSEncodingImpl<BE, F>` over the new `api::CKKSEncodingScalar`, whose per-precision plans resolve through the HAL plan cache so `f64` and `Quad` coexist on one module.
- Add the `CKKSEncodingBuffer<D, F>` layout family: a backend-resident planar `[re|im]` workspace with views, scratch carving, and host transfer helpers.
- **Breaking:** `CKKSPlaintextVecHostCodec` drops its separate sparse codec; `encode_host_floats` / `decode_host_floats` are the unified stride-aware path driven by `CKKSMeta::log_sparsity`.
- **Breaking:** `EvalMod::from_literal` is replaced by the free generic `compile_eval_mod` / `compile_eval_mod_exp`, which encode on any backend.

### `poulpy-core` / `poulpy-ckks`: polynomial evaluation
- Add explicit parity folding for real and complex BSGS polynomials through `Polynomial::fold_parity`, `Polynomial::decompose_bsgs_folded_with`, and `EncodeBSGS::encode_bsgs_folded_with`: even/odd monomial polynomials become `Q(x²)` / `x·Q(x²)`, while even/odd Chebyshev polynomials become `Q(T₂(x))` / `x·Q(T₂(x))`. The encoded degree is halved, the input transform is carried by `BSGSPolynomial`, and depth/budget accounting includes the initial square or `T₂` evaluation and the odd final multiplication. Folding remains opt-in because it can increase multiplicative depth for some degree/split-strategy combinations.
- Avoid EvalMod's scratch-input → owned-input copy for identity base polynomials by transferring the relabelled owned input directly into the shared power basis. Plans without an inverse no longer reserve a full scratch ciphertext for that copy; inverse plans retain only their post-composition working copy.

### `poulpy-ckks`: hardening, typed errors, and plan validation
- **Breaking:** every fallible op-trait method across all four layers now returns `CKKSResult<T>` (= `Result<T, CKKSError>`) instead of `anyhow::Result<T>`. `CKKSError` is `#[non_exhaustive]` with `Composition(CKKSCompositionError)` and `Internal(anyhow::Error)` variants, a `composition()` accessor for recoverable conditions, and a downcasting `From<anyhow::Error>` bridge; `CKKSCompositionError` is now `#[non_exhaustive]` and gains `InvalidPlan` and `PreparedOperandLayoutMismatch`. Callers using `?` into `anyhow` keep working; explicit `anyhow::Result` signatures must be updated.
- **Breaking:** naming sweep on the api layer: `CKKSEncrypt` → `CKKSEncryptOps`, `CKKSDecrypt` → `CKKSDecryptOps`, `DFTOps` → `CKKSDFTOps`, aligning every op family on the `CKKS*Ops` convention.
- **Breaking:** `ckks_encrypt_sk` (and `BootstrappingContext::generate_keys`) now take the randomness sources in the order `(source_xe, source_xa)` — a positional swap of the two `&mut Source` parameters; call sites must be updated by hand since the types are identical.
- **Breaking:** `ckks_decrypt` now decrypts into the **destination plaintext's preset** `(log_delta, log_budget)` frame instead of stamping the ciphertext's metadata onto it, so a caller can extract at a different precision; preset `pt.set_meta(ct.meta())` for the previous behavior. Frame mismatches surface as typed `PlaintextAlignmentImpossible` / `PlaintextBase2KMismatch` errors.
- **Breaking:** the CKKS multiplications now produce the result at the **destination's requested `dst.k()`** — rounding the low bits value-preservingly when `k` is narrower than the natural product — instead of filling the destination to its limb-aligned buffer capacity (`dst.max_k()`). This is uniform across `ct × ct` (`ckks_mul_*` / `ckks_square_*`, `get_mul_ct_params`), `ct × pt` (`ckks_mul_pt_vec_*`, `get_mul_pt_params`), the prepared `ckks_mul_prepared_assign`, and the diagonal `ct × pt` inside linear transformations. Allocate the destination at exactly the `k` you want the product at; a max-width destination is unchanged. This is what lets a leveled consumer (the PaCo blind rotation) evaluate its whole downstream circuit at a lower, cheaper torus width, and it fixes value corruption when the reduction target is non-`base2k`-aligned under a wide scale (e.g. `Quad`'s `log_delta > base2k`). The `CKKSMulOps` metadata rule is documented against `dst.k()`.
- **Breaking:** `DFTPlan` is now constructor-validated: fields are private, plans are built with `DFTPlan::new(kind, schedule, format, meta)` + `with_scaling` (rejects non-finite/non-positive values) + `with_bit_reversed`, the per-factor schedule is the new `FactorSchedule` of `FactorStep { depth, giant_step }` (making the parallel-array length invariant unrepresentable), the `meta` field is renamed `coeffs_meta` and carries the new lightweight `CoeffsMeta` (`k` + `CKKSMeta`) instead of a full `CKKSLayout`, `DFTPlan::check` is removed, and the schedule-introspection queries (`galois_elements`, `diagonal_indexes`, `num_diagonals`) are infallible instead of panicking on malformed plans.
- Add `DFTPlan::with_optimal_bsgs(log_n)` to derive structure-aware, rotation-count-balanced BSGS widths for every DFT factor without changing factor depth or modulus consumption.
- **Breaking:** `BootstrappingPlan` is a constructor-validated recipe: `new` requires the pipeline and explicit techniques, checks stage directions and sparse-secret weight, and enforces EvalRound+'s power-of-two `f_mod_interval` constraint. Sparse-secret encapsulation is recipe-owned; `EncapsulationKeysLayout` now contains only the two switching-key shapes, and key generation/execution reject layouts or key bundles whose presence disagrees with the recipe.
- Add S2C-first (slim) CKKS bootstrapping as `BootstrappingPipeline::S2CFirst`: the standard `ckks_bootstrap` orchestrator evaluates `SlotsToCoeffs → ModUp → CoeffsToSlots → EvalMod` with scratch-carved intermediates and the shared ModUp, DFT, and EvalMod blocks. `BootstrappingPlan::{pre_mod_up_consumed_bits, post_mod_up_consumed_bits, input_k, bootstrap_k}` place the bottom S2C cost below ModUp and the C2S plus EvalMod cost at the top of the raised modulus. `ckks_bootstrap_real` skips the imaginary nonlinear branch for known-real slots. Backend-generic end-to-end tests cover both slot domains, output precision and metadata, and the typed insufficient-input-budget error.
- Add CKKS functional bootstrapping (`ckks_functional_bootstrap` and the shared-power-basis `ckks_functional_bootstrap_multi`) on the standard S2C-first `BootstrappingPlan` / `BootstrappingContext`. The implementation reuses slim bootstrapping's S2C, ModUp, C2S, key generation, scratch layout, and recipe budget accounting, replacing EvalMod with encoded general or binary LUT evaluation; `ckks_functional_bootstrap_real` skips the imaginary LUT branch for known-real slots. LUT arity automatically determines `log_msg_ratio`, `functional_bootstrap_k` accounts for general and binary LUTs without charging binary evaluation for an unused EvalMod, and `ckks_functional_bootstrap_tmp_bytes` includes the LUT coefficient layout. Backend-generic tests cover the general, multi-LUT, binary, real-slot, mismatched-ratio, invalid-context, layout, scratch-sizing, and insufficient-input paths.
- Support EvalRound+ with S2C-first CKKS bootstrapping: after the low-modulus SlotsToCoeffs and ModUp, the orchestrator evaluates the low- and high-precision CoeffsToSlots branches and cancels the raised integer and low-precision transform error.
- Bootstrapping, eval_mod, and the linear-transformation wrappers now carve every working ciphertext from scratch instead of heap-allocating (up to seven owned ciphertexts per bootstrap previously); the new `ckks_bootstrap_tmp_bytes` sizing entry point is wired through all four layers and charges the full pipeline, and `ckks_eval_mod_tmp_bytes` charges its working copy. `ensure_uniform_diagonal_scale` rejects hand-built linear transformations with heterogeneous diagonal scales.
- Generalize DFT-matrix and eval-mod table generation over the scalar via the new `DftScalar` trait: per-factor scale roots use `nth_root_scalar` (f64 seed + Newton–Raphson refinement in `F`), the 256-bit CosHK solve lands through the mantissa-exact `fbig_to_scalar` triple-double decomposition, and `approximate_cos` is scalar-generic — so `Quad` matrices are binary128-exact instead of f64-rounded. `Quad` is now `bytemuck::Pod`, with new full-mantissa parity tests.
- Make `Quad`'s complete libm-backed `Float` surface portable on targets without a native binary128 libm: algebraic operations use pure-Rust `libm`, while transcendentals use guarded arbitrary-precision evaluation through `astro-float-num` followed by one round-to-nearest-even conversion to binary128. The `f128`/libquadmath dependency and its parity tests are now positively gated to x86_64 Linux GNU, fixing Intel macOS builds without changing `Quad`'s public type or ABI; the optional `libquadmath` feature remains the accelerated backing on that supported target and is a no-op elsewhere.
- **Breaking:** the `crate::leveled::*` backwards-compatibility re-export shim is removed; use the canonical `api::` / `oep::` / `delegates::` / `default::` / `layouts::` paths.
- **Breaking:** the backend-conformance `test_suite` module is now gated behind the new `test-utils` feature (backends enable it from dev-dependencies; production dependents no longer compile ~17k lines of test code), and the dead `enable-ckks` / `enable-avx` / `enable-neon` feature flags are removed from `poulpy-ckks` itself (backend crates keep their own opt-in features). Dependencies: `rand` / `rand_distr` dropped, `num-traits` and `paste` added.
- The bootstrapping conformance test now asserts a measured precision floor (`MIN_AVG_LOG2_PREC = 24.0` bits, ~4 bits under the observed 27.5–28.3) on every configuration, and the retired host reim encoder survives only as the test-suite reference oracle (`test_suite/reference_encoder.rs`).
- **Breaking:** with the `poulpy-core` two-quantity collapse, CKKS ciphertexts no longer compact: the per-op `dst.compact()` calls, the compacted-destination re-expansion in `ckks_mul_pt_vec`, and the `Compact`/`SetSize` impls and bounds are all removed — a ciphertext's buffer stays at its allocated width for its whole life, and `max_k()` uniformly means the allocation. In its place the **effective-`k` width rule is uniform across the ciphertext-side API**: the ct×ct tensor intermediates (`ckks_mul_*`/`ckks_square_*`/`ckks_mul_add_*`) are carved at the operands' effective `max(a.k, b.k)` (matching the `cnv_offset` rule, which was already expressed on effective `k`), the add/sub/copy/neg/rotate/pow2/composite alignment offsets are computed against the destination's requested `dst.k()` (previously the buffer's limb-aligned allocation), and `ckks_mod_up_into` widens to `dst.k()` — everything narrows by allocating (or relabeling) the destination at the `k` you want, and the api-doc offset formulas are updated accordingly. Plaintext operands are the one *typed* exception: they are integer polynomials, not Torus elements, and every plaintext-consuming path (ct×pt masking and `cnv_offset`, codec digit alignment, LT diagonal-scale uniformity, carry-verb constants, the encoding ops) reads their `IntPolyInfos::encoded_k()` — the declared width of the encoded integer — with the operand bounds requiring the trait across the api/oep/delegates/default layers. Masking an integer polynomial at its `k` (which labels claimed precision only) would destroy data limbs; the trait makes that width a stated property instead of a `max_k` convention.
- Fixed, uncovered by the width-model collapse: the ct×ct tensor intermediate was sized from the physical width while its own `cnv_offset` rule used effective `k` (masked by compaction; surfaced as a measured 12-15% EvalMod cost once compaction was removed, until the sizing was aligned); `validate_backend_storage_capacity` compared the key's reported stored width against its backend view's allocation and passed only because keys were full-width; and the add/sub/copy/rotate/pow2 families aligned results to the destination's allocation while the mul family used the requested `dst.k()`. Measured on the standard bootstrapping pipeline, every stage's precision is bit-identical to the compacting implementation (C2S snr 44.65/46.29, EvalMod snr 27.09/28.11, bootstrap avg 27.56/28.05) at unchanged EvalMod timing.

### `poulpy-ckks`: PaCo branch count derived from the input

- **Breaking:** `ckks_paco_bootstrap{,_direct,_parallel,_parallel_direct}_into` no longer take `kappa`; it is derived as `N/(C*2^log_sparsity)`, the branch count that refreshes every live coefficient of the input and no known-zero one. Sparse inputs are accepted (the `log_sparsity == 0` requirement is gone), and the output carries the input's own sparsity.

### `poulpy-ckks`: slot-kind metadata

- **Breaking:** `CKKSMeta` gains `slots: SlotsKind` (`Real` / `Complex`, defaulting to `Complex`) recording which subfield the slots are known to live in; every `CKKSMeta` literal must name it. Ops compose it with `SlotsKind::join`, keeping `Real` only when every operand is `Real`; `mul_i` and linear transformations yield `Complex`, and the conjugate folds restamp `Real`. Read with `CKKSInfos::slots`, set with `SetCKKSInfos::set_slots`.
- **Breaking:** the plaintext-assign add/sub methods on the default and OEP layers now require `SetCKKSInfos` on their destination, matching the api layer.

### `poulpy-ckks`: four-layer consolidation

- `CKKSInfos` is a supertrait of `LWEInfos` with provided `log_delta` / `log_budget` / `log_sparsity`; implementors supply only `fn meta()`. New bound-alias traits `CKKSCtBounds` and `CKKSAtkBounds<BE>` replace the spelled-out clusters.
- The add and sub Default/OEP families are generated from one body by `ckks_carry_verb_default!` / `ckks_carry_verb_oep!`; the tensor-multiply variants share `tensor_mul_core`; the multiply-then-accumulate composites share `mul_then_combine`; `impl_ckks_infos!` generates the metadata/view bundle.
- The DFT family joins Pattern A (default-bodied `DFTDefault` / `DFTMatrixDefault`, `impl_ckks_dft_defaults!` reduced to two marker impls, partial overrides now possible), and `ckks_new_dft_matrix` moves onto the scalar-generic `DFTMatrixImpl<BE, F>` / `DFTMatrixDefault` pair.
- `CKKSModuleAlloc` is fully default-bodied over `ModuleCoreAlloc` and gains `ckks_ciphertext_alloc_with_rank`. The `CKKSImpl` bundle also covers `DFTImpl` and `CKKSEvalModImpl`; bootstrapping is a no-OEP composition over `CKKSBootstrappingOps` and `BootstrappingKeys`.
- `UnnormalizedCKKSCiphertextRefMut` is sealed (public for OEP signatures, no public constructor). Shared layout validation helpers move to `layouts/validation.rs`.

### `poulpy-cpu-ref` / `poulpy-cpu-avx` / `poulpy-cpu-avx512` / `poulpy-cpu-arm`

- Add `ckks_encoding.rs`, the CPU encoding implementation, behind `impl_ckks_encoding_fft64_f64!`, `impl_ckks_encoding_owned_for!` and `impl_ckks_encoding_owned!`. Every backend wires `f64` and `Quad`.
- **Breaking:** backend ring handles hold plan *sets* covering every power-of-two sub-dimension: `FFT64PlanSet` / `get_fft_plan(n)` replace `get_fft_table` / `get_ifft_table`, and `NttPlanSet` / `get_ntt_plan(n)` replace `get_ntt_table` / `get_intt_table`. Handles carry the new `ModulePlanCache`; the old cache names remain as aliases.
- Add `ckks_paco.rs` and `ckks_ship.rs` with the `impl_ckks_paco_coeff_encoding!` and `impl_ckks_ship_coeff_encoding!` macros, invoked by every backend.
- Add benchmarks binaries based on Criterion and the new `poulpy-bench` harness.
- Fix `poulpy-cpu-avx512` compiling with `enable-ckks,enable-avx512f` but without `enable-ifma`.

### `poulpy-core`: fused VMP and tensoring overrides

- Add the Core-owned `GGLWEProductDigitsStridedImpl` backend hook, including scratch sizing, which applies every gadget digit directly from an interleaved DFT input. The portable default materializes one digit at a time; accelerated backends can fuse the passes while preserving the reference digit-width schedule.
- **Breaking for backend implementations:** tensoring becomes an explicit `GLWETensoringImpl` opt-in through `impl_glwe_tensoring_default!` instead of a blanket implementation, allowing optimized backends to override the complete operation and its scratch bounds.
- The AVX-512 NTT backends specialize rank-one tensor apply and square at `n = 2^15` and `n = 2^16` with a direct three-product DFT path, avoiding the coefficient-domain diagonal cache and its associated scratch allocation; other shapes use the canonical all-rank implementation.

### `poulpy-cpu-avx` / `poulpy-cpu-avx512`: NTT, VMP and convolution optimization

- Pack `NTT3x42Ifma` transform-domain vectors and prepared convolution operands into two `u64` words per coefficient instead of three. The existing packed VMP representation is retained, reducing `VecZnxDft` and `CnvPVecL/R` storage and bandwidth by one third.
- Fuse multi-digit VMP on `NTT4x30Avx`, `NTT4x30Avx512` and `NTT3x42Ifma`: digit rows are read directly from their interleaved source, matrix data is streamed once per output block, and intermediate digit buffers are removed. The AVX accumulating save is also vectorized.
- Vectorize packed `NTT3x42Ifma` scalar-vector products, enlarge the cache-resident NTT base case, and fuse the forward NTT stages specialized for `n = 2^15` and `n = 2^16`.
- Fuse rank-one tensor convolution on the AVX-512 NTT backends, computing all three output columns while the prepared inputs remain resident.

### `poulpy-hal`: word genericity delivered

- **Breaking:** `VecZnx`, `ScalarZnx` and `MatZnx` lose the `= i64` default on their word parameter. Sizing follows, and `ZnxWord` gains `from_i64` so `FillUniform` and the secret samplers are word-generic. `layouts::encoding` stays bound to `ZnxWord = i64`.
- `Backend` gains provided sizing methods `size_of_znx_word`, `bytes_of_vec_znx`, `bytes_of_scalar_znx` and `bytes_of_mat_znx`, so sizing routes through the backend that owns the memory.
- Fixed: the `MatZnx` backend entry-view helpers computed offsets from the dense host formula while allocation goes through `Backend::bytes_of_mat_znx`; the stride now calls `B::bytes_of_vec_znx`. The same defect is fixed in the test suite's upload helpers and in `word_compat`, which did not assert matching `ZnxWord`.
- The generic test suites drop 11 `Backend<OwnedBuf = Vec<u8>>` bounds; `TestBackend` remains the single documented `ZnxWord = i64` pin.

### `poulpy-core`: backend-routed sizing

- **Breaking:** the cross-backend layout transfers (`ModuleTransfer::upload_*` / `download_*` and the `to_backend` wrappers) require both backends to share a coefficient word. Moving a value between backends whose limb axis differs is a re-encoding, left to a follow-up.
- Fixed: `lwe_bytes_of_from_infos` summed the body and mask sizes with no padding, under-counting the scratch alignment gap and surfacing as a scratch-exhaustion panic at small limb counts.
- Add `api::GLWEBytesOf<BE>` on `Module<BE>`, replacing host-pinned `Type::<Vec<u8>, i64>::bytes_of` calls at scratch-sizing sites.
- **Breaking:** `ModuleCoreAlloc` gains `type ZnxWord`, and `ModuleCoreCompressedAlloc` gains `type OwnedBuf` and `type ZnxWord`, returning `Self::OwnedBuf` across its constructors instead of `Vec<u8>`.
- The layout family is word-generic throughout; a handful of methods stay i64 by construction and say so.

### `poulpy-ckks`: word-generic containers

- **Breaking:** `CKKSCiphertext<D, W, S = Normalized>` and `CKKSPlaintext<D, W>` carry the coefficient word, the latter also dropping its `D = Vec<u8>` default. The new `CKKSCiphertextOwned<BE>` / `CKKSPlaintextOwned<BE>` aliases absorb the change at most call sites.
- Threading the two containers collapses 316 `Backend<ZnxWord = i64>` bounds to 10, all in the host float codec path.
- `CKKSConjugateImpl` states the `GLWEShift<BE>` bound it actually needs instead of pinning the word.
- Fixed: `delegates::eval_mod` sized backend scratch with the host `VecZnx::bytes_of` rather than `BE::bytes_of_vec_znx`.

### `poulpy-bin-fhe`: word-generic containers and host de-pinning

- **Breaking:** the storage types carry the coefficient word: `FheUint`, `FheUintPreparedDebug`, `LookupTable`, `BlindRotationKey` (both forms), `CircuitBootstrappingKey`, `BDDKey`, `GLWEBlindRetriever` and `LookupTableFactory`; `LookupTable` and `GLWEBlindRetriever` also drop their host defaults.
- 75 bounds state `Backend<OwnedBuf: HostDataMut + HostDataRef>` instead of `OwnedBuf = Vec<u8>`, and key allocation moves off host storage.
- What stays pinned is genuinely i64: the BDD and CGGI evaluation path writes message bits into i64 coefficients, so those items keep `ZnxWord = i64` where the requirement is created.
- Fixed: the compressed `CircuitBootstrappingKey` was generic over its buffer but held host key material, so a device instantiation would have compiled while silently holding it; and the CGGI blind rotation sized backend scratch with the host `VecZnx::bytes_of`.

### `poulpy-bench`: new generic benchmarking API

- Completely reworked the exisiting crate into a composable API for the backend crates to instantiate their benchmarks easily and with minimal boilerplate. The crate defines a generic runner for each backend operation, and a generic sweep driver that turns a table of runners plus a param sweep into Criterion groups. Backends can compose their own benchmark suites by selecting the operations they implement and the sweeps they want to run. See `poulpy-bench/README.md`.

### Build & Docs

- Add `docs/paco.md` and `docs/spec/paco_dft_convention.md`, and `docs/ship.md`; link them from `docs/README.md`.
- Update `docs/bootstrapping.md` for the constructor-validated plans, and add a Toolchain section to the `poulpy-ckks` README (nightly `#![feature(f128)]`, the x86_64 Linux GNU `libquadmath` fast path, the portable fallback elsewhere).
- Add native Intel macOS and Apple Silicon CI smoke coverage for the portable binary128 math surface and `Quad` CKKS encoding.

## [0.7.0] - 2026-07-09

Builds the full leveled-CKKS evaluation stack on the backend-generic core: BSGS polynomial evaluation, slot-domain linear transformations, the homomorphic DFT, EvalMod, and the complete bootstrapping pipeline. Adds the AArch64 NEON backend crate (`poulpy-cpu-arm`), a round of NTT/convolution/VMP performance work, and renames the multi-prime NTT families to `<primes>x<bits>` (entries below use the name current when the change landed).

### Renamed

- Rename the multi-prime NTT backend families to a `<primes>x<bits>` scheme: `NTT120` → `NTT4x30` and `NTT126` → `NTT3x42`, across the backend marker types, modules, CKKS test parameters and documentation. The `Q120` / `Q126` moduli and `Primes30` / `Primes42` prime sets keep their names.

### `poulpy-hal`

- Add scalar-automorphism and packed-matrix APIs: `ScalarZnxAutomorphismBackend`, `ScalarZnxAutomorphismAssignBackend`, `VecZnxTransposeBackend`, `VecZnxBigColWeightedSum`.
- Add reusable DFT-domain automorphism planning via `VecZnxDftAutomorphismPlan` and `VecZnxDftAutomorphism`.
- Add the accumulating convolution apply `Convolution::cnv_apply_dft_accumulate`, bit-identical to `cnv_apply_dft` followed by a DFT-domain add and reusing its scratch contract.
- Add the `VecZnxLshAddCoeffToCoeffBackend` / `VecZnxLshSubCoeffToCoeffBackend` hooks for left-shift-aligned coefficient accumulation.

### `poulpy-core`

- Add a scale-agnostic Baby-Step/Giant-Step polynomial-evaluation engine and the `GLWEPolynomialEvaluation` API. The engine owns only the combinatorial schedule and delegates every arithmetic operation, with all precision bookkeeping, to the scheme through `BSGSBabyOps` / `BSGSGiantOps`.
- Add the scheme-agnostic layouts `Polynomial`, `BSGSPolynomial`, `PowerBasis`, `Basis`, `Parity` and `SplitStrategy`: monomial and Chebyshev bases, interpolation, arbitrary intervals, Paterson-Stockmeyer decomposition under the `MinDepth` / `MinMult` strategies, and ahead-of-time accounting via `bsgs_consumed_bits` and `bsgs_eval_depth`.
- Factor the GLWE tensor product into `glwe_tensor_apply_loop` and add the hoisted-operand primitives `glwe_prepare_right` / `glwe_tensor_apply_prepared_right`.
- Allow GLWE plaintext add/sub alignment to left-shift when the encoded precision exceeds the ciphertext budget.
- Add the scheme-agnostic linear-transformation engine: `LinearTransformation` / `GiantStep` / `Diagonal`, the BSGS schedule types (`LinearTransformationLayout`, `Plan`, `Strategy`, `optimal_bsgs_giant_step`) and the `GLWELinearTransformations` trait. It evaluates `M·v = Σ_k diag_k ⊙ rot(v, k)` over raw GLWE and carries no scale notion.
- Add the convolution-domain prepared operand `PreparedDiagonal`, so the resident transform is `LinearTransformation<PreparedDiagonal>`, the same container as the streamed plaintext form. The left operand stays the separate `LinearTransformationBabySteps`.
- Add a streamed (unprepared-RHS) evaluation path preparing each diagonal through scratch. The per-giant product is dispatched by the `DiagonalProd` trait, so both paths share one driver behind a single `glwe_eval_linear_transformation_into` generic over `P`.
- Add the clear evaluator `Diagonals<T>` with the `Evaluate` / `DiagonalArithmetic` traits, used as the plaintext reference.
- Add packed LWE matrix support: `LWEMatrix`, `LWEMatrixLayout`, `LWEMatrixInfos`, `BackendLWEMatrix`, and the `lwe_matrix_alloc*` constructors.
- Add `GLWEExpandLWEMatrix` (GLWE into a matrix of LWE samples) and `LWEMatrixDecrypt`.
- Add the `GLWEMaskFill` / `LWEFillMask` traits for backend-generic mask generation from a `Source` or seed; compressed decompression uses them, and `GLWECompressed` exposes `data()` / `data_mut()`.
- Accumulate the `dsize > 1` keyswitch digits in place through `vmp_apply_dft_to_dft_accumulate`, dropping the per-digit scratch buffer.

### `poulpy-ckks`

- Add the scale-aware polynomial-evaluation bridge over the core engine: `PolynomialEvaluation`, the `EncodeBSGS` / `PowerBasisGen` extension traits, and `CKKSBSGSOps` dispatching onto the CKKS API so all scale math lives in the CKKS multiply ops.
- Add complex-coefficient evaluation via `ckks_eval_poly_complex_const_coeffs_from_power_basis`, taking a single `&ComplexBSGSPolynomial` and combining matched baby steps as `re + i·im`.
- Add the one-shot `ckks_eval_poly_real_const_coeffs` / `ckks_eval_poly_complex_const_coeffs`, which build the power basis internally.
- Add the hoisted `ct×ct` multiply: `CKKSMulOps::ckks_prepare_right` into the backend-resident `CKKSPreparedRight`, and `ckks_mul_prepared_assign`.
- Add backend-generic `eval_mod` (homomorphic `x mod 1`): `EvalModPlan` over `SinContinuous` / `CosContinuous` / `CosDiscrete` / `Exp`, with double-angle composition, optional arcsine, and a caller-selected BSGS split strategy. It runs at its own `f_mod_log_delta` through a budget-neutral scale round-trip, and `EvalModPlan::consumed_bits` charges the budget deterministically.
- Add the complex-exponential `Exp` variant, held by the new `EvalModBsgs::{Real, Complex}`.
- Add eval_mod OEP/delegate wiring (`CKKSEvalModImpl`, `CKKSEvalModOps`) and conformance tests for every variant.
- Generalize the fused multiply-add/plaintext paths over backend-owned layouts.
- Add conformance tests for CKKS polynomial evaluation.
- Add the CKKS linear-transformation API `LinearTransformationOps`. The CKKS layer owns the scale math and delegates evaluation to the core engine; the prepared-vs-streamed choice is which diagonal type `P` you pass, not a separate method. `ComplexDiagonals<T>` and `ckks_encode_linear_transformation_from_diagonals` build the encoded transform.
- Strengthen `assert_decrypt_precision` to run a ring-domain noise bound and the canonical-embedding per-slot assertion off one decryption.
- Add the homomorphic DFT (CoeffsToSlots / SlotsToCoeffs) via `DFTOps`: `ckks_new_dft_matrix` builds a factorized transform from a `DFTPlan` as the unprepared `DFTMatrix`, and `ckks_prepare_dft_matrix` promotes it to the resident `DFTMatrixPrepared`. Both forms share the unified linear-transformation evaluator, and the BSGS schedule is caller-supplied per factor.
- `DFTMatrix<BE, Dir, Fmt, R>` carries its direction (`Encode`/`Decode`) and output format (`Standard`/`Split`/`Repack`) as type-state, making a mismatch a compile error. Conformance tests compare each transform against an independent plaintext reference built from the dual encoding, not an `Encode∘Decode` round trip.
- Add budget/compaction maintenance for leveled pipelines: `SetCKKSInfos::compact_in_place` and `ckks_set_log_delta`. Every budget-consuming op compacts its result.
- Add CKKS bootstrapping (`CKKSBootstrappingOps`): the `ModUp → CoeffsToSlots → EvalMod → SlotsToCoeffs` pipeline. The only new primitive is ModUp (`ckks_mod_up_into`), a base-`2^base2k` digit shift rather than an RNS basis extension. `BootstrappingPlan` bundles the stage parameterization and budget accounting, and `BootstrappingContext::compile` lowers it once into resident matrices plus the encoded EvalMod.

### `poulpy-cpu-ref` / `poulpy-cpu-avx` / `poulpy-cpu-avx512`

- Add the `bootstrap_trace` example: the end-to-end bootstrapping pipeline as a standalone binary for sampling profilers.
- Implement the new transpose, weighted-sum, automorphism and packed LWE matrix defaults, with AVX and AVX-512 overrides for the automorphism paths.
- Rework the FFT64 convolution applies into fused column kernels behind `Reim4Convolution::reim4_convolution_apply` / `_pairwise_apply`, tiling output limbs over a sliding window and staging outputs so each destination cache line is written once. 2.3-3.5x faster on `FFT64Avx512`/`FFT64Avx`.
- Interleave the `NTT120Avx512` VMP prepared-matrix prime planes per chunk so the apply streams the matrix sequentially: `vmp_apply_dft_to_dft` up to 3.4x faster, `glwe_keyswitch` at n = 2^15 from 41.5 ms to 24.7 ms.
- Switch the NTT-family prepared convolution operands to block-major rows with the right operand in reversed limb order, deleting the per-apply pack/gather passes and moving the left operand's reduction to prepare time. The applies tile four output limbs per pass and reduce once per output. At 32768x14: `NTT126Ifma` apply -36% / pairwise -62%, `NTT120Avx512` `mul_ct` -15%.
- Fuse the `NTT126Ifma` forward level-0 twist into the first butterfly level and run the last three levels as one radix-8 pass (~5% faster); add n = 2^12 cross-backend idft conformance tests.

### `poulpy-cpu-avx`

- Fuse NTT levels in the `NTT120Avx` by-level phase: the forward level-0 twist and the inverse final normalization fold into adjacent butterfly levels, and above `FUSE_MIN_N = 2^15` pairs of levels fuse into radix-4 passes. Bit-identical to the prior radix-2 passes; forward/inverse ~34%/~26% faster at n = 2^16.
- Batch the q120b → i128 CRT reconstruction four coefficients at a time, transposing each residue block so the weighted sum accumulates vertically. Bit-identical and ~15% faster on the kernel.
- Vectorize the `b_to_znx128_avx2` final CRT reduction as a planar 4-wide fold mirroring the AVX-512 path. Bit-identical to the prior finalize.

### `poulpy-cpu-avx512`

- Migrate the `NTT126Ifma` DFT-domain representation from a 4-lane array-of-structs to a planar 3-prime layout, removing the wasted lane and the hand-written assembly CRT kernel.
- Optimize the planar transforms for pass efficiency (masked sub-8 remainders, transposed radix-8 head/tail, fused upper-level pairs, folded normalization/untwist), roughly halving the raw NTT/iNTT cost at log n = 15.
- Vectorize the planar CRT-to-i128 reconstruction with AVX-512-IFMA base-2^52 limb accumulation.
- Replace the `NTT126Ifma` forward/inverse kernels with radix-2 Cooley-Tukey / Gentleman-Sande passes over the three planes, under a lazy reduction feeding the Harvey/Shoup multiply directly. Above `BASE_NTT_SIZE` each plane is split depth-first to stay cache-resident.
- Add a `lazy_output` mode to the `NTT126Ifma` forward NTT, opted into by the prepare paths whose consumers re-reduce; the public DFT contract keeps the full reduction.
- Vectorize the `NTT120Avx512` `b_to_znx128` final CRT reduction with a planar 8-wide fold. Byte-identical to the reference.
- Replace the scalar modular accumulate in the `NTT120Avx512` VMP output save with a lazy SIMD fold reproducing it byte-for-byte.
- Pack the `NTT3x42Ifma` prepared VMP matrix into two u64 words per coefficient instead of three, cutting the streamed bandwidth by a third.

### `poulpy-cpu-arm`

- Add the `poulpy-cpu-arm` NEON/ASIMD backend for AArch64, exposing `FFT64Neon` and `NTT120Neon`. Hand-written kernels cover every accelerated HAL family; the rest inherit the portable `poulpy-cpu-ref` defaults, and the core/CKKS wiring is inherited through the `impl_*_defaults` macros. Opt-in via `enable-neon`, `compile_error!`ing off AArch64. Integer operations are bit-identical to the reference and FFT operations match within ULP. A `neon` CI lane runs the suite natively.
- Add the canonical-x NEON convolution-apply kernel `NttMulBbc1ColX2::ntt_mul_bbc_tile4_x2`, the analog of the AVX2/AVX-512 tiled kernels.

### `poulpy-bench`

- Add the `ckks_poly_eval` benchmark (degree and split-strategy sweep, reporting baby-step size and budget consumption), later extended to `NTT120Avx` and `NTT126Ifma`.
- Add the `ckks_eval_mod` benchmark over the `EvalModType` variants.
- Enable the `poulpy-cpu-avx` CKKS implementations in the `ckks-bench` feature.
- Add the `cnv_apply_dft_accumulate` sweep and a `glwe_tensor_relinearize` benchmark.

### Build & Docs

- Add `docs/linear_transformation.md` and `docs/polynomial_evaluation.md`.
- Refresh the `ckks_poly2` example onto Chebyshev interpolation, `PowerBasis` and the BSGS evaluator.
- Overhaul `docs/`: add `README.md` (index), `backends.md`, `bootstrapping.md`, `grafting-vs-bivariate.md`, and the `docs/spec/` linear-transformation specification; refresh `getting-started.md` and trim the root README.

## [0.6.0] - 2026-05-18

Completes the migration from the host-oriented HAL/backend plumbing to backend-generic HAL and core layers, so backends own buffers, scratch, and transfer paths explicitly, and adds the `poulpy-cpu-avx512` crate with three accelerated backends (`FFT64Avx512`, `NTT120Avx512`, `NTT126Ifma`).

### `poulpy-hal`

- The layout family stores private shape snapshots instead of public mutable fields, with explicit getters (`shape()`, `n()`, `cols()`, `size()`, `max_size()`) and metadata-only `with_size` / `set_size`.
- `Module` becomes the canonical allocation entrypoint for the raw coefficient-domain layouts and `VecZnxBig`; the host-owned `alloc` constructors and the explicit-`n` allocator surface are removed, and host staging goes through `Module::<HostBytesBackend>::new(...)`.
- **Breaking:** HAL compute traits take backend-native borrows and scratch explicitly: `Scratch` → `ScratchArena<'_, BE>`, `*ToRef` / `*ToMut` → `*ToBackendRef<BE>` / `*ToBackendMut<BE>`, and trait names move to backend-explicit forms (`VecZnxAddIntoBackend`, …).
- Add backend-owned interop: `Backend::{OwnedBuf, BufRef, BufMut}`, `HostBackend` / `DeviceBackend`, `TransferFrom<From>`, per-layout view aliases and reborrow traits, the `ScalarZnxAlloc` / `VecZnxAlloc` / `MatZnxAlloc` allocators, and `api::reim::{NegacyclicFFT, NegacyclicFFTNew}`.
- Add `VmpApplyDftToDftAccumulate` (fused `res += a · pmat` with limb offset), replacing the scattered apply-then-fold in `gglwe_product_dft`.
- Fix the convolution API: the output shift is renamed `cnv_offset` and moved to the front of the apply calls.
- **Breaking:** `Convolution::cnv_by_const_apply` takes a backend `VecZnx` plus `(b_col, b_coeff)` selectors instead of a raw coefficient slice.
- Replace the monolithic `oep::HalImpl` with per-family OEP traits (`HalModuleImpl`, `HalVecZnxImpl`, `HalVecZnxBigImpl`, `HalVecZnxDftImpl`, `HalSvpImpl`, `HalVmpImpl`, `HalConvolutionImpl`) and per-family defaults, so a backend overrides only the families it owns.
- Make `WriterTo` for `MatZnx` / `VecZnx` emit the canonical logical byte length and error on short backing storage; fix `ScalarZnx::write_to` to emit the full `n * cols` span.
- **Breaking:** remove `ReaderFrom` / `WriterTo` and `from_bytes` for the prepared DFT layouts (`SvpPPol`, `VmpPMat`), whose alignment assumes a power-of-two degree.

### `poulpy-core`

- Fix #158: `VecZnxScalarProduct` now computes the element-wise Hadamard product into `VecZnxBig`; callers needing the inner sum follow up with `VecZnxBigInnerSumBackend`, as LWE encryption now does.
- Add rank>1 support to `GLWEExpandLWE::glwe_expand_lwe` (with a new `_tmp_bytes`) and to `SecretConversion::lwe_secret_from_glwe_secret`, which produces the concatenated LWE key.
- Add `ModuleCoreAlloc` / `ModuleCoreCompressedAlloc` and migrate every allocation site; the wrapper `alloc` constructors become crate-private.
- **Breaking:** core traits follow the HAL backend-native convention (`ScratchArena<'_, BE>`, `GLWEToBackendRef/Mut<BE>`, prepared-key `...ToBackendRef<BE>`), and no extension point assumes host-slice views.
- Add `api::ModuleTransfer` for typed upload/download of every core object across backends.
- Thread the corrected convolution-offset semantics through the multiply and tensoring paths, pass explicit effective-`k` into them, and mask partial bottom limbs instead of assuming full stored width.
- Fix tensoring noise blowup when the output operand was smaller than the input.
- Split the public API into `api` / `delegates` / `oep` layers, reorganize the operation modules to match, and re-export the top level for ergonomics.
- **Breaking:** backend-default wiring is exported per family (`impl_encryption_defaults_full!`, `impl_glwe_trace_defaults_full!`, …); add `impl_glwe_rotate_impl_from!` for backends forwarding rotation.
- Move backend conformance suites into `src/test_suite`.
- Standardize prepared allocations on `DeviceBuf`, and rename the Module allocation/prepare helpers to struct-first names (`gglwe_prepared_alloc`, `glwe_secret_prepare`).
- **Breaking:** remove `ReaderFrom` / `WriterTo` for `LWESecret` and `GLWESecret`; secret material should use seeds or application-level transfer.

### `poulpy-cpu-ref` / `poulpy-cpu-avx`

- Refresh the FFT64/NTT120 references and glue for the private-shape layouts and the corrected `cnv_offset` API.
- Optimize NTT120 convolution on AVX by wiring the prep paths to backend kernels and restructuring the applies around prepacked x2 blocks.
- Add a row-prime-major prepared-matrix layout for the `NTT120Avx` VMP, streaming one prime plane at a time and reusing extracted input rows across the output-column loop.
- Reorganize backends around `hal_impl` / `hal_defaults` modules mirroring the per-family HAL surface, removing the legacy per-family modules.
- Export the FFT-table types the CKKS encoder needs (`FFT64ReimTable`, `FFT64AvxReimTable`), move the CKKS `poly2` example and backend tests into `poulpy-cpu-ref`, and add `poulpy-cpu-avx`'s opt-in `enable-ckks`.

### `poulpy-cpu-avx512`

- Add `FFT64Avx512`: f64 complex-FFT backend gated on `enable-avx512f`, mixing AVX-512F REIM butterflies with AVX2+FMA REIM4 kernels.
- Add `NTT120Avx512`: Q120 NTT over four ~30-bit primes, gated on `enable-avx512f`, with AVX-512F butterflies, row-prime-major VMP and convolution overrides, and a fused iNTT + Garner compaction hook.
- Add `NTT126Ifma`: Q126 NTT over three ~42-bit primes on AVX-512-IFMA, gated on `enable-ifma`. Implements every HAL family against IFMA kernels; the post-iNTT CRT-to-i128 reconstruction is a hand-written assembly kernel fusing IFMA Garner reduction with a BMI2/ADX carry chain.
- `enable-avx512f` / `enable-ifma` `compile_error!` when the matching target features are missing instead of emitting binaries that SIGILL; with neither feature the crate compiles to an empty shell.
- Opt-in `enable-ckks` mirrors `poulpy-cpu-avx`, running the full CKKS suite against the three backends.

### `poulpy-ckks`

- Implement a fully backend-generic leveled CKKS evaluator: every operation is generic over any `poulpy-hal` backend.
- Organize the interface into the same four layers as `poulpy-core` (`api` / `oep` / `delegates` / `default`), with backends opting into defaults via `impl_ckks_*_defaults!`.
- **Breaking:** collapse the plaintext family split into a unified `CKKSPlaintext<D>` plus `CKKSPlaintextVecHostCodec<F>` for host float encode/decode and `CKKSModuleAlloc` for module-first allocation.
- Add api trait families for copy, affine/composite helpers, and imaginary-unit operations.
- Add the normalization typestate `CKKSCiphertext<D, S = Normalized>` with `UnnormalizedCKKSCiphertext<D>` for fused linear operations before an explicit `normalize`, plus `CKKSCiphertextViewMut` for allocation-free in-place writes.
- Add `ScratchArenaTakeCKKS`, the backend-view aliases, and the `CKKSAddOpsUnnormalized` / `CKKSSubOpsUnnormalized` / `CKKSAffineOps` / `CKKSImagOps` / `CKKSCopyOps` traits.
- Move the conformance suite into `src/test_suite/`, wired into `poulpy-cpu-ref` via `ckks_backend_test_suite!`.
- Remove the concrete-backend dependencies from `poulpy-ckks`; backend crates opt in from their side.
- Preserve the historical `crate::leveled::api` paths as a re-export shim.
- **Breaking:** CKKS backend wiring moves to per-family traits and macros (`CKKSAddImpl`, `impl_ckks_add_defaults!`, …).
- Document unnormalized operations: signed-digit behavior, worst-case O(n) growth, Irwin-Hall O(√n) typical growth, and the `n ≤ 2^(63 − base2k)` bound against i64 overflow.

### `poulpy-bin-fhe`

- **Breaking:** bin-FHE traits follow the backend-owned surface (`ScratchArena<'_, BE>`, `...ToBackendRef/Mut<BE>`), and many entrypoints require `BE: Backend<OwnedBuf = Vec<u8>>` plus `ModuleCoreAlloc`.
- Move the public constructors to module-first forms (`FheUint::alloc`, `LookupTable::alloc`, `GLWEBlindRetriever::alloc`, `CircuitBootstrappingKey::alloc_from_infos`), add `LookupTable::to_backend`, and align the key/prepared layouts with the refactored core.
- Remove the unnecessary `BE: 'static` and redundant `ScratchArenaTakeCore` bounds.
- Add `ReaderFrom` / `WriterTo` for `CircuitBootstrappingKey` and `BDDKey<Vec<u8>>`, with stable sorted-Galois ATK serialization.
- **Note:** the crate is not yet backend-agnostic; it still depends unconditionally on `poulpy-cpu-ref` and exposes host bounds in several public APIs.

### `poulpy-bench`

- Update the suites for the new HAL/core APIs, remove the remaining layout-field assumptions, and add shared host-upload / `ModuleTransfer` staging helpers so fixtures can target any backend.
- Split the bench opt-ins by family (`hal-bench`, `core-bench`, `bin-fhe-bench`, `ckks-bench`) instead of one monolithic feature, and add `enable-avx512f` / `enable-ifma`.

### Build & Docs

- Refresh the READMEs and document the shared `api` / `oep` / `delegates` / `default` layering and backend-integration flow.
- Extend CI with CKKS-focused test steps in both AVX and portable configurations, and add an `avx512` lane that type-checks and lints the AVX-512F + IFMA configuration (tests need self-hosted silicon).
- Add a hugepage hint in the aligned allocator: on Linux, allocations ≥ 2 MB issue `madvise(MADV_HUGEPAGE)` (~5% on `ntt120-avx` at large rings), overridable via `POULPY_HUGEPAGE_MIN_BYTES`.
- Add acknowledgements for PZ, EF, and ENS in the root README.

### Fixes

- Avoid under-allocating scratch in the bin-FHE scheme tests via new tmp-bytes helpers.
- Make the AVX backend optional (`enable-avx`) so non-AVX machines build.

### Migration

HAL backend wiring moves from one `HalImpl` to per-family OEP traits, with `poulpy-cpu-ref` exposing per-family `hal_impl_*!` macros and `*Defaults` traits so a backend inherits the reference path for cold methods and overrides only the hot ones:

```rust
// before
unsafe impl HalImpl<FFT64Avx> for FFT64Avx { hal_impl_vec_znx!(); hal_impl_module_fft64!(); }

// after: one impl per family, overriding individual methods inline
unsafe impl HalVecZnxImpl<FFT64Avx> for FFT64Avx { poulpy_cpu_ref::hal_impl_vec_znx!(); }
unsafe impl HalVmpImpl<FFT64Avx> for FFT64Avx { poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefaults); }
```

Core and CKKS wiring likewise replaces `CoreImpl` + `impl_core_default_methods!` with per-family `impl_*_defaults_full!` (and `impl_ckks_*_defaults!`) macros, one call per family, overriding any method by writing it after the call.

## [0.5.0] - 2026-03-31

### `poulpy-bench` (new crate)

- Consolidate every benchmark suite into `poulpy-bench` under `bench_suite::{hal, core, schemes}`, removing the `bench_suite` modules from the other crates.
- Add the `standard` binary: one representative run across all layers at fixed parameters, for version-to-version regression tracking.
- Add JSON-configurable parameters via `POULPY_BENCH_PARAMS` (file path or inline). Every sweep range and layout constant is overridable, with built-in defaults for omitted fields: `hal.sweeps`, `cnv.sweeps`, `vmp.sweeps`, `svp_prepare.log_n`, and `core.{n, base2k, k, rank, dsize}`.
- Add the `run` field (bench binaries or Criterion function filters) and the `backends` field (`fft64-ref`, `ntt120-ref`, `fft64-avx`, `ntt120-avx`); listing an AVX backend enables `enable-avx` and the matching `RUSTFLAGS`.
- Replace the per-group `measurement_time` overrides with a shared `criterion_config()` (100 samples, 5 s).
- Add `examples/custom_params.json` and `examples/run_custom_params.sh`.

### `poulpy-hal`

- Remove `VmpApplyDftToDftAdd` / `SvpApplyDftToDftAdd` and merge the additive variant into `VmpApplyDftToDft` / `SvpApplyDftToDft` via a `limb_offset` parameter, with their OEP and delegate plumbing. Accumulating into a scattered output caused severe cache misses; a contiguous temporary folded with `VecZnxDftAddAssign` is ~2x faster.
- Add family defaults for `vec_znx_big`, `vec_znx_dft`, `svp_ppol`, `vmp_pmat` and `convolution`, and portable `scratch` / `vec_znx` defaults in `HalImpl`, removing the legacy per-family OEP traits.

### `poulpy-cpu-ref` / `poulpy-cpu-avx`

- Update the FFT64 and NTT120 `vmp_apply_dft_to_dft` to accept `limb_offset` directly, replacing the `_add` codepath.
- NTT120 AVX2: add `reduce_b_and_apply_crt`, fusing the CRT multiply into the Barrett reduction pass, roughly halving the instruction count of `compact_all_blocks`.
- Drop the legacy backend-specific VMP, convolution, scratch, `vec_znx`, `svp` and `vec_znx_dft` impl modules in favor of the HAL family defaults.

### `poulpy-core`

- Rewrite the external-product and keyswitching inner loops to write per-digit VMP results into a temporary before accumulating with `VecZnxDftAddAssign`, avoiding scattered-write cache thrashing.
- Add the `keyswitch::gglwe` bench module and the `keyswitch_glwe` benchmark on NTT120, replacing the FFT64-specific one.

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

- Add the `GLWEMulPlain`, `GLWEMulConst` and `GLWETensoring` traits (apply, assign, and relinearize forms, each with `_tmp_bytes`), plus the `test_glwe_tensoring` method test.

### `poulpy-hal`

- Add the `CnvPVecL` / `CnvPVecR` structs with the `CnvPVecBytesOf` / `CnvPVecAlloc` traits, and the `Convolution` trait regrouping prepare-left/right, `cnv_apply_dft`, `cnv_pairwise_apply_dft` and `cnv_by_const_apply` (each with `_tmp_bytes`).
- Add the `Reim4Convolution*` and `i64*` convolution traits with their FFT64 reference implementations, rename `Reim4Extract1Blk` to `Reim4Extract1BlkContiguous`, and add `take_cnv_pvec_left` / `take_cnv_pvec_right` to `ScratchTakeBasic`, plus convolution tests and benchmarks.
- **Breaking:** the normalization API and OEP (`VecZnxNormalize`, `VecZnxBigNormalize`) take `res_offset: i64`, a positive or negative bit-shift applied before the mod-1 reduction, with inputs reordered for consistency. This completes cross-base2k normalization at arbitrary offset; correctness is ensured, optimality is not.

### `poulpy-cpu-ref`

- Implement the convolution OEP on `FFT64Ref`, with tests and benchmarks.

### `poulpy-cpu-avx`

- Implement the convolution OEP on `FFT64Avx` with AVX FFT64 kernels (`reim4_save_1blk_to_reim_contiguous_avx`, `reim4_convolution_1coeff_avx`, `reim4_convolution_2coeffs_avx`), with tests and benchmarks.

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
