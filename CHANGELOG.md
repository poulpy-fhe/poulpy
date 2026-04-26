# CHANGELOG

## [Unreleased]

### `poulpy-ckks` (new crate — first iteration, API subject to change)

`poulpy-ckks` is the first Poulpy crate implementing the CKKS (Cheon-Kim-Kim-Song) approximate homomorphic encryption scheme. This release is an **initial iteration**: the core evaluator is functional and tested, but the public API is not yet stable, naming conventions may shift, and several performance optimizations are still pending.

**Representation.** The implementation uses a *bivariate Torus* representation instead of the RNS decomposition used by other CKKS libraries. Precision and homomorphic capacity are tracked at the bit level (`log_delta`, `log_budget`) rather than through a prime chain. This gives bit-granular capacity consumption, trivial scale management expressed in bits, compact plaintexts that do not expand to a full RNS basis, and circuit-independent evaluation-key parameterization.

**Architecture.** The crate is organized in four layers that mirror the structure of `poulpy-core` and `poulpy-hal`:
- `api/` — public user-facing operation traits implemented on `Module<BE>`.
- `delegates/` — blanket implementations that route method calls through the dispatch layer.
- `leveled/oep/` — internal dispatch traits (`CKKSPlaintextZnxOep`, etc.) that separate user traits from backend hooks.
- `default/` — default algorithm implementations, used when a backend does not provide a specialized override.

Backends override CKKS algorithms by implementing `unsafe trait CKKSImpl<BE>`, the scheme-level analogue of `HalImpl` and `CoreImpl`. The `impl_ckks_*_default_methods!` macros provide full default wiring so backends can opt into only the overrides they need.

**Plaintext types.** Four plaintext layouts are supported:
- `CKKSPlaintextVecZnx` — the primary internal format (multi-limb Znx vector).
- `CKKSPlaintextVecRnx<F>` — floating-point slot vectors; supports `f64` and `f128`.
- `CKKSPlaintextCstZnx` / `CKKSPlaintextCstRnx<F>` — constant plaintext variants backed by a single-column layout.

`CKKSPlaintextVecRnx` conversion to/from `CKKSPlaintextVecZnx` is handled via `CKKSPlaintextConversion` and uses an `i64`-path for encodings that fit in 64-bit limbs and an `i128`-path for higher precision.

**Leveled operations.** The evaluator exposes the following trait groups, all dispatched through `Module<BE>`:
- `CKKSEncrypt` / `CKKSDecrypt` — secret-key encryption and decryption.
- `CKKSAddOps` / `CKKSAddOpsUnsafe` / `CKKSSubOps` / `CKKSSubOpsUnsafe` — ciphertext-ciphertext and ciphertext-plaintext addition and subtraction. Plaintext operand families: `vec_znx`, `vec_rnx`, `const_znx`, `const_rnx`. Each family provides an `_into` (out-of-place), `_assign` (in-place), and two `_unsafe` variants (unnormalized; caller must normalize before overflow).
- `CKKSMulOps` — ciphertext-ciphertext multiplication, squaring, and ciphertext-plaintext multiplication for all four plaintext families.
- `CKKSNegOps` / `CKKSConjugateOps` — negation and complex conjugation.
- `CKKSRotateOps` — slot rotation via automorphisms and an evaluation key.
- `CKKSRescaleOps` — scale management.
- `CKKSPow2Ops` — `div_pow2` and `mul_pow2` power-of-two scaling.
- `CKKSPlaintextZnxOps` — plaintext-level `extract_pt_znx` for pulling a compact `CKKSPlaintextVecZnx` out of a raw `GLWEPlaintext` after decryption.
- `CKKSAllOpsTmpBytes` — a single scratch-sizing entry point that returns the maximum scratch required across all operations for a given layout.
- `CKKSAddManyOps` / `CKKSMulManyOps` — tree-reduction helpers for adding or multiplying a slice of ciphertexts with minimal depth.
- `CKKSDotProductOps` / `CKKSMulAddOps` / `CKKSMulSubOps` — fused multiply-accumulate composites that save a normalization pass compared to separate multiply + add.

**Fused kernels.** Inner loops for composite operations use fused `VecZnxBig` normalize-add / normalize-sub primitives to reduce the number of normalization passes in tree-reductions and dot products.

**Backends.** Tested against `NTT120Ref` and `FFT64Ref`; naturally all backends implementing poulpy-hal will enable the full capabilities of the scheme by the default dispatches.

**Test coverage.** 439 unit and integration tests covering all operation families, plaintext types, alignment edge cases, capacity exhaustion, and composite helpers.

**Known limitations and upcoming work.**
- The API is not yet stable. Trait names, method signatures, and plaintext type layouts may change before a stabilization release.
- Linear transformations, polynomial evaluation, and homomorphic DFT are not yet implemented.
- Further performance work is planned: notably more granular low-level API over different output formats (vec_znx, vec_znx_big, vec_znx_dft), additional fused kernels and backend-specific overrides for hot paths.
- Bootstrapping is on the roadmap but not in scope for this iteration.

### `poulpy-hal`
- **Breaking:** Rename all in-place operation methods from `_assign` to `_assign` across all operation families (`vec_znx`, `vec_znx_big`, `vec_znx_dft`, `svp_ppol`, GLWE operations, etc.) to establish a uniform workspace-wide naming convention where `_assign` denotes in-place mutation of the first operand.
- Fix the convolution API by renaming the output-shift parameter to `cnv_offset`, moving it to the front of the apply calls, and updating delegates and conformance tests to match the corrected calling convention.
- Replace legacy OEP modules with the unified `oep::HalImpl` entrypoint to provide one consistent extension surface for backends.
- Add family defaults for `vec_znx`, `vec_znx_big`, `vec_znx_dft`, `svp_ppol`, `vmp_pmat`, and `convolution` to reduce backend boilerplate and make overrides explicit.
- Remove legacy OEP traits and per-family OEP modules; update delegates to route through `HalImpl` and simplify dispatch.
- Update layouts and encoding helpers to match the new dispatch surface.
- Refresh HAL test suites to align with the new defaults and dispatch.
- Add family-level module/scratch defaults to cut backend boilerplate and centralize scratch sizing.
- Make `WriterTo` for `MatZnx` and `VecZnx` emit the canonical logical byte length from layout metadata, write only that prefix, and error when backing storage is shorter than the coefficient span.
- Fix `ScalarZnx::write_to` to emit the full `n * cols` coefficient byte span (aligned `i64` layout).
- **Breaking:** Remove `ReaderFrom` / `WriterTo` for prepared DFT layouts (`SvpPPol`); remove `SvpPPolFromBytes`, `VmpPMatFromBytes`, and `from_bytes` on the corresponding prepared types. Document that `SvpPPol` / `VmpPMat` DFT alignment assumes a power-of-two ring degree.

### `poulpy-core`
- **Breaking:** Rename all in-place GLWE and LWE operation methods from `_assign` to `_assign` (`glwe_normalize_assign`, `glwe_sub_assign`, `glwe_automorphism_assign`, etc.) to match the workspace-wide naming convention.
- Thread the corrected convolution-offset semantics through GLWE constant/plaintext multiply and tensoring paths so scratch sizing, truncation, and normalization all use the same convention.
- Pass explicit effective-k information into convolution-backed multiply/tensor routines and mask partial bottom limbs correctly instead of assuming every input uses its full stored limb width.
- Refresh GLWE tensor tests to cover the updated convolution API and the corrected effective-width handling.
- Fix tensoring noise blowup when output operand had a smaller size than the input operand.
- Split public APIs into `api` trait modules backed by `delegates` and `oep` layers to separate user-facing traits from backend hooks and dispatch.
- Reorganize encryption, decryption, conversions, keyswitching, external products, and operations to match the new API structure.
- Move backend conformance suites into `src/test_suite` and keep unit tests separate.
- Refresh layouts, noise helpers, and utilities to align with the new API surface.
- Re-export top-level modules to preserve public API ergonomics while routing through the new `api` traits.
- Standardize prepared allocations on `DeviceBuf` for backend-owned buffers to make data ownership explicit.
- Rename Module allocation/prepare helpers to struct-first names (e.g. `gglwe_prepared_alloc`, `glwe_secret_prepare`) to match the rest of the API.
- **Breaking:** Remove `ReaderFrom` / `WriterTo` for `LWESecret` and `GLWESecret`; secret material should use seeds or application-level transfer, not library binary I/O.

### `poulpy-cpu-ref` / `poulpy-cpu-avx`
- **Breaking:** Rename all in-place internal helpers from `_assign` to `_assign` (e.g. `vec_znx_sub_assign`, `reim_sub_assign`, `ntt_negate_assign`, `svp_apply_dft_to_dft_assign`) to match the workspace-wide naming convention. Internal NTT120 normalization helpers that previously used `_assign` to denote a generic out-of-place write are renamed to `_into` (`nfc_middle_step_into`, `nfc_final_step_into`) to restore the distinction.
- Update FFT64 and NTT120 convolution implementations, references, and tests to the corrected `cnv_offset` API.
- Optimize NTT120 convolution on the AVX backend by wiring the prep paths to backend-specific kernels and restructuring `cnv_apply_dft` / `cnv_pairwise_apply_dft` around prepacked x2 blocks, substantially reducing GLWE tensoring time on large `ntt120-avx` workloads.
- Reorganize backend implementations around `hal_impl` modules and `hal_defaults` to mirror the new HAL entrypoint and reduce duplication.
- Remove legacy per-family FFT64/NTT120 modules; route implementations through the new HAL defaults to keep a single source of truth.
- Update FFT64/NTT120 reference kernels, normalization, and shift helpers to keep behavior aligned with the new dispatch path.
- Flatten AVX test module paths to remove redundant crate prefixes.
- Split backend code into family-specific `hal_impl/*` modules (module/scratch/vec_znx/vmp/svp/convolution) for clearer override points.

### `poulpy-schemes`
- **Breaking:** Update all call sites to use the renamed `_assign` methods (e.g. `ggsw_external_product_assign`, `glwe_automorphism_assign`, `ggsw_blind_rotation_assign`) following the workspace-wide rename from `_assign`.
- Update bin-FHE BDD arithmetic, blind rotation, and test suites for the new core/HAL APIs.
- Refresh scheme examples and library wiring; remove the redundant `poulpy-schemes/README.md`.
- Align bin-FHE key/prepared layouts and circuit helpers with the refactored core layouts.
- Add `ReaderFrom` / `WriterTo` for `CircuitBootstrappingKey` and `BDDKey<Vec<u8>>` (optional `ks_glwe` encoded with a presence tag), with stable ATK map serialization (sorted Galois keys).

### `poulpy-bench`
- Update core and HAL convolution benchmarks to the new convolution API.
- Align benchmark suites with the new HAL/core APIs and update parameter examples.

### Build & Docs
- Refresh root and crate READMEs (naming, examples, and links); update docs references to reduce drift after the refactor.
- Update `rust-toolchain.toml` (nightly toolchain) to keep build expectations aligned.
- Add acknowledgements for PZ, EF, and ENS in the root README.

### Fixes
- Avoid under-allocating scratch space in bin-FHE scheme tests via new FheUint/BDD tmp-bytes helpers.
- Make AVX backend optional (`enable-avx`) to prevent build failures on non-AVX machines.

### Migration (before/after)

**HAL backend wiring** moved from per-family OEP traits to a single `HalImpl` entrypoint with defaults.

Before (legacy OEP traits):

```rust
use poulpy_hal::oep::{VecZnxImpl, VecZnxTmpBytesImpl};

unsafe impl VecZnxImpl<FFT64Avx> for FFT64Avx {
    fn vec_znx_add_into<R, A, B>(/* ... */) { /* AVX impl */ }
}

unsafe impl VecZnxTmpBytesImpl<FFT64Avx> for FFT64Avx {
    fn vec_znx_add_tmp_bytes(/* ... */) -> usize { /* ... */ }
}
```

After (unified `HalImpl` + defaults):

```rust
use poulpy_hal::oep::HalImpl;

unsafe impl HalImpl<FFT64Avx> for FFT64Avx {
    hal_impl_vec_znx!();      // default VecZnx wiring
    hal_impl_module_fft64!(); // FFT64-specific hooks
    // override only the hot paths you need
}
```

**Core API override hooks**: `poulpy-core` dispatches through `poulpy-hal::Module<BE>` by default, but a backend can override core algorithms directly by implementing `CoreImpl`.

Before (default core behavior via HAL + core APIs):

```rust
use poulpy_core::api::GLWEAdd;
use poulpy_hal::layouts::Module;

// Uses default core algorithms routed through Module<BE>
module.glwe_add(&mut out, &a, &b);
```

After (override selected core ops in the backend):

```rust
use poulpy_core::oep::{CoreImpl, impl_core_default_methods};

unsafe impl CoreImpl<MyBackend> for MyBackend {
    impl_core_default_methods!(MyBackend); // keep defaults

    fn glwe_add<R, A, B>(module: &Module<MyBackend>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        B: GLWEToRef + GLWEInfos,
    {
        // custom fast path here
    }
}
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
 - Fix typo in impl of GGLWEToRef for GLWEAutomorphismKey that required the data to be mutable.

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
