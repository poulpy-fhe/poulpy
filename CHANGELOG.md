# CHANGELOG

## [Unreleased]

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
- Update bin-FHE BDD arithmetic, blind rotation, and test suites for the new core/HAL APIs.
- Refresh blind-rotation / circuit-bootstrapping staging helpers for the new `ScalarZnx` view API.
- Refresh scheme examples and library wiring to match the crate split and the new backend-generic APIs.
- **Note:** `poulpy-bin-fhe` is not yet backend agnostic: it still depends unconditionally on `poulpy-cpu-ref` and exposes host `Vec<u8>` / `HostBackend` bounds in several public APIs. Full backend-agnosticity for this crate is deferred to a follow-up.
- **Breaking:** Bin-FHE traits and helpers now follow the backend-owned core/HAL surface: methods take `ScratchArena<'_, BE>`, use `...ToBackendRef<BE>` / `...ToBackendMut<BE>` bounds for ciphertexts and prepared keys, and many generic entrypoints now require `BE: Backend<OwnedBuf = Vec<u8>>` plus `ModuleCoreAlloc`.
- Move public constructors/allocation helpers to module-first forms across the crate: `FheUint::alloc[_from_infos](module, ...)`, `LookupTable::alloc(module, ...)`, `GLWEBlindRetriever::alloc(module, ...)`, and `CircuitBootstrappingKey::alloc_from_infos(module, ...)`.
- Add `LookupTable::to_backend` for explicit backend transfer of LUT storage and keep prepared blind-rotation / circuit-bootstrapping factories on backend-owned output types via `ScratchArena`.
- Align bin-FHE key/prepared layouts and circuit helpers with the refactored core layouts.
- Add `ReaderFrom` / `WriterTo` for `CircuitBootstrappingKey` and `BDDKey<Vec<u8>>` (optional `ks_glwe` encoded with a presence tag), with stable ATK map serialization (sorted Galois keys).

### `poulpy-bench`
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
