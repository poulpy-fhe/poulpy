# poulpy-bench review rules

This file adds the rules specific to `poulpy-bench` and to the bench binaries of the backend crates.

## 1. Crate boundaries

- **Generic over the backend.** poulpy-bench depends, among poulpy crates, only on the library crates it benches, never on a `poulpy-cpu-*` crate, its compiled code names no concrete backend type and it has no bench target.
- **No features.** poulpy-bench has no `[features]`, and its `[dependencies]` hold only what its code uses, test-only features included (`poulpy-ckks/test-utils`).
- **Files per layer.** Runners live in `tools/poulpy-bench/src/{hal,core,schemes,blas}/` (`schemes` for the scheme crates), sweep structs and `pub fn default_bench_params_<name>() -> Vec<P>` in `<layer>/params.rs`, tables and targets in (or re-exported from) `<layer>/suites.rs`, and a new layer adds in one diff its `{mod,params,suites}.rs`, `pub mod <layer>;` in `lib.rs` and its tier-binary registration.
- **Minimal public surface.** Only runners, sweep structs and `default_bench_params_*`, tables, targets, harness-free entry points (section 4), `BenchOp`, `bench_ops`, the Criterion configurations, bound-bundle traits of public signatures (`BlasBenchModule`) and the `heap` items a binary or test reads (`Counting`, `peak`, `reset_peak`, `counting`) are `pub`, generation and upload helpers staying private.
- **Nothing new in `old_benches`.** A benchmark from `tools/poulpy-bench/old_benches/` is ported to a registered runner in `src/<layer>/`, and nothing under `old_benches/` is added or updated.

## 2. Runners

### 2.1. Signature and names

- **Runner signature.** A listed runner is `pub fn runner_<name><BE, M: Measurement>(bencher: &mut Bencher<'_, M>, params: &P)`, `M` never fixed to `WallTime`.
- **Names follow the timed method.** `BenchOp::name` is the one API method the runner times on an explicitly fixed path, the runner is `runner_<name>`, and a path or variant name adds `_<variant>` (`blas_matmul_pt_left_jkls`), in the table entry only when a type or const parameter of a shared runner selects it (`runner_cnv_apply_dft_sum::<B, M, true>` as `cnv_apply_dft_sum_embedded`).
- **Variants share one body.** Variants of one operation are a type or const parameter of one runner or one-line public wrappers over a private shared body, never a runtime argument, and a richer sweep struct gets a `_sweep` sibling the plain runner delegates to with defaults (`runner_vec_znx_normalize_sweep`).

### 2.2. Setup

- **Module at the point's degree.** A runner or its shared body builds its own `Module::<BE>::new(..)` at the point's degree and allocates operands there, an embedded operand's degree asserted or raised to at least `BE::MIN_DEGREE` before allocation.
- **Layouts from the point.** Layouts take every field the sweep point carries (degree, radix, precision, rank, digit width) from it, and key shapes come from `tools/poulpy-bench/src/core/params.rs` (`switching_key_dnum_k_aux(input_k, ..)` for a key decomposing an input, `key_dnum_k_aux(k, ..)` for a key covering a precision) or a runner helper with a one-line reason.
- **No host bounds.** A runner, table or target bounds `BE: Backend` only by the associated types its calls fix (`ZnxWord = i64`), `'static` and the module traits it calls, never by `OwnedBuf = AlignedBuf`, `AsRef<[u8]>` or `AsMut<[u8]>` on `OwnedBuf`, `BufRef` or `BufMut`, `TestContextBackend` or a host bound.
- **Operands staged once.** HAL runners build operands on the host (`random_host_*`) and upload them once (`upload_host_*`, `BE::from_host_bytes`); core and higher runners fill them on the backend (`fill_glwe_from_source`) or make them through the API, never through host staging, and no runner writes backend storage through a host view.
- **Real operands.** Every operand the timed call reads, destinations and keys included, holds uploaded or backend-filled uniform data or an API output, never a fresh zeroed buffer or an unprepared key; `random_backend_*` bytes in a HAL DFT, big or prepared layout, whose representation no contract fixes, are valid where the operation's domain does not bound them.
- **Inputs inside the contract domain.** Inputs satisfy the `domain` line of the timed operation's contract block, `vec_znx_normalize*`, `vec_znx_lsh*` and `vec_znx_rsh*` reading `random_normalization_vec_znx` or `random_normalization_vec_znx_big` data.
- **Canonical inputs to DFT readers.** A DFT-domain reader (key switching, automorphism, external product, products, decryption, preparation) gets a canonical operand (`fill_glwe_from_source`, the previous in-place iteration's canonical output) unless the runner's name or doc says it times the flag-clear path.
- **Only the keys it reads.** A runner generates exactly the deduplicated keys the timed call looks up: rotation keys for its information query's Galois elements (`BlasPlan::*_galois_elements(order)`), and a tensor key only for a ciphertext by ciphertext product.
- **Fixed seeds, one per role.** Fixed test seeds per role hold for runners, while uniform fills of value-agnostic operands may share one.
- **Scratch from the queries.** Each arena is sized by the maximum selected `_tmp_bytes` of every operation borrowing it, setup included, at the layouts each receives, and no object is allocated only to feed a query.
- **Fallible calls are unwrapped.** A runner unwraps every `Result` (`CKKSResult`, setup `anyhow::Result`), inside `bencher.iter` too, and sets CKKS operands' metadata from the point (`set_meta_checked`, `set_meta`) before timing.

### 2.3. The timed region

- **Only the call is timed.** Everything but the call happens before `bencher.iter`, and the closure allocates nothing: it only takes backend views, borrows the arena (`&mut scratch.borrow()`), calls the operation (per column for a per-column one), then `module.synchronize()` (`poulpy-hal/docs/backend_safety_contract.md`).
- **Always iterate.** A runner calls `bencher.iter` on every invocation, a point it cannot run being filtered from the sweep in the suite before `bench_ops` (dense `blas_matmul`: `batch() > 1`) and the runner doc stating that precondition.
- **Build an expensive point once.** A runner whose setup far exceeds one call builds its point once per id and reuses it across Criterion's calls (`timed` in `tools/poulpy-bench/src/blas/matmul.rs`), dropping the previous point before building the next.
- **Setup stays in the runner.** Modules, operands and keys are built in the runner or a closure Criterion calls, never in a table builder or a `bench_*` target body.

### 2.4. Checks inside runners

- **Verify before timing.** A runner with a precision contract (BLAS products and transpose, bootstrapping presets through `BootstrappingPresetRun`) checks its result against the documented bound (`error_bound`, advertised precision) and `base2k` against any radix helper (`blas_*_max_base2k` at `FAILURE_BITS = 128`), prints one `PRECISION` line and panics outside the bound, a new runner verifying its first call before timing it.
- **Runner checks are not tests.** A library property a runner asserts (precision bound, radix model) is also pinned by an owning-crate `test_suite` test (`test_blas_max_base2k`, `bootstrapping_presets_meet_precision`), a full-size one ignored beside an enabled small case.

## 3. Tables, targets and tiers

- **Unlisted runners never run.** Every `pub fn runner_*` is in a `BenchOp` table of its layer's `suites.rs` or in a focused target, and every table runs from a `bench_*` target, directly or through an aggregation, or from a backend binary's `bench_ops` call.
- **Tables carry exact bounds.** A family table is `pub fn <family>_ops<BE, M: Measurement>() -> [BenchOp<M, P>; N]`, never touching Criterion, one per API family or accepted point set (`blas_dense_ops`), and its where clause, like an aggregation's (`all_ops`) or a target's, is exactly the union of what it includes.
- **Targets take one backend.** A target is `pub fn bench_<name>(c: &mut Criterion<WallTime>)` over one backend plus at most a scheme marker (`BRA`) or const fixture parameter (`FIXTURE_BASE2K`), named `bench_<layer>_<role>` (role `ckks` or `binfhe`) in hal and core, `bench_<layer>` above (`bench_ckks`, `bench_binfhe`, `bench_blas`) and `bench_<topic>` if focused or full-only, passing `bench_ops` a `PhantomData` of that backend.
- **Tiers filter one sweep.** Each layer's `suites.rs` holds the full tier, running each table over its `default_bench_params_*` sweep, and same-named `pub mod standard` and `pub mod light` functions running the layer's cross-section (`standard_ops`; all BLAS tables) over that sweep filtered by `is_standard_n` or `is_light_n`, optionally on other fields, with no point list, literal degree or sweep argument of their own; every tier pins its `*_binfhe` functions to `bin_fhe_n()`, and `bench_binfhe` has one tier.
- **Sweeps cover the tier degrees.** A default sweep feeding the tiers has a point at each `STANDARD_N` and `LIGHT_N` degree and, where a `*_binfhe` function reads it, at `bin_fhe_n()`, every point lying in `[Backend::MIN_DEGREE, 2^17]` of each registering backend.
- **Focused targets keep their sweep.** A focused target (`bench_normalization`) runs its own `default_bench_params_<topic>()` and is registered only in its own binary (`normalize.rs`).
- **One id grammar.** Every id is `<backend>/<layer>/<name>/<params>`, built by `bench_ops` or, for a runner's own group, as `format!("{}/<layer>/<name>", backend_name::<BE>())` with `BenchmarkId::from_parameter(..)`, and `BenchOp::layer` is the timed API's crate layer (`"hal"`, `"core"`, `"ckks"`, `"bin_fhe"`, `"blas"`, a new layer its crate name).
- **Stable parameter ids.** A sweep struct's `Display` prints every field varying across its default sweep, a later-added field only when it departs from the value existing rows implied, and op names, layer labels and `Display` formats change only with an API rename.
- **Shared Criterion configurations.** Groups run under `poulpy_bench::criterion_config()` except those whose call takes seconds (the BLAS groups under `poulpy_bench::blas_criterion_config()`, the hand-built bootstrapping group at `sample_size(10)`), and binaries reference these functions instead of building a `Criterion`.

## 4. Backend bench binaries

- **Binaries only register.** A backend `benches/*.rs` binary, harness-free ones aside, only picks backend types and registers poulpy-bench targets or tables in `criterion_group!`, every runner, sweep and check living in poulpy-bench.
- **Harness-free binaries.** A check whose single call takes minutes or more memory than a Criterion sample affords runs from a harness-free binary (`blas_scale`), whose `main` installs `heap::Counting`, collects the arguments not starting with `--` as a label filter, calls poulpy-bench entry points once per backend type (a feature-gated type under `#[cfg(feature = ..)]`, by full path) and panics after the last if any check failed; the entry points take the filter, label each check `<backend>/<layer>/<name>/<params>` and return the failed labels.
- **One `[[bench]]` entry each.** Each new `benches/<name>.rs` comes with a `[[bench]]` entry with `name = "<name>"`, `harness = false` and `required-features` holding the crate's ISA feature (`enable-avx`, `enable-avx512f`, `enable-neon`; none on poulpy-cpu-portable and poulpy-cpu-rayon) and, directly or implied, its layer features: `enable-ckks` and `enable-bin-fhe` for tier binaries, `enable-ckks` for CKKS-only ones (`bootstrapping`), `enable-core` for core-only ones (`encrypt_pk`) on poulpy-cpu-portable and poulpy-cpu-rayon only, none for HAL-only ones (`normalize`), and `enable-blas` only for BLAS-only ones (poulpy-cpu-rayon `blas`).
- **The tier grid.** poulpy-cpu-portable, poulpy-cpu-avx, poulpy-cpu-avx512 and poulpy-cpu-arm each keep `benches/full.rs`, `standard.rs` and `light.rs` (targets from `suites`, `suites::standard` and `suites::light`, single-tier ones from `suites`), registering `bench_hal_ckks`, `bench_core_ckks`, `bench_ckks` and, in the BLAS group, `bench_blas` on the NTT backend, and `bench_hal_binfhe`, `bench_core_binfhe` and `bench_binfhe::<_, CGGI>` on the FFT64 backend; a three-tier target enters all twelve binaries in one diff, a full-only one (`bench_ckks_bootstrapping`) `full.rs` or its own binary, and extra backend types come on top.
- **Mains come from `criterion_main!`.** A Criterion binary ends with `criterion_main!`, its BLAS group a second `criterion_group!` named `blas_benches` under `#[cfg(feature = "enable-blas")]`, selected by `#[cfg(not(feature = "enable-blas"))] criterion_main!(benches);` and `#[cfg(feature = "enable-blas")] criterion_main!(benches, blas_benches);`.
- **Gated types through wrappers.** A backend type behind an extra feature (`enable-rayon`, `enable-ifma`) is registered through a wrapper `fn <name>(_c: &mut Criterion)` listed unconditionally in `targets`, its `#[cfg(feature = ..)]`-gated body naming the type by full path or an import under the same `#[cfg]`.
- **Check feature-off arms by hand.** A diff touching a feature-off main or empty wrapper no lane compiles states a `cargo clippy -p <crate> --benches --features <the binary's required-features> -- -D warnings` run with the lane's ISA target flags.
- **BLAS on NTT backends.** `bench_blas`, `blas_ops` and `blas_dense_ops` are registered on NTT backend types only.

## 5. Measurements and documentation

- **Claims run registered runners.** Cited ids are `bench_ops` ids of a registered runner run through a backend binary, or labels of a harness-free check (section 4).
