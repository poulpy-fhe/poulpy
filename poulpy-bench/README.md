# poulpy-bench

Shared benchmarking library for the poulpy workspace. `poulpy-bench` does not
ship its own runnable benchmarks — it provides the runners, sweep
parameters, and composition primitives that a backend crate (e.g.
`poulpy-cpu-portable`) uses to build its own `benches/*.rs` binaries.

## Why a library, not binaries

A backend crate depends on `poulpy-bench`, not the other way around. This
lets each backend assemble exactly the benchmarks it can run: one that
implements only part of the HAL, core, or CKKS surface can still benchmark
the part it supports, rather than being forced into an all-or-nothing suite.

## Core abstraction

Every benchmarked operation is a **runner**:

```rust
fn(&mut Bencher<'_, M>, &P)
```

which sets up its inputs once and times the operation via `bencher.iter(...)`.
Runners are grouped into tables of `BenchOp<M, P>` — each entry carries a
`layer` (`"hal"`, `"core"`, `"ckks"`, `"bin_fhe"`, ...) alongside its `name`
and `runner`. `bench_ops` drives a table against an iterator of sweep
parameters, producing one Criterion group per op, named
`<backend>/<layer>/<name>`. Both live in [`src/lib.rs`](src/lib.rs). Most
backend crates don't need to call `bench_ops` directly — see
[Using it from a backend crate](#using-it-from-a-backend-crate) below.

Each module exposes its runner suites as builder functions of the form:

```rust
pub fn <group>_ops<Backend, Measurement>() -> [BenchOp<M, P>; N]
```

so a caller can filter, reorder, or `.extend()` several tables together
before running them — no Criterion involved until `bench_ops` is called.
Most modules also expose `all_ops()` (every group in that layer, concatenated)
and/or `standard_ops()` (a small representative cross-section), built from
the same underlying `*_ops` tables.

## Layout

| Module | Covers |
|---|---|
| [`hal`](src/hal) | `poulpy-hal` trait operations: `vec_znx`, `vec_znx_dft`, `vec_znx_big`, `svp`, `vmp`, `convolution`, `reim` |
| [`core`](src/core) | `poulpy-core` GLWE/GGSW operations: encryption, decryption, keyswitch, automorphism, external product, tensoring |
| [`schemes`](src/schemes) | `poulpy-ckks` (CKKS) and `poulpy-bin-fhe` (blind rotation, circuit bootstrapping) |
| [`blas`](src/blas) | `poulpy-blas` encrypted matrix operations on one `d x d` block: `blas_matmul` by a dense and by a replicated right factor, `blas_matmul_pt_right` and `blas_matmul_pt_left` under `Diagonal` and `Jkls`, `blas_transpose` |

Each of `hal`, `core`, `schemes` and `blas` has its own `params` submodule (e.g.
[`hal::params`](src/hal/params.rs)) holding that layer's sweep-parameter
structs and `default_bench_params_*` builders, so every backend sweeps the
same reasonable defaults.

## Using it from a backend crate

For a backend implementing the full HAL/core/CKKS/bin-fhe surface, add
`poulpy-bench` as a dev-dependency and register the ready-made `bench_*`
functions directly in `criterion_group!` — each layer's `suites` module
(`hal::suites`, `core::suites`, `schemes::suites`, and `blas::suites` for the
BLAS layer, see [below](#blas-benchmarks)) exposes them for all three tiers:

- `bench_hal_ckks`/`bench_hal_binfhe`, `bench_core_ckks`/`bench_core_binfhe`,
  `bench_ckks`, `bench_binfhe` (top of each `suites` module): the **full**
  tier — every op in that layer.
- `suites::standard::{bench_hal_ckks, bench_hal_binfhe, ...}`: the
  **standard** tier — a smaller, representative cross-section of ops.
- `suites::light::{...}`: same shape as `standard`, but sweeps a single set of params.


These functions enable a backend to build a complete benchmark binary with just a few
lines of code:

```rust
use criterion::{criterion_group, criterion_main};
use poulpy_bench::core::suites::{bench_core_binfhe, bench_core_ckks};
use poulpy_bench::hal::suites::{bench_hal_binfhe, bench_hal_ckks};
use poulpy_bench::schemes::suites::{bench_binfhe, bench_ckks};
use poulpy_bin_fhe::blind_rotation::CGGI;

use my_backend_crate::{FftBackend, NttBackend};

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
     bench_hal_ckks::<NttBackend>,
     bench_hal_binfhe::<FftBackend>,
     bench_core_ckks::<NttBackend>,
     bench_core_binfhe::<FftBackend>,
     bench_ckks::<NttBackend>,
     bench_binfhe::<FftBackend, CGGI>
}

criterion_main!(benches);
```

See [`poulpy-cpu-portable/benches`](../poulpy-cpu-portable/benches) — or any of
`poulpy-cpu-avx`/`-avx512`/`-arm`'s `benches/` — for the complete, minimal
`full.rs` / `standard.rs` / `light.rs` binaries this produces.

### Composing your own tables

The crate exposes a deeper API than the `bench_*` functions, so a backend can compose its
own tables of `BenchOp`s and call `bench_ops` directly. For example, a backend that
implements the HAL but not the core layer can still benchmark the HAL ops it supports, and
a backend that implements only a subset of the HAL can still benchmark the ops it
supports. A backend can also filter, reorder, or extend the tables before passing them to
`bench_ops`.

```rust
use std::marker::PhantomData;

use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};
use poulpy_bench::{bench_ops, hal::{params::default_bench_params_hal, suites::vec_znx_ops}};

fn hal<BE: MyBackendBound>(c: &mut Criterion<WallTime>) {
    bench_ops(PhantomData::<BE>, &vec_znx_ops::<BE, WallTime>(), default_bench_params_hal(), c);
}

criterion_group!(benches, hal::<MyBackend>);
criterion_main!(benches);
```

## Running benchmarks

Bench binaries live in the backend crate, not here, and are usually gated
behind feature flags for the layers they exercise (so a partial-support
backend doesn't need to compile ops it can't run). Run them with
`cargo bench -p <backend-crate> --bench <binary>`, enabling whatever
features the binary needs.

For example, the `poulpy-cpu-portable` backend ships three binaries — `full` (every op, full
parameter grid), `standard` (a representative cross-section), and `light`
(a single-parameter test) — all three require the `enable-ckks` feature
(which pulls in `enable-core`):

```sh
# Everything in the standard suite
cargo bench -p poulpy-cpu-portable --bench standard --features enable-ckks

# The full grid — long running; only do this when you mean it
cargo bench -p poulpy-cpu-portable --bench full --features enable-ckks
```

### Running a subset: layer or op filters

Criterion treats any trailing arguments after `--` as a filter matched
against the benchmark's full id (`<backend>/<layer>/<op name>/<params>`), so
you can scope a run to one layer, backend, or op by passing a matching
prefix or substring. `--list` (optionally combined with a filter) prints the matching ids
without running them.

```sh
# Only the HAL layer, on the NTT4x30Portable backend
cargo bench -p poulpy-cpu-portable --bench standard --features enable-ckks -- "NTT4x30Portable/hal"

# One op, across every backend/layer that has it
cargo bench -p poulpy-cpu-portable --bench standard  --features enable-ckks -- "vec_znx_add"

# See what a filter would run, without running it
cargo bench -p poulpy-cpu-portable --bench standard --features enable-ckks -- --list "core"
```

### CKKS bootstrapping presets

`schemes::suites::bench_ckks_bootstrapping` benchmarks every preset in
`poulpy_ckks::presets::bootstrapping::all()`, including
`n15_d35_k180_p18_c2s` (four recovered levels, 18-bit minimum precision)
and the logN16 presets. The reference, AVX and AVX-512 `full` binaries register this runner.
Each case checks its measured precision against the preset's advertised minimum.

```sh
# List the logN15 cases without allocating bootstrapping keys
cargo bench -p poulpy-cpu-portable --bench full --features enable-ckks -- n15_d35_k180_p18_c2s --list

# Run the logN15 preset on the portable NTT backend
cargo bench -p poulpy-cpu-portable --bench full --features enable-ckks -- 'NTT4x30Portable/ckks/ckks_bootstrapping/n15_d35_k180_p18_c2s'
```

### BLAS benchmarks

`blas::suites` benchmarks `poulpy-blas` on NTT backends, on one `d x d` block per operand, at radix `base2k` 52, `k` 208
and scales `log_delta = p = 40`:

| Operation | Variant |
|---|---|
| `blas_matmul` | a dense right factor, distinct matrices per matrix slot (not run where one matrix fills the slots) |
| `blas_matmul_replicated` | a replicated right factor, one matrix whose compact encryption proves it |
| `blas_matmul_pt_right_diagonal`, `blas_matmul_pt_right_jkls` | one public matrix shared by the batch, on the right |
| `blas_matmul_pt_left_diagonal`, `blas_matmul_pt_left_jkls` | one public matrix shared by the batch, on the left |
| `blas_transpose` | |

The sweep is over the ring degree `N` and the block dimension `d`, which fill the `N / 2` slots with `N / (2 d^2)`
matrices: light `(2^14, 64)`; standard `(2^13, 32)`, `(2^14, 64)`, `(2^15, 128)`; full adds `(2^15, 64)`, `(2^16, 64)`
and `(2^16, 128)`.

A runner builds its point once, since Criterion calls it again for every sample: only the rotation keys the operation
looks up (and the tensor key of `blas_matmul`), the encrypted inputs, the context or the public operand, and the scratch
its `*_tmp_bytes` reports. Before timing, it checks `base2k` against the operation's radix helper at 128 failure bits,
runs the call once, decrypts and asserts every entry within `blas::matmul::error_bound` of the cleartext, and prints
the measured error in a `PRECISION` line. The group runs under `blas_criterion_config()`, ten samples.

A backend binary registers the group next to its main one, behind its `enable-blas` feature:

```rust
#[cfg(feature = "enable-blas")]
criterion_group! {
    name = blas_benches;
    config = poulpy_bench::blas_criterion_config();
    targets = bench_blas::<Ntt>
}

#[cfg(not(feature = "enable-blas"))]
criterion_main!(benches);
#[cfg(feature = "enable-blas")]
criterion_main!(benches, blas_benches);
```

```sh
# The light tier of the BLAS group only
cargo bench -p poulpy-cpu-portable --bench light --features enable-ckks,enable-bin-fhe,enable-blas -- blas
```

Memory: a point holds its context, keys and operands at once, and its rotation keys, tens of MB each at `N = 2^15`
and `2^16`, dominate. The key set of `blas_matmul` is 125 keys at `(2^13, 32)`, about 250 at `d = 64`, 381 at
`(2^15, 128)` and 509 at `(2^16, 128)`; the process peaks at about 11 GB at `(2^15, 128)` and 28 GB at `(2^16, 128)`, so
the full tier needs a host sized for that. The keys of one point are freed before the next is built.

## Normalization sweep

`hal::params::default_bench_params_normalize()` covers input and output bases
`{17, 19, 21, 50, 51}`, offsets `{-a_base2k, 0, 1, a_base2k}`, equal input/output
sizes `{4, 8, 16}`, and ring degrees `{4096, 65536}`, with one column, plus the rank-1 key
switching shape: degree 4096, two columns, base 52, `k = 108`, sources of 3 and 4 limbs
into 3 (rows ending in `drop=48` and `drop=48/a_size=4`), and one many-limb row: degree 16384,
32 limbs, base 50 (`16384x(1x32)`). The Criterion
runners `runner_vec_znx_normalize_sweep` and
`runner_vec_znx_big_normalize_sweep` accept these parameters.

The `normalize` benchmark compares small and big normalization on FFT64 and
NTT4x30, with reference and AVX2 backends, for 4,800 cases. Input generation
and allocation happen outside the timed loop, using the documented input bounds.

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench \
  -p poulpy-cpu-avx --features enable-avx --bench normalize
```

Append `-- 'k=17->19/offset=0'` to select a cross-base conversion, or
`-- --list` to list cases. Results use Criterion's standard reports and baselines.
