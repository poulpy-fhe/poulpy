# poulpy-bench

Shared benchmarking library for the poulpy workspace. `poulpy-bench` does not
ship its own runnable benchmarks — it provides the runners, sweep
parameters, and composition primitives that a backend crate (e.g.
`poulpy-cpu-ref`) uses to build its own `benches/*.rs` binaries.

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

Each of `hal`, `core`, and `schemes` has its own `params` submodule (e.g.
[`hal::params`](src/hal/params.rs)) holding that layer's sweep-parameter
structs and `default_bench_params_*` builders, so every backend sweeps the
same reasonable defaults.

## Using it from a backend crate

For a backend implementing the full HAL/core/CKKS/bin-fhe surface, add
`poulpy-bench` as a dev-dependency and register the ready-made `bench_*`
functions directly in `criterion_group!` — each layer's `suites` module
(`hal::suites`, `core::suites`, `schemes::suites`) exposes them for all
three tiers:

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

See [`poulpy-cpu-ref/benches`](../poulpy-cpu-ref/benches) — or any of
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

For example, the `poulpy-cpu-ref` backend ships three binaries — `full` (every op, full
parameter grid), `standard` (a representative cross-section), and `light`
(a single-parameter test) — all three require the `enable-ckks` feature
(which pulls in `enable-core`):

```sh
# Everything in the standard suite
cargo bench -p poulpy-cpu-ref --bench standard --features enable-ckks

# The full grid — long running; only do this when you mean it
cargo bench -p poulpy-cpu-ref --bench full --features enable-ckks
```

### Running a subset: layer or op filters

Criterion treats any trailing arguments after `--` as a filter matched
against the benchmark's full id (`<backend>/<layer>/<op name>/<params>`), so
you can scope a run to one layer, backend, or op by passing a matching
prefix or substring. `--list` (optionally combined with a filter) prints the matching ids
without running them. 

```sh
# Only the HAL layer, on the NTT4x30Ref backend
cargo bench -p poulpy-cpu-ref --bench standard --features enable-ckks -- "NTT4x30Ref/hal"

# One op, across every backend/layer that has it
cargo bench -p poulpy-cpu-ref --bench standard  --features enable-ckks -- "vec_znx_add_into"

# See what a filter would run, without running it
cargo bench -p poulpy-cpu-ref --bench standard --features enable-ckks -- --list "core"
```
