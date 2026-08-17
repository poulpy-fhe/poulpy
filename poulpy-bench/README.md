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
`<backend>/<layer>/<name>`. Both live in [`src/lib.rs`](src/lib.rs).

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

Add `poulpy-bench` as a dev-dependency, then write a `benches/*.rs` binary
that picks the ops it wants and drives them with `bench_ops`. `bench_ops`
takes the backend as a value (a zero-sized marker type, e.g. a unit struct
implementing `poulpy_hal::layouts::Backend`) rather than a type parameter, so
the backend name is inferred and doesn't need turbofish:

```rust
use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};
use poulpy_bench::{bench_ops, hal::{params::default_bench_params_hal, suites::vec_znx_ops}};

fn hal(c: &mut Criterion) {
    bench_ops(MyBackend, &vec_znx_ops::<MyBackend, WallTime>(), default_bench_params_hal(), c);
}

criterion_group!(benches, hal);
criterion_main!(benches);
```

Each resulting Criterion group is named `<backend>/<layer>/<op name>`, e.g.
`MyBackend/hal/vec_znx_add_into`, with sweep parameters as the benchmark id
inside that group.

See [`poulpy-cpu-ref/benches`](../poulpy-cpu-ref/benches) for a complete
worked example, including its three-tier `full` / `standard` / `light`
binaries.

## Running benchmarks

Bench binaries live in the backend crate, not here, and are usually gated
behind feature flags for the layers they exercise (so a partial-support
backend doesn't need to compile ops it can't run). Run them with
`cargo bench -p <backend-crate> --bench <binary>`, enabling whatever
features the binary needs.

For exemple, the `poulpy-cpu-ref` backend ships three binaries — `full` (every op, full
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
