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
Runners are grouped into tables of `BenchOp<M, P>`, and `bench_ops` drives a
table against a slice of sweep parameters `&[P]`, producing one Criterion
group per op. Both live in [`src/lib.rs`](src/lib.rs).

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
| [`params`](src/params.rs) | Sweep-parameter structs and `default_bench_params_*` builders shared across backends |

## Using it from a backend crate

Add `poulpy-bench` as a dev-dependency, then write a `benches/*.rs` binary
that picks the ops it wants and drives them with `bench_ops`:

```rust
use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};
use poulpy_bench::{bench_ops, hal::suites::vec_znx_ops, params::default_bench_params_hal};

fn hal(c: &mut Criterion) {
    bench_ops(&vec_znx_ops::<MyBackend, WallTime>(), default_bench_params_hal().as_slice(), "my-backend", c);
}

criterion_group!(benches, hal);
criterion_main!(benches);
```

See [`poulpy-cpu-ref/benches`](../poulpy-cpu-ref/benches) for a complete
worked example, including its three-tier `full` / `standard` / `light`
binaries.
