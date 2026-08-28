# poulpy-bin-fhe

`poulpy-bin-fhe` is the binary and gate-level FHE crate built on top of
`poulpy-core` and `poulpy-hal`.

It provides:

- blind rotation
- circuit bootstrapping
- BDD-based encrypted integer arithmetic

## Tests And Backend Integration

`poulpy-bin-fhe` exposes its public API as soon as the crate is imported, and
depends on no backend. Its tests are backend-generic: each backend crate
instantiates them with `bin_fhe_backend_test_suite!`, so they run from there:

```sh
cargo test -p poulpy-cpu-ref --features enable-core bin_fhe
```

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo test -p poulpy-cpu-avx --features enable-avx bin_fhe
```

For Rayon-scheduled NEON acceleration on AArch64 targets, instantiate the
suite from the backend crate:

```sh
cargo test -p poulpy-cpu-arm --features enable-rayon bin_fhe
```

The runnable examples live in `poulpy-cpu-ref/examples` (`bdd_arithmetic`,
`circuit_bootstrapping`, `max_array`), behind its `enable-core` feature.

## Backend Status

Public traits and helpers use the backend-owned HAL/core surface
(`ScratchArena<'_, BE>`, `...ToBackendRef<BE>`, and `...ToBackendMut<BE>`),
and the crate depends on no backend: it names only `poulpy-hal` and
`poulpy-core`. Backend crates instantiate its tests through
`bin_fhe_backend_test_suite!`. Several host `Vec<u8>` / `HostBackend` bounds
remain and are follow-up work.
