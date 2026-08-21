# poulpy-bin-fhe

`poulpy-bin-fhe` is the binary and gate-level FHE crate built on top of
`poulpy-core` and `poulpy-hal`.

It provides:

- blind rotation
- circuit bootstrapping
- BDD-based encrypted integer arithmetic

## Tests And Backend Integration

`poulpy-bin-fhe` exposes its public API as soon as the crate is imported.
Backend crates own the feature flags that wire concrete implementations into
that API. For this crate's local tests and examples, enable the reference
backend integration:

```sh
cargo test -p poulpy-bin-fhe --features enable-bin-fhe
```

To include examples and test targets in a compile check:

```sh
cargo check -p poulpy-bin-fhe --all-targets --features enable-bin-fhe
```

For AVX2/FMA acceleration on x86_64 targets:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo test -p poulpy-bin-fhe --features enable-avx
```

Without `enable-bin-fhe`, the public API still builds, but this crate's
backend-backed examples are skipped.

## Backend Status

Public traits and helpers use the backend-owned HAL/core surface
(`ScratchArena<'_, BE>`, `...ToBackendRef<BE>`, and `...ToBackendMut<BE>`),
and the crate depends on no backend: it names only `poulpy-hal` and
`poulpy-core`. Backend crates instantiate its tests through
`bin_fhe_backend_test_suite!`. Several host `Vec<u8>` / `HostBackend` bounds
remain and are follow-up work.
