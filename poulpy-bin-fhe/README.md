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

Most public traits and helpers now use the backend-owned HAL/core surface
(`ScratchArena<'_, BE>`, `...ToBackendRef<BE>`, and
`...ToBackendMut<BE>`). The crate is still not fully backend-agnostic in
v0.6.0: it keeps an unconditional `poulpy-cpu-ref` dependency and several
host `Vec<u8>` / `HostBackend` bounds. Full backend-agnostic cleanup is
planned as follow-up work.
