# poulpy-release-bench

Release measurements of the CKKS bootstrapping presets, for the published benchmark tables.
It is driven by [`tools/release_bench.py`](../release_bench.py), which builds this binary, runs it and writes the record of a host.

```sh
# On every machine of the release, from the repository root
python3 tools/release_bench.py run --label ryzen-9950x

# Then on one of them
python3 tools/release_bench.py merge release.json ryzen-9950x.json m5-pro.json
```

## What is measured

The release set is three presets at `N = 2^16`: `n16_d35_k636_p20_c2s`, `n16_d35_k733_p20_s2c` and the fastest, `n16_d35_k685_p19_s2c_ffr_ci`.
Each one is measured with the rotation keys the preset derives and with power-of-two rotation keys (`pow2_keys` in the record), at the radix `preset_for_backend` selects for a failure probability of `2^-128`.

A measurement sets the preset up with the shared `BootstrappingPresetRun` driver, runs 3 warm-up bootstraps and 20 timed ones, then decrypts the last output and measures its precision.
The timed region is one bootstrap call followed by a device synchronization.
Setup (context, keys, encryption and the one bootstrap the driver runs to check the output width) is timed apart, and decryption is excluded.
The GPU backends are also measured with the bootstrap recorded once as a graph and replayed (`mode: "replay"`), each iteration being one replay followed by a device synchronization.
On a runtime that records no graph, that row has `status: "unavailable"` and the direct measurement stands.
A graph dropped during the iterations fails the row rather than mixing replayed and direct timings.
A precision below the one the preset advertises is reported, not fatal.

## Builds

The driver builds this crate once per set of target flags, each in its own target directory under `target/release-bench`, so that no backend is compiled for a wider instruction set than its own:

| Build | Features | `RUSTFLAGS` | Backends |
|---|---|---|---|
| `base` (x86) | `portable-rayon`, GPU | none | `NTT4x30Portable`, `NTT4x30PortableRayon`, GPU |
| `avx` | `avx` | `+avx2,+fma` | `NTT4x30Avx`, `NTT4x30AvxRayon` |
| `avx512` | `avx512` | `+avx2,+fma,+avx512f` | `NTT4x30Avx512`, `NTT4x30Avx512Rayon` |
| `ifma` | `ifma` | `+avx2,+fma,+avx512f,+avx512ifma,+avx512vl` | `NTT3x42Ifma`, `NTT3x42IfmaRayon` |
| `base` (ARM) | `portable-rayon`, `neon`, GPU | none | portable, `NTT4x30Neon`, `NTT4x30NeonRayon`, GPU |

A build is skipped when the host lacks its instruction sets.
The GPU feature is `metal` on Apple Silicon, `cuda` when `nvcc` is found, `hip` when ROCm is.
`--gpu` overrides it and `--gpu none` skips it.
The GPU backends use `NTT4x30Portable` as the client that generates the keys and encrypts.

Rayon backends run on 16 threads on x86 and on every thread on ARM, capped at what the host has (`--threads` overrides).

The crate is outside the workspace, so workspace builds and CI leave it out.
Its `Cargo.lock` pins the versions of the workspace lock.
After the workspace lock changes, refresh it from the repository root:

```sh
cp Cargo.lock tools/poulpy-release-bench/Cargo.lock
cargo metadata --format-version 1 --manifest-path tools/poulpy-release-bench/Cargo.toml > /dev/null
```

## The binary

```sh
poulpy-release-bench list
poulpy-release-bench run --backend NTT4x30NeonRayon --preset n16_d35_k733_p20_s2c [--pow2] [--replay] [--warmup 3] [--iters 20] [--device 0]
```

`list` prints the backends of the build and the release presets.
`run` prints one JSON object: the backend, its crate and features, the thread count, the preset and its parameters, the setup time, the time of every warm-up and timed iteration, and the precision of the real and imaginary parts.
