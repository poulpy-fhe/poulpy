# poulpy-release-bench

Release measurements of the CKKS bootstrapping presets, of `poulpy-libm` functions and of `poulpy-blas` operations, for the published benchmark tables.
It is driven by [`tools/release_bench.py`](../release_bench.py), which builds this binary, runs it and writes the record of a host.

```sh
# On every machine of the release, from the repository root
python3 tools/release_bench.py run --label ryzen-9950x

# One or two suites only
python3 tools/release_bench.py run --label ryzen-9950x --suites libm blas

# Then on one of them
python3 tools/release_bench.py merge release.json ryzen-9950x.json m5-pro.json
```

## What is measured

Three suites, each selected with `--suites` (all by default): `bootstrapping`, `libm` and `blas`.
Every measurement reports its parameters, its modulus, its setup time, the time of every warm-up and timed call, and the precision of its output.

### Bootstrapping

The release set is three presets at `N = 2^16`: `n16_d35_k636_p20_c2s`, `n16_d35_k733_p20_s2c` and the fastest, `n16_d35_k685_p19_s2c_ffr_ci`.
Each one is measured with the rotation keys the preset derives and with power-of-two rotation keys (`pow2_keys` in the record), at the radix `preset_for_backend` selects for a failure probability of `2^-128`.

A measurement sets the preset up with the shared `BootstrappingPresetRun` driver, runs 3 warm-up bootstraps and 20 timed ones, then decrypts the last output and measures its precision.
The timed region is one bootstrap call followed by a device synchronization.
Setup (context, keys, encryption and the one bootstrap the driver runs to check the output width) is timed apart, and decryption is excluded.
The GPU backends are also measured with the bootstrap recorded once as a graph and replayed (`mode: "replay"`), each iteration being one replay followed by a device synchronization.
On a runtime that records no graph, that row has `status: "unavailable"` and the direct measurement stands.
A graph dropped during the iterations fails the row rather than mixing replayed and direct timings.
A precision below the one the preset advertises is reported, not fatal.

### libm

One ciphertext at `N = 2^16`, `base2k = 52`, a scale of `2^50`, every slot holding a value of the domain.
At `2^40` the noise caps the results near 21 bits relative to the largest output, below the targets of `rsqrt_r256` and `inv_r256`.

| Case | Function | Domain | Target |
|---|---|---|---|
| `rsqrt_r16` | `ckks_rsqrt` | [1/16, 1] | 20 bits |
| `rsqrt_r256` | `ckks_rsqrt` | [1/256, 1] | 20 bits |
| `softmax_b20` | `ckks_softmax` | [-1, 1] | 20 bits |
| `inv_r256` | `ckks_inverse_positive_domain` | [1/256, 1] | 20 bits |

Targets are absolute-error bits, as libm's options count them.
`ckks_rsqrt` only refines an estimate, so the rsqrt cases evaluate a minimax seed of `1/sqrt(x)` first, then Newton steps: the seed and step count of least depth that reach the target, both in the record.
The inverse runs in place, so its timed call starts with a copy of the input.
The ciphertext is as wide as the function consumes plus `2 log_delta` of headroom and two limbs of guard, rounded up to a limb.
Keys use digits of 4 limbs, which keeps a softmax point, with its tensor key and 15 rotation keys, under 6 GB.

### blas

One `d x d` block per operand at `N = 2^16`, so a ciphertext holds `2^15 / d^2` matrices, each of distinct entries.
The points take the parameters of the poulpy-bench sweep (`base2k = 52`, `k = 208`, entries at scale `2^40`, maps and public matrices at `2^40`), at the degree and block sizes above, and are built by its point builders.

| Op | d = 32 (32 matrices) | d = 64 (8 matrices) | d = 128 (2 matrices) |
|---|---|---|---|
| `blas_matmul` (ct x ct) | `matmul_d32` | `matmul_d64` | `matmul_d128` |
| `blas_matmul_pt_right` (Jkls) | `matmul_pt_right_d32` | `matmul_pt_right_d64` | `matmul_pt_right_d128` |
| `blas_transpose` | `transpose_d32` | `transpose_d64` | `transpose_d128` |
| `blas_sum_rows` | `sum_rows_d32` | `sum_rows_d64` | `sum_rows_d128` |

A builder checks the radix against the lazy sums of its operation and measures the error of the first result, which the record compares with the error bound of poulpy-bench (`meets_target`), then the same call is timed.
The record also gives the time per matrix.

The libm and blas suites run on every backend.
As in the bootstrapping suite, a GPU backend evaluates and `NTT4x30Portable` generates the keys, encrypts and decrypts, the keys and ciphertexts moving between them.
A CPU backend is its own client.
A libm or blas case builds its point, calls it once, then moves the result to the client, decrypts and decodes it before the timed iterations, so its `setup_ns` includes that checking call.

### Modulus

A libm or blas record reports, under `modulus`, the width of a fresh ciphertext (`log_q`), the bits its operation consumed, measured from the budget of its input to that of its output (a libm record also gives the plan's count, `planned_consumed_bits`), the budget left in its output, and the key layout: its full width (`key_log_qp`), digit size, digit count and guard.
A bootstrapping record reports its preset's moduli under `preset`.

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

Nothing else builds it, so a change to an API it calls (the CKKS test-suite presets, the libm plans, the poulpy-bench point builders, a backend type) or to the workspace lock breaks it silently.
Before a release, and after such a change, check that it still builds:

```sh
cargo check --locked --manifest-path tools/poulpy-release-bench/Cargo.toml --features portable-rayon
```

## The binary

```sh
poulpy-release-bench list
poulpy-release-bench run --backend NTT4x30NeonRayon --preset n16_d35_k733_p20_s2c [--pow2] [--replay] [--warmup 3] [--iters 20] [--device 0]
poulpy-release-bench run --backend NTT4x30NeonRayon --libm rsqrt_r16 [--warmup 3] [--iters 20] [--device 0]
poulpy-release-bench run --backend NTT4x30NeonRayon --blas matmul_d64 [--warmup 3] [--iters 20]
```

`list` prints the backends of the build with the suites each one runs, the release presets and the libm and blas cases.
`run` prints one JSON object: the suite, the backend, its crate and features, the thread count, the preset or case and its parameters, the modulus of a case, the setup time, the time of every warm-up and timed iteration, and the precision.
