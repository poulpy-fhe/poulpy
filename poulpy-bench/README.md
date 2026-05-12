# poulpy-bench

Consolidated [Criterion](https://bheisler.github.io/criterion.rs/book/) benchmark suite for the poulpy workspace.

Each benchmark binary covers one subsystem. Binaries that operate on generic polynomial or FHE operations run against **all available backends**; binaries that target transform-domain-specific operations (DFT, convolution, VMP/SVP) are restricted to the **FFT64 family**.

## Backends and feature flags

| Backend | Type | Feature flag | Label in output |
|---|---|---|---|
| `FFT64Ref` | FFT64, portable | *(always enabled)* | `fft64-ref` |
| `NTT120Ref` | NTT120, portable | *(always enabled)* | `ntt120-ref` |
| `FFT64Avx` | FFT64, AVX2/FMA | `enable-avx` | `fft64-avx` |
| `NTT120Avx` | NTT120, AVX2/FMA | `enable-avx` | `ntt120-avx` |

The `enable-avx` flag enables the `poulpy-cpu-avx` backend and requires `target_arch = "x86_64"`.

Benchmark binaries are also feature-gated by family:

| Feature | Enables |
|---|---|
| `hal-bench` | HAL-level polynomial, transform, convolution, FFT, and NTT benchmarks |
| `core-bench` | `poulpy-core` benchmarks |
| `bin-fhe-bench` | `poulpy-bin-fhe` scheme benchmarks |
| `ckks-bench` | CKKS benchmarks |

Useful compile/test commands:

```sh
cargo check -p poulpy-bench --all-targets --features hal-bench
cargo check -p poulpy-bench --all-targets --features core-bench
cargo check -p poulpy-bench --all-targets --features bin-fhe-bench
cargo check -p poulpy-bench --all-targets --features ckks-bench

cargo clippy -p poulpy-bench --all-targets \
  --features hal-bench,core-bench,bin-fhe-bench,ckks-bench -- -D warnings
```

## Benchmark binaries

### HAL-level (backend-agnostic polynomial arithmetic)

| Binary | Subsystem | Backends |
|---|---|---|
| `vec_znx` | `VecZnx` add / sub / negate / rotate / automorphism / shift | all |
| `vec_znx_big` | `VecZnxBig` add / sub / negate / normalize / automorphism | all |
| `vec_znx_dft` | DFT-domain add / sub / apply / iDFT | FFT64 only |
| `convolution` | Polynomial convolution (prepare + apply) | FFT64 only |
| `svp` | Scalar-vector product (prepare, DFT-to-DFT) | FFT64 only |
| `vmp` | Vector-matrix product (prepare, DFT-to-DFT) | FFT64 only |
| `fft` | Raw FFT / iFFT primitive | hardcoded (ref + avx) |
| `ntt` | Raw NTT / iNTT primitive | hardcoded (ref + avx) |

### Core-level (scheme-agnostic FHE operations)

| Binary | Subsystem | Backends |
|---|---|---|
| `operations` | GLWE add / sub / normalize / mul-plain | all |
| `encryption` | GLWE / GGSW / automorphism-key encryption | all |
| `decryption` | GLWE decryption | all |
| `automorphism` | GLWE automorphism | all |
| `external_product` | GLWE external product (inplace + out-of-place) | all |
| `keyswitch` | GLWE key-switching | all |

### Scheme-level (end-to-end FHE primitives)

| Binary | Subsystem | Backends |
|---|---|---|
| `blind_rotate` | Blind rotation (CGGI / AP) | hardcoded |
| `circuit_bootstrapping` | Circuit bootstrapping | hardcoded |
| `bdd_prepare` | BDD key preparation | hardcoded |
| `bdd_arithmetic` | BDD homomorphic arithmetic | hardcoded |

### Regression tracking

| Binary | Purpose |
|---|---|
| `standard` | One representative run across **all layers** with fixed parameters — used for version-to-version regression tracking |

The `standard` binary uses a single parameter set (`N=4096`, `base2k=18`, `k=54`, `rank=1`) for all HAL and core benchmarks, and the standard parameter sets embedded in the scheme crates for the scheme-level benchmarks. Its results can be saved as named baselines for direct comparison across releases (see [Save and compare baselines](#save-and-compare-baselines)).

## Configuring parameters via JSON

All sweep ranges and layout parameters are overridable at runtime through the `POULPY_BENCH_PARAMS` environment variable. Set it to either a **path to a JSON file** or an **inline JSON string**. Any field omitted from the JSON falls back to the built-in default.

### JSON schema

```json
{
  "run": ["vec_znx_big", "vmp", "external_product"],
  "hal": {
    "sweeps": [[10,2,2],[11,2,4],[12,2,8],[13,2,16],[14,2,32]]
  },
  "cnv": {
    "sweeps": [[10,1],[11,2],[12,4],[13,8],[14,16],[15,32],[16,64]]
  },
  "vmp": {
    "sweeps": [[10,2,1,2,3],[11,4,1,2,5],[12,7,1,2,8],[13,15,1,2,16],[14,31,1,2,32]]
  },
  "svp_prepare": {
    "log_n": [10,11,12,13,14]
  },
  "core": {
    "n": 4096, "base2k": 18, "k": 54, "rank": 1, "dsize": 1
  }
}
```

Field reference:

| Section | Field | Applies to | Description |
|---|---|---|---|
| `backends` | `["label", ...]` | shell script | Backends to run: `fft64-ref`, `ntt120-ref`, `fft64-avx`, `ntt120-avx`. AVX feature is auto-enabled when an AVX backend is listed. Omit to run all compiled-in backends. |
| `run` | `["name", ...]` | shell script | What to run. Binary names (e.g. `"vec_znx_big"`) run the whole binary; function names (e.g. `"vec_znx_big_add_into"`) are used as a Criterion filter across the default binary set. Mix freely. Omit or leave empty to run the default set in full. |
| `hal.sweeps` | `[[log_n, cols, size], ...]` | `vec_znx_big`, `vec_znx_dft`, `svp` | Sweep points for generic HAL ops |
| `cnv.sweeps` | `[[log_n, size], ...]` | `convolution` | Sweep points for convolution |
| `vmp.sweeps` | `[[log_n, rows, cols_in, cols_out, size], ...]` | `vmp` | Sweep points for VMP |
| `svp_prepare.log_n` | `[log_n, ...]` | `svp` prepare | Ring degrees for SVP prepare |
| `core.n` | power of two | all core/scheme/standard | Ring degree `N` |
| `core.base2k` | integer | all core/scheme/standard | Limb bit-width |
| `core.k` | integer | all core/scheme/standard | Total torus precision |
| `core.rank` | integer | all core/scheme/standard | GLWE rank |
| `core.dsize` | integer | all core/scheme/standard | Decomposition size |

### Examples

**Single ring degree — benchmark `vec_znx_big` only at N=4096:**

```sh
POULPY_BENCH_PARAMS='{"hal":{"sweeps":[[12,2,8]]}}' \
  cargo bench -p poulpy-bench --bench vec_znx_big --features hal-bench
```

**Custom core params — run `standard` at a smaller parameter set:**

```sh
POULPY_BENCH_PARAMS='{"core":{"n":1024,"base2k":14,"k":42,"rank":1,"dsize":1}}' \
  cargo bench -p poulpy-bench --bench standard --features hal-bench,core-bench,bin-fhe-bench
```

**From a file — full custom sweep for a profiling run:**

```sh
# bench_params.json
cat > bench_params.json <<'EOF'
{
  "hal":  { "sweeps": [[10,2,2],[12,2,8],[14,2,32]] },
  "cnv":  { "sweeps": [[10,1],[12,4],[14,16]] },
  "vmp":  { "sweeps": [[10,2,1,2,3],[12,7,1,2,8]] },
  "core": { "n": 4096, "base2k": 18, "k": 54, "rank": 1, "dsize": 1 }
}
EOF

POULPY_BENCH_PARAMS=bench_params.json \
  cargo bench -p poulpy-bench --features hal-bench,core-bench,bin-fhe-bench,ckks-bench,enable-avx
```

**Regression baseline at a specific parameter set:**

```sh
POULPY_BENCH_PARAMS='{"core":{"n":4096,"base2k":18,"k":54,"rank":1,"dsize":1}}' \
  cargo bench -p poulpy-bench --bench standard --features hal-bench,core-bench,bin-fhe-bench -- --save-baseline v0.4.4

# later, compare against it with the same params
POULPY_BENCH_PARAMS='{"core":{"n":4096,"base2k":18,"k":54,"rank":1,"dsize":1}}' \
  cargo bench -p poulpy-bench --bench standard --features hal-bench,core-bench,bin-fhe-bench -- --baseline v0.4.4
```

## Running benchmarks

### All benchmarks, reference backends only

```sh
cargo bench -p poulpy-bench --features hal-bench,core-bench,bin-fhe-bench,ckks-bench
```

### All benchmarks with AVX acceleration

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo bench -p poulpy-bench --features hal-bench,core-bench,bin-fhe-bench,ckks-bench,enable-avx
```

### One binary

```sh
# run only the vec_znx binary
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench

# with AVX
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench,enable-avx
```

### One group or function within a binary

Criterion accepts a regex filter as a trailing argument (after `--`).
The benchmark ID format is `<group_name>/<parameter>` where the group name
encodes the operation and backend label.

```sh
# all vec_znx benchmarks on the ntt120-ref backend
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench -- ntt120-ref

# only the add benchmark, all backends
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench -- vec_znx_add_into

# one specific backend × operation
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench -- "vec_znx_add_into::fft64-ref"

# all encryption benchmarks, AVX only
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo bench -p poulpy-bench --bench encryption --features core-bench,enable-avx -- avx
```

### Standard regression binary

```sh
# run the standard binary (ref backends)
cargo bench -p poulpy-bench --bench standard --features hal-bench,core-bench,bin-fhe-bench

# with AVX acceleration
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo bench -p poulpy-bench --bench standard --features hal-bench,core-bench,bin-fhe-bench,enable-avx
```

### Save and compare baselines

```sh
# save a named baseline (e.g. tagging a release)
cargo bench -p poulpy-bench --bench standard --features hal-bench,core-bench,bin-fhe-bench -- --save-baseline v0.4.4

# run again later and compare against it
cargo bench -p poulpy-bench --bench standard --features hal-bench,core-bench,bin-fhe-bench -- --baseline v0.4.4
```

The same `--save-baseline` / `--baseline` flags work on any bench binary:

```sh
# save a baseline named "before"
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench -- --save-baseline before

# bench again and compare
cargo bench -p poulpy-bench --bench vec_znx --features hal-bench -- --baseline before
```

Criterion HTML reports are written to `target/criterion/`.

## Adding a new backend

1. Add the backend crate to `[dependencies]` in `Cargo.toml` (behind an optional feature if needed).
2. Add one entry to the appropriate private family macro in [`src/lib.rs`](src/lib.rs):
   - `for_each_fft_backend_family!` for an FFT64 backend
   - `for_each_ntt_backend_family!` for an NTT120 backend
3. No bench files need to change.
