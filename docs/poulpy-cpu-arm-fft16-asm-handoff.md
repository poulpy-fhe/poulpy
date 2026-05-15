# `fft16` / `ifft16` AArch64 Assembly — Handoff

This document brieves a follow-up port: hand-written AArch64 assembly
kernels for the size-16 FFT and inverse FFT leaves of `FFT64Neon`. It is
**optional**: a Rust-intrinsics `fft16_neon` / `ifft16_neon` already
lives in `poulpy-cpu-arm/src/neon/fft.rs` and is correct on its own.
This handoff is only worth taking on once profiling on Apple Silicon
(or another AArch64 host) shows the intrinsic version is the bottleneck
for real FHE workloads.

## Why these kernels matter

`fft16` and `ifft16` are the inner-most leaves of the size-`m` FFT
recursion in `poulpy-cpu-arm/src/neon/fft.rs`. The dispatcher in the
AVX backend, copied verbatim into the NEON port, ends every BFS level
at `m = 16` with a single call to the leaf:

```rust
// poulpy-cpu-avx/src/fft64/reim/fft_avx2_fma.rs:26
fn fft_avx2_fma(m: usize, omg: &[f64], data: &mut [f64]) {
    if m < 16            { fft_ref(m, omg, data); return }
    if m == 16           { fft16_avx2_fma(...) }
    else if m <= 2048    { fft_bfs_16_avx2_fma(...) }   // calls fft16 at every leaf
    else                 { fft_rec_16_avx2_fma(...) }   // → BFS → fft16 leaves
}
```

So `fft16` is hit **once per `m / 16` leaves**, every FFT call, for any
ring degree `n ≥ 32` (since `m = n / 2`). This is every realistic FHE
parameter:

| Scheme         | Source                                                     | `n`     | `m`    | `fft16` calls / FFT |
|----------------|------------------------------------------------------------|---------|--------|---------------------|
| CKKS           | `poulpy-ckks/examples/poly2.rs`                            | 4096    | 2048   | 128                 |
| bin-FHE GLWE   | `poulpy-bin-fhe/examples/bdd_arithmetic.rs` (`N_GLWE=1024`) | 1024    | 512    | 32                  |
| bin-FHE GLWE   | `poulpy-bin-fhe/src/blind_rotation/tests/fft64_avx.rs`     | 32, 512 | 16, 256 | 1, 16              |

The AVX backend's docstring claims the hand-written assembly leaves are
"**~2× over intrinsics**" (`poulpy-cpu-avx/src/fft64/mod.rs:28`). That
gain compounds across every leaf: a 2× win on `fft16` is roughly a 2×
win on the entire FFT for any `m` where the leaf phase dominates the
runtime (which it does at typical FHE sizes — the BFS phase is mostly
loads, stores, and a handful of cross-block twiddles per stage, while
the leaf does all the inner butterflies).

So the kernels **are** on the hot path for real schemes. The Rust
intrinsic version produced by the butterfly handoff is correct, but
this assembly port is the realistic path to closing the gap with the
AVX backend's published numbers.

## What exists in the AVX backend

| File | Lines | What |
|---|---|---|
| `poulpy-cpu-avx/src/fft64/reim/fft16_avx2_fma.s` | 162 | x86-64 GAS, AVX2/FMA, `extern "sysv64"` |
| `poulpy-cpu-avx/src/fft64/reim/ifft16_avx2_fma.s` | 180 | same, inverse FFT |
| `poulpy-cpu-avx/src/fft64/reim/mod.rs:35` | — | `global_asm!(include_str!("fft16_avx2_fma.s"), include_str!("ifft16_avx2_fma.s"))` |
| `poulpy-cpu-avx/src/fft64/reim/fft_avx2_fma.rs:46-55` | — | Rust `extern "sysv64"` declarations + thin wrapper that calls `fft16_avx2_fma_asm(re_ptr, im_ptr, omg_ptr)` |

The two AVX `.s` files are direct ports of x86 assembly from
`spqlios-arithmetic` and carry an Apache-2.0 disclaimer at the top
crediting that origin. **Spqlios has no AArch64 backend**, so the
AArch64 work briefed here is not a port of any existing spqlios source.
See §"License" below.

The ABI is three pointers in the SysV registers `%rdi` / `%rsi` /
`%rdx`: `re: *mut f64` (16 doubles), `im: *mut f64` (16 doubles),
`omg: *const f64` (read-only twiddles).

## What to produce

Two new files:

- `poulpy-cpu-arm/src/neon/fft64_reim/fft16_neon.s` — AArch64 GAS, AAPCS64 ABI.
- `poulpy-cpu-arm/src/neon/fft64_reim/ifft16_neon.s` — same, inverse FFT.

Plus wiring:

- `poulpy-cpu-arm/src/neon/fft.rs` — replace the existing Rust intrinsic
  `fft16_neon` / `ifft16_neon` with thin wrappers that call into the
  assembly:

  ```rust
  unsafe extern "C" {
      unsafe fn fft16_neon_asm(re: *mut f64, im: *mut f64, omg: *const f64);
      unsafe fn ifft16_neon_asm(re: *mut f64, im: *mut f64, omg: *const f64);
  }
  // global_asm!(include_str!("fft64_reim/fft16_neon.s"),
  //             include_str!("fft64_reim/ifft16_neon.s"));
  ```

  Use `extern "C"` rather than the AVX backend's `extern "sysv64"` —
  AAPCS64 is what `extern "C"` on AArch64 produces; there is no
  separately-named "sysv64" calling convention on AArch64.

## ABI

AAPCS64 register usage (analogous to the AVX SysV mapping):

| Argument           | AVX SysV reg | AArch64 AAPCS64 reg |
|--------------------|--------------|---------------------|
| `re: *mut f64`     | `%rdi`       | `x0`                |
| `im: *mut f64`     | `%rsi`       | `x1`                |
| `omg: *const f64`  | `%rdx`       | `x2`                |

Caller-saved NEON registers `v0`–`v7` and `v16`–`v31` (32 in total)
are free to clobber. `v8`–`v15`'s low 64 bits are callee-saved per
AAPCS64 — the AVX kernel uses 16 of its 16 ymm registers, so the
AArch64 port should be able to stay within the 24 caller-saved
NEON regs. If `v8`–`v15` are needed, save/restore the low 64 bits at
the entry/exit (`stp d8, d9, [sp, #-16]!` etc.).

## Algorithm shape (do not redesign)

The AVX kernel is a 4-stage Cooley–Tukey radix-2 FFT over 16 complex
points, layout `[re_0..re_15, im_0..im_15]`. Each stage is:

1. Load 8 complex points into split (re, im) pairs of `__m256d`
   (4 lanes each). On AArch64 these become 8 split pairs of
   `float64x2_t` (2 lanes each), so each AVX `ymm` register splits
   into a pair of NEON `q` registers.
2. Broadcast a complex twiddle `(ωr, ωi)` via `vshufpd` (AVX) or
   `dup` / `ext` (AArch64). The AVX file uses
   `vinsertf128 + vshufpd $0/$15` to splat the real and imag parts
   across all four lanes; AArch64 has direct `dup v.d[0]/d[1]`.
3. Cross-product butterfly: `t = ω · b`, `b' = a − t`, `a' = a + t`.
   AVX uses `vmulpd` + `vfmsub231pd` + `vfmadd231pd` to do the four
   `t.re = ωr·br − ωi·bi`, `t.im = ωr·bi + ωi·br` lanes in two FMAs
   per output lane.
4. Store back to the same `[re, im]` slots.

The NEON port differs only in:

- **Lane width**: 2 doubles per register vs 4 → roughly twice as many
  load/store/FMA instructions per stage. The 32-register file (vs AVX's
  16) absorbs the doubled register pressure.
- **FMA mnemonic**: `fmla v.d, v.d, v.d` (`d = d + a·b`) and
  `fmls v.d, v.d, v.d` (`d = d − a·b`). Note: AVX's `vfmsub231pd a, b,
  c = a·b − c` has **opposite sign** from NEON's `fmls` — see the
  butterfly handoff §"Sign trap" for the table.
- **Twiddle splat**: `dup v.d[0]` / `dup v.d[1]` extract a single
  lane and broadcast. The AVX `vshufpd $0` (lo, lo) and `$15` (hi, hi)
  patterns map directly.

Translate stage-by-stage. Do not invent a new schedule — the AVX
schedule is hand-tuned and tested.

## Step-by-step plan for the implementer

1. **Read the AVX source line-by-line.** Open
   `poulpy-cpu-avx/src/fft64/reim/fft16_avx2_fma.s` next to your
   AArch64 file. Each AVX `vmovupd` becomes two AArch64 `ld1` /
   `st1` instructions (one per `q`-register half). Each AVX FMA
   becomes one AArch64 FMA over a pair of `q` registers.
2. **Set up a register map up front.** AVX uses `ymm0..ymm15` and
   never spills. Allocate `v0..v23` (24 caller-saved q-regs)
   doubled-up: `v0/v1` ↔ AVX `ymm0`, `v2/v3` ↔ `ymm1`, etc. Document
   the map in a comment at the top.
3. **Stage 0: bulk load.** AVX loads 8 ymm regs from `[re]` and
   `[im]`. AArch64 needs 16 NEON loads (`ld1 {v0.2d}, [x0]`,
   `ld1 {v1.2d}, [x0, #16]`, …). Use `ldp` for paired loads where
   possible: `ldp q0, q1, [x0]` reads 32 bytes in one instruction.
4. **Stages 1-4: butterflies.** For each AVX stage (~30 instructions),
   produce ~50-60 AArch64 instructions. The first stage is the
   simplest twiddle (single complex broadcast); later stages have
   compound twiddle layouts that need careful translation of the
   AVX `vinsertf128` / `vshufpd` choreography.
5. **Stage 4: bulk store.** Mirror stage 0 with `stp q0, q1, [x0]`
   etc.
6. **Wire into Rust.** Replace the Rust intrinsic body of
   `fft16_neon` / `ifft16_neon` in `poulpy-cpu-arm/src/neon/fft.rs`
   with `extern "C" fn` declarations and a `global_asm!` invocation.
   Keep the function signatures byte-compatible with the intrinsic
   version so the dispatcher in `fft_neon` doesn't need to change.
7. **Repeat for `ifft16`.** The inverse kernel is structurally similar
   but with twiddles applied **before** the butterfly add/sub (Gentleman–
   Sande) and a final pass that bakes in `n^{-1}`.

## Build wiring

The AVX backend uses `std::arch::global_asm!` with `include_str!`. AArch64
behaves identically — no `build.rs` or `cc` build step needed:

```rust
// in poulpy-cpu-arm/src/neon/fft.rs (or a sub-module)
use std::arch::global_asm;
global_asm!(
    include_str!("fft64_reim/fft16_neon.s"),
    include_str!("fft64_reim/ifft16_neon.s"),
);
```

`rustc` assembles the GAS source with the system assembler. On the dev
box this resolves to `aarch64-linux-gnu-as` for the `aarch64-unknown-
linux-gnu` target and to LLVM's integrated assembler for the
`aarch64-unknown-linux-musl` target (via `rust-lld`). Both work without
extra configuration.

## Testing

The infrastructure shipped with the butterfly handoff covers this port
unchanged:

```bash
cargo test -p poulpy-cpu-arm --features enable-neon \
    --target aarch64-unknown-linux-musl
```

When this handoff lands, the existing per-kernel `neon::fft::tests::*`
unit tests (`fft_intt_identity_neon`, `fft_neon_vs_ref`,
`fft_convolution_neon`) and the `fft64::tests::*` HAL families will all
exercise the assembly path automatically — no test changes needed,
because the entry points keep the same Rust signatures.

The `cross_backend_test_suite!` HAL families are the tightest
correctness gate: any divergence between assembly fft16 and
`fft_ref` will show up as a `test_vec_znx_dft_*` mismatch.

## Definition of done

- `poulpy-cpu-arm/src/neon/fft64_reim/fft16_neon.s` and `ifft16_neon.s`
  exist and are the only definition of `fft16_neon_asm` /
  `ifft16_neon_asm`.
- `poulpy-cpu-arm/src/neon/fft.rs` invokes `global_asm!` and replaces
  the Rust intrinsic implementations of `fft16_neon` / `ifft16_neon`
  with thin `extern "C"` wrappers.
- All FFT64 tests pass under qemu (already-shipped command above).
- A bench delta is recorded in the PR description: run a representative
  CKKS bench (`cargo bench -p poulpy-bench --bench ckks_mul --features
  enable-neon`) on Apple Silicon **before and after** the assembly
  swap. The handoff is worth landing if the delta is ≥ 1.5× on
  `fft64-neon` rows; otherwise the maintenance cost (assembly is
  harder to read and review) outweighs the gain.
- This handoff file is deleted (its job is done) — but only if the
  recorded bench delta justifies keeping the assembly. If the
  intrinsic version is within ~10% of the assembly on the target
  hardware, abandon the port and update the README to remove this
  follow-up entirely.

## Out of scope

- **No SVE / SVE2.** AArch64 NEON is fixed 128-bit; SVE is
  vector-length agnostic. A future SVE backend (`FFT64Sve`) would have
  its own size-16 leaf — do not mix.
- **No new FMA scheduling.** Copy the AVX schedule. Only after the
  port lands and benches show a gap should anyone reschedule.
- **No Apple-specific tuning.** The kernel must work on any AArch64
  AAPCS64 host. Apple-specific micro-optimisations (e.g. assuming
  Firestorm's pipeline) belong in a separate, profiled patch.

## License

There is no spqlios-arithmetic AArch64 backend. The AArch64 `.s` files
produced by this handoff are **new work** — not a port of any existing
spqlios source. The instruction sequences themselves are authored from
scratch against the AArch64 NEON ISA.

What is inherited is the **algorithmic shape** (4-stage radix-2
butterfly schedule, register allocation pattern, loop layout). That
shape comes from the AVX `.s` files in `poulpy-cpu-avx/src/fft64/reim/`,
which themselves credit `spqlios-arithmetic`. The poulpy crate is
already Apache-2.0, the same license spqlios uses, so this lineage
needs no special handling — algorithmic ideas are not copyrightable in
the first place.

Recommended header for the new AArch64 `.s` files: a short comment
noting "AArch64 port; schedule derived from
`poulpy-cpu-avx/src/fft64/reim/{fft16,ifft16}_avx2_fma.s`". Do not
copy the spqlios disclaimer verbatim — it would suggest a direct port
that does not exist.
