# GPU backend crate review rules

Read with `docs/review/REVIEW.md`; this file adds the rules specific to `poulpy-gpu-portable`. The rules of `docs/review/backends.md` cover the CPU crates; they apply here only where a rule below cites them.

## 0. Scope and terms

- **Terms.** *Runtime*: a CubeCL runtime marker of the crate (`Cuda`, `Hip`, `Wgpu`), implementing the sealed `GpuRuntime`; *client*: the one CubeCL client of a runtime; *launch*: one kernel dispatch through `launch!`; *view*: a byte range of a device allocation (`GpuRef`, `GpuMut`), bound to a kernel as a `Binding`; *unit*: one kernel thread; *host stage*: work done on the host with the portable implementation and moved to or from the device; *recording* and *replay*: a body of backend calls captured once and rerun as one CubeCL graph (`Replay`).

## 1. Crate role and dependencies

Generic rule: `docs/review/workspace.md` section 1 (dependency direction).

- **Dependencies.** The crate takes poulpy-hal, poulpy-cpu-portable, CubeCL, `cubecl-environment`, `bytemuck` and `pollster` unconditionally, poulpy-core and poulpy-ckks only behind `enable-core` and `enable-ckks` (each also enabling the same poulpy-cpu-portable feature), `anyhow` only behind `enable-ckks`, and no other `poulpy-cpu-*` crate.
- **Definitions come from portable.** The prime set and every constant derived from it come from `poulpy_cpu_portable::kernels::ntt4x30::primes::Primes30`, host stages call portable's kernels and encoder (section 6), and no prime, root, CRT constant or rounding routine is copied into the crate.
- **CubeCL is pinned.** `cubecl` and `cubecl-environment` are pinned to one exact release (`=x.y.z`) in the workspace manifest, with the comment saying why; a bump moves both, states the runtimes and devices the suites then passed on and the bootstrap benchmark before and after (`docs/review/bench.md` section 5), and updates every comment naming the release.
- **A compile lane.** A step of `.github/workflows/ci.yml` lints the crate with `-D warnings` and every target on a runtime that builds without a vendor toolkit (`enable-vulkan`) with `enable-ckks`, and every lane excluding the crate states its reason beside the exclusion.

## 2. Types, runtimes and features

Generic rules: `docs/review/workspace.md` sections 12 and 14.

- **One marker, one alias per runtime.** The backend is the one generic `NTT4x30Gpu<RT: GpuRuntime, R: Ring = Standard>`, and each runtime has the alias `NTT4x30<Runtime><R = Standard> = NTT4x30Gpu<runtime::<Runtime>, R>` behind its feature, documented in `lib.rs`; no code is written per runtime outside the `gpu_runtime!` marker and its registration lines.
- **Sealed runtimes.** Runtime markers come only from `gpu_runtime!` in `runtime/device.rs` (`GpuRuntime: RuntimeDevice`), `RuntimeDevice` is never exported, and each comes with its feature, name, workgroup size (`RuntimeDevice::GROUP`) and transform geometry (`NttGeometry`).
- **A new runtime lands whole.** A new runtime adds in one diff its feature, marker and alias, the `# Features` list of `lib.rs`, the README table, the GPU table of `docs/guides/backends.md`, the registrations of both rings in `core_impl.rs` and `ckks_impl.rs`, the test runtime selection of `src/tests/mod.rs` and, if it builds without a vendor toolkit, `[package.metadata.docs.rs]`.
- **One device per runtime.** A runtime owns one device for the life of the process, chosen by `GpuRuntime::select_device` before the first module or buffer of the runtime and the first one otherwise, and only `RuntimeDevice::client` creates a client.
- **Features.** The crate's features are one `enable-<runtime>` per runtime (`enable-metal`, `enable-vulkan` and `enable-wgpu` all selecting `Wgpu`) plus `enable-core` and `enable-ckks`, code needing a runtime is gated on the `any(..)` of the runtime features, and the tests are gated the same way.

## 3. Arithmetic contract

Generic rules: `docs/review/workspace.md` section 2 (the reference is the implementation) and `docs/review/hal.md` section 9.

- **Bit-identical with portable.** Every operation returns the words of `NTT4x30Portable` (`NTT4x30CIPortable` on the conjugate-invariant ring) for every input: integer arithmetic only, no float kernel on the device, normalization following the HAL definition (one rounding, ties to even) for every radix, and no tolerance in any test.
- **The DFT word.** `DftWord` is `CrtWord<Primes30, u32>`: each `(limb, column)` block of a DFT-domain vector is four planes of `n` `u32`, one per prime, canonical in `[0, q)`, and `DFT_LIMBS_CONTIGUOUS = true` holds (limb `j` of column `i` at block `j * cols + i`, `docs/review/backends.md` section 5.7); a kernel writing DFT words keeps them canonical.
- **Tables and constants on the host.** Transform tables are built on the host from `Primes30` and uploaded once per degree in the module handle (`ntt4x30::Tables`), kernels receive constants as `#[comptime]` arguments or literals computed by `ntt4x30/primes.rs`, and `Backend::MIN_DEGREE` stays tied to the tables' smallest degree by the `const` assertion of `HalModuleImpl::new`.
- **The slot law.** Forward slots follow the CPU slot law (`docs/review/backends.md` section 6), the conjugate-invariant transform being the negacyclic one of the same degree behind the basis change documented in `ntt4x30/mod.rs`.
- **Radix model.** `MaxBase2k` is implemented once over `R: Ring` with `max_base2k_ntt` and `max_base2k_ntt_squares` on `Primes30::LOG_Q_PRODUCT`, as portable's (`docs/review/hal.md` section 6).

## 4. Storage, ordering and device access

Generic rules: `docs/review/hal.md` sections 6 (deferred execution) and 7, and `poulpy-hal/docs/backend_safety_contract.md`.

- **Device storage.** `OwnedBuf = GpuBuf<RT>`, `BufRef = GpuRef`, `BufMut = GpuMut` and `Location = Device`; owned allocations are padded as host ones (`DEFAULTALIGN`) so that a host-staged transfer moves an owned buffer byte for byte, the transfer hooks keep the prefix semantics of `docs/review/backends.md` section 5.7 (`write_prefix` zeroes the rest), and `test_transfer_padded_lengths` stays registered.
- **One stream, ordered traffic.** Every client is pinned to the crate's stream, so launches, transfers and replays are ordered whatever thread issues them, and every launch, transfer, allocation and synchronization takes `runtime::access::shared()`, recordings and replays taking `access::exclusive()`; new code reaches the client only through the `runtime` helpers that take the access.
- **Calls queue, few things wait.** A backend call returns once its launches are queued; only `HalModuleImpl::synchronize` and a transfer to the host wait for queued work, and library code never synchronizes or reads back inside an operation (a read is allowed only in a host stage, section 6).
- **Temporaries.** Temporaries come from the caller's scratch arena (`docs/review/workspace.md` section 5), and `GpuHandle::with_temp` serves only the in-place permutations, the inverse transform that keeps its source and the wide-radix normalization of an inverse transform, zeroing its temporary before releasing it; any other per-call device allocation states why the arena cannot hold it.

## 5. Kernels and launches

Generic rules: `docs/review/workspace.md` sections 4 (checks before the first effect) and 7.

- **Through `launch!`.** A kernel is a `#[cube(launch_unchecked)]` function launched only through `runtime::launch!`, under its own `Label` (a new kernel adds one), with the bytes it reads plus writes as its accounted size, its arguments in the order buffers, scalars, compile-time arguments.
- **Checked on the host.** Kernels launch unchecked, so every bound is checked on the host before the launch through the operand helpers of `hal_impl/operand.rs` and `DftColumn::limbs`: the column (`"operand column out of range"`), the storage covering the layout (`"device storage shorter than its layout"`) and every word index in the 32-bit range; a kernel guards its own units (`if pos < units`) and relies on nothing else for bounds.
- **Aliasing through `BufferArgs`.** Views of one allocation that may overlap are passed as aliases of one buffer argument (`BufferArgs`), never as distinct arguments, and a kernel writing a buffer it also reads either reads each word before writing it at the same index (its doc saying so) or goes through a temporary.
- **Geometry from the runtime.** Workgroup sizes and transform tiling come from `RT::GROUP` and `RT::NTT`, never literals in a kernel or launch, and a change to them or to a tile, fold or stage count states its measurement (device, runtime, degree, speedup) beside the constant (`docs/review/backends.md` section 10, gate constants).
- **Exactness by construction.** A kernel's accumulation bound (products per reduction, fold depth, limbs per pass) is a named constant whose doc derives it from the operand bounds, never checked in the data.

## 6. Host stages

- **What goes through the host.** Sampling (`SamplingImpl`, the portable samplers drawing the same values as `NTT4x30Portable`), CKKS encoding (`CKKSEncodingImpl`, the PaCo and SHIP coefficient encodings, all through the portable encoder on the canonical tables), the uniform fill of a coefficient column, and normalizations of an `i128` source or of an inverse transform at a radix above the kernel's 62 bits; nothing else runs on the host.
- **Staged as portable computes.** A host stage computes with the portable implementation on host copies of the operands and moves each column in one transfer each way, and its result is the portable one by construction.
- **Staged secrets are wiped.** A host stage a secret or a sample reaches (sampling, a normalization on a decryption path) zeroes the host copies and device buffers it owns before releasing them, the staging copy CubeCL owns for each transfer being out of its reach.
- **Keep composites recordable.** A composite operation (a key switch, a linear transformation, a bootstrap) issues no host stage, read, host write or synchronization when the data it needs is on the device; a constant a reference body uploads at every call is built once per module and kept in its handle (`ckks_add_sub.rs`, `ckks_impl::one_plaintext`).

## 7. Registration and overrides

Generic rules: `docs/review/core.md` section 1, `docs/review/ckks.md` section 1 and `docs/review/backends.md` sections 3.3 (hand-kept lists move together) and 4 (core-level overrides).

- **Every family on both rings.** Every HAL family is implemented for `NTT4x30Gpu<RT, R>` over both rings, standard-only families (`HalVecZnxMonomialImpl`, `HalVecZnxCIImpl`) on `Standard` only, and every scheme family registers per runtime and ring through the crate's macros: `impl_core!`, `impl_tensoring!` and `impl_core_standard!` in `core_impl.rs`, `impl_ckks!` and `impl_ckks_standard!` (with the PaCo and SHIP encodings) in `ckks_impl.rs`; a new upstream family enters these macros in the same PR.
- **Overrides only for fused kernels.** A family is hand-written only where the crate has a fused kernel or must avoid a host transfer (`GGLWEProductDigitsStridedImpl`, `GLWETensoringImpl::glwe_mul_columns`, the cached add-one and sub-one of `ckks_add_sub.rs`), every other method of that family, queries included, forwarding to its `*_reference` body, and the override asserting what the reference asserts before its first launch.
- **Queries match the bodies.** A HAL method whose kernels take no scratch reports 0, an override's query is the reference query or covers its own carve, and a query never depends on the runtime.

## 8. Recording and replay

- **The `Replay` contract.** The contract on `Replay` (same operands, same control flow, device work only) and its summary in `docs/guides/backends.md` change together, and a recording is valid only while every buffer its body received is still owned (`GraphRecording::valid`).
- **Recordability is a property to keep.** A change that adds a host read, host write, synchronization or allocation to a composite operation makes it unrecordable; it states that cost, and `Replay::check` or the host events (`Replay::host_events`) of a test show the operation records.
- **Panic-safe capture.** Recording code holds the exclusive access from `graph_prepare` to `stop_capture`, ends a capture on drop, and never records a body whose warm-up run read, wrote or synchronized.

## 9. Tests

Generic rules: `docs/review/workspace.md` section 15 and `docs/review/hal.md` section 11.

- **Portable is the attested link.** Every cross-backend registration takes `NTT4x30Portable` (`NTT4x30CIPortable` for the conjugate-invariant ring) as `backend_ref`, except the CKKS encryption parity: the core encryption parity compares draws against `NTT4x30Portable` (same samplers, section 6), and the CKKS encryption parity against `ControlledSamplingFFT64Portable` (`ControlledSamplingFFT64CIPortable`) under the controlled-sampling scope (`docs/review/workspace.md` section 10).
- **The HAL lists mirror portable.** The HAL registrations mirror portable's NTT4x30 lists for both rings, a test renamed or added upstream moving here in the same PR, minus the operand re-view variants registered on portable and the oracle alone (`docs/review/backends.md` section 13.2), plus the large degrees the transform tiles change at (`ntt_n4096` to `ntt_n131072`, `ci_ntt_n8192`) and `test_transfer_padded_lengths`.
- **Scheme suites.** The core parity list is complete (`docs/review/backends.md` section 13.2, complete core parity lists), with `core_encryption_parity_test_suite!` and the conjugate-invariant core parity, and with `enable-ckks` the CKKS parity modules `f64`, `quad`, `rank2` and a full-degree one, encryption parity at ranks 1 and 2, and the conjugate-invariant CKKS modules.
- **Local tests for what no suite reaches.** `src/tests/local.rs` compares each fused product with its reference composition, at a small degree and at the full degree of a bootstrap, plus views the suites do not build (limb ranges, windows) and the host stages at the radices only they serve; a new kernel path gets a case there when no suite reaches it.
- **Replay tests.** `src/tests/replay.rs` keeps a case per clause of the `Replay` contract (a repeated closure, a changed one, a host-touching one, a dropped operand, another thread, a panic, nesting).
- **Running the suites.** A PR touching the crate states the runtimes and devices its suites ran on (`cargo test --release -p poulpy-gpu-portable --features enable-<runtime>,enable-ckks`), and the limits listed in `docs/guides/backends.md` stay true.

## 10. Docs and benches

Generic rules: `docs/review/workspace.md` section 13 and `docs/review/bench.md` sections 4 and 5.

- **Three places, one story.** The `lib.rs` crate doc (features and backend types), the README (feature table, staging, commands) and the GPU section of `docs/guides/backends.md` (runtimes, ordering, replay, limits) describe the same behaviour and change together.
- **Benches.** `benches/standard.rs` registers the shared HAL, core and CKKS suites and `benches/bootstrapping.rs` the shared bootstrap runners with a portable client, `bench_ckks_bootstrapping_with_client` and `bench_ckks_bootstrapping_replayed_with_client` recording through `Replay`, every iteration ending with a synchronization; a performance claim names the device, the runtime, the CubeCL release and the preset.
