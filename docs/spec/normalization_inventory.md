# Normalization migration inventory (PR 0)

Companion to [normalization_typestate.md](normalization_typestate.md). This document records the PR 0 deliverables: the frozen normalization contracts, the workspace-wide inventory of every site the migration must touch or guard, and the baselines against which later PRs are measured.

Baseline snapshot: branch `jp/move_normalize_flag`, commit `0827a5dc`, 2026-09-03. All counts below were produced with the fixed-string counting rule of `scripts/normalization_denylist.sh` (grep over `poulpy-*/src` and `poulpy-*/examples`, one count per matching line, definitions included) unless a section states otherwise. Refresh counts when rebasing this inventory.

## 1. Reference definitions

These restate the normative definitions of spec §3 so reviewers can sign off on the numbers themselves (exit-gate item 1).

**Normalized bounds.** For digit base `base2k = b`, a limb word `d` is normalized iff `d ∈ [-2^(b-1), 2^(b-1))`. A column is `Normalized` iff every live limb word satisfies this interval for the column's `base2k`. The interval is asymmetric: `-2^(b-1)` is included, `+2^(b-1)` is excluded.

**Bottom-limb orientation.** Limb index grows toward less significant digits: limb `0` holds the most significant digit and the bottom live limb `L-1` holds the least significant digit, where `L = ceil(represented_k / base2k)`.

**`represented_k`.** The torus precision in bits that the layout claims to represent. Every context is constructed with `represented_k > 0`, so `L >= 1` and the bottom live limb always exists.

**Canonical projection `P_p`.** With `p = base2k - (represented_k % base2k)` when `represented_k` is not a multiple of `base2k`, and `p = 0` otherwise, `P_p(d) = from_bits(to_bits(d) & (!0 << p))` applied to every word of the bottom live limb only. `P_p` preserves the normalized interval (the endpoint `-2^(b-1)` is `2^p`-grid-aligned) and commutes with normalization for a fixed representation context: `normalize(P_p(x)) == P_p(normalize(x))`.

## 2. Frozen normalization API family

These are the signatures that must remain source-shaped through the migration apart from new state bounds (spec §5.2). They are frozen by compile-time shape tests: [poulpy-hal/src/api/shape_tests.rs](../../poulpy-hal/src/api/shape_tests.rs) and [poulpy-core/src/api/shape_tests.rs](../../poulpy-core/src/api/shape_tests.rs). Any signature edit fails those files first and requires a spec §5.2 amendment in the same PR.

| Trait | Method | Class |
|---|---|---|
| `VecZnxNormalizeTmpBytes` | `vec_znx_normalize_tmp_bytes` | scratch sizing |
| `VecZnxNormalize` | `vec_znx_normalize` | out-of-place |
| `VecZnxNormalizeAssignBackend` | `vec_znx_normalize_assign_backend` | assign |
| `VecZnxNormalizeCoeffBackend` | `vec_znx_normalize_coeff_backend` | coefficient |
| `VecZnxNormalizeCoeffAssignBackend` | `vec_znx_normalize_coeff_assign_backend` | coefficient assign |
| `VecZnxBigNormalizeTmpBytes` | `vec_znx_big_normalize_tmp_bytes` | scratch sizing |
| `VecZnxBigNormalize` | `vec_znx_big_normalize` | big |
| `VecZnxIdftNormalizeConsumeTmpBytes` | `vec_znx_idft_normalize_consume_tmp_bytes` | scratch sizing |
| `VecZnxIdftNormalizeConsume` | `vec_znx_idft_normalize_consume` | fused (IDFT + normalize, optional addend) |
| `GLWENormalize` | `glwe_normalize_tmp_bytes`, `glwe_normalize`, `glwe_normalize_assign` | aggregate out-of-place / assign |

Deliberately NOT frozen, because the migration removes them: the receiver forms `VecZnx::normalize`, `GLWE::normalize`, `CKKSCiphertext::normalize`, the scratch-view receiver normalizers, and `SetNormalizationState` (spec PRs 3, 5, 6).

State-effect classification under the new algebra (spec §5.3, §6.2): out-of-place, coefficient, big, and fused paths write `Coeff<Normalized, NonCanonical>` stores; assign paths are state-preserving and non-promoting; none of them repairs padding; only `make_canonical` / `make_canonical_consume` produce `Canonical`.

## 3. Site inventories

### 3.1 `into_unnormalized` (184 lines, 54 files)

Classification groups follow spec §6.4: **A** scratch accumulator (becomes `Unwritten` scratch root), **B** caller-provided normalized destination (becomes compute-in-scratch plus out-of-place normalize), **C** full writer (becomes sealed `Unwritten` builder), **P** the primitive itself and its OEP plumbing, **T** test/bench/example harness (mechanical retyping).

| Area | Lines | Group |
|---|---|---|
| `poulpy-hal/src/test_suite/vec_znx.rs` | 42 | T |
| `poulpy-hal` layouts, scratch, oep (primitive definitions) | 9 | P |
| `poulpy-ckks` carry verb (oep + default + api + delegates for add/sub/composite) | 36 | B |
| `poulpy-ckks` layouts, oep, lib, test_suite | 15 | P, T |
| `poulpy-core` operations, packing, encryption, noise, linear transformation defaults | 20 | A, B |
| `poulpy-core` layouts, oep, scratch views | 10 | P |
| `poulpy-core` test_suite | 9 | T |
| `poulpy-bin-fhe` eval, fhe_uint, blind rotation | 13 | A |
| `poulpy-bench` | 15 | T |
| CPU backend crates (reference kernels, core_impl, tuning, examples) | 11 | C, T |

The per-file counts behind this table are reproducible with `grep -rc --include='*.rs' into_unnormalized poulpy-*/src poulpy-*/examples`.

### 3.2 Receiver normalization (23 sites)

Every `.normalize(` receiver call, all slated for replacement by module out-of-place normalization or scratch transactions:

- `poulpy-core/src/layouts/glwe.rs:141` and `poulpy-core/src/layouts/scratch_views.rs:146` (the aggregate receiver plumbing itself).
- `poulpy-core/src/default/glwe_packing.rs:62`, `poulpy-core/src/default/encryption/glwe.rs:480,579,616` (group B call sites).
- `poulpy-ckks/src/layouts/ciphertext.rs:507` (plumbing), `poulpy-ckks/src/delegates/composite.rs:458` (group A).
- `poulpy-ckks/src/test_suite/add_unsafe.rs`, `sub_unsafe.rs` (8 sites, group T).
- `poulpy-bin-fhe/src/blind_rotation/algorithms/cggi/algorithm.rs:684`, `bdd_arithmetic/eval.rs:625,710,849,915,965`, `bdd_arithmetic/ciphertexts/fhe_uint.rs:634` (group A).

### 3.3 Public raw mutation

`as_scalar_znx_mut` has exactly one occurrence: its definition at `poulpy-hal/src/layouts/vec_znx.rs:235`. There are zero external callers, so the PR 2 deletion is free.

`ZnxViewMut::at_mut` / `raw_mut` reach typed storage safely. Backend crates use them legitimately inside kernels; the audit surface is the 54 call lines in scheme crates (`poulpy-core`, `poulpy-ckks`, `poulpy-bin-fhe`), concentrated in test suites (noise/parity/blind-rotation harnesses, 38 lines) with the production remainder in layouts constructors (`glwe_secret`, `ggsw`, `gglwe`, compressed forms), `poulpy-ckks/src/layouts/ship/keyset.rs`, and `poulpy-bin-fhe` LUT/blind-rotation utilities. Reproduce with `grep -rn '\.at_mut(\|\.raw_mut(' poulpy-core/src poulpy-ckks/src poulpy-bin-fhe/src`. Under spec §7.1 these routes become `Raw`/`Unwritten`-only; each production site must migrate to a full-write builder or a `Raw` transaction.

`data` / `data_mut` accessors on `ZnxInfos` implementors (`znx_base.rs:114,119`) and the per-layout mirrors are part of the same boundary and follow the same rule.

### 3.4 Unchecked constructors and raw ingestion

| Constructor | Lines | Disposition |
|---|---|---|
| `from_data(` | 218 | raw ingestion; becomes `Raw`-producing (spec §7.2); most sites are layout plumbing and backend tests |
| `from_bytes(` | 39 | same rule |
| `from_data_like` | 5 | state-forwarding loophole; closed in PR 2 |
| `map_data_mut` | 3 | state-forwarding loophole; closed in PR 2 |
| `from_data_with_state` | 8 | crate-private to `poulpy-hal`; callers must remain the normalize family and OEP |
| `relabel_unchecked` | 7 | crate-private relabel primitive; sole callers are the normalize family and `SetNormalizationState` |

### 3.5 `ReaderFrom` implementations

3 in `poulpy-hal`, 23 in `poulpy-core`, 4 in `poulpy-bin-fhe`. All currently overwrite storage while keeping the receiver's arithmetic state; under spec §7 (PR 7) readers construct `Raw` instead. Reproduce with `grep -rln 'impl.*ReaderFrom' poulpy-*/src`.

### 3.6 Scratch takes

35 distinct `take_*` methods across `poulpy-hal`, `poulpy-core`, `poulpy-ckks` (list reproducible with `grep -rhn 'fn take_[a-z_]*' -o poulpy-*/src | sort -u`). All become `Unwritten` takes in the new model (spec §8.1). Two state-forging takes remain and are on the deny-list ratchet: `take_unnormalized_ckks_ciphertext_scratch` and `take_unnormalized_ckks_ciphertext_like_scratch` (`take_unnormalized_vec_znx_scratch` was already removed on this branch).

### 3.7 Metadata setters

`set_k` / `set_base2k` on typed roots, replaced in PR 7 by module conversions or immutable construction: `poulpy-core/src/layouts/glwe_plaintext.rs`, `lwe.rs`, `lwe_plaintext.rs`, `glwe_tensor.rs`, `glwe.rs` (`set_base2k`). No setters exist in `poulpy-hal` or `poulpy-ckks`.

### 3.8 Padding-sensitive consumers (`msb_mask_bottom_limb`, 25 call lines)

These are today's defensive masks; under spec §6.3 each becomes either a typed `Canonical` requirement or a documented mask-on-read compatibility path.

- `poulpy-core/src/default/operations/glwe.rs` (10 lines): add/sub with mismatched `k`; candidates for `Canonical` input bounds.
- `poulpy-core/src/default/linear_transformation/` `inner_product.rs`, `baby_steps.rs`, `prepare.rs` (3 sites): plaintext preparation; candidates for canonical-on-construction.
- `poulpy-core/src/test_suite/noise/linear_transformation.rs` (1 site): test-side mirror of the above.
- `poulpy-ckks/src/layouts/ship/keyset.rs`, `poulpy-ckks/src/default/ship/masking.rs` (2 sites): SHIP masking; precision-sensitive, candidates for `Canonical` requirements.
- `poulpy-cpu-avx512/src/core_impl.rs` (3 sites): fused tensoring kernel; stays a kernel-internal mask, documented under the §9.1 capability audit.

### 3.9 Working-width narrowing (spec §6.6)

Coefficient-domain narrowing exists in exactly one primitive: `vec_znx_backend_mut_with_size` (`poulpy-hal/src/layouts/vec_znx.rs:795`), which shrinks the view's `size` field over unchanged storage. Its only callers are the crate-private `VecZnxShape::with_size` and the test-suite helper `vec_znx_backend_mut_sized` (`poulpy-hal/src/test_suite/mod.rs:64`).

DFT-domain narrowing (`VecZnxDft::with_size_mut`, used by `poulpy-core/src/default/keyswitching/glwe.rs:274` and `poulpy-core/src/default/external_product/glwe.rs:117` to cap the compute size) carries no coefficient state and is out of scope for the `N`/`C` axes; converting back to coefficients declares a fresh state (spec §8.3).

Confirmation required by §6.6: narrowing today only changes which limbs an operation touches, never the bytes of untouched limbs, so adding state bounds leaves runtime behavior byte-identical. The narrowed-view rule (conservatively `NonCanonical` unless the bottom live limb is preserved, normalization marker preserved) applies only to the single coefficient-domain primitive above. Status: confirmed for the two known caller families by inspection; PR 1 must re-verify byte-identical output against the §5 baselines after the state bounds land.

## 4. Baselines

**Correctness.** At the baseline snapshot: `cargo fmt --all --check` clean; `cargo clippy --workspace --all-targets -- -D warnings` clean on the default, CI-feature, AVX, and AVX-512 lanes; workspace tests on the CI feature lane: 1072 passed, 0 failed; `cargo check -p poulpy-cpu-arm --lib --features enable-rayon,enable-ckks --target aarch64-unknown-linux-gnu` clean.

**Performance, scratch-size, and binary-size.** To be captured on dedicated benchmark hardware before PR 1 merges (this development machine cannot run the heavy suites). Commands: `cargo bench -p poulpy-bench` filtered to the `normalize` and keyswitch groups for the §12.1 gates; record `*_tmp_bytes` outputs for the standard parameter presets for the scratch baseline; `cargo build --release` artifact sizes for the §12.3 gate. Record results in this section when captured; PR 1 and later must not merge without them.

## 5. Deny-list ratchet

`scripts/normalization_denylist.sh`, wired into the CI portable job, fails when any bypass pattern exceeds its recorded baseline and reminds the author to lower the baseline when a count drops. Current baselines:

| Pattern | Baseline |
|---|---|
| `into_unnormalized` | 184 |
| `.normalize(` | 23 |
| `set_normalized(` | 9 |
| `set_unnormalized(` | 6 |
| `from_data_like` | 5 |
| `map_data_mut` | 3 |
| `as_scalar_znx_mut` | 1 |
| `take_unnormalized_` | 3 |
| `relabel_unchecked` | 7 |
| `from_data_with_state` | 8 |

PR 8 turns the surviving entries into hard zero-or-frozen assertions.

## 6. Exit-gate checklist

- [ ] Reviewers agree on the normalized interval `[-2^(b-1), 2^(b-1))` and its asymmetry (§1).
- [ ] Reviewers agree on the bottom-limb orientation and that `represented_k > 0` is a construction invariant (§1).
- [ ] Reviewers agree on the meaning of `represented_k` for every root layout (`VecZnx`, `GLWE`, plaintexts, keys, compressed forms).
- [ ] Reviewers agree on `P_p` including the aligned case `p = 0` (§1).
- [ ] Reviewers agree that §2 is the exact list of normalization APIs that remain source-shaped, and that the receiver forms are removed.
- [ ] The §3.9 narrowing confirmation stands, or the design is amended first.
- [ ] The §4 performance baselines are captured before PR 1 merges.
